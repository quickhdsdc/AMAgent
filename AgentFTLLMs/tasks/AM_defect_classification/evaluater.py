from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from accelerate import Accelerator
from accelerate.utils import gather_object

# canonical order, same as training + GPT5 eval
LABEL_ORDER = ["none", "lof", "balling", "keyhole"]
VALID_SET = set(LABEL_ORDER)

def _canon_from_int(y_int: int) -> str:
    # map 0->"none", 1->"lof", 2->"balling", 3->"keyhole"
    try:
        yi = int(y_int)
        if 0 <= yi < len(LABEL_ORDER):
            return LABEL_ORDER[yi]
    except Exception:
        pass
    return "none"

def _normalize_pred_str(s: str) -> str:
    """
    Map arbitrary string to one of the canonical labels
    ("none", "lof", "balling", "keyhole") if possible.
    """
    s_lower = s.lower().strip()

    # handle expansions / synonyms
    if "lack of fusion" in s_lower:
        return "lof"
    if "lo f" in s_lower or "lo-f" in s_lower or "lof" in s_lower:
        return "lof"

    # direct matches
    for cand in LABEL_ORDER:
        if cand in s_lower:
            return cand

    # heuristic fallbacks
    if "keyhole" in s_lower:
        return "keyhole"
    if "ball" in s_lower:  # catches "balling", "balling-induced"
        return "balling"
    if (
        "none" in s_lower
        or "no defect" in s_lower
        or "no major defect" in s_lower
        or "no significant defect" in s_lower
        or "stable window" in s_lower
        or "dense / stable" in s_lower
        or "no major porosity" in s_lower
    ):
        return "none"

    # default
    return "none"


import re

def _extract_label_robust(full_gen: str) -> str:
    """
    Robustly extract the LAST valid label from the generated text.
    Handles CoT/thinking processes by preferring the conclusion.
    """
    s_lower = full_gen.lower()
    
    # If explicit structure is used ([LABEL] ... [/LABEL]), use content before [/LABEL]
    if "[/label]" in s_lower:
        s_lower = s_lower.split("[/label]")[0]

    # Map synonyms/variations to canonical keys
    # Order matters: check longer/specific ones first if they overlap
    mapping = {
        "lack of fusion": "lof",
        "lo-f": "lof",
        "lof": "lof",
        "keyhole": "keyhole",
        "balling": "balling",
        "ball": "balling",
        "none": "none",
        "no defect": "none",
        "no major defect": "none",
        "stable": "none",
    }
    
    # 1. Find all occurrences of known labels with their positions
    hits = []
    for snippet, canon in mapping.items():
        # Iterate over all matches of this snippet
        for m in re.finditer(re.escape(snippet), s_lower):
            hits.append((m.start(), canon))
    
    # 2. Sort by position (descending) to find the LAST one
    hits.sort(key=lambda x: x[0], reverse=True)
    
    if hits:
        return hits[0][1]
    
    return "none"


class Evaluator:
    """
    - Load LoRA adapter checkpoint, merge into base weights
    - Run generation on test prompts
    - Save per-row predictions
    - Compute accuracy and macro-F1 (print & save)
    - Return (accuracy, macro_f1)
    """

    def __init__(self, accelerator: Accelerator = None):
        self.accelerator = accelerator
        if not self.accelerator or self.accelerator.is_main_process:
            print("Evaluating the model...")

    def merge_model(self, finetuned_model_dir: Path):
        """
        Load LoRA adapter (PEFT), merge into base model weights, unload adapters.
        Return a plain causal LM model ready for inference + the tokenizer.
        """
        finetuned_model_dir = Path(finetuned_model_dir)

        tokenizer = AutoTokenizer.from_pretrained(str(finetuned_model_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # Switch to left padding for generation

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Handle device map for DDP
        device_map = None
        if self.accelerator:
            device_map = {"": self.accelerator.process_index}
        else:
            device_map = "auto"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

        model = AutoPeftModelForCausalLM.from_pretrained(
            str(finetuned_model_dir),
            torch_dtype=compute_dtype,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            return_dict=True,
            device_map=device_map,
        )

        if  getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    def predict(
        self,
        test_ds,
        model,
        tokenizer,
        output_dir: Path,
        experiment_name: str,
        max_new_tokens: int = 30,
        temperature: float = 0.1,
    ) -> pd.DataFrame:
        """
        For each example in test_ds:
        - take sample["text"] (prompt-only, no gold label block),
        - generate model output,
        - parse predicted label,
        - store GT + pred.
        """
        # Ensure left-padding for generation
        tokenizer.padding_side = "left"

        output_dir = Path(output_dir)
        
        # Prepare valid device for pipeline
        device = None
        if self.accelerator:
            device = self.accelerator.device
        
        # We'll run the pipeline distributedly using split_between_processes
        # First, prepare the data as a list of dicts that we can iterate
        # Assuming test_ds is indexable and has "text" and "label"
        
        # Convert to list if it's a dataset
        if not isinstance(test_ds, list):
             # Try to convert to list of dicts if it's a Dataset
             try:
                 data_list = [
                     {"idx": i, "text": test_ds[i]["text"], "label": test_ds[i]["label"]} 
                     for i in range(len(test_ds))
                 ]
             except:
                 # If it breaks, assume it's already a list
                 data_list = list(test_ds)
        else:
            data_list = test_ds
            
        local_results = []
        
        # Setup context for distributed splitting
        # If accelerator is None, this context might fail if we don't handle it
        # But loop logic handles accelerator=None by just iterating full list usually if we simulate it, 
        # or we just guard it.
        
        if self.accelerator:
             ctx_mgr = self.accelerator.split_between_processes(data_list)
        else:
             # Fake context for single process
             from contextlib import nullcontext
             ctx_mgr = nullcontext(data_list)

        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine rank
        rank = self.accelerator.process_index if self.accelerator else 0
        
        # Temp file for this rank
        temp_file = output_dir / f"temp_pred_{experiment_name}_rank_{rank}.csv"
        
        # Check resumption
        processed_indices = set()
        if temp_file.exists():
            try:
                existing_df = pd.read_csv(temp_file)
                if "row_idx" in existing_df.columns:
                    processed_indices = set(existing_df["row_idx"].tolist())
                print(f"[Rank {rank}] Resuming: found {len(processed_indices)} processed.")
            except:
                pass

        # Batch size for generation (per GPU)
        BATCH_SIZE = 8

        with ctx_mgr as batch_data:
            num_samples = len(batch_data)
            
            for i in tqdm(range(0, num_samples, BATCH_SIZE), desc=f"Eval Rank {rank}", disable=not (not self.accelerator or self.accelerator.is_local_main_process)):
                batch_items = batch_data[i : i + BATCH_SIZE]

                # Filter items that are already done
                to_process = [item for item in batch_items if item.get("idx", -1) not in processed_indices]
                
                if not to_process:
                    continue 

                prompts = [item["text"] for item in to_process]
                
                # Tokenize
                inputs = tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=1024
                ).to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=(temperature > 0),
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                # Decode
                input_len = inputs.input_ids.shape[1]
                new_tokens = outputs[:, input_len:]
                gen_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                
                new_rows = []
                for idx, item in enumerate(to_process):
                    full_gen = gen_texts[idx]
                    prompt_tail = item["text"][-100:]
                    gt = item["label"]
                    
                    # Robust Parse
                    pred_canon = _extract_label_robust(full_gen)
                    
                    # Try to get canonical GT
                    try:
                        gt_int = int(gt)
                        gt_canon = _canon_from_int(gt_int)
                    except:
                        gt_int = -1
                        gt_canon = str(gt)

                    row = {
                        "row_idx": item.get("idx", -1),
                        "prompt": prompt_tail,
                        "generated": full_gen,
                        "pred_label": pred_canon,
                        "gt_label": gt_canon,
                        "gt_int": gt_int,
                    }
                    new_rows.append(row)
                
                # SAVE IMMEDIATELY
                df_batch = pd.DataFrame(new_rows)
                write_header = not temp_file.exists()
                df_batch.to_csv(temp_file, mode='a', header=write_header, index=False)
            
            torch.cuda.empty_cache()

        # Gather (Synchronization point)
        if self.accelerator:
            self.accelerator.wait_for_everyone()

        # Only main process needs to save/merge
        final_df = pd.DataFrame()
        
        if not self.accelerator or self.accelerator.is_main_process:
            print("Merging temporary files...")
            all_dfs = []
            
            # Pattern search
            saved_temps = list(output_dir.glob(f"temp_pred_{experiment_name}_rank_*.csv"))
            for f in saved_temps:
                try:
                    d = pd.read_csv(f)
                    all_dfs.append(d)
                except Exception as e:
                    print(f"Warning: could not read {f}: {e}")
            
            if all_dfs:
                final_df = pd.concat(all_dfs, ignore_index=True)
                if "row_idx" in final_df.columns:
                    final_df = final_df.sort_values("row_idx")

                eval_file = output_dir / f"eval_pred_{experiment_name}.csv"
                if eval_file.exists(): 
                     pass # Overwrite or append? We overwrite the FINAL file because it's a merge of all temps
                     
                final_df.to_csv(eval_file, index=False)
                print(f"Saved merged predictions to {eval_file}")
                
                # Cleanup temps
                for f in saved_temps:
                     f.unlink()
            
            else:
                print("No temp files found to merge.")

        return final_df

    def compute_metrics_and_save(
        self,
        df_pred: pd.DataFrame,
        output_dir: Path,
        experiment_name: str,
    ) -> Tuple[float, float]:
        """
        Compute accuracy and macro-F1 on the 4 defect classes.
        - map both gt_label and pred_label (strings) into 0..3 via LABEL_ORDER
        - compute accuracy + macro-F1
        - dump report + confusion matrix
        - return (accuracy, macro_f1)
        """
        # Run only on main process
        if self.accelerator and not self.accelerator.is_main_process:
            return 0.0, 0.0

        def to_int(label_str: str) -> int:
            label_str = str(label_str).lower().strip()
            if label_str in VALID_SET:
                return LABEL_ORDER.index(label_str)
            # fallback: treat unknown as "none" (class 0)
            return 0

        y_true_ints = [to_int(s) for s in df_pred["gt_label"].tolist()]
        y_pred_ints = [to_int(s) for s in df_pred["pred_label"].tolist()]

        # metrics
        macro_f1 = f1_score(y_true_ints, y_pred_ints, average="macro")
        acc = accuracy_score(y_true_ints, y_pred_ints)

        print(f"\n=== {experiment_name} Evaluation ===")
        print(f"accuracy: {acc:.4f}")
        print(f"macro-F1: {macro_f1:.4f}")

        # classification report per class label (also includes 'accuracy' key)
        class_report = classification_report(
            y_true_ints,
            y_pred_ints,
            target_names=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        )

        # confusion matrix
        cm = confusion_matrix(
            y_true_ints,
            y_pred_ints,
            labels=list(range(len(LABEL_ORDER))),
        )
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{lbl}" for lbl in LABEL_ORDER],
            columns=[f"pred_{lbl}" for lbl in LABEL_ORDER],
        )

        # save metrics to disk
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / f"eval_metrics_{experiment_name}.json"
        cm_path = output_dir / f"eval_confusion_{experiment_name}.csv"

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": float(acc),
                    "macro_f1": float(macro_f1),
                    "report": class_report,  # sklearn report also includes 'accuracy'
                },
                f,
                indent=2,
            )
        cm_df.to_csv(cm_path, index=True)

        print("Confusion matrix:")
        print(cm_df)

        return acc, macro_f1

    def run_full_eval(
        self,
        experiment_name: str,
        finetuned_model_dir: Path,
        test_ds,   # HF Dataset already preprocessed with is_train=True (prompt only)
        output_dir: Path,
        model=None,
        tokenizer=None,
    ) -> Tuple[float, float]:
        """
        High-level helper:
        1. load merged model (if model/tokenizer not provided)
        2. generate predictions
        3. compute accuracy + macro-F1 and save artifacts
        Returns: (accuracy, macro_f1)
        """
        
        output_dir = Path(output_dir)
        if not self.accelerator or self.accelerator.is_main_process:
             output_dir.mkdir(parents=True, exist_ok=True)
             
        # Wait for dir creation? 
        if self.accelerator:
            self.accelerator.wait_for_everyone()

        if model is None or tokenizer is None:
             model, tokenizer = self.merge_model(finetuned_model_dir)

        df_pred = self.predict(
            test_ds=test_ds,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            experiment_name=experiment_name,
            max_new_tokens=1024,
            temperature=0.1,
        )

        acc, macro_f1 = self.compute_metrics_and_save(
            df_pred=df_pred,
            output_dir=output_dir,
            experiment_name=experiment_name,
        )
        
        # Cleanup if we loaded the model locally
        if model is not None:
             del model
        if tokenizer is not None:
             del tokenizer
        import gc
        torch.cuda.empty_cache()
        gc.collect()

        return acc, macro_f1



