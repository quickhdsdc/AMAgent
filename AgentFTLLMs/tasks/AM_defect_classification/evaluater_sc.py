import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from accelerate import Accelerator
from accelerate.utils import gather_object

LABEL_ORDER = ["none", "lof", "balling", "keyhole"]
_NUM_LABELS = len(LABEL_ORDER)


def _maps():
    label2id = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
    id2label = {i: lbl for i, lbl in enumerate(LABEL_ORDER)}
    return label2id, id2label


def gpu_supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


class EvaluatorSeqCls:
    def __init__(self, max_length: int = 256, batch_size: int = 32, accelerator: Accelerator = None, force_single_process: bool = False):
        self.max_length = max_length
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.force_single_process = force_single_process

    def load_model(self, finetuned_model_dir: Path):
        finetuned_model_dir = Path(finetuned_model_dir)

        tokenizer = AutoTokenizer.from_pretrained(str(finetuned_model_dir))
        if tokenizer.pad_token is None:
            # for Llama/ChatGLM-style tokenizers
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Store (optional reference)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # ---- 4-bit config that actually gets used ----
        compute_dtype = torch.bfloat16 if gpu_supports_bf16() else torch.float16
        
        # Determine device map for DDP
        device_map = None
        if self.accelerator:
             device_map = {"": self.accelerator.process_index}
        else:
             device_map = "auto"
             
        # FORCE DEVICE IF single process requested but accelerator is present
        # (This is implicitly handled by device_map logic above if accelerator is passed)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,  # avoid bf16 if unsupported
        )

        if self.accelerator:
            print(f"[Rank {self.accelerator.process_index}] Loading model from {finetuned_model_dir}...")
        
        # IMPORTANT:
        # - Do NOT pass torch_dtype=torch.float32 here (it defeats quantization).
        # - Use device_map="auto" to shard/offload if needed.
        model = AutoPeftModelForSequenceClassification.from_pretrained(
            str(finetuned_model_dir),
            quantization_config=bnb_config,
            device_map=device_map,
            low_cpu_mem_usage=True,
            return_dict=True,
            num_labels=_NUM_LABELS,
        )
        model.eval()
        model.config.use_cache = False
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

        # label maps (optional but nice to have)
        label2id = {l: i for i, l in enumerate(LABEL_ORDER)}
        id2label = {i: l for l, i in label2id.items()}
        model.config.num_labels = len(label2id)
        model.config.label2id = label2id
        model.config.id2label = id2label
        
        if self.accelerator:
             print(f"[Rank {self.accelerator.process_index}] Model loaded.")
        
        return model, tokenizer, label2id, id2label

    @torch.no_grad()
    def predict(
        self,
        test_ds,
        model,
        tokenizer,
        output_dir: Path,
        experiment_name: str
    ) -> pd.DataFrame:
        output_dir = Path(output_dir)
        
        # Prepare context for splitting
        if not isinstance(test_ds, list):
             # Try to convert to list if possible
             try:
                 all_items = [test_ds[i] for i in range(len(test_ds))]
             except:
                 all_items = list(test_ds)
        else:
             all_items = test_ds
        
        local_results = []
        
        # SPLIT LOGIC
        if self.accelerator and not self.force_single_process:
             context = self.accelerator.split_between_processes(all_items)
             rank = self.accelerator.process_index
        else:
             from contextlib import nullcontext
             context = nullcontext(all_items)
             # If accelerator is present but forced single process, use its rank, otherwise 0
             rank = self.accelerator.process_index if self.accelerator else 0

        # Temp file for this rank
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_file = output_dir / f"temp_pred_sc_{experiment_name}_rank_{rank}.csv"
        # Always clean temp file on start if forced single process (treating as fresh run)
        if temp_file.exists():
             try:
                 temp_file.unlink()
             except:
                 pass

        # We will write in chunks
             
        with context as batch_ds_list:
             # Process this shard
             # Batch the shard locally
             bs = self.batch_size
             n = len(batch_ds_list)
             if self.accelerator:
                 print(f"[Rank {self.accelerator.process_index}] Starting predict loop for {n} items...")
             
             # if temp file exists and we want to resume, we'd need global indices. 
             # For now, let's just process.
             
             for start in tqdm(range(0, n, bs), desc=f"Eval Rank {rank}", disable=not (not self.accelerator or self.accelerator.is_local_main_process)):
                end = min(start + bs, n)
                batch_items = batch_ds_list[start:end]
                
                texts = [item["text"] for item in batch_items]
                mats = [item.get("material", "") for item in batch_items]
                gts = [int(item.get("label", -1)) for item in batch_items]

                enc = tokenizer(
                    texts,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
                
                # Move to correct device
                device = model.device
                for k in enc:
                    enc[k] = enc[k].to(device)

                amp_dtype = torch.bfloat16 if gpu_supports_bf16() else torch.float16
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                    outputs = model(**enc)
                    logits = outputs.logits

                preds = torch.argmax(logits, dim=-1).tolist()

                chunk_rows = []
                for i in range(len(texts)):
                    gt_int = gts[i]
                    gt_lbl = LABEL_ORDER[gt_int] if 0 <= gt_int < _NUM_LABELS else "none"
                    pred_lbl = LABEL_ORDER[preds[i]]
                    chunk_rows.append({
                        # "row_idx": ??? hard to get global idx easily without passed-in info
                        "material": mats[i],
                        "text": texts[i],
                        "gt_int": gt_int,
                        "gt_label": gt_lbl,
                        "pred_int": preds[i],
                        "pred_label": pred_lbl,
                    })

                del enc, outputs, logits
                
                # Append to temp file
                df_chunk = pd.DataFrame(chunk_rows)
                # header only if file didn't exist (or we just started and cleaned it? No, if we append, checks file)
                # To be safe, let's say if start==0 and not resumption, we write header.
                # Actually, simpler: check if file exists.
                write_header = not temp_file.exists()
                df_chunk.to_csv(temp_file, mode='a', header=write_header, index=False)
             
        # Wait for all
        if self.accelerator and not self.force_single_process:
             print(f"[Rank {self.accelerator.process_index}] Waiting for everyone after shard processing...")
             self.accelerator.wait_for_everyone()
             print(f"[Rank {self.accelerator.process_index}] Passed wait_for_everyone.")
             
        # Merge
        final_df = pd.DataFrame()
        if not self.accelerator or self.accelerator.is_main_process:

            all_dfs = []
            saved_temps = list(output_dir.glob(f"temp_pred_sc_{experiment_name}_rank_*.csv"))
            for f in saved_temps:
                try:
                    d = pd.read_csv(f)
                    all_dfs.append(d)
                except Exception as e:
                    print(f"Warning: could not read {f}: {e}")
            
            if all_dfs:
                final_df = pd.concat(all_dfs, ignore_index=True)
                
                out_csv = output_dir / f"eval_pred_{experiment_name}.csv"
                final_df.to_csv(out_csv, index=False)

                
                # Cleanup
                for f in saved_temps:
                    f.unlink()
                pass
                
        return final_df

    def compute_metrics_and_save(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        experiment_name: str
    ) -> Tuple[float, float]:
        
        # Run only on main process
        if self.accelerator and not self.accelerator.is_main_process:
             return 0.0, 0.0
             
        # guard against missing/NaN
        y_true = [int(x) if pd.notna(x) else -1 for x in df["gt_int"].tolist()]
        y_pred = [int(x) if pd.notna(x) else -1 for x in df["pred_int"].tolist()]

        # filter to valid classes 0..3 (should already be)
        mask = [(0 <= yt < _NUM_LABELS) and (0 <= yp < _NUM_LABELS) for yt, yp in zip(y_true, y_pred)]
        y_true = [yt for yt, m in zip(y_true, mask) if m]
        y_pred = [yp for yp, m in zip(y_pred, mask) if m]

        # metrics
        acc = accuracy_score(y_true, y_pred) if len(y_true) else float("nan")
        macro_f1 = f1_score(y_true, y_pred, average="macro") if len(y_true) else float("nan")

        report = classification_report(
            y_true, y_pred,
            target_names=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        ) if len(y_true) else {}

        cm = confusion_matrix(
            y_true, y_pred,
            labels=list(range(_NUM_LABELS)),
        ) if len(y_true) else np.zeros((_NUM_LABELS, _NUM_LABELS), dtype=int)

        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in LABEL_ORDER],
            columns=[f"pred_{l}" for l in LABEL_ORDER],
        )

        output_dir = Path(output_dir)
        (output_dir / f"eval_metrics_{experiment_name}.json").write_text(
            json.dumps(
                {
                    "accuracy": float(acc),
                    "macro_f1": float(macro_f1),
                    "report": report,  # note: sklearn's report also contains an "accuracy" field
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        cm_df.to_csv(output_dir / f"eval_confusion_{experiment_name}.csv")

        print(f"\n=== {experiment_name} ===")
        print(f"accuracy:  {acc:.4f}" if np.isfinite(acc) else "accuracy: NaN")
        print(f"macro-F1:  {macro_f1:.4f}" if np.isfinite(macro_f1) else "macro-F1: NaN")
        print("Confusion matrix:\n", cm_df)
        return acc, macro_f1

    def run_full_eval(
        self,
        experiment_name: str,
        finetuned_model_dir: Path,
        test_ds,
        output_dir: Path
    ) -> Tuple[float, float]:
        
        output_dir = Path(output_dir)
        if not self.accelerator or self.accelerator.is_main_process:
             output_dir.mkdir(parents=True, exist_ok=True)
             
        if self.accelerator:
             self.accelerator.wait_for_everyone()
        
        model, tokenizer, _, _ = self.load_model(finetuned_model_dir)
        df_pred = self.predict(test_ds, model, tokenizer, output_dir, experiment_name)
        acc, macro_f1 = self.compute_metrics_and_save(df_pred, output_dir, experiment_name)
        
        # Cleanup
        del model
        del tokenizer
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        return acc, macro_f1





