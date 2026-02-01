import os, gc, torch, csv
from pathlib import Path
import shutil
import shutil
from accelerate import Accelerator
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"


from tasks.AM_defect_classification.task_data_loader import TaskDataLoader
from tasks.AM_defect_classification.model_loader_sc import ModelLoader as ModelLoaderSeqCls
from tasks.AM_defect_classification.data_preprocessor_sc import DataPreprocessor as DataPreprocessorSeqCls
from tasks.AM_defect_classification.model_finetuner_sc import ModelFinetuner
from tasks.AM_defect_classification.evaluater_sc import EvaluatorSeqCls
from typing import Dict
# -------------------- CONFIG --------------------
EXPERIMENTS = [
    "Exp_ID_1", "Exp_OOD_1",
    "Exp_ID_2", "Exp_OOD_2",
    "Exp_ID_3", "Exp_OOD_3",
    "Exp_ID_4", "Exp_OOD_4",
]

MODEL_KEY = "llama3.1-8b"   # choose: "llama3.1-8b", "llama3.1-8b-Instruct", "llama2-13b-chat", "llama2-13b"

USE_4BIT = True

# LoRA / training
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
BIAS = "none"
TASK_TYPE = "SEQ_CLS"     # sequence classification
LR = 2e-4
BATCH_SIZE = 2
EPOCHS = 16
TARGET_MODULES = "all-linear"

MAX_LEN = 256
RESET_DIR = True          # True = wipe experiment dir before training
EVAL_BEST_OF_CHECKPOINTS = False  # seq-cls often keeps only final; set True to scan checkpoints

BASE_RESULTS = Path("./results/AM")

SUMMARY_CSV = BASE_RESULTS / "all_experiments_summary_sc.csv"

MODEL_NAME_VERSION = {
    "llama3.1-8b": {
        "model_name": "llama3.1-8b",
        "model_version": "8b",
        "model_root_path": "meta-llama/Llama-3.1-8B",
    },
    "Qwen2.5-14B": {
        "model_name": "Qwen2.5-14B",
        "model_version": "14B",
        "model_root_path": "Qwen/Qwen2.5-14B",
    },
}
# ------------------------------------------------

import re, gc  # make sure these are imported

def _ckpt_sort_key(p):
    """
    Sort checkpoints numerically if there's a step number in the name
    (e.g., 'checkpoint-10' < 'checkpoint-100'). Falls back to name.
    """
    m = re.search(r'(\d+)(?!.*\d)', p.name)  # last number in the name
    return (0, int(m.group())) if m else (1, p.name.lower())



def run_one_experiment_sc(experiment_name: str, accelerator: Accelerator) -> float:
    if accelerator.is_main_process:
        print(f"\n==============================")
        print(f" [SC] Running {experiment_name}")
        print(f"==============================")

    # 1) Data
    loader = TaskDataLoader(experiment_name=experiment_name)
    train_ds = loader.load_train()
    val_ds   = loader.load_val()
    test_ds  = loader.load_test()
    labels, label2id, id2label = loader.get_labels()  # ["none","lof","balling","keyhole"]

    # 2) Model
    cfg = MODEL_NAME_VERSION[MODEL_KEY]
    model_loader = ModelLoaderSeqCls(use_4bit=USE_4BIT, accelerator=accelerator)
    model, tokenizer = model_loader.load_model_from_path_name_version(
        model_root_path=cfg["model_root_path"],
        model_name=cfg["model_name"],
        model_version=cfg["model_version"],

        num_labels=len(labels),
        label_order=labels,
    )

    # 3) Preprocess (plain string inputs + label ints)
    prep = DataPreprocessorSeqCls()
    # Ensure shuffle is consistent or handled; for training usually shuffle is fine to differ, 
    # but distributed sampler handles splitting.
    train_fmt = prep.preprocess_data(tokenizer, train_ds, max_length=MAX_LEN, shuffle=True)
    val_fmt   = prep.preprocess_data(tokenizer, val_ds,   max_length=MAX_LEN, shuffle=False)
    test_fmt  = prep.preprocess_data(tokenizer, test_ds,  max_length=MAX_LEN, shuffle=False)

    # 4) Output dirs
    exp_dir = BASE_RESULTS / experiment_name
    model_tag = MODEL_KEY.split('-')[0] + "_sc" + f"_lora-r{LORA_R}-a{LORA_ALPHA}"
    out_dir = exp_dir / model_tag
    
    if accelerator.is_main_process:
        exp_dir.mkdir(parents=True, exist_ok=True)
        if RESET_DIR and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    
    accelerator.wait_for_everyone()

    # 5) Train
    finetuner = ModelFinetuner()
    finetuner.fine_tune(
        model, tokenizer, train_fmt, val_fmt,
        LORA_R, LORA_ALPHA, LORA_DROPOUT, BIAS, TASK_TYPE,
        BATCH_SIZE, out_dir, EPOCHS,
        target_modules=TARGET_MODULES, learning_rate=LR,
        accelerator=accelerator
    )

    # free train objects
    del model; del finetuner
    torch.cuda.empty_cache(); gc.collect()
    
    accelerator.wait_for_everyone()

    # 6) Evaluate
    # Run evaluation ONLY on the main process to avoid distributed hangs
    # Other ranks will wait at the barrier after this block.
    
    best_f1 = float("-inf")
    best_acc = 0.0
    best_ckpt = None

    if accelerator.is_main_process:
        print(f"[Rank 0] Starting Single-Process Evaluation...")
        
        # Instantiate evaluator WITH accelerator but FORCE SINGLE PROCESS
        # This ensures we get the correct device map but NO split logic.
        evaluator = EvaluatorSeqCls(accelerator=accelerator, batch_size=BATCH_SIZE, force_single_process=True)

        # Build checkpoint list
        ckpt_dirs = [
            d for d in out_dir.iterdir()
            if d.is_dir() and d.name.lower().startswith("checkpoint")
        ]
        ckpt_dirs = sorted(set(ckpt_dirs), key=_ckpt_sort_key)
        ckpt_dirs = ckpt_dirs[len(ckpt_dirs)//2:]  # Skip first half

        print(f"Found {len(ckpt_dirs)} checkpoint dirs under {out_dir}:")
        for d in ckpt_dirs:
            print(" -", d.name)
        
        import json

        for d in ckpt_dirs:
            metric_file = d / f"eval_metrics_{experiment_name}.json"
            
            should_skip = False
            if metric_file.exists():
                should_skip = True
                print(f"Skipping {d.name}, found {metric_file.name}")
                # Try load existing
                try:
                    with open(metric_file, "r") as f:
                        data = json.load(f)
                    acc = data.get("accuracy", 0.0)
                    macro_f1 = data.get("macro_f1", 0.0)
                except Exception as e:
                    print(f"[WARN] Failed to read metrics: {e}")
                    acc, macro_f1 = 0.0, 0.0
            else:
                print(f"Evaluating {d}")
                acc, macro_f1 = evaluator.run_full_eval(
                    experiment_name=experiment_name,
                    finetuned_model_dir=d,
                    test_ds=test_fmt,
                    output_dir=d,
                )

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_ckpt = d.name
                best_acc = acc

            torch.cuda.empty_cache()
            gc.collect()

        # Fallback if no checkpoints
        if (not ckpt_dirs) and out_dir.exists():
            print("No checkpoints found; evaluating base finetuned dir...")
            acc, macro_f1 = evaluator.run_full_eval(
                experiment_name=experiment_name,
                finetuned_model_dir=out_dir,
                test_ds=test_fmt,
                output_dir=out_dir,
            )
            best_f1 = macro_f1
            best_ckpt = out_dir.name
            best_acc = acc

    # -------------------------------------------------------
    # CRITICAL BARRIER: Ensure all ranks wait for Rank 0 to finish eval
    # -------------------------------------------------------
    accelerator.wait_for_everyone()

    # 7) Append summary row
    if accelerator.is_main_process:
        with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["experiment", "model_key", "output_dir", "best_checkpoint", "macro_f1", "accuracy"])
            writer.writerow([experiment_name, MODEL_KEY, str(out_dir), best_ckpt or "-", f"{best_f1:.4f}", f"{best_acc:.4f}"])

        print(f"\n{experiment_name} DONE. Best macro-F1={best_f1:.4f} @ {best_ckpt}\n")
    
    return best_f1, best_acc



def _fmt_num(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "nan"

def main():
    accelerator = Accelerator()
    if accelerator.is_main_process:
        BASE_RESULTS.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    all_scores: Dict[str, Dict[str, float]] = {}

    for stem in EXPERIMENTS:
        try:
            best_f1, best_acc = run_one_experiment_sc(stem, accelerator)  # now returns (f1, acc)
            all_scores[stem] = {"macro_f1": best_f1, "accuracy": best_acc}
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[ERROR] {stem} failed: {e}")
                # write a failure row to the summary with both metrics as NaN
                with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:
                        writer.writerow(["experiment", "model_key", "output_dir", "best_checkpoint", "macro_f1", "accuracy"])
                    writer.writerow([stem, MODEL_KEY, "<failed>", "<failed>", "nan", "nan"])
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    if accelerator.is_main_process:
        print("\n=== FINAL SUMMARY ===")
        for k, vals in all_scores.items():
            print(f"{k}: macro-F1={_fmt_num(vals['macro_f1'])}, acc={_fmt_num(vals['accuracy'])}")


if __name__ == "__main__":
    main()
