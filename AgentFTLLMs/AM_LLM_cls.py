from pathlib import Path
import os
import shutil
import gc
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from tasks.AM_defect_classification.task_data_loader import TaskDataLoader
from tasks.AM_defect_classification.model_loader import ModelLoader
from tasks.AM_defect_classification.data_preprocessor import DataPreprocessor
from tasks.AM_defect_classification.model_finetuner import ModelFinetuner
from tasks.AM_defect_classification.evaluater import Evaluator
import csv
from typing import Dict
import re

EXPERIMENTS = [
    "Exp_ID_1",  "Exp_OOD_1",
    "Exp_ID_2",  "Exp_OOD_2",
    "Exp_ID_3",  "Exp_OOD_3",
    "Exp_ID_4",  "Exp_OOD_4",
]

MODEL_KEY = "Llama-3.3-70B-Instruct"
DO_FINE_TUNE = False
LOAD_IN_4BIT = True 

LORA_R = 128
LORA_ALPHA = 128
LORA_DROPOUT = 0.1
BIAS = "none"
TASK_TYPE = "CAUSAL_LM" 
LR = 2e-4
BATCH_SIZE = 4
EPOCHS = 16
TARGET_MODULES = "all-linear"

RESET_DIR = True

BASE_RESULTS = Path("./results/AM")

MODEL_NAME_VERSION = {
    "llama3.1-8b-Instruct": {
        "model_name": "llama3.1-8b-Instruct",
        "model_version": "8b-Instruct",
        "model_root_path": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "Qwen3-30B-A3B-Thinking-2507": {
        "model_name": "Qwen3-30B-A3B-Thinking-2507",
        "model_version": "30B-Thinking",
        "model_root_path": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    },
    "Qwen2.5-14B-Instruct": {
        "model_name": "Qwen2.5-14B-Instruct",
        "model_version": "14B-Instruct",
        "model_root_path": "Qwen/Qwen2.5-14B-Instruct",
    },
    "Llama-3.3-70B-Instruct": {
        "model_name": "Llama-3.3-70B-Instruct",
        "model_version": "70B-Instruct",
        "model_root_path": "meta-llama/Llama-3.3-70B-Instruct",
    }
}

SUMMARY_CSV = BASE_RESULTS / "all_experiments_summary.csv"

def _ckpt_sort_key(p):
    """
    Sort checkpoints numerically if there's a step number in the name
    (e.g., 'checkpoint-10' < 'checkpoint-100'). Falls back to name.
    """
    m = re.search(r'(\d+)(?!.*\d)', p.name) 
    return (0, int(m.group())) if m else (1, p.name.lower())


def run_one_experiment(experiment_name: str, accelerator: Accelerator) -> float:
    if accelerator.is_main_process:
        print(f"\n==============================")
        print(f" Running {experiment_name}")
        print(f"==============================")
    
    accelerator.wait_for_everyone()

    loader = TaskDataLoader(experiment_name=experiment_name)
    train_ds = loader.load_train()
    val_ds   = loader.load_val()
    test_ds  = loader.load_test()
    labels, label2id, id2label = loader.get_labels()

    m = MODEL_NAME_VERSION[MODEL_KEY]
    model_loader = ModelLoader(accelerator=accelerator, load_in_4bit=LOAD_IN_4BIT)
    model, tokenizer = model_loader.load_model_from_path_name_version(
        m["model_root_path"], m["model_name"], m["model_version"]
    )

    prep = DataPreprocessor()
    train_fmt = prep.preprocess_data(tokenizer, train_ds, is_train=True,  max_length=512)
    val_fmt   = prep.preprocess_data(tokenizer, val_ds,   is_train=True,  max_length=512)
    test_fmt  = prep.preprocess_data(
        tokenizer, 
        test_ds,  
        is_train=False, 
        max_length=512, 
        enforce_structured_output=(not DO_FINE_TUNE)
    )

    exp_dir = BASE_RESULTS / experiment_name
    
    if DO_FINE_TUNE:
        model_folder_name = m["model_name"].split('-')[0] + '-' + m["model_version"] + f"_lora-r{LORA_R}-a{LORA_ALPHA}"
    else:
        model_folder_name = m["model_name"].split('-')[0] + '-' + m["model_version"] + "_no_ft"
        
    out_dir = exp_dir / model_folder_name
    
    if accelerator.is_main_process:
        exp_dir.mkdir(parents=True, exist_ok=True)
        if RESET_DIR and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    
    accelerator.wait_for_everyone()

    best_f1 = float("-inf")
    best_acc = 0.0
    best_ckpt = None
    
    evaluator = Evaluator(accelerator=accelerator)

    if DO_FINE_TUNE:
        finetuner = ModelFinetuner()
        finetuner.fine_tune(
            model, tokenizer, train_fmt, val_fmt,
            LORA_R, LORA_ALPHA, LORA_DROPOUT, BIAS, TASK_TYPE,
            BATCH_SIZE, out_dir, EPOCHS,
            target_modules=TARGET_MODULES, learning_rate=LR,
            accelerator=accelerator
        )
        
        accelerator.wait_for_everyone()
        
        del model
        del finetuner
        torch.cuda.empty_cache()
        gc.collect()

        ckpt_dirs = []
        if accelerator.is_main_process:
            ckpt_dirs = [
                d for d in out_dir.iterdir()
                if d.is_dir() and d.name.lower().startswith("checkpoint")
            ]
            ckpt_dirs = sorted(set(ckpt_dirs), key=_ckpt_sort_key)
        
        accelerator.wait_for_everyone()
        ckpt_dirs = [
            d for d in out_dir.iterdir()
            if d.is_dir() and d.name.lower().startswith("checkpoint")
        ]
        ckpt_dirs = sorted(set(ckpt_dirs), key=_ckpt_sort_key)
        ckpt_dirs = ckpt_dirs[len(ckpt_dirs)//2:]  

        if accelerator.is_main_process:
            print(f"Found {len(ckpt_dirs)} checkpoint dirs under {out_dir}:")
            for d in ckpt_dirs:
                print(" -", d.name)

        for d in ckpt_dirs:
            if accelerator.is_main_process:
                print(f"Evaluating {d}")
            
            try:
                accelerator.wait_for_everyone()
                
                acc, macro_f1 = evaluator.run_full_eval(
                    experiment_name=experiment_name,
                    finetuned_model_dir=d,
                    test_ds=test_fmt,
                    output_dir=d,
                )
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"[WARN] Skipping {d.name} due to error: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_ckpt = d.name
                best_acc = acc

            torch.cuda.empty_cache()
            gc.collect()

        if (not ckpt_dirs) and out_dir.exists():
            if accelerator.is_main_process:
                print("No checkpoints found; evaluating base finetuned dir...")
            
            accelerator.wait_for_everyone()
            acc, macro_f1 = evaluator.run_full_eval(
                experiment_name=experiment_name,
                finetuned_model_dir=out_dir,
                test_ds=test_fmt,
                output_dir=out_dir,
            )
            best_f1 = macro_f1
            best_ckpt = out_dir.name
            best_acc = acc
            
    else:
        if accelerator.is_main_process:
            print("Skipping fine-tuning. Evaluating base model...")
        
        accelerator.wait_for_everyone()
        
        acc, macro_f1 = evaluator.run_full_eval(
             experiment_name=experiment_name,
             finetuned_model_dir=None, 
             test_ds=test_fmt,
             output_dir=out_dir,
             model=model,
             tokenizer=tokenizer
        )
        best_f1 = macro_f1
        best_acc = acc
        best_ckpt = "base_model"

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
    kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    
    if accelerator.is_main_process:
        BASE_RESULTS.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    all_scores: Dict[str, Dict[str, float]] = {}

    for stem in EXPERIMENTS:
        try:
            best_f1, best_acc = run_one_experiment(stem, accelerator) 
            all_scores[stem] = {"macro_f1": best_f1, "accuracy": best_acc}
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[ERROR] {stem} failed: {e}")
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
