from __future__ import annotations

import csv
import gc
import os
import re
import shutil
from pathlib import Path
from typing import Dict

import torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

TASK = os.environ.get("AM_TASK", "binary")
os.environ["AM_TASK"] = TASK

from tasks.AM_defect_classification.task_data_loader import TaskDataLoader
from tasks.AM_defect_classification.model_loader import ModelLoader
from tasks.AM_defect_classification.data_preprocessor import DataPreprocessor
from tasks.AM_defect_classification.model_finetuner import ModelFinetuner
from tasks.AM_defect_classification.evaluater import Evaluator


EXPERIMENTS = ["Exp_OOD_5", "Exp_OOD_6", "Exp_OOD_7"]

METHOD = os.environ.get("AM_METHOD", "C4")

_METHODS = {
    "C4": ("Llama-3.1-8B-Instruct",   "AM_C4_clm_binary"),
    "C6": ("Qwen2.5-14B-Instruct",    "AM_C6_clm_binary"),
}
MODEL_KEY, _RESULTS_SUBDIR = _METHODS[METHOD]

USE_4BIT = True
LORA_R = 128
LORA_ALPHA = 128
LORA_DROPOUT = 0.1
BIAS = "none"
TASK_TYPE = "CAUSAL_LM"
LR = 2e-4
BATCH_SIZE = 4
EPOCHS = 16
TARGET_MODULES = "all-linear"

MAX_LEN = 512
RESET_DIR = False

BASE_RESULTS = Path("./results_AM") / _RESULTS_SUBDIR
SUMMARY_CSV = BASE_RESULTS / "all_experiments_summary_clm.csv"

MODEL_NAME_VERSION = {
    "Llama-3.1-8B-Instruct": {
        "model_name": "Llama-3.1-8B-Instruct",
        "model_version": "8b-Instruct",
        "model_root_path": os.environ.get("AM_LLAMA_31_INSTRUCT_PATH", "meta-llama/Llama-3.1-8B-Instruct"),
    },
    "Qwen2.5-14B-Instruct": {
        "model_name": "Qwen2.5-14B-Instruct",
        "model_version": "14B-Instruct",
        "model_root_path": os.environ.get("AM_QWEN_25_INSTRUCT_PATH", "Qwen/Qwen2.5-14B-Instruct"),
    },
}


def _ckpt_sort_key(p: Path):
    m = re.search(r"(\d+)(?!.*\d)", p.name)
    return (0, int(m.group())) if m else (1, p.name.lower())


def _model_tag() -> str:
    cfg = MODEL_NAME_VERSION[MODEL_KEY]
    return (
        cfg["model_name"].split("-")[0] + "-" + cfg["model_version"]
        + f"_lora-r{LORA_R}-a{LORA_ALPHA}"
    )


def run_one_experiment_clm(experiment_name: str) -> tuple[float, float]:
    """Train (if needed) + evaluate one (experiment) CLM trial.

    Returns (macro_f1, accuracy).
    """

    print(
        "\n=============================="
        f"\n [{METHOD} / CLM-binary] {experiment_name}"
        "\n=============================="
    )

    out_dir = BASE_RESULTS / experiment_name / _model_tag()
    existing_ckpts = []
    if out_dir.exists():
        existing_ckpts = [
            d for d in out_dir.iterdir()
            if d.is_dir() and d.name.lower().startswith("checkpoint")
        ]
    skip_training = bool(existing_ckpts) and not RESET_DIR

    loader = TaskDataLoader(experiment_name=experiment_name)
    test_ds = loader.load_test()
    labels, label2id, id2label = loader.get_labels()

    cfg = MODEL_NAME_VERSION[MODEL_KEY]

    if skip_training:
        print(
            f"Found {len(existing_ckpts)} checkpoints in {out_dir}; "
            f"skipping training, running eval only."
        )
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_root_path"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prep = DataPreprocessor()
        test_fmt = prep.preprocess_data(
            tokenizer, test_ds, is_train=False, max_length=MAX_LEN,
        )
    else:
        train_ds = loader.load_train()
        val_ds = loader.load_val()

        model_loader = ModelLoader(accelerator=None, load_in_4bit=USE_4BIT)
        model, tokenizer = model_loader.load_model_from_path_name_version(
            cfg["model_root_path"], cfg["model_name"], cfg["model_version"]
        )

        prep = DataPreprocessor()
        train_fmt = prep.preprocess_data(tokenizer, train_ds, is_train=True,  max_length=MAX_LEN)
        val_fmt   = prep.preprocess_data(tokenizer, val_ds,   is_train=True,  max_length=MAX_LEN)
        test_fmt  = prep.preprocess_data(tokenizer, test_ds, is_train=False, max_length=MAX_LEN)

        out_dir.parent.mkdir(parents=True, exist_ok=True)
        if RESET_DIR and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        finetuner = ModelFinetuner()
        finetuner.fine_tune(
            model, tokenizer, train_fmt, val_fmt,
            LORA_R, LORA_ALPHA, LORA_DROPOUT, BIAS, TASK_TYPE,
            BATCH_SIZE, out_dir, EPOCHS,
            target_modules=TARGET_MODULES, learning_rate=LR,
            accelerator=None,
        )
        del model
        del finetuner
        torch.cuda.empty_cache()
        gc.collect()

    best_f1 = float("-inf")
    best_acc = 0.0
    best_ckpt = None

    evaluator = Evaluator(accelerator=None)

    ckpt_dirs = [
        d for d in out_dir.iterdir()
        if d.is_dir() and d.name.lower().startswith("checkpoint")
    ]
    ckpt_dirs = sorted(set(ckpt_dirs), key=_ckpt_sort_key)
    ckpt_dirs = ckpt_dirs[len(ckpt_dirs) // 2:]

    import json
    for d in ckpt_dirs:
        metric_file = d / f"eval_metrics_{experiment_name}.json"
        if metric_file.exists():
            try:
                data = json.load(open(metric_file))
                acc = data.get("accuracy", 0.0)
                macro_f1 = data.get("macro_f1", 0.0)
            except Exception as e:
                print(f"[WARN] failed to read {metric_file}: {e}")
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

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists() or SUMMARY_CSV.stat().st_size == 0
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "experiment", "model_key", "output_dir",
                "best_checkpoint", "macro_f1", "accuracy",
            ])
        writer.writerow([
            experiment_name, MODEL_KEY, str(out_dir),
            best_ckpt or "-", f"{best_f1:.4f}", f"{best_acc:.4f}",
        ])
    print(
        f"\n{experiment_name} DONE. Best macro-F1={best_f1:.4f} @ {best_ckpt}\n"
    )
    return best_f1, best_acc


def _fmt(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "nan"


def main():
    BASE_RESULTS.mkdir(parents=True, exist_ok=True)
    all_scores: Dict[str, Dict[str, float]] = {}

    model_tag = _model_tag()
    train_stem = EXPERIMENTS[0]
    print(
        f"=== {METHOD} CLM-binary runs {EXPERIMENTS} "
        f"(train once on {train_stem}) ==="
    )

    for i, stem in enumerate(EXPERIMENTS):
        try:
            if i > 0:
                src = (BASE_RESULTS / train_stem / model_tag).resolve()
                dst = BASE_RESULTS / stem / model_tag
                if src.exists() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(src, dst)
                    print(f"[binary] reusing {train_stem} model for {stem} "
                          f"(symlink {dst} -> {src})")
            f1, acc = run_one_experiment_clm(stem)
            all_scores[stem] = {"macro_f1": f1, "accuracy": acc}
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {stem} failed: {e}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\n=== FINAL SUMMARY ({METHOD}) ===")
    for k, vals in all_scores.items():
        print(f"{k}: macro-F1={_fmt(vals['macro_f1'])}, acc={_fmt(vals['accuracy'])}")


if __name__ == "__main__":
    main()
