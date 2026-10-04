from __future__ import annotations

import csv
import gc
import json
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

from tasks.AM_defect_classification.data_preprocessor_sc_physics import (
    DataPreprocessor as DataPreprocessorSeqClsPhysics,
)
from tasks.AM_defect_classification.evaluater_sc_physics import EvaluatorSeqClsPhysics
from tasks.AM_defect_classification.model_finetuner_sc_physics import ModelFinetunerPhysics
from tasks.AM_defect_classification.model_loader_sc import ModelLoader as ModelLoaderSeqCls
from tasks.AM_defect_classification.task_data_loader import TaskDataLoader


_EXPERIMENTS_BY_TASK = {
    "defect4": [
        "Exp_ID_1", "Exp_OOD_1",
        "Exp_ID_2", "Exp_OOD_2",
        "Exp_ID_3", "Exp_OOD_3",
        "Exp_ID_4", "Exp_OOD_4",
    ],
    "binary": ["Exp_OOD_5", "Exp_OOD_6", "Exp_OOD_7"],
}
EXPERIMENTS = _EXPERIMENTS_BY_TASK[TASK]

RUN_MODE = "production"
PILOT_EXPERIMENT = "Exp_OOD_1"

METHOD = os.environ.get("AM_METHOD", "C10")

_METHODS = {
    "C5":       (False,        0.0,    "Llama-3.1-8B", "AM_C5_sc"),
    "C7":       (False,        0.0,    "Qwen2.5-14B",  "AM_C7_sc"),
    "C10":      (True,         0.0,    "Qwen2.5-14B",  "AM_C10_physics"),
    "C11":      (False,        2.0,    "Qwen2.5-14B",  "AM_C11_physics"),
    "C10+C11":  (True,         2.0,    "Qwen2.5-14B",  "AM_C10C11_ablation"),
}
USE_PHYSICS_FEATURES, METHOD_LAMBDA, MODEL_KEY, _METHOD_SUBDIR = _METHODS[METHOD]
_RESULTS_SUBDIR = _METHOD_SUBDIR + ("_binary" if TASK == "binary" else "")

LAMBDA_SWEEP_VALUES = [0.1, 0.5, 1.0, 2.0]
LAMBDA_PHYS_PRODUCTION = METHOD_LAMBDA

USE_4BIT = True

LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
BIAS = "none"
TASK_TYPE = "SEQ_CLS"
LR = 2e-4
BATCH_SIZE = 64
EPOCHS = 16
TARGET_MODULES = "all-linear"

MAX_LEN = 256
RESET_DIR = False

BASE_RESULTS = Path("./results_AM") / _RESULTS_SUBDIR
SUMMARY_CSV = BASE_RESULTS / "all_experiments_summary_sc_physics.csv"
SWEEP_SUMMARY_CSV = BASE_RESULTS / "lambda_sweep" / "_summary.csv"

MODEL_NAME_VERSION = {
    "Llama-3.1-8B": {
        "model_name": "Llama-3.1-8B",
        "model_version": "8b",
        "model_root_path": os.environ.get("AM_LLAMA_31_INSTRUCT_PATH", "meta-llama/Llama-3.1-8B-Instruct"),
    },
    "Qwen2.5-14B": {
        "model_name": "Qwen2.5-14B",
        "model_version": "14B",
        "model_root_path": os.environ.get("AM_QWEN_25_BASE_PATH", "Qwen/Qwen2.5-14B"),
    },
}


def _ckpt_sort_key(p: Path):
    m = re.search(r"(\d+)(?!.*\d)", p.name)
    return (0, int(m.group())) if m else (1, p.name.lower())


def _build_out_dir(experiment_name: str, lambda_phys: float, mode: str) -> Path:
    if mode == "lambda_sweep":
        lam_tag = f"lambda_{lambda_phys}".replace(".", "p")
        return BASE_RESULTS / "lambda_sweep" / f"{experiment_name}_{lam_tag}"
    return BASE_RESULTS / experiment_name


def run_one_experiment_physics(
    experiment_name: str,
    lambda_phys: float,
    mode: str,
) -> tuple[float, float, float]:
    """Train + evaluate one (experiment, lambda) trial.

    Returns (macro_f1, accuracy, physics_violation_rate).
    """

    print(
        "\n=============================="
        f"\n [{METHOD} / {mode}] {experiment_name} (lambda={lambda_phys})"
        "\n=============================="
    )

    base_out = _build_out_dir(experiment_name, lambda_phys, mode)
    model_tag = MODEL_KEY.split("-")[0] + "_sc_physics" + f"_lora-r{LORA_R}-a{LORA_ALPHA}"
    out_dir = base_out / model_tag

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

    if skip_training:
        print(
            f"Found {len(existing_ckpts)} checkpoints in {out_dir}; "
            f"skipping training, running eval only."
        )
        from transformers import AutoTokenizer
        cfg = MODEL_NAME_VERSION[MODEL_KEY]
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_root_path"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        prep = DataPreprocessorSeqClsPhysics(use_physics_features=USE_PHYSICS_FEATURES)
        test_fmt = prep.preprocess_data(tokenizer, test_ds, max_length=MAX_LEN, shuffle=False)
    else:
        train_ds = loader.load_train()
        val_ds = loader.load_val()

        cfg = MODEL_NAME_VERSION[MODEL_KEY]
        model_loader = ModelLoaderSeqCls(use_4bit=USE_4BIT)
        model, tokenizer = model_loader.load_model_from_path_name_version(
            model_root_path=cfg["model_root_path"],
            model_name=cfg["model_name"],
            model_version=cfg["model_version"],
            num_labels=len(labels),
            label_order=labels,
        )

        prep = DataPreprocessorSeqClsPhysics(use_physics_features=USE_PHYSICS_FEATURES)
        train_fmt = prep.preprocess_data(tokenizer, train_ds, max_length=MAX_LEN, shuffle=True)
        val_fmt = prep.preprocess_data(tokenizer, val_ds, max_length=MAX_LEN, shuffle=False)
        test_fmt = prep.preprocess_data(tokenizer, test_ds, max_length=MAX_LEN, shuffle=False)

        base_out.mkdir(parents=True, exist_ok=True)
        if RESET_DIR and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        finetuner = ModelFinetunerPhysics()
        finetuner.fine_tune(
            model, tokenizer, train_fmt, val_fmt,
            LORA_R, LORA_ALPHA, LORA_DROPOUT, BIAS, TASK_TYPE,
            BATCH_SIZE, out_dir, EPOCHS,
            target_modules=TARGET_MODULES, learning_rate=LR,
            lambda_phys=lambda_phys,
        )
        del model
        del finetuner
        torch.cuda.empty_cache()
        gc.collect()

    best_f1 = float("-inf")
    best_acc = 0.0
    best_violation = float("nan")
    best_ckpt = None

    print("Single-process eval ...")
    evaluator = EvaluatorSeqClsPhysics(batch_size=BATCH_SIZE)

    ckpt_dirs = [
        d for d in out_dir.iterdir()
        if d.is_dir() and d.name.lower().startswith("checkpoint")
    ]
    ckpt_dirs = sorted(set(ckpt_dirs), key=_ckpt_sort_key)
    ckpt_dirs = ckpt_dirs[len(ckpt_dirs) // 2:]

    for d in ckpt_dirs:
        metric_file = d / f"eval_metrics_{experiment_name}.json"
        if metric_file.exists():
            try:
                with open(metric_file, "r") as f:
                    data = json.load(f)
                acc = data.get("accuracy", 0.0)
                macro_f1 = data.get("macro_f1", 0.0)
                violation = data.get("physics_violation_rate", float("nan"))
            except Exception as e:
                print(f"[WARN] Failed to read metrics: {e}")
                acc, macro_f1, violation = 0.0, 0.0, float("nan")
        else:
            acc, macro_f1 = evaluator.run_full_eval(
                experiment_name=experiment_name,
                finetuned_model_dir=d,
                test_ds=test_fmt,
                output_dir=d,
            )
            try:
                with open(metric_file, "r") as f:
                    data = json.load(f)
                violation = data.get("physics_violation_rate", float("nan"))
            except Exception:
                violation = float("nan")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_ckpt = d.name
            best_acc = acc
            best_violation = violation
        torch.cuda.empty_cache()
        gc.collect()

    if (not ckpt_dirs) and out_dir.exists():
        print("No checkpoints found; evaluating base finetuned dir ...")
        acc, macro_f1 = evaluator.run_full_eval(
            experiment_name=experiment_name,
            finetuned_model_dir=out_dir,
            test_ds=test_fmt,
            output_dir=out_dir,
        )
        metric_file = out_dir / f"eval_metrics_{experiment_name}.json"
        try:
            with open(metric_file, "r") as f:
                data = json.load(f)
            violation = data.get("physics_violation_rate", float("nan"))
        except Exception:
            violation = float("nan")
        best_f1, best_acc, best_violation = macro_f1, acc, violation
        best_ckpt = out_dir.name

    summary_csv = SWEEP_SUMMARY_CSV if mode == "lambda_sweep" else SUMMARY_CSV
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_csv.exists() or summary_csv.stat().st_size == 0
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "experiment", "model_key", "lambda_phys", "output_dir",
                "best_checkpoint", "macro_f1", "accuracy", "physics_violation_rate",
            ])
        writer.writerow([
            experiment_name, MODEL_KEY, f"{lambda_phys}", str(out_dir),
            best_ckpt or "-", f"{best_f1:.4f}", f"{best_acc:.4f}",
            f"{best_violation:.4f}" if best_violation == best_violation else "nan",
        ])
    print(
        f"\n{experiment_name} (lambda={lambda_phys}) DONE. "
        f"Best macro-F1={best_f1:.4f} @ {best_ckpt}, "
        f"physics_violation_rate={best_violation:.4f}\n"
    )

    return best_f1, best_acc, best_violation


def _fmt(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "nan"


def _model_tag() -> str:
    return MODEL_KEY.split("-")[0] + "_sc_physics" + f"_lora-r{LORA_R}-a{LORA_ALPHA}"


def run_binary_novel_material():
    """Tab. 8 binary runs: Exp_OOD_5/6/7 share one 1200-row training pool.

    Train once on Exp_OOD_5, then reuse the trained model (via a symlink so
    the skip-training path kicks in) to evaluate Exp_OOD_6 and Exp_OOD_7 on
    their own novel-alloy test sets.
    """
    all_scores: Dict[str, Dict[str, float]] = {}
    model_tag = _model_tag()
    train_stem = EXPERIMENTS[0]

    print(
        f"=== {METHOD} binary novel-material runs {EXPERIMENTS} "
        f"at lambda={LAMBDA_PHYS_PRODUCTION} (train once on {train_stem}) ==="
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
            f1, acc, viol = run_one_experiment_physics(
                stem, lambda_phys=LAMBDA_PHYS_PRODUCTION, mode="production",
            )
            all_scores[stem] = {
                "macro_f1": f1, "accuracy": acc, "physics_violation_rate": viol,
            }
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {stem} failed: {e}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()
    return all_scores


def main():
    BASE_RESULTS.mkdir(parents=True, exist_ok=True)

    all_scores: Dict[str, Dict[str, float]] = {}

    if TASK == "binary":
        all_scores = run_binary_novel_material()

    elif RUN_MODE == "lambda_sweep":
        print(
            f"=== {METHOD} lambda sweep on {PILOT_EXPERIMENT}: "
            f"{LAMBDA_SWEEP_VALUES} ==="
        )
        for lam in LAMBDA_SWEEP_VALUES:
            key = f"{PILOT_EXPERIMENT}_lambda_{lam}"
            try:
                f1, acc, viol = run_one_experiment_physics(
                    PILOT_EXPERIMENT, lambda_phys=lam, mode="lambda_sweep"
                )
                all_scores[key] = {
                    "macro_f1": f1, "accuracy": acc, "physics_violation_rate": viol,
                }
            except Exception as e:
                print(f"[ERROR] {key} failed: {e}")
            finally:
                torch.cuda.empty_cache()
                gc.collect()

    elif RUN_MODE == "production":
        print(
            f"=== {METHOD} production runs on all 8 stems "
            f"at lambda={LAMBDA_PHYS_PRODUCTION} ==="
        )
        for stem in EXPERIMENTS:
            try:
                f1, acc, viol = run_one_experiment_physics(
                    stem, lambda_phys=LAMBDA_PHYS_PRODUCTION, mode="production",
                )
                all_scores[stem] = {
                    "macro_f1": f1, "accuracy": acc, "physics_violation_rate": viol,
                }
            except Exception as e:
                print(f"[ERROR] {stem} failed: {e}")
                SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
                with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:
                        writer.writerow([
                            "experiment", "model_key", "lambda_phys", "output_dir",
                            "best_checkpoint", "macro_f1", "accuracy",
                            "physics_violation_rate",
                        ])
                    writer.writerow([
                        stem, MODEL_KEY, f"{LAMBDA_PHYS_PRODUCTION}", "<failed>",
                        "<failed>", "nan", "nan", "nan",
                    ])
            finally:
                torch.cuda.empty_cache()
                gc.collect()
    else:
        raise ValueError(
            f"RUN_MODE must be 'lambda_sweep' or 'production', got {RUN_MODE!r}"
        )

    print(f"\n=== FINAL SUMMARY ({METHOD}) ===")
    for k, vals in all_scores.items():
        print(
            f"{k}: macro-F1={_fmt(vals['macro_f1'])}, "
            f"acc={_fmt(vals['accuracy'])}, "
            f"physics_violation_rate={_fmt(vals['physics_violation_rate'])}"
        )


if __name__ == "__main__":
    main()
