from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from .evaluater_sc import EvaluatorSeqCls, LABEL_ORDER
from .physics_loss import (
    NUM_LABELS,
    build_forbidden_mask,
    compute_ved,
)


_RAW_PARAM_COLS = ["Power", "Velocity", "Hatch spacing", "layer thickness"]


def _row_to_raw_params(row) -> list[float]:
    out = []
    for col in _RAW_PARAM_COLS:
        v = row.get(col)
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _per_row_violation(pred_int: int, raw_params: list[float]) -> tuple[float, bool, bool]:
    """Return (ved, violated, valid). `violated` is False if `valid` is False."""

    rp = torch.tensor([raw_params], dtype=torch.float32)
    ved = compute_ved(rp[:, 0], rp[:, 1], rp[:, 2], rp[:, 3]).item()
    if not np.isfinite(ved):
        return float("nan"), False, False

    forbidden = build_forbidden_mask(
        torch.tensor([ved], dtype=torch.float32), num_labels=NUM_LABELS
    )
    violated = bool(forbidden[0, int(pred_int)].item() > 0.5)
    return float(ved), violated, True


class EvaluatorSeqClsPhysics(EvaluatorSeqCls):
    """Adds VED + physics-violation columns to the prediction CSV and
    a `physics_violation_rate` field to the metrics JSON.

    Assumes the test dataset was preprocessed with
    `data_preprocessor_sc_physics.DataPreprocessor`, so each row still
    carries `Power`, `Velocity`, `Hatch spacing`, and `layer thickness`.
    """

    def run_full_eval(
        self,
        experiment_name: str,
        finetuned_model_dir: Path,
        test_ds,
        output_dir: Path,
    ) -> Tuple[float, float]:
        """Like parent, but enriches df_pred with raw process-parameter
        columns from test_ds before calling compute_metrics_and_save.

        The parent's predict() only records {material, text, gt_int,
        gt_label, pred_int, pred_label}. The raw numerical columns
        (Power, Velocity, Hatch spacing, layer thickness) are kept in
        test_ds by DataPreprocessorSeqClsPhysics but are not forwarded
        through predict(). Without this override, compute_metrics_and_save
        would receive NaN for all raw params, making the VED computation
        and physics_violation_rate silently output NaN.
        """
        output_dir = Path(output_dir)
        if not self.accelerator or self.accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        if self.accelerator:
            self.accelerator.wait_for_everyone()

        model, tokenizer, _, _ = self.load_model(finetuned_model_dir)
        df_pred = self.predict(test_ds, model, tokenizer, output_dir, experiment_name)

        if (not self.accelerator or self.accelerator.is_main_process) and not df_pred.empty:
            try:
                available = [c for c in _RAW_PARAM_COLS if c in test_ds.column_names]
                if available:
                    texts = test_ds["text"]
                    col_vals = {col: test_ds[col] for col in available}
                    seen: dict = {}
                    rows = []
                    for i, t in enumerate(texts):
                        if t not in seen:
                            seen[t] = True
                            rows.append({"text": t, **{col: col_vals[col][i] for col in available}})
                    param_df = pd.DataFrame(rows)
                    df_pred = df_pred.merge(param_df, on="text", how="left")
            except Exception as e:
                print(f"[WARN C10] Could not enrich df_pred with raw params: {e}")

        acc, macro_f1 = self.compute_metrics_and_save(df_pred, output_dir, experiment_name)

        del model
        del tokenizer
        import gc
        torch.cuda.empty_cache()
        gc.collect()

        return acc, macro_f1

    def compute_metrics_and_save(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        experiment_name: str,
    ) -> Tuple[float, float]:
        acc, macro_f1 = super().compute_metrics_and_save(df, output_dir, experiment_name)

        if self.accelerator and not self.accelerator.is_main_process:
            return acc, macro_f1

        metrics_path = Path(output_dir) / f"eval_metrics_{experiment_name}.json"
        if not metrics_path.exists():
            return acc, macro_f1

        eval_pred_path = Path(output_dir) / f"eval_pred_{experiment_name}.csv"
        if not eval_pred_path.exists():
            return acc, macro_f1
        pred_df = pd.read_csv(eval_pred_path)

        merge_df = df.copy()
        for col in _RAW_PARAM_COLS:
            if col not in merge_df.columns:
                merge_df[col] = np.nan

        keep = ["text"] + _RAW_PARAM_COLS
        keep = [c for c in keep if c in merge_df.columns]
        merge_df = merge_df[keep].drop_duplicates(subset="text", keep="first")
        merged = pred_df.merge(merge_df, on="text", how="left")

        violations = []
        veds = []
        valids = []
        for _, row in merged.iterrows():
            raw_params = _row_to_raw_params(row)
            ved, violated, valid = _per_row_violation(int(row["pred_int"]), raw_params)
            veds.append(ved)
            violations.append(violated)
            valids.append(valid)

        merged["ved"] = veds
        merged["physics_violated"] = violations
        merged["ved_valid"] = valids
        merged.to_csv(eval_pred_path, index=False)

        n_valid = int(sum(valids))
        n_violated = int(sum(1 for v, va in zip(violations, valids) if va and v))
        violation_rate = float(n_violated) / n_valid if n_valid > 0 else float("nan")

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_blob = json.load(f)
        metrics_blob["physics_violation_rate"] = violation_rate
        metrics_blob["n_valid_phys_rows"] = n_valid
        metrics_blob["n_phys_violations"] = n_violated
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_blob, f, indent=2)

        return acc, macro_f1
