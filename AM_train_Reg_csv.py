#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Optional: XGBoost regression
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    print("[WARN] xgboost not available; XGB will be skipped.")

# Optional: skops safer serialization
try:
    import skops.io as sio
    _HAS_SKOPS = True
except Exception:
    _HAS_SKOPS = False
    print("[WARN] skops not available; .skops artifacts will be skipped.")

# --------------------------
# Configuration
# --------------------------
DATA_DIR = "data"
OUT_DIR_MODELS = "ml_models/Reg"
OUT_DIR_ERRORS = "ml_models/errorCase"
os.makedirs(OUT_DIR_MODELS, exist_ok=True)
os.makedirs(OUT_DIR_ERRORS, exist_ok=True)

materials = ["Ti-6Al-4V", "SS316L", "SS17-4PH", "IN718"]  # IN625 excluded as in your setup
input_features = [
    # 'Hatch spacing',      # intentionally excluded
    "Velocity",
    "Power",
    "beam D",
    "layer thickness",
    # 'Cp', 'k',            # intentionally excluded
]
targets = [
    "depth of melt pool",
    "width of melt pool",
    "length of melt pool",
]

RANDOM_STATE = 42

# GP kernel (can be tuned)
gpr_kernel = (ConstantKernel() * RBF() + WhiteKernel()) + (ConstantKernel() * RBF() + WhiteKernel())

models = {
    "RF": RandomForestRegressor(random_state=RANDOM_STATE),
    "GP": GaussianProcessRegressor(kernel=gpr_kernel, n_restarts_optimizer=10, random_state=RANDOM_STATE),
    "SVR": SVR(kernel="rbf", gamma="scale", C=5),
    "RR": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE),
    "GB": GradientBoostingRegressor(n_estimators=600, learning_rate=0.03, max_depth=5, random_state=RANDOM_STATE),
    "NN": MLPRegressor(max_iter=1200, random_state=RANDOM_STATE),
    "Bootstrap": BaggingRegressor(random_state=RANDOM_STATE),
}
if _HAS_XGB:
    models["XGB"] = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
    )

# --------------------------
# Utilities
# --------------------------
def _material_short(name: str) -> str:
    # "Ti-6Al-4V" -> "Ti6Al4V"
    return name.replace("-", "")

def _features_token(features: list[str]) -> str:
    # e.g., ['Velocity','Power','beam D','layer thickness'] -> 'V_P_D_T'
    toks = []
    for f in features:
        if f == "beam D":
            toks.append("D")
        elif f == "layer thickness":
            toks.append("T")
        else:
            toks.append(f[0])
    return "_".join(toks)

def _target_token(t: str) -> str:
    # "depth of melt pool" -> "dep", "width of melt pool" -> "wid", "length of melt pool" -> "len"
    t = t.lower().strip()
    if t.startswith("depth"):
        return "dep"
    if t.startswith("width"):
        return "wid"
    if t.startswith("length"):
        return "len"
    return t.replace(" ", "")[:3]

def _save_artifacts(model, material_name: str, target: str, estimator_key: str, features: list[str]):
    mat_short = _material_short(material_name)
    feat_tok = _features_token(features)
    tgt_tok = _target_token(target)
    base = f"bestMAE_{mat_short}_{estimator_key}_{tgt_tok}_{feat_tok}"

    # 1) joblib (always)
    joblib_path = os.path.join(OUT_DIR_MODELS, f"{base}.joblib")
    joblib.dump(model, joblib_path)
    print(f"Saved (joblib): {joblib_path}")

    # 2) safer format
    if estimator_key == "XGB" and _HAS_XGB and hasattr(model, "save_model"):
        xgb_json = os.path.join(OUT_DIR_MODELS, f"{base}.json")
        try:
            model.save_model(xgb_json)
            print(f"Saved XGBoost native JSON: {xgb_json}")
        except Exception as e:
            print(f"[WARN] Failed saving XGB JSON: {e}")
    else:
        if _HAS_SKOPS:
            try:
                skops_path = os.path.join(OUT_DIR_MODELS, f"{base}.skops")
                sio.dump(model, skops_path)
                print(f"Saved (.skops): {skops_path}")
            except Exception as e:
                print(f"[WARN] Failed saving .skops: {e}")
        else:
            print("[INFO] skops not installed; skipped .skops export.")

    # 3) metadata
    meta = {
        "material": material_name,
        "target": target,
        "estimator": estimator_key,
        "features": features,
        "random_state": RANDOM_STATE,
    }
    meta_path = os.path.join(OUT_DIR_MODELS, f"{base}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {meta_path}")

def _save_error_cases(material_name: str, target: str, X_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray):
    # mark cases with |err| > 20% * |true|
    thr = 0.2 * np.abs(y_true)
    wrong_idx = np.where(np.abs(y_pred - y_true) > thr)[0]
    if len(wrong_idx) == 0:
        return
    dfw = X_test.iloc[wrong_idx].copy()
    dfw["true_value"] = y_true[wrong_idx]
    dfw["pred_value"] = y_pred[wrong_idx]
    tgt_tok = _target_token(target)
    mat_short = _material_short(material_name)
    out_csv = os.path.join(OUT_DIR_ERRORS, f"errorCase_{tgt_tok}_MAE_{mat_short}.csv")
    dfw.to_csv(out_csv, index=False)
    print(f"Saved error cases: {out_csv}")

# --------------------------
# Train / Evaluate (MAE only)
# --------------------------
for target in targets:
    print(f"\n===== Target: {target} =====")

    # summary CSV for this target
    summary_csv = f"Result_reg_{_target_token(target)}_summary.csv"
    rows_for_csv = []

    # header row: model + materials
    header = ["model"] + materials
    rows_for_csv.append(header)

    # keep MAE per model per material
    per_model_mae = {m: {} for m in models.keys()}
    best_model_for_material = {}

    for material_name in materials:
        path = os.path.join(DATA_DIR, f"material_{material_name}_m.csv")
        if not os.path.exists(path):
            print(f"[WARN] Missing CSV for {material_name}: {path}")
            # mark N/A
            for m in models.keys():
                per_model_mae[m][material_name] = None
            continue

        df = pd.read_csv(path)

        # keep only rows with the target present
        df = df[df[target].notnull()].copy()

        # ensure feature completeness (do NOT drop rows on target - already filtered)
        df = df.dropna(subset=input_features)

        if df.empty:
            print(f"[WARN] No valid rows after filtering for {material_name} ({target}).")
            for m in models.keys():
                per_model_mae[m][material_name] = None
            continue

        X = df[input_features].copy()
        y = df[target].astype(float).copy()

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE
        )

        # Train & evaluate (MAE)
        mae_scores = {}
        trained = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae = float(mean_absolute_error(y_test, y_pred))
                mae_scores[name] = mae
                trained[name] = model
            except Exception as e:
                print(f"[WARN] Training {name} failed for {material_name} ({target}): {e}")
                mae_scores[name] = np.inf

        # record MAE per model for summary
        for name in models.keys():
            per_model_mae[name][material_name] = mae_scores.get(name, None)

        # pick best by MAE (min)
        best_name = min(mae_scores, key=mae_scores.get)
        best_model = trained.get(best_name, None)
        best_model_for_material[material_name] = (best_name, mae_scores.get(best_name, None))

        # save artifacts & error cases
        if best_model is not None:
            _save_artifacts(best_model, material_name, target, best_name, input_features)

            # error cases based on 20% relative threshold
            try:
                y_pred_best = best_model.predict(X_test)
                _save_error_cases(material_name, target, X_test, y_test.values, np.array(y_pred_best))
            except Exception as e:
                print(f"[WARN] Could not save error cases for {material_name} ({target}): {e}")

    # build summary rows (one per model)
    for name in models.keys():
        row = [name]
        for mat in materials:
            v = per_model_mae[name].get(mat, None)
            row.append("" if (v is None or not np.isfinite(v)) else f"{v:.3f}")
        rows_for_csv.append(row)

    # final row: BEST(model,MAE) per material
    best_row = ["BEST(model,MAE)"]
    for mat in materials:
        bname, bmae = best_model_for_material.get(mat, (None, None))
        best_row.append("" if bname is None or bmae is None else f"{bname},{bmae:.3f}")
    rows_for_csv.append(best_row)

    # write summary CSV for this target
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows_for_csv)
    print(f"Summary written: {summary_csv}")

print("\n--------- Finished (MAE-only regression) ---------")
