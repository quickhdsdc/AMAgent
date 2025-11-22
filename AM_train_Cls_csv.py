#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Optional: XGBoost (only needed if you use XGB model)
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    print("[WARN] xgboost not available; XGB model will be skipped.")

# Optional: skops for safer serialization of sklearn models
try:
    import skops.io as sio
    _HAS_SKOPS = True
except Exception:
    _HAS_SKOPS = False
    print("[WARN] skops not available; .skops artifacts will not be written.")


# --------------------------
# Configuration
# --------------------------
DATA_DIR = "data"
OUT_DIR_MODELS = "ml_models/Cls"
OUT_DIR_ERRORS = "ml_models/errorCase"
os.makedirs(OUT_DIR_MODELS, exist_ok=True)
os.makedirs(OUT_DIR_ERRORS, exist_ok=True)

materials = ["Ti-6Al-4V", "SS316L", "SS17-4PH", "IN718"]  # "IN625" intentionally excluded as in your script
input_features = [
    # 'Hatch spacing',   # intentionally excluded as in your script
    "Velocity",
    "Power",
    "beam D",
    "layer thickness",
    # 'Cp', 'k',         # intentionally excluded as in your script
]
randomState = 42

# Gaussian Process kernel (from your code)
ckernel = (ConstantKernel() * RBF() + WhiteKernel()) + (ConstantKernel() * RBF() + WhiteKernel())

# Model zoo
models = {
    "RF": RandomForestClassifier(random_state=randomState),
    "GP": GaussianProcessClassifier(kernel=ckernel, multi_class="one_vs_rest",
                                    n_restarts_optimizer=10, max_iter_predict=500, random_state=21),
    "SVM": SVC(kernel="rbf", gamma="scale", C=12, random_state=21),  # probability=False (ok for training & predict)
    "GB": GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, max_depth=20,
                                     random_state=21, loss="log_loss"),
    "LR": LogisticRegression(max_iter=1000, random_state=randomState),
    "NN": MLPClassifier(max_iter=1000, random_state=randomState),
    "Bootstrap": BaggingClassifier(random_state=randomState),
}

# Add XGB if available
if _HAS_XGB:
    models["XGB"] = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=randomState)


# --------------------------
# Utilities
# --------------------------
def _material_short(name: str) -> str:
    # e.g. "Ti-6Al-4V" -> "Ti6Al4V"
    return name.replace("-", "")


def _features_short_token(features: list[str]) -> str:
    # Use first letter; special-case for "beam D" -> "D", "layer thickness" -> "T"
    toks = []
    for f in features:
        if f == "beam D":
            toks.append("D")
        elif f == "layer thickness":
            toks.append("T")
        else:
            toks.append(f[0])
    return "_".join(toks)


def _save_model_artifacts(
    model,
    material_name: str,
    est_key: str,
    X_test: pd.DataFrame,
    y_test_encoded: np.ndarray,
    y_pred_best: np.ndarray,
    label_encoder: LabelEncoder,
    features: list[str],
):
    """
    Save model (.joblib), plus .skops for sklearn or .json for XGBoost,
    and metadata (.meta.json). Also save misclassified cases CSV.
    """
    mat_short = _material_short(material_name)
    feat_tok = _features_short_token(features)

    # Basename shared by all artifacts
    base = f"bestF1_{mat_short}_{est_key}_{feat_tok}"

    # 1) Save joblib (always)
    joblib_path = os.path.join(OUT_DIR_MODELS, f"{base}.joblib")
    joblib.dump(model, joblib_path)
    print(f"Saved best F1 model (joblib): {joblib_path}")

    # 2) Save safer format
    if est_key == "XGB" and _HAS_XGB and hasattr(model, "save_model"):
        # Native XGBoost
        xgb_json = os.path.join(OUT_DIR_MODELS, f"{base}.json")
        try:
            model.save_model(xgb_json)
            print(f"Saved XGBoost native model: {xgb_json}")
        except Exception as e:
            print(f"[WARN] Failed saving XGB native model: {e}")
    else:
        # sklearn → .skops
        if _HAS_SKOPS:
            try:
                skops_path = os.path.join(OUT_DIR_MODELS, f"{base}.skops")
                sio.dump(model, skops_path)
                print(f"Saved .skops model: {skops_path}")
            except Exception as e:
                print(f"[WARN] Failed saving .skops: {e}")
        else:
            print("[INFO] skops not installed; skipped .skops export.")

    # 3) Save metadata
    meta = {
        "features": features,
        "label_classes": [str(c) for c in label_encoder.classes_],  # original class names
        "estimator": est_key,
        "material": material_name,
    }
    meta_path = os.path.join(OUT_DIR_MODELS, f"{base}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {meta_path}")

    # 4) Save misclassified test cases
    wrong_idx = np.where(y_pred_best != y_test_encoded)[0]
    if len(wrong_idx) > 0:
        y_test_inv = label_encoder.inverse_transform(y_test_encoded)
        y_pred_inv = label_encoder.inverse_transform(y_pred_best)
        wrong_df = X_test.iloc[wrong_idx].copy()
        wrong_df["true_label"] = y_test_inv[wrong_idx]
        wrong_df["pred_label"] = y_pred_inv[wrong_idx]
        err_csv = os.path.join(OUT_DIR_ERRORS, f"errorCase_cls_{mat_short}.csv")
        wrong_df.to_csv(err_csv, index=False)
        print(f"Saved misclassified cases: {err_csv}")


# --------------------------
# Training & Evaluation
# --------------------------
summary_f1 = {}        # {material: {model_name: f1}}
best_models = {}       # {material: (best_name, best_model_object)}
summary_dims = {}      # {material: (n_rows_after_filter, n_features)}

for material_name in materials:
    path = os.path.join(DATA_DIR, f"material_{material_name}_m.csv")
    if not os.path.exists(path):
        print(f"[WARN] Missing CSV for {material_name}: {path}")
        continue

    # Load and filter
    df = pd.read_csv(path)
    # keep rows with defect label
    df = df[df["defect_label"].notnull()].copy()
    # drop rows missing any of the selected features
    df = df.dropna(subset=input_features)

    if df.empty:
        print(f"[WARN] No valid rows after filtering for {material_name}.")
        continue

    X = df[input_features].copy()
    y = df["defect_label"].copy()

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=randomState, stratify=y_enc
    )

    # Train all models, compute F1
    f1_scores = {}
    trained = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="weighted")
            f1_scores[name] = float(f1)
            trained[name] = model
        except Exception as e:
            print(f"[WARN] Training {name} failed for {material_name}: {e}")
            # mark as very bad so it won't be selected
            f1_scores[name] = -1.0

    # Pick best by F1
    best_name = max(f1_scores, key=f1_scores.get)
    best_model = trained.get(best_name, None)
    best_models[material_name] = (best_name, best_model)
    summary_f1[material_name] = f1_scores
    summary_dims[material_name] = (X.shape[0], X.shape[1])

    # Save artifacts for the best model
    if best_model is not None:
        y_pred_best = best_model.predict(X_test)
        _save_model_artifacts(
            model=best_model,
            material_name=material_name,
            est_key=best_name,
            X_test=X_test,
            y_test_encoded=y_test,
            y_pred_best=y_pred_best,
            label_encoder=le,
            features=input_features,
        )
    else:
        print(f"[WARN] No trained best model object for {material_name} (best was {best_name}).")


# --------------------------
# Write summary CSV
# --------------------------
summary_csv = "Result_cls_summary.csv"
all_model_names = sorted({mn for m in summary_f1.values() for mn in m.keys()})

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Header
    writer.writerow(["model"] + materials)

    # Rows per model
    for mn in all_model_names:
        row = [mn]
        for mat in materials:
            f1_val = summary_f1.get(mat, {}).get(mn, "")
            row.append("" if f1_val == "" else f"{f1_val:.3f}")
        writer.writerow(row)

    # Final row: BEST (model name + score)
    best_row = ["BEST(model,score)"]
    for mat in materials:
        if mat in best_models and mat in summary_f1:
            bname, _ = best_models[mat]
            score = summary_f1[mat].get(bname, None)
            best_row.append(f"{bname},{score:.3f}" if score is not None else "")
        else:
            best_row.append("")
    writer.writerow(best_row)

print(f"\nWrote summary: {summary_csv}")
for mat in materials:
    dims = summary_dims.get(mat)
    if dims:
        print(f"  {mat}: rows={dims[0]}, features={dims[1]}")
