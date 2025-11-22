import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

# --------------------------
# Configuration
# --------------------------
DATA_DIR = "data"
OUT_DIR_MODELS = "ml_models/Reg_single_GB"
os.makedirs(OUT_DIR_MODELS, exist_ok=True)

material_name = "Ti-6Al-4V"  # only this material
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

# Single model type: GB regressor (same hyperparams as before)
def make_gb():
    return GradientBoostingRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        random_state=RANDOM_STATE,
    )

# --------------------------
# Small helpers
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
    # "depth of melt pool" -> "dep", etc.
    t = t.lower().strip()
    if t.startswith("depth"):
        return "dep"
    if t.startswith("width"):
        return "wid"
    if t.startswith("length"):
        return "len"
    return t.replace(" ", "")[:3]

def _save_model(model, target: str):
    mat_short = _material_short(material_name)
    feat_tok = _features_token(input_features)
    tgt_tok = _target_token(target)
    base = f"GB_{mat_short}_{tgt_tok}_{feat_tok}"

    joblib_path = os.path.join(OUT_DIR_MODELS, f"{base}.joblib")
    joblib.dump(model, joblib_path)
    print(f"Saved GB model: {joblib_path}")

    meta = {
        "material": material_name,
        "target": target,
        "estimator": "GB",
        "features": input_features,
        "random_state": RANDOM_STATE,
    }
    meta_path = os.path.join(OUT_DIR_MODELS, f"{base}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {meta_path}")

# --------------------------
# Main: train GB on Ti-6Al-4V and predict 2 samples
# --------------------------
def main():
    path = os.path.join(DATA_DIR, f"material_{material_name}_m.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV for {material_name} not found: {path}")

    df_all = pd.read_csv(path)

    # Two inference samples (same for every target, only y changes)
    inference_samples = [
        {'Power': 350, 'Velocity': 800,  'beam D': 100, 'layer thickness': 30},
        {'Power': 190, 'Velocity': 1200, 'beam D': 100, 'layer thickness': 30},
    ]

    print(f"Loaded data for material: {material_name}")
    print(f"Input features: {input_features}")
    print(f"Targets: {targets}")
    print()

    for target in targets:
        print(f"===== Target: {target} =====")

        # Filter rows where this target is present
        df = df_all[df_all[target].notnull()].copy()
        # Ensure feature completeness
        df = df.dropna(subset=input_features)

        if df.empty:
            print(f"[WARN] No valid rows for target '{target}' after filtering.")
            continue

        X = df[input_features].copy()
        y = df[target].astype(float).copy()

        # Train GB on FULL data (no train/test split, focus is inference)
        gb = make_gb()
        gb.fit(X, y)
        print(f"Trained GB on {len(df)} samples for target '{target}'.")

        # Save model + metadata (optional, but keeps your original spirit)
        _save_model(gb, target)

        # Build dataframe for inference in correct column order
        X_new = pd.DataFrame(inference_samples)
        # Ensure columns are exactly in input_features order
        X_new = X_new[input_features]

        # Predict
        y_pred = gb.predict(X_new)

        for i, sample in enumerate(inference_samples):
            print(f"Sample #{i+1}: {sample}")
            print(f"  Predicted {target}: {y_pred[i]:.4f}")
        print()

    print("---- Done (GB on Ti-6Al-4V, 2-sample inference) ----")

if __name__ == "__main__":
    main()

