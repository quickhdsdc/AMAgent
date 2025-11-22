#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

CLEAN_DIR = "data_clean"     # where *_classification_clean.csv live
OUT_DIR   = "data_exp"       # where we'll write experiment splits
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Helper: load each material classification dataset once
# Assumes files are named "<material>_classification_clean.csv"
# and contain columns:
#   Power, Velocity, beam D, [maybe layer thickness for some mats], defect_label
# ------------------------------------------------------------------

MATERIALS = [
    "Ti-6Al-4V",
    "SS316L",
    "SS17-4PH",
    "IN718",
]

def load_cleaned_classification(material: str) -> pd.DataFrame:
    path = os.path.join(CLEAN_DIR, f"{material}_classification_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected cleaned file not found: {path}")
    df = pd.read_csv(path)
    # Tag origin material, for analysis/debug later
    df["material"] = material
    return df

material_dfs = {m: load_cleaned_classification(m) for m in MATERIALS}

# quick sanity print
for m, dfm in material_dfs.items():
    print(f"[INFO] {m}: {len(dfm)} rows loaded for classification")

# ------------------------------------------------------------------
# Experiment definitions
# For each experiment pair:
#   - id_name:      Exp_ID_i
#   - ood_name:     Exp_OOD_i
#   - train_mats:   list of 3 materials used to train
#   - ood_test_mats: list of held-out materials for OOD test set (usually 1)
#
# ID setting:
#   80/20 stratified split within merged(train_mats)
#
# OOD setting:
#   train = ALL rows from merged(train_mats)
#   test  = ALL rows from merged(ood_test_mats)
#
# We'll save:
#   OUT_DIR/<exp_name>_train.csv
#   OUT_DIR/<exp_name>_test.csv
# ------------------------------------------------------------------

EXPERIMENTS = [
    {
        "id_name":  "Exp_ID_1",
        "ood_name": "Exp_OOD_1",
        "train_mats": ["Ti-6Al-4V", "SS316L", "SS17-4PH"],
        "ood_test_mats": ["IN718"],
    },
    {
        "id_name":  "Exp_ID_2",
        "ood_name": "Exp_OOD_2",
        "train_mats": ["Ti-6Al-4V", "SS316L", "IN718"],
        "ood_test_mats": ["SS17-4PH"],
    },
    {
        "id_name":  "Exp_ID_3",
        "ood_name": "Exp_OOD_3",
        "train_mats": ["Ti-6Al-4V", "SS17-4PH", "IN718"],
        "ood_test_mats": ["SS316L"],
    },
    {
        "id_name":  "Exp_ID_4",
        "ood_name": "Exp_OOD_4",
        "train_mats": ["SS316L", "SS17-4PH", "IN718"],
        "ood_test_mats": ["Ti-6Al-4V"],
    },
]

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def merge_materials(mats, material_dfs):
    """Concat classification-clean dataframes for the given list of materials."""
    frames = []
    for m in mats:
        if m not in material_dfs:
            raise KeyError(f"Material {m} not loaded")
        frames.append(material_dfs[m])
    merged = pd.concat(frames, ignore_index=True)
    return merged

def make_id_split(df_all, random_state=42, test_size=0.2, label_col="defect_label"):
    """
    Stratified 80/20 split on the merged in-distribution data.
    Returns (train_df, test_df).
    """
    if label_col not in df_all.columns:
        raise KeyError(f"Label column '{label_col}' not found in dataframe")

    y = df_all[label_col]
    train_df, test_df = train_test_split(
        df_all,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    # reset index for neatness
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    return train_df, test_df

def save_df(df, out_path):
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[WRITE] {out_path} ({len(df)} rows)")


# ------------------------------------------------------------------
# Main experiment loop
# ------------------------------------------------------------------

summary_rows = []  # we'll collect counts for printing at the end

for exp in EXPERIMENTS:
    id_name  = exp["id_name"]
    ood_name = exp["ood_name"]
    train_mats = exp["train_mats"]
    ood_test_mats = exp["ood_test_mats"]

    print("\n==============================")
    print(f" Processing {id_name} / {ood_name}")
    print(f"   Train mats: {train_mats}")
    print(f"   OOD test mats: {ood_test_mats}")

    # --- Build merged training pool ---
    df_pool = merge_materials(train_mats, material_dfs)

    # --- Exp_ID_i split (80/20 on same pool) ---
    id_train_df, id_test_df = make_id_split(df_pool, random_state=42, test_size=0.2,
                                            label_col="defect_label")

    id_train_path = os.path.join(OUT_DIR, f"{id_name}_train.csv")
    id_test_path  = os.path.join(OUT_DIR, f"{id_name}_test.csv")
    save_df(id_train_df, id_train_path)
    save_df(id_test_df, id_test_path)

    # --- Build OOD test set (held-out material(s)) ---
    df_ood_test = merge_materials(ood_test_mats, material_dfs)

    # OOD train = full in-distribution pool (no split)
    ood_train_df = df_pool.reset_index(drop=True)
    ood_test_df  = df_ood_test.reset_index(drop=True)

    ood_train_path = os.path.join(OUT_DIR, f"{ood_name}_train.csv")
    ood_test_path  = os.path.join(OUT_DIR, f"{ood_name}_test.csv")
    save_df(ood_train_df, ood_train_path)
    save_df(ood_test_df, ood_test_path)

    # Store summary info
    summary_rows.append({
        "id_name": id_name,
        "ood_name": ood_name,
        "train_mats": ",".join(train_mats),
        "ood_test_mats": ",".join(ood_test_mats),
        "Exp_ID_train_rows": len(id_train_df),
        "Exp_ID_test_rows": len(id_test_df),
        "Exp_OOD_train_rows": len(ood_train_df),
        "Exp_OOD_test_rows": len(ood_test_df),
    })

# ------------------------------------------------------------------
# Print summary table of row counts
# ------------------------------------------------------------------

print("\n=== Summary of generated experiment splits ===")
print(
    f"{'Experiment Pair':<15} "
    f"{'ID_train':>10} {'ID_test':>10} "
    f"{'OOD_train':>10} {'OOD_test':>10} "
    f"{'TrainMats':>30} {'OODTestMats':>15}"
)

for row in summary_rows:
    pair_label = f"{row['id_name']}/{row['ood_name']}"
    print(
        f"{pair_label:<15} "
        f"{row['Exp_ID_train_rows']:>10} {row['Exp_ID_test_rows']:>10} "
        f"{row['Exp_OOD_train_rows']:>10} {row['Exp_OOD_test_rows']:>10} "
        f"{row['train_mats']:>30} {row['ood_test_mats']:>15}"
    )

print(f"\nAll experiment datasets written to {OUT_DIR}")
