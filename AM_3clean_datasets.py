import os
import pandas as pd

# --------------------------------------------
# Paths
# --------------------------------------------
RAW_DATA_DIR = "data"         # contains material_<MAT>_m.csv (original, unfixed)
COMPLETED_DIR = "data"        # contains material_<MAT>_m_completed.csv (filled by LLM)
OUT_DIR = "data_clean"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------
# Materials to process  ->  <material key> : base filename stem
# --------------------------------------------
MATERIAL_FILES = {
    "IN718":     "material_IN718_m",
    "SS17-4PH":  "material_SS17-4PH_m",
    "SS316L":    "material_SS316L_m",
    "Ti-6Al-4V": "material_Ti-6Al-4V_m",
}

# --------------------------------------------
# Columns to keep and output
# --------------------------------------------
INPUT_COLS  = ["Power", "Velocity", "beam D", "layer thickness", "Hatch spacing"]
TARGET_COLS = ["defect_label"]
KEEP_COLS   = INPUT_COLS + TARGET_COLS

# --------------------------------------------
# Label normalization
# --------------------------------------------
LABEL_ORDER = ["desirable", "LOF", "balling", "keyhole"]  # -> 0,1,2,3
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_ORDER)}
LABEL_MAP_CANON = {
    "desirable": "desirable",
    "good": "desirable",
    "none": "desirable",
    "lof": "LOF",
    "lack of fusion": "LOF",
    "lack-of-fusion": "LOF",
    "balling": "balling",
    "ball": "balling",
    "keyhole": "keyhole",
    "key hole": "keyhole",
}

def normalize_shape_label(x: str) -> int | None:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    s = s.replace("\u00a0", " ")  # non-breaking space
    s = " ".join(s.split())       # collapse whitespace
    canon = LABEL_MAP_CANON.get(s)
    if canon is None:
        # tolerant fallbacks
        if "lack" in s and "fusion" in s:
            canon = "LOF"
        elif "key" in s and "hole" in s:
            canon = "keyhole"
        elif s in {"ok", "stable", "conduction"}:
            canon = "desirable"
    if canon is None:
        return None
    return LABEL_TO_ID[canon]

# --------------------------------------------
# Core routine
# --------------------------------------------
def build_clean_classification_for_material(material: str, stem: str) -> None:
    """
    Steps:
      1) Read ORIGINAL (unfilled) file: data/<stem>.csv
      2) Record indices where 'meltpool shape' is NOT missing
      3) Read COMPLETED (filled) file: data/<stem>_completed.csv
      4) Select rows by those indices (preserving original order)
      5) Map 'meltpool shape' -> integer 'defect_label'
      6) Keep only desired columns and drop rows that still have NaNs in KEEP_COLS
      7) Save to data_clean/<material>_classification_clean.csv
    """
    orig_path = os.path.join(RAW_DATA_DIR, f"{stem}.csv")
    comp_path = os.path.join(COMPLETED_DIR, f"{stem}_completed.csv")
    out_path  = os.path.join(OUT_DIR, f"{material}_classification_clean.csv")

    if not os.path.exists(orig_path):
        print(f"[SKIP] {material}: missing original file: {orig_path}")
        return
    if not os.path.exists(comp_path):
        print(f"[SKIP] {material}: missing completed file: {comp_path}")
        return

    df_orig = pd.read_csv(orig_path, low_memory=False)
    if "meltpool shape" not in df_orig.columns:
        print(f"[SKIP] {material}: original file lacks 'meltpool shape': {orig_path}")
        return

    # 1) find indices in ORIGINAL where meltpool shape is present
    mask_has_label = df_orig["meltpool shape"].notna() & (df_orig["meltpool shape"].astype(str).str.strip() != "")
    idxs = df_orig.index[mask_has_label].tolist()

    print(f"[{material}] original rows: {len(df_orig)} | rows with 'meltpool shape': {len(idxs)}")

    # 2) read COMPLETED and take only those indices (intersection for safety)
    df_comp = pd.read_csv(comp_path, low_memory=False)
    max_idx = len(df_comp) - 1
    idxs_valid = [i for i in idxs if 0 <= i <= max_idx]

    if len(idxs_valid) != len(idxs):
        print(f"[{material}] WARNING: {len(idxs) - len(idxs_valid)} indices out of range for completed file; they will be ignored.")

    df_sel = df_comp.iloc[idxs_valid].copy()

    # 3) create integer defect label from 'meltpool shape'
    if "meltpool shape" in df_sel.columns:
        df_sel["defect_label"] = df_sel["meltpool shape"].apply(normalize_shape_label)
    else:
        df_sel["defect_label"] = pd.NA

    # 4) keep only required columns (ensure presence)
    for col in INPUT_COLS:
        if col not in df_sel.columns:
            df_sel[col] = pd.NA

    # 5) drop rows that still have NaNs in required inputs or target
    df_out = df_sel[KEEP_COLS].dropna(subset=KEEP_COLS).reset_index(drop=True)

    print(f"[{material}] selected rows in completed: {len(df_sel)} | cleaned rows kept: {len(df_out)}")
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] wrote {out_path}")

def main():
    for material, stem in MATERIAL_FILES.items():
        build_clean_classification_for_material(material, stem)

    print("\nAll cleaned classification datasets saved in:", OUT_DIR)

if __name__ == "__main__":
    main()

