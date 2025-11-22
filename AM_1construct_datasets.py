import os
import pandas as pd

SRC_PATH = "./data/meltpoolnet_classification.csv"
OUT_DIR  = "./data"

# Columns to KEEP (order preserved). We will resolve them case/space-insensitively.
KEEP_COLS = [
    "Power",
    "Velocity",
    "Material",
    "Hatch spacing",
    "Sub-process",
    "beam D",
    "layer thickness",
    "Cp",
    "k",
    "meltpool shape",
]

# Output files per requested materials
TARGET_FILES = {
    "IN718":       "material_IN718_m.csv",
    "SS17-4PH":    "material_SS17-4PH_m.csv",
    "SS316L":      "material_SS316L_m.csv",
    "Ti-6Al-4V":   "material_Ti-6Al-4V_m.csv",
}

def _resolve_columns(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    """
    Map desired logical column names (case/space tolerant) to actual DataFrame columns.
    Raises KeyError if any cannot be found.
    """
    # Build normalization dict
    norm_lookup = {c.strip().lower(): c for c in df.columns}
    resolved = []
    missing = []
    for w in wanted:
        key = w.strip().lower()
        if key in norm_lookup:
            resolved.append(norm_lookup[key])
        else:
            missing.append(w)
    if missing:
        raise KeyError(f"Missing required columns in source CSV: {missing}")
    return resolved

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load CSV
    df = pd.read_csv(SRC_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    # Filter rows: Process == PBF and Sub-process == SLM (keep all samples otherwise; no dropna)
    # Resolve these filter columns robustly
    proc_col = _resolve_columns(df, ["Process"])[0]
    subp_col = _resolve_columns(df, ["Sub-process"])[0]
    df = df[(df[proc_col] == "PBF") & (df[subp_col] == "SLM")].copy()

    # Keep only requested columns (resolved to actual names)
    keep_cols_actual = _resolve_columns(df, KEEP_COLS)
    df_keep = df[keep_cols_actual].copy()

    # Rename columns back to the canonical names in KEEP_COLS for consistent outputs
    rename_map = {actual: canonical for actual, canonical in zip(keep_cols_actual, KEEP_COLS)}
    df_keep.rename(columns=rename_map, inplace=True)

    # Split by material and save
    written = set()
    for material, fname in TARGET_FILES.items():
        df_mat = df_keep[df_keep["Material"] == material].copy()
        if not df_mat.empty:
            df_mat.to_csv(os.path.join(OUT_DIR, fname), index=False)
            written.add(material)

    # Everything else -> material_others.csv
    df_others = df_keep[~df_keep["Material"].isin(written)].copy()
    if not df_others.empty:
        df_others.to_csv(os.path.join(OUT_DIR, "material_others.csv"), index=False)

    # Console summary
    print(f"Kept columns: {KEEP_COLS}")
    for m in TARGET_FILES:
        n = int((df_keep["Material"] == m).sum())
        print(f"{m}: {n} rows saved to {TARGET_FILES[m] if n>0 else '(no file, zero rows)'}")
    print(f"Others: {len(df_others)} rows saved to material_others.csv" if not df_others.empty else "Others: 0 rows")

if __name__ == "__main__":
    main()
