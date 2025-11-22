import os
import re
import csv
import json
import time
from typing import Any, Optional, Tuple

import pandas as pd

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

from app.config import config, LLMSettings
from openai import OpenAI, AzureOpenAI

# ==============================
# LLM client helpers
# ==============================
def get_llm_settings(profile: Optional[str] = None) -> LLMSettings:
    profiles = config.llm
    if profile is None:
        profile = "default"
    if profile not in profiles:
        raise KeyError(f"Unknown LLM profile '{profile}'. Available: {list(profiles.keys())}")
    return profiles[profile]

def make_chat_client(profile: Optional[str] = "default") -> Tuple[Any, str, dict]:
    llm = get_llm_settings(profile)
    api_type = llm.api_type

    if api_type == "azure":
        client = AzureOpenAI(
            api_key=llm.api_key,
            api_version=llm.api_version,
            azure_endpoint=llm.base_url,
        )
        deployment = llm.model
        default_kwargs = {
            "max_completion_tokens": llm.max_completion_tokens,
            "temperature": llm.temperature,
        }
        return client, deployment, default_kwargs

    elif api_type == "Openai":
        client = OpenAI(api_key=llm.api_key)
        model = llm.model
        default_kwargs = {
            "model": model,
            "max_completion_tokens": llm.max_completion_tokens,
            "temperature": llm.temperature,
        }
        return client, model, default_kwargs

    else:
        raise ValueError(f"Unsupported api_type: {llm.api_type!r}")

# ==============================
# JSON parsing helpers
# ==============================
def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _as_obj(text: str) -> Any:
    s = _strip_fences(text)
    decoder = json.JSONDecoder()
    i = 0
    n = len(s)
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        if s[i] in "{[":
            try:
                obj, end = decoder.raw_decode(s, i)
                return obj
            except json.JSONDecodeError:
                i += 1
                continue
        i += 1
    raise ValueError("No complete JSON object/array found in input.")

# ==============================
# Config
# ==============================
DATA_DIR = "./data"
INPUT_FILES = {
    "IN718":       "material_IN718_m.csv",
    "SS17-4PH":    "material_SS17-4PH_m.csv",
    "SS316L":      "material_SS316L_m.csv",
    "Ti-6Al-4V":   "material_Ti-6Al-4V_m.csv",
    # optionally: "material_others.csv"
}
OUTPUT_SUFFIX = "_completed.csv"
PARTIAL_SUFFIX = "_partial_llm_fill.csv"  # per-file rolling checkpoint
PROFILE = "default"    # LLM profile name from your config.llm
RATE_LIMIT_SLEEP = 0   # seconds between calls, set if you hit rate limits

# Columns we will read and (potentially) fill
COL_MATERIAL        = "Material"
COL_POWER           = "Power"
COL_VELOCITY        = "Velocity"
COL_BEAMD           = "beam D"
COL_LAYER_THICK     = "layer thickness"
COL_HATCH           = "Hatch spacing"
COL_DEFECT          = "meltpool shape"

RELEVANT_COLS = [
    COL_MATERIAL,
    COL_POWER,
    COL_VELOCITY,
    COL_BEAMD,
    COL_LAYER_THICK,
    COL_HATCH,
    COL_DEFECT,
]

# System prompt
SYSTEM_MSG = (
    "You are a cautious LPBF (laser powder bed fusion) assistant that fills in missing process parameters according to other given parameters and meltpool shape (potential defect label) for 3D printing runs. "
    "You MUST return a single strict JSON object with the exact keys and formats requested. Do not include markdown, backticks, commentary, or any text outside the JSON. "
    "All the values must be filled as a number. Never overwrite provided numeric inputs. Only fill missing fields. "
    "## Output JSON schema (all keys required) "
    "{"
    "\"Think\": string, "
    "\"Material\": string, "
    "\"Power\": number, "
    "\"Velocity\": number, "
    "\"beam D\": number, "
    "\"layer thickness\": number, "
    "\"Hatch spacing\": number, "
    "\"meltpool shape\": string"
    "}"
)

# ==============================
# Prompt construction
# ==============================
def _format_user_prompt_row(row: pd.Series) -> str:
    def fmt(v):
        return "" if pd.isna(v) else str(v)

    material = fmt(row.get(COL_MATERIAL))
    power = fmt(row.get(COL_POWER))
    vel = fmt(row.get(COL_VELOCITY))
    beam = fmt(row.get(COL_BEAMD))
    layer = fmt(row.get(COL_LAYER_THICK))
    hatch = fmt(row.get(COL_HATCH))
    label = fmt(row.get(COL_DEFECT))

    prompt = (
        "Given a printed part with:\n"
        f"Material: {material}\n"
        f"Power (W): {power}\n"
        f"Velocity (mm/s): {vel}\n"
        f"Beam D (µm): {beam}\n"
        f"Layer Thickness (µm): {layer}\n"
        f"Hatch spacing (µm): {hatch}\n"
        f"Meltpool shape: {label}\n"
        "Hints for estimation. 1) Convert units: µm → mm where needed. 2) Compute: "
        "- LED_J_per_mm = Power_W / Velocity_mm_s "
        "- VED_J_per_mm3 = Power_W / (Velocity_mm_s * Hatch_mm * LayerThickness_mm) "
        "3) If one or more of {Hatch, Beam D, Layer Thickness} are missing, estimate them using: "
        "- Label→VED target bands (tuneable, defaults below): "
        "LoF: VED < 60–70 "
        "Balling: ~70–90 (plus instability indicators) "
        "None: ~90–140 "
        "Keyhole: >140–160+ "
        "- Practical priors (adjustable): "
        "LayerThickness_mm ∈ [0.02, 0.06] "
        "BeamD_mm ∈ [0.07, 0.10] "
        "Hatch_mm ≈ β * BeamD_mm, β ∈ [0.7, 0.9] (30–50% overlap) "
    )
    return prompt

# ==============================
# GPT helpers
# ==============================
def _fill_missing_from_gpt(row: pd.Series, gpt_json: dict) -> pd.Series:
    """
    Update row IN-PLACE ONLY for missing numeric fields using GPT outputs.
    Do not overwrite provided values.
    """
    try:
        if pd.isna(row.get(COL_BEAMD)):
            val = gpt_json.get("beam D", None)
            if isinstance(val, (int, float)):
                row[COL_BEAMD] = float(val)

        if pd.isna(row.get(COL_LAYER_THICK)):
            val = gpt_json.get("layer thickness", None)
            if isinstance(val, (int, float)):
                row[COL_LAYER_THICK] = float(val)

        if pd.isna(row.get(COL_HATCH)):
            val = gpt_json.get("Hatch spacing", None)
            if isinstance(val, (int, float)):
                row[COL_HATCH] = float(val)

        if pd.isna(row.get(COL_POWER)):
            val = gpt_json.get("Power", None)
            if isinstance(val, (int, float)):
                row[COL_POWER] = float(val)

        if pd.isna(row.get(COL_VELOCITY)):
            val = gpt_json.get("Velocity", None)
            if isinstance(val, (int, float)):
                row[COL_VELOCITY] = float(val)

        if pd.isna(row.get(COL_DEFECT)) or str(row.get(COL_DEFECT)).strip() == "":
            val = gpt_json.get("meltpool shape", None)
            if isinstance(val, str) and val.strip():
                row[COL_DEFECT] = val.strip()

        if pd.isna(row.get(COL_MATERIAL)) or str(row.get(COL_MATERIAL)).strip() == "":
            val = gpt_json.get("Material", None)
            if isinstance(val, str) and val.strip():
                row[COL_MATERIAL] = val.strip()

    except Exception:
        pass

    return row

# ==============================
# Pre-filter relevant columns & missing checks
# ==============================
def _extract_relevant(df: pd.DataFrame) -> pd.DataFrame:
    for c in RELEVANT_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[RELEVANT_COLS].copy()

def _missing_mask(df_rel: pd.DataFrame) -> pd.Series:
    check_cols = [COL_POWER, COL_VELOCITY, COL_BEAMD, COL_LAYER_THICK, COL_HATCH]
    return df_rel[check_cols].isna().any(axis=1)

def _report_missing(df_rel: pd.DataFrame) -> None:
    total = len(df_rel)
    print("Missing values report (relevant columns):")
    for c in RELEVANT_COLS:
        n_miss = int(df_rel[c].isna().sum())
        if n_miss > 0:
            print(f"  - {c}: {n_miss}/{total} missing")
    print("")

# ==============================
# Partial save/append/resume
# ==============================
def _partial_path_for(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(DATA_DIR, f"{stem}{PARTIAL_SUFFIX}")

def _load_partial_results(filename: str) -> pd.DataFrame:
    """
    If we already have partial fills for this file, load them.
    Otherwise return empty DataFrame with right columns.
    """
    partial_path = _partial_path_for(filename)
    if os.path.exists(partial_path):
        try:
            df_partial = pd.read_csv(partial_path, low_memory=False)
            # Ensure expected columns exist
            for c in ["row_idx"] + RELEVANT_COLS + ["gpt_raw_response"]:
                if c not in df_partial.columns:
                    df_partial[c] = pd.NA
            return df_partial[["row_idx"] + RELEVANT_COLS + ["gpt_raw_response"]].copy()
        except Exception:
            pass

    # Empty template
    return pd.DataFrame(columns=["row_idx"] + RELEVANT_COLS + ["gpt_raw_response"])

def _append_partial_result(filename: str, row_idx: int, row: pd.Series, gpt_raw_response: str) -> None:
    """
    Append a single row's (possibly updated) values to the partial CSV on disk.
    Creates the file with header if it doesn't exist yet, else appends without header.
    """
    partial_path = _partial_path_for(filename)
    write_header = not os.path.exists(partial_path)

    record = {
        "row_idx": row_idx,
        COL_MATERIAL: row.get(COL_MATERIAL),
        COL_POWER: row.get(COL_POWER),
        COL_VELOCITY: row.get(COL_VELOCITY),
        COL_BEAMD: row.get(COL_BEAMD),
        COL_LAYER_THICK: row.get(COL_LAYER_THICK),
        COL_HATCH: row.get(COL_HATCH),
        COL_DEFECT: row.get(COL_DEFECT),
        "gpt_raw_response": gpt_raw_response,
    }

    with open(partial_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["row_idx"] + RELEVANT_COLS + ["gpt_raw_response"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(record)

# ==============================
# Main processing
# ==============================
def process_file(material_key: str, filename: str, out_dir: str = DATA_DIR, profile: str = PROFILE):
    """
    Steps:
      - Read file & extract relevant columns
      - Report missing values
      - Load partial file & figure out which rows are already done
      - Build a mask of rows with missing values (in core params)
      - ONLY send those rows (that also aren't already done) to GPT to fill
      - After each row is processed, append to partial CSV
      - Merge: for rows already in partial, use their saved values
      - Save final completed CSV
    """
    in_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(in_path):
        print(f"[WARN] {material_key}: file not found: {in_path}")
        return

    df_raw = pd.read_csv(in_path, low_memory=False)
    df_rel = _extract_relevant(df_raw)
    _report_missing(df_rel)

    # Mask rows that need fill (have missing key params)
    mask_need = _missing_mask(df_rel)

    # Load partial (resume) and compute already-finished indices
    df_partial = _load_partial_results(filename)
    done_row_idxs = set()
    if not df_partial.empty and "row_idx" in df_partial.columns:
        done_row_idxs = set(int(i) for i in df_partial["row_idx"].dropna().astype(int).tolist())

    n_need = int(mask_need.sum())
    n_total = len(df_rel)
    n_done = len(done_row_idxs)
    print(f"{material_key}: {n_need}/{n_total} rows have missing core params; {n_done} rows already saved in partial.")

    # Reuse one client for speed
    client, model, default_kwargs = make_chat_client(profile=profile)

    # Prepare container for final output (we will assemble in original row order)
    final_rows = []

    # Iterate over rows in original order with a progress bar
    # Only call GPT for rows that need fill AND are not already done in partial
    for idx in tqdm(df_rel.index, desc=f"Filling {material_key}", total=n_total):
        if idx in done_row_idxs:
            # Use the already saved filled row
            saved = df_partial[df_partial["row_idx"] == idx]
            if not saved.empty:
                # Replace current with saved values
                row_saved = saved.iloc[0]
                row_out = pd.Series({c: row_saved.get(c, pd.NA) for c in RELEVANT_COLS})
                final_rows.append(row_out)
                continue

        # Current fresh row
        row = df_rel.loc[idx].copy()

        # If this row doesn't need fill, keep as-is and append to partial (so it's resumable)
        if not mask_need.loc[idx]:
            # Save to partial (with empty raw response) and to final
            _append_partial_result(filename, int(idx), row, gpt_raw_response="")
            final_rows.append(row)
            if RATE_LIMIT_SLEEP > 0:
                time.sleep(RATE_LIMIT_SLEEP)
            continue

        # Build prompt and call GPT with basic retries
        prompt = _format_user_prompt_row(row)
        tries = 0
        gpt_data = None
        gpt_raw = ""
        while tries < 3:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": prompt},
                    ],
                    **default_kwargs
                )
                gpt_raw = resp.choices[0].message.content or ""
                gpt_data = _as_obj(gpt_raw)
                break
            except Exception:
                tries += 1
                time.sleep(1.0 * tries)

        # Fill if we got JSON
        if gpt_data is not None:
            row = _fill_missing_from_gpt(row, gpt_data)

        # Append to partial immediately (checkpoint)
        _append_partial_result(filename, int(idx), row, gpt_raw_response=gpt_raw)

        # Push to final
        final_rows.append(row)

        if RATE_LIMIT_SLEEP > 0:
            time.sleep(RATE_LIMIT_SLEEP)

    # Build final DataFrame and save completed CSV
    df_out = pd.DataFrame(final_rows)
    out_path = os.path.join(out_dir, filename.replace(".csv", OUTPUT_SUFFIX))
    df_out.to_csv(out_path, index=False)
    print(f"[OK] {material_key}: wrote {out_path} ({len(df_out)} rows)")
    print(f"[OK] {material_key}: rolling checkpoint at {_partial_path_for(filename)}")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for mat, fname in INPUT_FILES.items():
        process_file(mat, fname, out_dir=DATA_DIR, profile=PROFILE)

if __name__ == "__main__":
    main()
