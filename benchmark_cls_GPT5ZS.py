import os
import re
import csv
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Any
from sklearn.metrics import f1_score, accuracy_score  # <-- added accuracy_score
from openai import OpenAI, AzureOpenAI
from app.config import config, LLMSettings

LABEL_ORDER = ["none", "lof", "balling", "keyhole"]  # 0,1,2,3
VALID_LABELS = set(LABEL_ORDER)  # {"none","lof","balling","keyhole"}

def get_llm_settings(profile: Optional[str] = None) -> LLMSettings:
    """
    Return the merged LLMSettings for a profile (or 'default').
    Raises KeyError if the profile doesn't exist.
    """
    profiles = config.llm
    if profile is None:
        profile = "default"
    if profile not in profiles:
        raise KeyError(f"Unknown LLM profile '{profile}'. Available: {list(profiles.keys())}")
    return profiles[profile]


def make_chat_client(profile: Optional[str] = "default") -> tuple[Any, str, dict]:

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

# ----------------------------------------------------
# Config
# ----------------------------------------------------

EXP_DIR = "data_exp"        # where Exp_ID_1_test.csv etc. live
LABEL_COL = "defect_label"  # ground-truth column
META_COL = "material"       # used in prompt
randomState = 42            # unused here but kept for consistency

# All experiments we want to score GPT5 on
EXPERIMENTS = [
    "Exp_ID_1",
    "Exp_OOD_1",
    "Exp_ID_2",
    "Exp_OOD_2",
    "Exp_ID_3",
    "Exp_OOD_3",
    "Exp_ID_4",
    "Exp_OOD_4",
]

# We will lowercase all labels for evaluation and map synonyms if needed.
VALID_LABELS = {"none", "lof", "balling", "keyhole"}


# ----------------------------------------------------
# Utilities
# ----------------------------------------------------

def _load_exp_split(stem: str) -> pd.DataFrame:
    """
    Load ONLY the test split for the given experiment stem.
    We'll evaluate GPT5 zero-shot on this test split directly.
    """
    test_path = os.path.join(EXP_DIR, f"{stem}_test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test split: {test_path}")
    df_test = pd.read_csv(test_path)
    return df_test


def _get_val_with_unit(row: pd.Series, col: str, unit: str, fallback: str = "unknown") -> str:
    """
    Safely grab row[col]; if missing or NaN, fall back to 'unknown'.
    Return f"{value} {unit}" with value as-is (no rounding).
    """
    if col in row and pd.notnull(row[col]):
        return f"{row[col]} {unit}"
    else:
        return f"{fallback} {unit}"


def _build_prompt_for_row(row: pd.Series) -> str:
    """
    Build the zero-shot classification prompt for one data sample.
    """
    material = row[META_COL] if META_COL in row and pd.notnull(row[META_COL]) else "unknown material"

    # Pull numeric process parameters with units
    power_str           = _get_val_with_unit(row, "Power",            "W")
    velocity_str        = _get_val_with_unit(row, "Velocity",         "mm/s")
    beam_diam_str       = _get_val_with_unit(row, "beam D",           "µm")
    layer_thickness_str = _get_val_with_unit(row, "layer thickness",  "µm")

    prompt = (
        "Your task is to assess in detail the potential imperfections for Laser Powder Bed Fusion printing "
        f"that arise in {material} manufactured at {power_str}, utilizing a {beam_diam_str} beam, "
        f"traveling at {velocity_str}, with a layer thickness of {layer_thickness_str}.\n"
        "Return ONLY the schema below. No extra text, no commentary, no explanations\n"
        "[THINK] {concise justification} [/THINK]\n"
        "[LABEL] {one of \"none\", \"lof\", \"balling\", and \"keyhole\"} [/LABEL]"
    )

    return prompt


def _call_gpt_zero_shot(prompt: str,
                        profile: Optional[str] = "default") -> str:
    """
    Use your chat completion stack to query GPT5-style model.
    We send a system message to reinforce formatting constraints.
    We return the raw .content string from the response.
    """
    client, model, default_kwargs = make_chat_client(profile=profile)

    system_msg = (
        "You are an LPBF process analysis assistant and act as an LPBF defect classification model."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
    )

    return resp.choices[0].message.content.strip()


_LABEL_REGEX = re.compile(
    r"\[LABEL\]\s*([^\[\]]+?)\s*\[/LABEL\]",
    flags=re.IGNORECASE | re.DOTALL,
)

def _extract_label_from_response(resp_text: str) -> str:
    """
    Extract the [LABEL] ... [/LABEL] block from GPT response.
    Normalize to one of LABEL_ORDER.
    If cannot parse or map, return "unknown".
    """
    m = _LABEL_REGEX.search(resp_text)
    if not m:
        return "unknown"

    raw = m.group(1).strip().lower()

    # collapse whitespace/hyphens
    norm = raw.replace("-", " ").strip()
    norm = re.sub(r"\s+", " ", norm)  # compress runs of spaces

    # try direct hits first
    if norm in ("none",):
        return "none"
    if norm in ("lof", "lack of fusion", "lack of fusion porosity", "lack of fusion defect"):
        return "lof"
    if norm in ("balling", "ball", "balling defect"):
        return "balling"
    if norm in ("keyhole", "keyhole porosity", "keyholing"):
        return "keyhole"

    return "unknown"


def _normalize_ground_truth_label(y) -> str:
    """
    Map the ground truth integer class (0-3) to its canonical string label
    using LABEL_ORDER.
    """
    if y is None or (isinstance(y, float) and pd.isna(y)):
        return "unknown"

    try:
        cls_idx = int(float(y))  # handles "2", 2, 2.0 safely
    except Exception:
        return "unknown"

    if 0 <= cls_idx < len(LABEL_ORDER):
        return LABEL_ORDER[cls_idx]

    return "unknown"

def _load_partial_results(stem: str) -> pd.DataFrame:
    """
    If we already have partial predictions for this experiment,
    load them. Otherwise return empty DataFrame with right columns.
    """
    partial_path = f"gpt5_raw_preds_{stem}.csv"
    if os.path.exists(partial_path):
        df_partial = pd.read_csv(partial_path)
    else:
        df_partial = pd.DataFrame(
            columns=[
                "row_idx",
                "material",
                "Power",
                "Velocity",
                "beam D",
                "layer thickness",
                "gt_label_norm",
                "gpt_raw_response",
                "pred_label_norm",
            ]
        )
    return df_partial


def _append_partial_result(stem: str, row_dict: dict) -> None:
    """
    Append a single row_dict to gpt5_raw_preds_{stem}.csv on disk.
    Creates the file with header if it doesn't exist yet,
    else appends without header.
    """
    out_path = f"gpt5_raw_preds_{stem}.csv"
    write_header = not os.path.exists(out_path)

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_idx",
                "material",
                "Power",
                "Velocity",
                "beam D",
                "layer thickness",
                "gt_label_norm",
                "gpt_raw_response",
                "pred_label_norm",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


def evaluate_gpt5_on_experiment_with_resume(
    stem: str,
    gpt_profile: str = "gpt5",
) -> Dict[str, float]:
    """
    Resumable GPT5 zero-shot eval for one experiment split.
    Now computes and returns both macro-F1 and overall accuracy.
    """

    # -----------------------
    # 1. Load test split
    # -----------------------
    df_test = _load_exp_split(stem)
    if LABEL_COL not in df_test.columns:
        raise RuntimeError(f"{stem}: '{LABEL_COL}' not found in test set.")

    # -----------------------
    # 2. Load partial predictions so far
    # -----------------------
    df_partial = _load_partial_results(stem)
    done_row_idxs = set(df_partial["row_idx"].tolist())

    # -----------------------
    # 3. Query GPT5 for missing rows
    # -----------------------
    for idx, row in df_test.iterrows():
        if idx in done_row_idxs:
            continue  # already processed

        # We'll log GT now (human-readable) just for trace/debug
        gt_norm_now = _normalize_ground_truth_label(row[LABEL_COL])

        row_record = {
            "row_idx": idx,
            "material": row.get("material", ""),
            "Power": row.get("Power", ""),
            "Velocity": row.get("Velocity", ""),
            "beam D": row.get("beam D", ""),
            "layer thickness": row.get("layer thickness", ""),
            "gt_label_norm": gt_norm_now,
            "gpt_raw_response": "",
            "pred_label_norm": "error",  # placeholder in case of failure
        }

        try:
            prompt = _build_prompt_for_row(row)
            resp_text = _call_gpt_zero_shot(prompt, profile=gpt_profile)

            pred_label_norm = _extract_label_from_response(resp_text)

            row_record["gpt_raw_response"] = resp_text
            row_record["pred_label_norm"] = pred_label_norm

        except Exception as e:
            row_record["gpt_raw_response"] = f"[ERROR] {e}"

        _append_partial_result(stem, row_record)

    # -----------------------
    # 4. Reload predictions (full) and join with df_test
    # -----------------------
    df_results = _load_partial_results(stem)
    df_results = df_results[df_results["row_idx"].isin(df_test.index)].copy()

    df_join = df_results.merge(
        df_test[[LABEL_COL, "material", "Power", "Velocity", "beam D", "layer thickness"]].copy(),
        left_on="row_idx",
        right_index=True,
        how="left",
        suffixes=("", "_true"),
    )

    # Canonicalize GT and prediction
    df_join["gt_label_canon"] = df_join[LABEL_COL].apply(_normalize_ground_truth_label)
    df_join["pred_label_canon"] = df_join["pred_label_norm"].astype(str).str.lower()

    # Derive a match flag
    df_join["match_flag"] = df_join["gt_label_canon"] == df_join["pred_label_canon"]

    # -----------------------
    # 5. Macro-F1 + Accuracy computation (on valid subset)
    # -----------------------
    preds = df_join["pred_label_canon"].tolist()
    gts   = df_join["gt_label_canon"].tolist()

    valid_mask = [
        (p in VALID_LABELS) and (g in VALID_LABELS)
        for p, g in zip(preds, gts)
    ]
    valid_mask = np.array(valid_mask, dtype=bool)

    if not np.any(valid_mask):
        macro_f1 = float("nan")
        accuracy = float("nan")
        n_scored = 0
    else:
        preds_valid = [preds[i] for i in range(len(preds)) if valid_mask[i]]
        gts_valid   = [gts[i]   for i in range(len(gts))   if valid_mask[i]]

        macro_f1 = f1_score(gts_valid, preds_valid, average="macro")
        accuracy = accuracy_score(gts_valid, preds_valid)  # <-- NEW
        n_scored = len(preds_valid)

    # -----------------------
    # 6. DEBUG OUTPUT (unchanged + show accuracy)
    # -----------------------

    # a) Make a compact debug dataframe for export & manual inspection
    df_debug = df_join.loc[:, [
        "row_idx",
        "material",
        "Power",
        "Velocity",
        "beam D",
        "layer thickness",
        LABEL_COL,            # raw integer 0-3
        "gt_label_canon",     # mapped to LABEL_ORDER
        "pred_label_canon",   # LLM normalized
        "match_flag",
        "gpt_raw_response",   # full assistant output for traceability
    ]].copy()

    debug_csv_path = f"gpt5_debug_{stem}.csv"
    df_debug.to_csv(debug_csv_path, index=False, encoding="utf-8")

    # b) Print a short confusion-style breakdown in stdout
    print(f"\n[DEBUG] Per-row comparison for {stem} written to {debug_csv_path}")
    print("[DEBUG] Sample mismatches (up to 10):")
    mismatches = df_debug[df_debug["match_flag"] == False].head(10)
    for _, r in mismatches.iterrows():
        print(
            f"  idx={r['row_idx']}  mat={r['material']}  GT={r['gt_label_canon']}  "
            f"PRED={r['pred_label_canon']}  P={r['Power']}  V={r['Velocity']}  "
            f"D={r['beam D']}  t={r['layer thickness']}"
        )

    # c) Also show simple class distribution comparison
    gt_counts = df_debug["gt_label_canon"].value_counts(dropna=False).to_dict()
    pred_counts = df_debug["pred_label_canon"].value_counts(dropna=False).to_dict()
    print("[DEBUG] Class distribution GT:", gt_counts)
    print("[DEBUG] Class distribution PRED:", pred_counts)

    # -----------------------
    # 7. Return summary (now includes accuracy)
    # -----------------------
    result = {
        "experiment": stem,
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),          # <-- NEW
        "n_test_total": len(df_test),
        "n_scored": n_scored,
        "n_completed_rows": len(df_results),
    }

    # Print concise summary line here as well
    print(
        f"{stem}: acc = {result['accuracy']:.4f} | macro-F1 = {result['macro_f1']:.4f} | "
        f"scored {result['n_scored']}/{result['n_test_total']} rows | "
        f"completed rows stored: {result['n_completed_rows']}"
    )

    return result



def main():
    results: List[Dict[str, float]] = []

    for stem in EXPERIMENTS:
        print("\n=======================================")
        print(f" Zero-shot GPT5 evaluation on {stem} (resumable)")
        print("=======================================")

        try:
            res = evaluate_gpt5_on_experiment_with_resume(
                stem,
                gpt_profile="default"
            )
        except FileNotFoundError as e:
            print(f"[ERROR] {stem}: {e}")
            continue
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
            continue

        results.append(res)

    # Write summary CSV at the end (now includes accuracy)
    out_csv = "gpt5_zero_shot_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "accuracy",          # <-- NEW
            "macro_F1",
            "n_scored",
            "n_test_total",
            "n_completed_rows",
        ])
        for r in results:
            writer.writerow([
                r["experiment"],
                f"{r['accuracy']:.6f}",   # <-- NEW
                f"{r['macro_f1']:.6f}",
                r["n_scored"],
                r["n_test_total"],
                r["n_completed_rows"],
            ])

    print(f"\nWrote GPT5 zero-shot summary: {out_csv}")


if __name__ == "__main__":
    main()
