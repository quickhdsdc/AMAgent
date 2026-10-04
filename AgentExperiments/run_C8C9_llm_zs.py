import csv
import glob
import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.getcwd())

from amcore import (
    EXPERIMENTS, LABEL_COL, META_COL,
    load_exp_split,
    normalize_ground_truth_label,
    extract_label_from_response,
    call_llm, build_zs_prompt,
)
from amcore.parsing import extract_think_block

LLM_PROFILE = "gpt5"
FIELDNAMES = ["row_idx", "material", "Power", "Velocity",
              "zs_response", "zs_label", "zs_think", "gt_label"]


def _out_path(stem: str) -> str:
    base_dir = f"./results_AM/AMagent_ZS_{LLM_PROFILE}"
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"ZS_{LLM_PROFILE}_preds_{stem}.csv")


def _append_result(stem: str, row_dict: dict) -> None:
    path = _out_path(stem)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(row_dict)


def run_experiment(stem: str):
    print(f"\n--- Running Zero-Shot for {stem} ---")
    df_test = load_exp_split(stem)
    path = _out_path(stem)
    processed = set()
    if os.path.exists(path):
        try:
            df_curr = pd.read_csv(path)
            if "row_idx" in df_curr.columns:
                processed = set(df_curr["row_idx"].astype(int))
        except Exception:
            pass

    total = 0
    for idx, row in df_test.iterrows():
        row_id = int(row["row_idx"]) if "row_idx" in row else int(idx)
        if row_id in processed:
            continue
        gt_norm = normalize_ground_truth_label(row.get(LABEL_COL, "unknown"))
        prompt = build_zs_prompt(row)
        try:
            resp = call_llm(prompt, profile=LLM_PROFILE)
            lbl = extract_label_from_response(resp)
            think = extract_think_block(resp)
        except Exception as e:
            print(f"  [ERROR] row {row_id}: {e}")
            resp, lbl, think = f"Error: {e}", "error", ""

        _append_result(stem, {
            "row_idx": row_id,
            "material": row.get(META_COL, ""),
            "Power": row.get("Power", ""),
            "Velocity": row.get("Velocity", ""),
            "zs_response": resp,
            "zs_label": lbl,
            "zs_think": think,
            "gt_label": gt_norm,
        })
        total += 1
        if total % 5 == 0:
            print(f"  Processed {total} new rows...")
    print(f"  Completed {stem}.")


def _summarize():
    pattern = f"./results_AM/AMagent_ZS_{LLM_PROFILE}/ZS_{LLM_PROFILE}_preds_*.csv"
    out = f"./results_AM/AMagent_ZS_{LLM_PROFILE}/zs_metrics_summary.csv"
    rows = []
    for f in sorted(glob.glob(pattern)):
        basename = os.path.basename(f)
        prefix = f"ZS_{LLM_PROFILE}_preds_"
        exp_name = (os.path.splitext(basename)[0].replace(prefix, "")
                    if basename.startswith(prefix) else basename)
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "gt_label" not in df.columns or "zs_label" not in df.columns:
            continue
        y_true = [normalize_ground_truth_label(l) for l in df["gt_label"].tolist()]
        y_pred = [normalize_ground_truth_label(l) for l in df["zs_label"].tolist()]
        rows.append({
            "experiment": exp_name,
            "n_samples": len(df),
            "zs_acc": round(accuracy_score(y_true, y_pred), 4),
            "zs_f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "zs_f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        })
    if rows:
        df_summary = pd.DataFrame(rows)[
            ["experiment", "n_samples", "zs_acc", "zs_f1_macro", "zs_f1_weighted"]
        ]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df_summary.to_csv(out, index=False)
        print(f"Saved summary to: {out}")
        print(df_summary)


def main():
    print(f"Starting Zero-Shot Benchmark (Profile: {LLM_PROFILE})...")
    for stem in EXPERIMENTS:
        run_experiment(stem)
    _summarize()
    print("All experiments done.")


if __name__ == "__main__":
    main()
