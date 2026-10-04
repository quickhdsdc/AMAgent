import argparse
import io
import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from amcore import (
    LABEL_COL, VALID_LABELS, LABEL_ORDER,
    load_exp_split,
    normalize_ground_truth_label,
    extract_label_from_response,
    call_llm, get_rag_loader,
)
from amcore.prompts import build_kd_agent_prompt

OUT_DIR = os.path.join(ROOT, "results_AM", "stats")
os.makedirs(OUT_DIR, exist_ok=True)

SUBSAMPLE_IDX = [
    1, 12, 13, 16, 20, 23, 29, 42, 44, 45,
    63, 65, 70, 71, 73, 75, 76, 85, 88, 92,
    96, 99, 101, 104, 107, 110, 112, 113, 114, 115,
    129, 183, 192, 202, 211, 220, 224, 234, 254, 257,
]

K_VALUES = [5, 10, 15, 20]
PROMPT_NAME = "v1_main_ks"
REASONING = "low"
STEM = "Exp_OOD_1"
PROFILE = "gpt5"


def _format_evidence(rag_context):
    if rag_context:
        return "".join(f"[{i+1}] {s}\n" for i, s in enumerate(rag_context))
    return "No specific literature found.\n"


def build_prompt_v1(row, rag_context):
    """Use the exact four-class KS prompt from the main experiment."""
    return build_kd_agent_prompt(row, rag_context=rag_context)

def run(stem, row_idx_list, k_values, out_path):
    df_full = load_exp_split(stem).copy()
    if "row_idx" not in df_full.columns:
        df_full["row_idx"] = df_full.index
    df_test = df_full[df_full["row_idx"].isin(row_idx_list)].copy()
    print(f"  rows: {len(df_test)}, k values: {k_values}")

    rag_loader = get_rag_loader()

    if os.path.exists(out_path):
        done = pd.read_csv(out_path)
        done_keys = set(zip(done["k"], done["row_idx"]))
        print(f"  resuming, {len(done_keys)} cells already done")
    else:
        done_keys = set()
        pd.DataFrame(columns=[
            "stem", "row_idx", "material", "k", "ks_label",
            "ks_response", "gt_label", "elapsed_s",
        ]).to_csv(out_path, index=False)

    total = len(k_values) * len(df_test)
    done_count = 0
    for k in k_values:
        print(f"\n  --- k={k} ---")
        for idx, row in df_test.iterrows():
            key = (k, int(idx))
            if key in done_keys:
                done_count += 1
                continue
            material = row.get("material", "")
            proc_params = {c: row.get(c) for c in
                          ["Power", "Velocity", "beam D",
                           "layer thickness", "Hatch spacing"]}
            rag_ctx = rag_loader.get_context(
                material, k=k, seed=int(idx) + 42,
                process_params=proc_params,
            )
            prompt = build_prompt_v1(row, rag_ctx)
            t0 = time.time()
            try:
                resp = call_llm(prompt, profile=PROFILE, reasoning_effort=REASONING)
                lab = extract_label_from_response(resp)
            except Exception as e:
                resp = f"[ERROR] {e}"
                lab = "unknown"
            dt = round(time.time() - t0, 1)
            gt = normalize_ground_truth_label(row.get(LABEL_COL))
            rec = {
                "stem": stem, "row_idx": int(idx), "material": material,
                "k": k, "ks_label": lab,
                "ks_response": resp.replace("\n", " "),
                "gt_label": gt, "elapsed_s": dt,
            }
            pd.DataFrame([rec]).to_csv(out_path, mode="a", header=False, index=False)
            done_count += 1
            ok = "OK" if lab == gt else "X "
            print(f"    [{done_count}/{total}] idx={int(idx):>3} GT={gt:<8} KS={lab:<8} {ok} ({dt:.1f}s)")

    return pd.read_csv(out_path)


def summarize(df):
    rows = []
    for k, grp in df.groupby("k"):
        valid = grp[grp["ks_label"].isin(VALID_LABELS) &
                    grp["gt_label"].isin(VALID_LABELS)]
        f1 = f1_score(valid["gt_label"], valid["ks_label"],
                      labels=LABEL_ORDER, average="macro", zero_division=0)
        parse_pct = len(valid) / len(grp) * 100 if len(grp) > 0 else 0
        rows.append({"k": k, "n_total": len(grp), "n_valid": len(valid),
                     "parse_pct": round(parse_pct, 1),
                     "macro_f1": round(f1, 4),
                     "median_elapsed_s": round(grp["elapsed_s"].median(), 1)})
    return pd.DataFrame(rows).sort_values("k")


def plot(summary, out_png):
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax2 = ax1.twinx()

    ax1.plot(summary["k"], summary["macro_f1"], "o-", color="#1f77b4",
             linewidth=2, markersize=7, label="macro-F1")
    ax1.axhline(0.529, color="#1f77b4", linestyle="--", linewidth=1,
                label="B3 reference (Tab. 10, k=5)")
    ax2.bar(summary["k"], summary["parse_pct"], alpha=0.25, color="grey",
            width=2.5, label="parse success %")
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("parse success (%)", fontsize=9)

    for _, r in summary.iterrows():
        ax1.annotate(f"{r.macro_f1:.3f}", (r["k"], r["macro_f1"]),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)

    ax1.set_xlabel("retrieved chunks  k", fontsize=9)
    ax1.set_ylabel("macro-F1", fontsize=9)
    ax1.set_xticks(summary["k"])
    ax1.set_title("KS retrieval depth k — Exp_OOD_1\n"
                  "(v1_baseline, reasoning=low, 40-row balanced subsample)",
                  fontsize=9)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    print(f"  figure -> {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stem", default=STEM)
    p.add_argument("--k", default=",".join(str(x) for x in K_VALUES))
    p.add_argument("--row-idx-list", default=None,
                   help="Override default 40-row subsample (comma-separated)")
    p.add_argument("--profile", default=PROFILE)
    args = p.parse_args()

    k_values = [int(x) for x in args.k.split(",")]
    row_idx_list = (
        [int(x.strip()) for x in args.row_idx_list.split(",")]
        if args.row_idx_list else SUBSAMPLE_IDX
    )

    out_raw = os.path.join(OUT_DIR, f"ksweep_corrected_{args.stem}.csv")
    out_sum = os.path.join(OUT_DIR, f"ksweep_summary_corrected_{args.stem}.csv")
    out_fig = os.path.join(OUT_DIR, f"fig_ksweep_corrected_{args.stem}.png")

    print(f"=== K-sweep on {args.stem} ===")
    print(f"  prompt={PROMPT_NAME}, reasoning={REASONING}, k={k_values}")
    print(f"  subsample: {len(row_idx_list)} rows")

    df = run(args.stem, row_idx_list, k_values, out_raw)
    summary = summarize(df)
    summary.to_csv(out_sum, index=False)
    print(f"\n=== Summary -> {out_sum} ===")
    print(summary.to_string(index=False))
    plot(summary, out_fig)


if __name__ == "__main__":
    main()
