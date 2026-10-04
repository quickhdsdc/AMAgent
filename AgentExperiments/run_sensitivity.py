import argparse
import io
import json
import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
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
from amcore.prompts import _row_params_strings, build_kd_agent_prompt

OUT_DIR = os.path.join(ROOT, "results_AM", "stats")
os.makedirs(OUT_DIR, exist_ok=True)


def _format_evidence(rag_context: Optional[List[str]]) -> str:
    if rag_context:
        return "".join(f"[{i+1}] {s}\n" for i, s in enumerate(rag_context))
    return "No specific literature found.\n"


def build_prompt_v1(row, rag_context):
    """Use the exact four-class KS prompt from the main experiment."""
    return build_kd_agent_prompt(row, rag_context=rag_context)


def build_prompt_v2(row, rag_context):
    """V2 — physics-leading reorder. Puts the energy-density calculation
    first, then literature comparison. Tests whether KS reasoning shifts
    when the chain-of-thought is anchored on physics first."""
    p = _row_params_strings(row)
    return (
        "You are an LPBF physics analyst. Before consulting literature, "
        "compute the volumetric energy density "
        "VED = Power / (Velocity x Hatch_mm x LayerThickness_mm), "
        "converting hatch and layer thickness from micrometers to mm, for the given "
        f"parameters ({p['power']}, {p['velocity']}, {p['hatch_spacing']}, "
        f"{p['layer_thickness']}). Then verify whether the VED, LED, and "
        f"individual parameters fall inside the published process window "
        f"for {p['material']}.\n\n"
        "Cross-check against the following literature evidence:\n"
        f"{_format_evidence(rag_context)}\n"
        "Decide the most likely defect class based on this combined "
        "physics-then-literature analysis. Use a 'none' label only when "
        "both the energy density AND the literature support a defect-free "
        "outcome.\n\n"
        "Return EXACTLY this schema:\n"
        "[THINK] {VED computation, window comparison, evidence cross-check} [/THINK]\n"
        "[RELIABILITY] {0.0-1.0} [/RELIABILITY]\n"
        "[BELIEF] {\"none\":X,\"lof\":X,\"balling\":X,\"keyhole\":X} [/BELIEF]\n"
        "[LABEL] {none|lof|balling|keyhole} [/LABEL]"
    )


def build_prompt_v3(row, rag_context):
    """V3 — engineer persona, short and concrete."""
    p = _row_params_strings(row)
    return (
        "Acting as a senior LPBF process engineer, predict the most likely "
        "defect class for the following print setting:\n"
        f"  material = {p['material']}\n"
        f"  laser power = {p['power']}\n"
        f"  scan speed = {p['velocity']}\n"
        f"  beam diameter = {p['beam_diameter']}\n"
        f"  layer thickness = {p['layer_thickness']}\n"
        f"  hatch spacing = {p['hatch_spacing']}\n\n"
        "Literature excerpts (which may lack measurements):\n"
        f"{_format_evidence(rag_context)}\n"
        "Base your decision on (a) where the parameters sit relative to "
        f"the {p['material']} process window and (b) any directly relevant "
        "evidence above. Avoid speculation: assign 'none' unless the "
        "evidence or physics clearly indicates a specific defect mechanism.\n\n"
        "Output the analysis using these tags only:\n"
        "[THINK] ... [/THINK]\n"
        "[RELIABILITY] 0.0-1.0 [/RELIABILITY]\n"
        "[BELIEF] {\"none\":x,\"lof\":x,\"balling\":x,\"keyhole\":x} [/BELIEF]\n"
        "[LABEL] none|lof|balling|keyhole [/LABEL]"
    )


PROMPT_VARIANTS = {
    "v1_main_ks": build_prompt_v1,
    "v2_physics_first": build_prompt_v2,
    "v3_engineer": build_prompt_v3,
}


def run_grid(stem: str, n_rows: int,
             k_values: List[int], reasoning_values: List[str],
             profile: str = "gpt5",
             row_idx_list: Optional[List[int]] = None) -> pd.DataFrame:
    df_full = load_exp_split(stem).copy()
    if "row_idx" not in df_full.columns:
        df_full["row_idx"] = df_full.index
    if row_idx_list is not None:
        df_test = df_full[df_full["row_idx"].isin(row_idx_list)].copy()
        print(f"  using explicit row-idx list ({len(df_test)} rows)")
    else:
        df_test = df_full.head(n_rows).copy()
    rag_loader = get_rag_loader()

    out_path = os.path.join(OUT_DIR, f"prompt_sensitivity_corrected_{stem}.csv")
    print(f"  output -> {out_path}")
    if os.path.exists(out_path):
        done = pd.read_csv(out_path)
        done_keys = set(zip(done["prompt"], done["k"], done["reasoning"],
                           done["row_idx"]))
        print(f"  resuming, {len(done_keys)} cells already done")
    else:
        done_keys = set()
        pd.DataFrame(columns=[
            "stem", "row_idx", "material", "prompt", "k", "reasoning",
            "ks_label", "ks_response", "gt_label", "elapsed_s",
        ]).to_csv(out_path, index=False)

    grid = [(pv, k, r) for pv in PROMPT_VARIANTS for k in k_values
            for r in reasoning_values]
    print(f"  total cells: {len(grid)} × rows {len(df_test)} = "
          f"{len(grid)*len(df_test)} calls")

    rows_done_total = 0
    for pv_name, k, reason in grid:
        prompt_fn = PROMPT_VARIANTS[pv_name]
        print(f"\n  --- prompt={pv_name}  k={k}  reasoning={reason} ---")
        for idx, row in df_test.iterrows():
            key = (pv_name, k, reason, int(idx))
            if key in done_keys:
                continue
            material = row.get("material", "")
            proc_params = {c: row.get(c) for c in
                          ["Power", "Velocity", "beam D",
                           "layer thickness", "Hatch spacing"]}
            rag_ctx = rag_loader.get_context(
                material, k=k, seed=int(idx) + 42,
                process_params=proc_params,
            )
            prompt = prompt_fn(row, rag_ctx)
            t0 = time.time()
            try:
                resp = call_llm(prompt, profile=profile,
                               reasoning_effort=reason)
                lab = extract_label_from_response(resp)
            except Exception as e:
                resp = f"[ERROR] {e}"
                lab = "unknown"
            dt = round(time.time() - t0, 1)
            gt = normalize_ground_truth_label(row.get(LABEL_COL))
            rec = {
                "stem": stem, "row_idx": int(idx), "material": material,
                "prompt": pv_name, "k": k, "reasoning": reason,
                "ks_label": lab, "ks_response": resp.replace("\n", " "),
                "gt_label": gt, "elapsed_s": dt,
            }
            pd.DataFrame([rec]).to_csv(out_path, mode="a", header=False,
                                      index=False)
            rows_done_total += 1
            print(f"    idx={idx:>3} mat={material:<10} GT={gt:<8} "
                  f"KS={lab:<8} ({dt:.1f}s)")

    print(f"\n  done {rows_done_total} new calls")
    return pd.read_csv(out_path)


def summarize(df: pd.DataFrame):
    """Per-cell macro-F1 across the (prompt, k, reasoning) grid."""
    rows = []
    for (pv, k, r), grp in df.groupby(["prompt", "k", "reasoning"]):
        valid = grp[grp["ks_label"].isin(VALID_LABELS) &
                   grp["gt_label"].isin(VALID_LABELS)]
        f1 = f1_score(valid["gt_label"], valid["ks_label"],
                     labels=LABEL_ORDER, average="macro", zero_division=0)
        rows.append({"prompt": pv, "k": k, "reasoning": r,
                    "n_valid": len(valid),
                    "macro_f1": round(f1, 4),
                    "median_elapsed_s": grp["elapsed_s"].median()})
    return pd.DataFrame(rows).sort_values(["prompt", "k", "reasoning"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stem", default="Exp_OOD_1")
    p.add_argument("--n-rows", type=int, default=15)
    p.add_argument("--k", default="3,5,8")
    p.add_argument("--reasoning", default="low,medium")
    p.add_argument("--profile", default="gpt5")
    p.add_argument("--row-idx-list", default=None,
                  help="Comma-separated explicit row_idx values, e.g. 28,50,60. "
                       "Overrides --n-rows when provided.")
    args = p.parse_args()
    k_values = [int(x) for x in args.k.split(",")]
    reasoning_values = [x.strip() for x in args.reasoning.split(",")]
    row_idx_list = None
    if args.row_idx_list:
        row_idx_list = [int(x.strip()) for x in args.row_idx_list.split(",")]
    print(f"=== Prompt sensitivity on {args.stem} ===")
    print(f"  rows: {args.n_rows}, k: {k_values}, reasoning: {reasoning_values}")
    if row_idx_list:
        print(f"  explicit row indices: {row_idx_list}")
    df = run_grid(args.stem, args.n_rows, k_values, reasoning_values,
                 profile=args.profile, row_idx_list=row_idx_list)
    summary = summarize(df)
    sum_path = os.path.join(OUT_DIR, f"prompt_sensitivity_summary_corrected_{args.stem}.csv")
    summary.to_csv(sum_path, index=False)
    print(f"\n=== Cell-level summary -> {sum_path} ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
