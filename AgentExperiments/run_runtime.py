import io
import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from amcore import (
    LABEL_COL, VALID_LABELS,
    load_exp_split, load_exp_train,
    normalize_ground_truth_label,
    extract_label_from_response,
    train_and_predict_proba, argmax_label_from_probs,
    calculate_entropy, ml_reliability_from_probs, deterministic_fusion,
    build_kd_agent_prompt, build_supervisor_prompt,
    call_llm, get_rag_loader,
)
from amcore.data_io import exp_type_from_stem
from amcore.labels import LABEL_ORDER

OUT_DIR = os.path.join(ROOT, "results_AM", "stats")
os.makedirs(OUT_DIR, exist_ok=True)

STEM = "Exp_OOD_1"
N_ROWS = 5
SUPERVISOR_PROFILE = "gpt5"
SUB_AGENT_PROFILE = "gpt5"


def _token_estimate(text: str) -> int:
    """Cheap token estimate (~4 chars/token for English)."""
    return max(1, len(text) // 4)


@contextmanager
def timer(name: str, sink: Dict[str, float]):
    t0 = time.perf_counter()
    yield
    sink[name] = round(time.perf_counter() - t0, 4)


def profile_one_stem(stem: str = STEM, n_rows: int = N_ROWS) -> pd.DataFrame:
    print(f"\n=== Profiling {stem} on {n_rows} rows ===")
    df_test = load_exp_split(stem).head(n_rows)
    df_train = load_exp_train(stem)
    exp_type = exp_type_from_stem(stem)
    is_ood = "OOD" in stem

    setup = {}
    with timer("dps_train_one_time_s", setup):
        ml_probs_by_row = train_and_predict_proba(stem, df_train, df_test)
    rag_loader = get_rag_loader()
    with timer("rag_warmup_s", setup):
        for mat in df_test["material"].dropna().unique():
            rag_loader.load_material_corpus(rag_loader.__class__.__name__
                                            if False else mat)
            rag_loader.get_context(mat, k=5, seed=42)
    print(f"  setup: ML training {setup['dps_train_one_time_s']:.2f}s, "
          f"RAG warm-up {setup['rag_warmup_s']:.2f}s")

    rows_out: List[Dict] = []
    for ridx, (idx, row) in enumerate(df_test.iterrows(), 1):
        print(f"\n  --- row {ridx}/{n_rows}  idx={idx}  material={row.get('material')} ---")
        t: Dict[str, float] = {}
        tokens: Dict[str, int] = {}
        material = row.get("material", "")

        with timer("end_to_end_s", t):
            with timer("dps_predict_s", t):
                ml_probs = ml_probs_by_row.get(int(idx), {})
                ml_pred_label = argmax_label_from_probs(ml_probs)
                ml_entropy = calculate_entropy(ml_probs, classes=tuple(LABEL_ORDER))
                ml_reliability = ml_reliability_from_probs(
                    ml_probs, is_ood, classes=tuple(LABEL_ORDER)
                )

            with timer("rag_retrieve_s", t):
                proc_params = {
                    "Power":           row.get("Power"),
                    "Velocity":        row.get("Velocity"),
                    "beam D":          row.get("beam D"),
                    "layer thickness": row.get("layer thickness"),
                    "Hatch spacing":   row.get("Hatch spacing"),
                }
                rag_context = rag_loader.get_context(
                    material, k=5, seed=int(idx) + 42,
                    process_params=proc_params,
                )

            with timer("ks_build_s", t):
                prompt_rag = build_kd_agent_prompt(row, rag_context=rag_context)
            tokens["ks_prompt_tok"] = _token_estimate(prompt_rag)
            with timer("ks_llm_s", t):
                try:
                    resp_text_rag = call_llm(prompt_rag, profile=SUB_AGENT_PROFILE)
                except Exception as e:
                    resp_text_rag = f"[ERROR] {e}"
            tokens["ks_response_tok"] = _token_estimate(resp_text_rag)

            with timer("fusion_s", t):
                fused_belief, fused_label = deterministic_fusion(
                    ml_probs_raw=ml_probs,
                    ml_reliability=ml_reliability,
                    rag_resp=resp_text_rag,
                    is_ood=is_ood,
                )

            with timer("sup_build_s", t):
                prompt_sup = build_supervisor_prompt(
                    row, ml_probs=ml_probs, ml_entropy=ml_entropy,
                    ml_reliability=ml_reliability,
                    rag_response=resp_text_rag,
                    fused_belief=fused_belief, fused_label=fused_label,
                    exp_type=exp_type,
                )
            tokens["sup_prompt_tok"] = _token_estimate(prompt_sup)
            with timer("sup_llm_s", t):
                try:
                    resp_text_sup = call_llm(prompt_sup, profile=SUPERVISOR_PROFILE)
                except Exception as e:
                    resp_text_sup = f"[ERROR] {e}"
            tokens["sup_response_tok"] = _token_estimate(resp_text_sup)

        rec = {"row_idx": int(idx), "material": material, **t, **tokens}
        rows_out.append(rec)
        print(f"    end-to-end: {t['end_to_end_s']:.2f}s   "
              f"ks_llm={t['ks_llm_s']:.2f}s   sup_llm={t['sup_llm_s']:.2f}s   "
              f"ks_tok={tokens['ks_prompt_tok']}+{tokens['ks_response_tok']}   "
              f"sup_tok={tokens['sup_prompt_tok']}+{tokens['sup_response_tok']}")

    return pd.DataFrame(rows_out), setup


def main():
    df, setup = profile_one_stem(stem=STEM, n_rows=N_ROWS)
    out_per_row = os.path.join(OUT_DIR, f"runtime_profile_{STEM}.csv")
    df.to_csv(out_per_row, index=False)
    print(f"\n[OK] per-row profile -> {out_per_row}")

    time_cols = [c for c in df.columns if c.endswith("_s") and c != "end_to_end_s"]
    tok_cols = [c for c in df.columns if c.endswith("_tok")]
    summary = {"setup_dps_train_s": setup["dps_train_one_time_s"],
               "setup_rag_warmup_s": setup["rag_warmup_s"],
               "end_to_end_s_median": df["end_to_end_s"].median(),
               "end_to_end_s_p95": df["end_to_end_s"].quantile(0.95)}
    for c in time_cols:
        summary[f"{c}_median"] = df[c].median()
    for c in tok_cols:
        summary[f"{c}_median"] = df[c].median()
    summary_df = pd.DataFrame([summary])
    out_summary = os.path.join(OUT_DIR, f"runtime_summary_{STEM}.csv")
    summary_df.to_csv(out_summary, index=False)
    print(f"[OK] summary       -> {out_summary}")

    print("\n=== Median per-component runtime ===")
    print(f"  Setup (one-time):")
    print(f"    DPS training:     {setup['dps_train_one_time_s']:.2f} s")
    print(f"    RAG warm-up:      {setup['rag_warmup_s']:.2f} s")
    print(f"  Per row:")
    print(f"    DPS predict:      {df['dps_predict_s'].median()*1000:.1f} ms")
    print(f"    RAG retrieve:     {df['rag_retrieve_s'].median()*1000:.1f} ms")
    print(f"    KS prompt build:  {df['ks_build_s'].median()*1000:.1f} ms")
    print(f"    KS LLM call:      {df['ks_llm_s'].median():.2f} s")
    print(f"    Fusion math:      {df['fusion_s'].median()*1000:.1f} ms")
    print(f"    Sup prompt build: {df['sup_build_s'].median()*1000:.1f} ms")
    print(f"    Sup LLM call:     {df['sup_llm_s'].median():.2f} s")
    print(f"    END-TO-END:       {df['end_to_end_s'].median():.2f} s   (p95: {df['end_to_end_s'].quantile(0.95):.2f} s)")
    print("\n=== Median token counts ===")
    for c in tok_cols:
        print(f"  {c:20s}: {df[c].median():.0f} tokens")
    print(f"\n  KS total tokens (prompt+response):  {df['ks_prompt_tok'].median()+df['ks_response_tok'].median():.0f}")
    print(f"  Sup total tokens (prompt+response): {df['sup_prompt_tok'].median()+df['sup_response_tok'].median():.0f}")


if __name__ == "__main__":
    main()
