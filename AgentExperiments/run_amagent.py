import glob
import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.getcwd())

from amcore import (
    EXPERIMENTS, LABEL_COL, VALID_LABELS,
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

from experiments._io import (
    AGENT_FIELDNAMES, get_output_paths,
    load_partial_results, append_partial_result,
)

RANDOM_STATE = 42


def evaluate_resume(stem: str,
                    supervisor_profile: str = "default",
                    sub_agent_profile: str = "deepseek") -> Dict[str, float]:
    df_test = load_exp_split(stem)
    if LABEL_COL not in df_test.columns:
        raise RuntimeError(f"{stem}: '{LABEL_COL}' not found in test set.")

    df_test = df_test.copy()
    if "row_idx" not in df_test.columns:
        df_test["row_idx"] = df_test.index

    df_train = load_exp_train(stem)
    exp_type = exp_type_from_stem(stem)
    ml_probs_by_row: Dict[int, Dict[str, float]] = {}
    if df_train is not None:
        try:
            ml_probs_by_row = train_and_predict_proba(stem, df_train, df_test)
        except Exception as e:
            print(f"[WARN] ML training failed for {stem}: {e}")

    df_partial = load_partial_results(stem, model_tag=supervisor_profile)
    done_row_idxs = set(df_partial["row_idx"].tolist())

    rag_loader = get_rag_loader()
    is_ood_exp = "OOD" in stem

    for idx, row in df_test.iterrows():
        if idx in done_row_idxs:
            continue

        gt_label_str = normalize_ground_truth_label(row[LABEL_COL])
        ml_probs = ml_probs_by_row.get(int(idx), {})
        ml_pred_label = argmax_label_from_probs(ml_probs)

        material_name = row.get("material", "")
        seed = int(idx) + RANDOM_STATE
        proc_params = {
            "Power":           row.get("Power"),
            "Velocity":        row.get("Velocity"),
            "beam D":          row.get("beam D"),
            "layer thickness": row.get("layer thickness"),
            "Hatch spacing":   row.get("Hatch spacing"),
        }
        rag_context = rag_loader.get_context(
            material_name, k=5, seed=seed, process_params=proc_params
        )

        ml_entropy = calculate_entropy(ml_probs, classes=tuple(LABEL_ORDER))
        ml_reliability = ml_reliability_from_probs(
            ml_probs, is_ood_exp, classes=tuple(LABEL_ORDER)
        )

        prompt_rag = build_kd_agent_prompt(row, rag_context=rag_context)
        try:
            resp_text_rag = call_llm(prompt_rag, profile=sub_agent_profile)
            agent_rag_label = extract_label_from_response(resp_text_rag)
            resp_text_rag_clean = resp_text_rag.replace("\n", " ").replace("\r", " ")
        except Exception as e:
            resp_text_rag_clean = f"[ERROR] {e}"
            agent_rag_label = "unknown"

        fused_belief, fused_label = deterministic_fusion(
            ml_probs_raw=ml_probs,
            ml_reliability=ml_reliability,
            rag_resp=resp_text_rag_clean,
            is_ood=is_ood_exp,
        )

        prompt_sup = build_supervisor_prompt(
            row,
            ml_probs=ml_probs,
            ml_entropy=ml_entropy,
            ml_reliability=ml_reliability,
            rag_response=resp_text_rag_clean,
            fused_belief=fused_belief,
            fused_label=fused_label,
            exp_type=exp_type,
        )
        try:
            resp_text_sup = call_llm(prompt_sup, profile=supervisor_profile)
            supervisor_label = extract_label_from_response(resp_text_sup)
            resp_text_sup_clean = resp_text_sup.replace("\n", " ").replace("\r", " ")
        except Exception as e:
            resp_text_sup_clean = f"[ERROR] {e}"
            supervisor_label = "unknown"

        prompts_debug = json.dumps({
            "kd_agent_prompt": prompt_rag,
            "supervisor_prompt": prompt_sup,
            "ml_stats": {
                "entropy": ml_entropy,
                "reliability": ml_reliability,
                "probs": ml_probs,
            },
        })

        row_record = {
            "row_idx": int(idx),
            "material": row.get("material", ""),
            "Power": row.get("Power", ""),
            "Velocity": row.get("Velocity", ""),
            "beam D": row.get("beam D", ""),
            "layer thickness": row.get("layer thickness", ""),
            "Hatch spacing": row.get("Hatch spacing", ""),
            "agent_rag_response": resp_text_rag_clean,
            "agent_rag_label": agent_rag_label,
            "supervisor_raw_response": resp_text_sup_clean,
            "supervisor_label": supervisor_label,
            "prompts_debug": prompts_debug,
            "ml_pred_label": ml_pred_label,
            "gt_label": gt_label_str,
        }
        append_partial_result(stem, row_record, model_tag=supervisor_profile)

    df_results = load_partial_results(stem, model_tag=supervisor_profile)
    df_results = df_results[df_results["row_idx"].isin(df_test.index)].copy()

    df_join = df_results.merge(
        df_test[[LABEL_COL, "material", "Power", "Velocity", "beam D", "layer thickness"]],
        left_on="row_idx", right_index=True,
        how="left", suffixes=("", "_true"),
    )
    df_join["gt_label_canon"] = df_join[LABEL_COL].apply(normalize_ground_truth_label)
    df_join["pred_label_canon"] = df_join["supervisor_label"].astype(str).str.lower()
    df_join["match_flag"] = df_join["gt_label_canon"] == df_join["pred_label_canon"]

    preds = df_join["pred_label_canon"].tolist()
    gts = df_join["gt_label_canon"].tolist()
    valid_mask = np.array(
        [(p in VALID_LABELS) and (g in VALID_LABELS) for p, g in zip(preds, gts)],
        dtype=bool,
    )

    if not np.any(valid_mask):
        macro_f1, n_scored = float("nan"), 0
    else:
        preds_valid = [preds[i] for i in range(len(preds)) if valid_mask[i]]
        gts_valid = [gts[i] for i in range(len(gts)) if valid_mask[i]]
        macro_f1 = f1_score(gts_valid, preds_valid, average="macro")
        n_scored = len(preds_valid)

    return {
        "experiment": stem,
        "macro_f1": float(macro_f1),
        "n_test_total": len(df_test),
        "n_scored": n_scored,
        "n_completed_rows": len(df_results),
    }


def _summarize_results(supervisor_profile: str):
    search_pattern = f"./results_AM/AMagent_{supervisor_profile}/{supervisor_profile}_raw_preds_*.csv"
    out_metrics = f"./results_AM/AMagent_{supervisor_profile}/metrics_summary.csv"
    files = glob.glob(search_pattern)
    if not files:
        print("No result files to summarize.")
        return

    targets = [
        ("ml_pred", "ml_pred_label"),
        ("agent_rag", "agent_rag_label"),
        ("supervisor", "supervisor_label"),
    ]
    summary_rows = []
    for f in sorted(files):
        basename = os.path.basename(f)
        prefix = f"{supervisor_profile}_raw_preds_"
        exp_name = (os.path.splitext(basename)[0].replace(prefix, "")
                    if basename.startswith(prefix) else basename)
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] Failed to read {f}: {e}")
            continue
        if "gt_label" not in df.columns:
            continue
        y_true = [normalize_ground_truth_label(l) for l in df["gt_label"].tolist()]
        row_metrics = {"experiment": exp_name, "n_samples": len(df)}
        try:
            df_test = load_exp_split(exp_name)
            row_metrics["n_total_test"] = len(df_test)
        except Exception:
            row_metrics["n_total_test"] = "unknown"

        for target_name, col_name in targets:
            resp_col = {
                "agent_rag": "agent_rag_response",
                "supervisor": "supervisor_raw_response",
            }.get(target_name)
            y_pred_derived = []
            if resp_col and resp_col in df.columns:
                for r in df[resp_col].fillna("").astype(str).tolist():
                    y_pred_derived.append(extract_label_from_response(r))
            elif col_name in df.columns:
                raw_labels = df[col_name].fillna("unknown").astype(str).tolist()
                y_pred_derived = [normalize_ground_truth_label(l) for l in raw_labels]
            else:
                continue
            y_pred = [normalize_ground_truth_label(l) for l in y_pred_derived]
            row_metrics[f"{target_name}_acc"] = round(accuracy_score(y_true, y_pred), 4)
            row_metrics[f"{target_name}_f1_macro"] = round(
                f1_score(y_true, y_pred, average="macro", zero_division=0), 4
            )
            row_metrics[f"{target_name}_f1_weighted"] = round(
                f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
            )
        summary_rows.append(row_metrics)

    if not summary_rows:
        print("No valid summary rows.")
        return
    df_summary = pd.DataFrame(summary_rows)
    base_cols = ["experiment", "n_samples", "n_total_test"]
    metric_cols = sorted([c for c in df_summary.columns if c not in base_cols])
    df_summary = df_summary[base_cols + metric_cols]
    os.makedirs(os.path.dirname(out_metrics), exist_ok=True)
    df_summary.to_csv(out_metrics, index=False)
    print(f"Saved metrics summary to: {out_metrics}")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df_summary)


def main():
    SUPERVISOR_PROFILE = "gpt5"
    SUB_AGENT_PROFILE = "gemini"
    print(f"Running benchmark with supervisor={SUPERVISOR_PROFILE}, sub_agents={SUB_AGENT_PROFILE}")

    for stem in EXPERIMENTS:
        print(f"\n===== {stem} =====")
        try:
            res = evaluate_resume(stem,
                                  supervisor_profile=SUPERVISOR_PROFILE,
                                  sub_agent_profile=SUB_AGENT_PROFILE)
            print(f"{stem}: macro-F1={res['macro_f1']:.4f}  "
                  f"scored {res['n_scored']}/{res['n_test_total']}  "
                  f"completed {res['n_completed_rows']}")
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")

    _summarize_results(SUPERVISOR_PROFILE)


if __name__ == "__main__":
    main()
