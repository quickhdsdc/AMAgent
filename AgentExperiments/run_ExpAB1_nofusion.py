import os
import sys
from typing import Optional

import pandas as pd

sys.path.insert(0, os.getcwd())

from amcore import (
    EXPERIMENTS, META_COL,
    load_exp_split,
    extract_label_from_response,
    call_llm,
)
from amcore.parsing import extract_json_block, extract_think_block
from amcore.prompts import get_val_with_unit

SUPERVISOR_PROFILE = "gpt5"


def build_supervisor_prompt_wofusion(row: pd.Series,
                                     ml_label: str,
                                     rag_response: str,
                                     is_ood: bool = False) -> str:
    exp_type = "Out-of-Distribution (OOD)" if is_ood else "In-Distribution (ID)"
    reliance_instruction = (
        "Since the ML model is trained on In-Distribution data, you should more rely "
        "on ML prediction, unless there are strong evidences against it."
        if not is_ood else ""
    )

    material = (row[META_COL]
                if META_COL in row and pd.notnull(row[META_COL])
                else "unknown material")
    power_str = get_val_with_unit(row, "Power", "W")
    velocity_str = get_val_with_unit(row, "Velocity", "mm/s")
    beam_diam_str = get_val_with_unit(row, "beam D", "µm")
    layer_thickness_str = get_val_with_unit(row, "layer thickness", "µm")
    hatch_str = get_val_with_unit(row, "Hatch spacing", "µm")

    rag_label = extract_label_from_response(rag_response)
    rag_think = extract_think_block(rag_response)

    ml_output_str = f"[LABEL] {ml_label} [/LABEL]"

    return (
        "You are a Senior AM Process Engineer (Supervisor). "
        "Your task is to assess in detail the potential imperfections for Laser Powder Bed Fusion printing "
        f"that arise in {material} manufactured at {power_str}, utilizing a {beam_diam_str} beam, "
        f"traveling at {velocity_str}, with a layer thickness of {layer_thickness_str} and hatch spacing of {hatch_str}. "
        "Review the analysis from two sub-agents and provide the final decision.\n\n"
        "Agent 1 (Data-Driven ML Analyst) - DIRECT PREDICTIONS:\n"
        f"{ml_output_str}\n"
        f"This model was trained on {exp_type} data.\n\n"
        "Agent 2 (Knowledge-Driven Analyst):\n"
        f"[THINK] {rag_think} [/THINK]\n"
        f"[LABEL] {rag_label} [/LABEL]\n\n"
        "Task:\n"
        "Synthesize the inputs from both agents. Make a final label prediction.\n"
        f"{reliance_instruction}\n"
        "Return ONLY the schema below:\n"
        "[THINK] {reasoning for final decision} [/THINK]\n"
        "[LABEL] {one of \"none\", \"lof\", \"balling\", \"keyhole\"} [/LABEL]"
    )


def rerun_stem(stem: str, supervisor_profile: str = SUPERVISOR_PROFILE):
    print(f"\n--- Running Ablation (No Fusion) for {stem} ---")
    base_dir = f"./results_AM/AMagent_{supervisor_profile}"
    res_path = os.path.join(base_dir, f"{supervisor_profile}_raw_preds_{stem}.csv")
    if not os.path.exists(res_path):
        print(f"Skipping {stem}: Result file not found at {res_path}")
        return

    df_results = pd.read_csv(res_path)
    df_test = load_exp_split(stem)
    modified = 0
    is_ood = "OOD" in stem

    for idx, row in df_results.iterrows():
        row_idx = row["row_idx"]
        try:
            test_row = df_test.loc[row_idx]
        except KeyError:
            print(f"  [WARN] Row {row_idx} not found in test set. Skipping.")
            continue

        resp_rag = str(df_results.at[idx, "agent_rag_response"]).replace("\n", " ")
        ml_pred_label = str(row.get("ml_pred_label", "unknown"))
        if ml_pred_label in ("nan", ""):
            ml_pred_label = "unknown"

        try:
            prompt_sup = build_supervisor_prompt_wofusion(
                test_row, ml_label=ml_pred_label, rag_response=resp_rag, is_ood=is_ood
            )
            resp_sup = call_llm(prompt_sup, profile=supervisor_profile)
            new_lbl = extract_label_from_response(resp_sup)
            df_results.at[idx, "supervisor_raw_response"] = resp_sup.replace("\n", " ")
            df_results.at[idx, "supervisor_label"] = new_lbl
            print(f"    [Row {row_idx}] Updated supervisor -> {new_lbl}")
            df_results.to_csv(res_path, index=False)
            modified += 1
        except Exception as e:
            print(f"    [ERROR] row {row_idx}: {e}")
            continue

    print(f"  Completed {stem}: modified {modified} rows.")


def main():
    print("Starting Ablation Study (No Fusion) rerun...")
    for stem in EXPERIMENTS:
        rerun_stem(stem)
    print("Done.")


if __name__ == "__main__":
    main()
