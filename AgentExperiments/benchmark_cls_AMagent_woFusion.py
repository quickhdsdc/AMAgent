import os
import sys
import pandas as pd
import numpy as np
import json
import re
# Ensure we can import from the current directory
sys.path.append(os.getcwd())

# Import from benchmark script (NO PHYSICS variant)
try:
    from benchmark_cls_AMagent import (
        EXPERIMENTS, EXP_DIR, VALID_LABELS, LABEL_ORDER, LABEL_COL,
        _load_exp_split, _load_exp_train,
        _extract_label_from_response, _normalize_ground_truth_label,
        _call_llm, get_rag_loader,
        _exp_type_from_stem, META_COL, _get_val_with_unit,
    )
    from app.config import config
except ImportError as e:
    print(f"Error importing from benchmark_cls_AMagent: {e}")
    sys.exit(1)

# Configuration
SUPERVISOR_PROFILE = "gpt5"      # Matches benchmark main()
SUB_AGENT_PROFILE = "gemini"     # Matches benchmark main()

def _extract_json_block(text: str, tag: str):
    """Extracts a JSON dict from within [TAG]...[/TAG]."""
    regex = re.compile(f"\[{tag}\]\s*(.+?)\s*\[/{tag}\]", flags=re.IGNORECASE | re.DOTALL)
    m = regex.search(text)
    if not m:
        return None
    try:
        # cleanup likely json issues (e.g. single quotes)
        content = m.group(1).replace("'", '"')
        return json.loads(content)
    except Exception:
        return None

def _extract_think_block(text: str) -> str:
    """Extracts content between [THINK] and [/THINK]."""
    regex = re.compile(r"\[THINK\]\s*(.+?)\s*\[/THINK\]", flags=re.IGNORECASE | re.DOTALL)
    m = regex.search(text)
    if m:
        return m.group(1).strip()
    return "Not available"

def _build_supervisor_prompt_wofusion(row: pd.Series,
                             ml_label: str,
                             rag_response: str,
                             is_ood: bool = False) -> str:
    """
    Supervisor Agent prompt WITHOUT deterministic fusion strategy.
    Just gives the supervising LLM the ML LABEL and RAG response/label.
    """
    exp_type = "Out-of-Distribution (OOD)" if is_ood else "In-Distribution (ID)"
    reliance_instruction = ""
    if not is_ood:
        reliance_instruction = "Since the ML model is trained on In-Distribution data, you should more rely on ML prediction, unless there are strong evidences against it."
    material = row[META_COL] if META_COL in row and pd.notnull(row[META_COL]) else "unknown material"
    power_str           = _get_val_with_unit(row, "Power",            "W")
    velocity_str        = _get_val_with_unit(row, "Velocity",         "mm/s")
    beam_diam_str       = _get_val_with_unit(row, "beam D",           "µm")
    layer_thickness_str = _get_val_with_unit(row, "layer thickness",  "µm")
    hatch_str           = _get_val_with_unit(row, "Hatch spacing",    "µm")

    # Extract RAG parts
    rag_belief = _extract_json_block(rag_response, "BELIEF")
    rag_label = _extract_label_from_response(rag_response)
    rag_think = _extract_think_block(rag_response)
    

    ml_output_str = f"[LABEL] {ml_label} [/LABEL]"

    prompt = (
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
        f"{reliance_instruction}\n" # Insert custom instruction
        "Return ONLY the schema below:\n"
        "[THINK] {reasoning for final decision} [/THINK]\n"
        "[LABEL] {one of \"none\", \"lof\", \"balling\", \"keyhole\"} [/LABEL]"
    )
    return prompt

def aug_results_rerun_for_stem(stem: str):
    print(f"\n--- Running Ablation (No Fusion) for {stem} ---")
    
    # Paths
    base_dir = f"./results_AM/AMagent_{SUPERVISOR_PROFILE}"
    fname = f"{SUPERVISOR_PROFILE}_raw_preds_{stem}.csv"
    res_path = os.path.join(base_dir, fname)
    
    if not os.path.exists(res_path):
        print(f"Skipping {stem}: Result file not found at {res_path}")
        return

    # Load Result Data
    df_results = pd.read_csv(res_path)
    
    # Load Test Data to get context if needed
    df_test = _load_exp_split(stem)
    
    modified_count = 0
    
    for idx, row in df_results.iterrows():
        row_idx = row["row_idx"]

        try:
             test_row = df_test.loc[row_idx]
        except KeyError:
             print(f"  [WARN] Row {row_idx} not found in test set. Skipping.")
             continue

        # 1. Get Agent Responses
        resp_rag = str(df_results.at[idx, "agent_rag_response"]).replace("\n", " ")
        ml_pred_label = str(row.get("ml_pred_label", "unknown"))
        if ml_pred_label == "nan" or ml_pred_label == "":
             ml_pred_label = "unknown"

        try:
            # 2. Build New Supervisor Prompt (No Fusion)
            is_ood = "OOD" in stem
            prompt_sup = _build_supervisor_prompt_wofusion(
                test_row, 
                ml_label=ml_pred_label,
                rag_response=resp_rag,
                is_ood=is_ood
            )
            
            # 3. Call LLM
            resp_sup = _call_llm(prompt_sup, profile=SUPERVISOR_PROFILE)
            new_sup_lbl = _extract_label_from_response(resp_sup)
            
            # 4. Update Result
            df_results.at[idx, "supervisor_raw_response"] = resp_sup.replace("\n", " ")
            df_results.at[idx, "supervisor_label"] = new_sup_lbl
            
            # Save strictly every 10 rows or at end? 
            # Original script saved every row. Let's stick to that to be safe against crashes.
            print(f"    [Row {row_idx}] Updated Supervisor (Label: {new_sup_lbl})")
            df_results.to_csv(res_path, index=False)
            modified_count += 1
            
        except Exception as e:
            print(f"    [FATAL] Error running Supervisor for row {row_idx}: {e}")
            # Continue or exit? Original exited.
            # sys.exit(1)
            continue
            
    print(f"  Completed {stem}. Modified {modified_count} rows.")

def main():
    print("Starting Ablation Study (No Fusion) Rerun...")
    # Iterate all experiments
    for stem in EXPERIMENTS:
        aug_results_rerun_for_stem(stem)
    print("Done.")

if __name__ == "__main__":
    main()

