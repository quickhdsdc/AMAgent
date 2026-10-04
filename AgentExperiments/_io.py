import csv
import os
from typing import List

import pandas as pd


def get_output_paths(stem: str, model_tag: str,
                    base: str = "./results_AM") -> str:
    base_dir = os.path.join(base, f"AMagent_{model_tag}")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{model_tag}_raw_preds_{stem}.csv")


AGENT_FIELDNAMES: List[str] = [
    "row_idx",
    "material", "Power", "Velocity", "beam D", "layer thickness", "Hatch spacing",
    "agent_rag_response", "agent_rag_label",
    "supervisor_raw_response", "supervisor_label",
    "prompts_debug",
    "ml_pred_label",
    "gt_label",
]


def load_partial_results(stem: str, model_tag: str,
                         fieldnames: List[str] = AGENT_FIELDNAMES) -> pd.DataFrame:
    out_path = get_output_paths(stem, model_tag)
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        for c in fieldnames:
            if c not in df.columns:
                df[c] = ""
        return df[fieldnames]
    return pd.DataFrame(columns=fieldnames)


def append_partial_result(stem: str, row_dict: dict, model_tag: str,
                          fieldnames: List[str] = AGENT_FIELDNAMES) -> None:
    out_path = get_output_paths(stem, model_tag)

    write_header = True
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
        write_header = header != fieldnames
        if write_header:
            df_old = pd.read_csv(out_path)
            for c in fieldnames:
                if c not in df_old.columns:
                    df_old[c] = ""
            df_old = df_old[fieldnames]
            df_old.to_csv(out_path, index=False, encoding="utf-8")
            write_header = False

    mode = "a" if os.path.exists(out_path) else "w"
    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not os.path.exists(out_path) or write_header:
            writer.writeheader()
        writer.writerow(row_dict)
