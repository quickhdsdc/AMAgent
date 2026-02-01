import os
import re
import csv
import pandas as pd
import glob
from typing import Optional, List, Any
from sklearn.metrics import f1_score, accuracy_score
from openai import OpenAI, AzureOpenAI
from app.config import config, LLMSettings

LABEL_ORDER = ["none", "lof", "balling", "keyhole"] 
VALID_LABELS = set(LABEL_ORDER)

def get_llm_settings(profile: Optional[str] = None) -> LLMSettings:
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
        client = OpenAI(api_key=llm.api_key, base_url=llm.base_url)
        model = llm.model
        default_kwargs = {
            "model": model,
            "max_completion_tokens": llm.max_completion_tokens,
            "temperature": llm.temperature,
        }
        return client, model, default_kwargs

    elif api_type.lower() == "google":
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Google GenAI package not installed. Please install it.")
        
        api_key = llm.api_key
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
            
        client = genai.Client(api_key=api_key)
        model = llm.model
        genai_config = types.GenerateContentConfig(
            temperature=llm.temperature,
            max_output_tokens=llm.max_tokens if hasattr(llm, 'max_tokens') else 8192,
            thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=1024) 
        )
        default_kwargs = {
            "config": genai_config
        }
        return client, model, default_kwargs

    else:
        raise ValueError(f"Unsupported api_type: {llm.api_type!r}")

EXP_DIR = "data_exp"
LABEL_COL = "defect_label"
META_COL = "material"
LLM_PROFILE = "gpt5" 

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

def _load_exp_split(stem: str) -> pd.DataFrame:
    test_path = os.path.join(EXP_DIR, f"{stem}_test.csv")
    print("Loading test split:", test_path)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test split: {test_path}")
    df_test = pd.read_csv(test_path)
    return df_test

def _get_val_with_unit(row: pd.Series, col: str, unit: str, fallback: str = "unknown") -> str:
    if col in row and pd.notnull(row[col]):
        return f"{row[col]} {unit}"
    else:
        return f"{fallback} {unit}"

_LABEL_REGEX = re.compile(
    r"\[LABEL\]\s*([^\[\]]+?)\s*\[/LABEL\]",
    flags=re.IGNORECASE | re.DOTALL,
)

def _extract_label_from_response(text: str) -> str:
    if not text or not isinstance(text, str):
        return "unknown"
    
    m = _LABEL_REGEX.search(text)
    if not m:
        return "unknown"
        
    raw = m.group(1).strip().lower()
    norm = re.sub(r"[\{\}\"\'\n\r]", "", raw).strip()
    if ":" in norm:
        norm = norm.split(":")[-1].strip()
    if norm in VALID_LABELS:
        return norm
        
    return "unknown"

def _normalize_ground_truth_label(y) -> str:
    if y is None or (isinstance(y, float) and pd.isna(y)):
        return "unknown"
    s_val = str(y).strip().lower()
    if s_val in VALID_LABELS:
        return s_val
    try:
        cls_idx = int(float(y))
        if 0 <= cls_idx < len(LABEL_ORDER):
            return LABEL_ORDER[cls_idx]
    except Exception:
        pass
    return "unknown"

def _extract_think_block(text: str) -> str:
    regex = re.compile(r"\[THINK\]\s*(.+?)\s*\[/THINK\]", flags=re.IGNORECASE | re.DOTALL)
    m = regex.search(text)
    if m:
        return m.group(1).strip()
    return ""

def _build_zs_prompt(row: pd.Series) -> str:
    material = row[META_COL] if META_COL in row and pd.notnull(row[META_COL]) else "unknown material"
    power_str           = _get_val_with_unit(row, "Power",            "W")
    velocity_str        = _get_val_with_unit(row, "Velocity",         "mm/s")
    beam_diam_str       = _get_val_with_unit(row, "beam D",           "µm")
    layer_thickness_str = _get_val_with_unit(row, "layer thickness",  "µm")
    hatch_str           = _get_val_with_unit(row, "Hatch spacing",    "µm")

    prompt = (
        "You are a Knowledge-Driven LPBF process analysis assistant. Your task is to assess detail the potential imperfections for"
        f"Laser Powder Bed Fusion printing that arise in {material} manufactured at {power_str}, utilizing a {beam_diam_str} beam,"
        f"traveling at {velocity_str}, with a layer thickness of {layer_thickness_str} and hatch spacing of {hatch_str}."
        f"Specifically, consider whether these parameters respect the typical process window for {material}. "
        "Predict the potential defect label by comparing the current process parameters against retrieved experimental findings and your internal physics knowledge.\n\n"
        "Return ONLY the schema below:\n"
        "[THINK] {literature comparison and knowledge inference} [/THINK]\n"
        "[LABEL] {one of \"none\", \"lof\", \"balling\", \"keyhole\"} [/LABEL]"
    )
    return prompt

def _call_llm(prompt: str, profile: Optional[str] = "default") -> str:
    client, model, default_kwargs = make_chat_client(profile=profile)
    system_msg = "You are an LPBF process analysis assistant."
    
    if "model" in default_kwargs:
        default_kwargs.pop("model")

    is_google = False
    try:
        from google import genai
        if isinstance(client, genai.Client):
            is_google = True
    except Exception:
        pass

    if is_google:
        full_prompt = f"{system_msg}\n\n{prompt}"
        resp = client.models.generate_content(
            model=model,
            contents=full_prompt,
            **default_kwargs
        )
        final_text = []
        if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    final_text.append(part.text)
        return "".join(final_text).strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        reasoning_effort="low",
        **default_kwargs
    )
    return resp.choices[0].message.content.strip()

def _get_output_path(stem: str) -> str:
    base_dir = f"./results_AM/AMagent_ZS_{LLM_PROFILE}"
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"ZS_{LLM_PROFILE}_preds_{stem}.csv")

def _append_result(stem: str, row_dict: dict) -> None:
    out_path = _get_output_path(stem)
    fieldnames = ["row_idx", "material", "Power", "Velocity", "zs_response", "zs_label", "zs_think", "gt_label"]
    
    write_header = not os.path.exists(out_path)
    
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)

def run_experiment(stem: str):
    print(f"\n--- Running Zero-Shot for {stem} ---")
    df_test = _load_exp_split(stem)
    
    out_path = _get_output_path(stem)
    processed_indices = set()
    if os.path.exists(out_path):
        try:
            df_curr = pd.read_csv(out_path)
            if "row_idx" in df_curr.columns:
                processed_indices = set(df_curr["row_idx"].astype(int))
        except Exception:
            pass
            
    correct_count = 0
    total_count = 0
    
    for idx, row in df_test.iterrows():
        row_id = idx 
        if "row_idx" in row:
             row_id = int(row["row_idx"])
        
        if row_id in processed_indices:
            continue
            
        gt_raw = row.get(LABEL_COL, "unknown")
        gt_norm = _normalize_ground_truth_label(gt_raw)
        
        prompt = _build_zs_prompt(row)
        
        try:
            resp = _call_llm(prompt, profile=LLM_PROFILE)
            lbl = _extract_label_from_response(resp)
            think = _extract_think_block(resp)
        except Exception as e:
            print(f"  [ERROR] LLM call failed for row {row_id}: {e}")
            resp = f"Error: {e}"
            lbl = "error"
            think = ""
            
        res_dict = {
            "row_idx": row_id,
            "material": row.get(META_COL, ""),
            "Power": row.get("Power", ""),
            "Velocity": row.get("Velocity", ""),
            "zs_response": resp,
            "zs_label": lbl,
            "zs_think": think,
            "gt_label": gt_norm
        }
        _append_result(stem, res_dict)
        
        if lbl == gt_norm and gt_norm != "unknown":
            correct_count += 1
        total_count += 1
        if total_count % 5 == 0:
            print(f"  Processed {total_count} rows...")

    print(f"  Completed {stem}.")


def main():
    print(f"Starting Zero-Shot Benchmark (Profile: {LLM_PROFILE})...")
    for stem in EXPERIMENTS:
        run_experiment(stem)
    
    print("\n--- Generating Metrics Summary ---")
    
    SEARCH_PATTERN = f"./results_AM/AMagent_ZS_{LLM_PROFILE}/ZS_{LLM_PROFILE}_preds_*.csv"
    OUT_METRICS = f"./results_AM/AMagent_ZS_{LLM_PROFILE}/zs_metrics_summary.csv"
    
    files = glob.glob(SEARCH_PATTERN)
    summary_rows = []
    
    for f in sorted(files):
        basename = os.path.basename(f)
        prefix = f"ZS_{LLM_PROFILE}_preds_"
        if basename.startswith(prefix):
            exp_name = os.path.splitext(basename)[0].replace(prefix, "")
        else:
            exp_name = basename
            
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] Failed to read {f}: {e}")
            continue
            
        if "gt_label" not in df.columns or "zs_label" not in df.columns:
            print(f"[WARN] Missing labels in {f}, skipping.")
            continue
            
        y_true = [_normalize_ground_truth_label(l) for l in df["gt_label"].tolist()]
        y_pred = [_normalize_ground_truth_label(l) for l in df["zs_label"].tolist()]
        
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        row_metrics = {
            "experiment": exp_name,
            "n_samples": len(df),
            "zs_acc": round(acc, 4),
            "zs_f1_macro": round(f1_macro, 4),
            "zs_f1_weighted": round(f1_weighted, 4)
        }
        summary_rows.append(row_metrics)
        
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        cols = ["experiment", "n_samples", "zs_acc", "zs_f1_macro", "zs_f1_weighted"]
        df_summary = df_summary[cols]
        
        os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
        df_summary.to_csv(OUT_METRICS, index=False)
        print(f"Saved summary to: {OUT_METRICS}")
        print(df_summary)
    else:
        print("No results found to summarize.")

    print("All experiments done.")

if __name__ == "__main__":
    main()
