import os
import re
import csv
import random
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from openai import OpenAI, AzureOpenAI
from app.config import config, LLMSettings
import glob
import json
from google import genai
from google.genai import types

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
randomState = 42

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

VALID_LABELS = {"none", "lof", "balling", "keyhole"}

BEST_MODELS = {
    "Exp_ID_1": "RF",
    "Exp_OOD_1": "RF",
    "Exp_ID_2": "RF",
    "Exp_OOD_2": "GB",
    "Exp_ID_3": "RF",
    "Exp_OOD_3": "RF",
    "Exp_ID_4": "RF",
    "Exp_OOD_4": "GB",
}

def _exp_type_from_stem(stem: str) -> str:
    return "in-distribution" if "_ID_" in stem or stem.endswith("_ID") else "out-of-distribution"


class RAGLoader:
    """
    Loads result.jsonl files from results_AM/{Material}_Search/results.jsonl
    and provides random sampling of summaries.
    """
    def __init__(self, base_dir: str = "./results_AM"):
        self.base_dir = base_dir
        self.cache: Dict[str, List[str]] = {}
        
        self.mat_to_folder = {
            "SS316L": "SS316L_Search",
            "Ti-6Al-4V": "Ti-6Al-4V_Search",
            "IN718": "IN718_Search",
            "17-4PH": "SS17-4PH_Search", 
        }

    def load_material_corpus(self, material_key: str) -> List[str]:
        if material_key in self.cache:
            return self.cache[material_key]
            
        folder = self.mat_to_folder.get(material_key)
        if not folder:
            return []
            
        path = os.path.join(self.base_dir, folder, "results.jsonl")
        if not os.path.exists(path):
            print(f"[WARN] RAG file not found: {path}")
            return []
            
        summaries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    summ = rec.get("summary", "").strip()
                    if len(summ) < 50: 
                        continue
                    if "No relevant parameter-defect information" in summ:
                        continue
                    summaries.append(summ)
                except Exception:
                    continue
        
        self.cache[material_key] = summaries
        print(f"[RAG] Loaded {len(summaries)} items for {material_key}")
        return summaries

    def get_context(self, material: str, k: int = 5, seed: int = 42) -> List[str]:
        """
        Returns k random summaries for the material.
        """
        canonical = _canonicalize_material(material)
        if not canonical:
            return []
            
        pool = self.load_material_corpus(canonical)
        if not pool:
            return []
            
        rng = random.Random(seed)
        k_actual = min(k, len(pool))
        return rng.sample(pool, k_actual)

_RAG_LOADER = None

def get_rag_loader() -> RAGLoader:
    global _RAG_LOADER
    if _RAG_LOADER is None:
        _RAG_LOADER = RAGLoader()
    return _RAG_LOADER

MATERIAL_ALIASES: Dict[str, str] = {
    "ss316l": "SS316L",
    "stainless steel 316l": "SS316L",
    "aisi 316l": "SS316L",
    "316l": "SS316L",
    "316l stainless steel": "SS316L",

    "ti-6al-4v": "Ti-6Al-4V",
    "ti6al4v": "Ti-6Al-4V",
    "ti 6al 4v": "Ti-6Al-4V",
    "ti64": "Ti-6Al-4V",
    "grade 5": "Ti-6Al-4V",

    "in718": "IN718",
    "inconel 718": "IN718",
    "alloy 718": "IN718",
    "nickel alloy 718": "IN718",
    "ni-based superalloy 718": "IN718",

    "ss17-4ph": "17-4PH",
    "17-4ph": "17-4PH",
    "17-4 ph": "17-4PH",
    "aisi 17-4ph": "17-4PH",
    "17-4 precipitation hardening steel": "17-4PH",
    "17-4ph stainless steel": "17-4PH",
}

def _canonicalize_material(name: Optional[str]) -> Optional[str]:
    if not name or (isinstance(name, float) and pd.isna(name)):
        return None
    key = str(name).strip().lower()
    return MATERIAL_ALIASES.get(key, None)


def _load_exp_split(stem: str) -> pd.DataFrame:
    test_path = os.path.join(EXP_DIR, f"{stem}_test.csv")
    print("Loading test split:", test_path)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test split: {test_path}")
    df_test = pd.read_csv(test_path)
    return df_test

def _load_exp_train(stem: str) -> Optional[pd.DataFrame]:
    train_path = os.path.join(EXP_DIR, f"{stem}_train.csv")
    print("Loading train split:", train_path)
    if not os.path.exists(train_path):
        return None
    return pd.read_csv(train_path)

def _get_val_with_unit(row: pd.Series, col: str, unit: str, fallback: str = "unknown") -> str:
    if col in row and pd.notnull(row[col]):
        return f"{row[col]} {unit}"
    else:
        return f"{fallback} {unit}"

def _calculate_entropy(probs: Dict[str, float], classes=None) -> float:
    """Shannon entropy in bits."""
    if not probs:
        return 0.0
    if classes is None:
        values = np.array(list(probs.values()), dtype=float)
    else:
        values = np.array([probs.get(c, 0.0) for c in classes], dtype=float)

    s = values.sum()
    if s <= 0:
        return 0.0
    values = values / s  

    values = values[values > 0]
    return float(-np.sum(values * np.log2(values)))

def _ml_reliability_from_probs(ml_probs: Dict[str, float], is_ood: bool,
                              classes=("none","lof","balling","keyhole")) -> float:
    values = np.array([ml_probs.get(c, 0.0) for c in classes], dtype=float)
    s = values.sum()
    if s <= 0:
        return 0.1
    values = values / s

    entropy = _calculate_entropy(ml_probs, classes=classes)
    K = len(classes)
    h_norm = entropy / np.log2(K) if K > 1 else 0.0  

    p_sorted = np.sort(values)[::-1]
    margin = float(p_sorted[0] - p_sorted[1]) if len(p_sorted) >= 2 else 1.0

    reliability = (1.0 - h_norm) * (0.5 + 0.5 * margin)

    if is_ood:
        reliability *= 0.5

    return float(np.clip(reliability, 0.05, 0.9))


def _build_kd_agent_prompt(row: pd.Series,
                             rag_context: Optional[List[str]] = None,
                             suggested_label: Optional[str] = None
                           ) -> str:
    """
    Agent 2: Knowledge-Driven Analyst.
    Focuses on process parameters, retrieved scientific literature, and internal knowledge.
    Outputs: [THINK], [ASSUMPTIONS], [RELIABILITY], [LABEL]
    """
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
        "Do not assume a defect is present unless evidence strongly favors a defect."
        "Retrieved Literature Evidence:\n"
    )

    if rag_context:
        for i, s in enumerate(rag_context, 1):
            prompt += f"[{i}] {s}\n"
    else:
        prompt += "No specific literature found.\n"

    prompt += (
        "\nTask:\n"
        "1. Compare the target parameters with the evidence and your internal knowledge.\n"
    )


    prompt += (
        "2. Check for the 'Process Window': If parameters fall within reported optimal ranges for high density, the label is 'none'.\n"
        "3. Assess your reliability:\n"
        "   - Use the retrieved evidence to validate your predictions. Direct matches increase reliability (0.7-1.0).\n"
        "   - If RAG evidence is missing or weak, rely on your internal physics knowledge. If your theoretical analysis is confident, you may assign MEDIUM to HIGH reliability (0.3-0.7).\n"
        "   - Only assign LOW reliability (0.1-0.3) if you lack both external evidence and internal theoretical confidence.\n"
        "4. List assumptions:\n"
        "   - Many papers focus on failures; do not assume a defect exists if parameters look nominal/standard.\n"
        "5. Estimate your belief distribution over defects, including 'none'.\n"
        "   - Unless evidence includes a clear mechanism indicating failure (e.g., very low overlap / extreme low energy, or explicit keyhole indicators), assign at least 0.1 probability to 'none'.\n"
        "6. Conclude with a single label.\n\n"
        "Return ONLY the schema below:\n"
        "[THINK] {literature comparison and knowledge inference} [/THINK]\n"
        "[ASSUMPTIONS] {list of assumptions and numeric mismatch warnings} [/ASSUMPTIONS]\n"
        "[RELIABILITY] {0.0 to 1.0} [/RELIABILITY]\n"
        "[BELIEF] {\"none\": 0.X, \"lof\": 0.X, \"balling\": 0.X, \"keyhole\": 0.X} [/BELIEF]\n"
        "[LABEL] {one of \"none\", \"lof\", \"balling\", \"keyhole\"} [/LABEL]"
    )
    return prompt


def _build_supervisor_prompt(row: pd.Series,
                             ml_probs: Dict[str, float],
                             ml_entropy: float,
                             ml_reliability: float,
                             rag_response: str,
                             fused_belief: Dict[str, float],
                             fused_label: str,
                             exp_type: str = "in-distribution") -> str:
    """
    Supervisor Agent: Senior Engineer.
    Validates and explains the deterministically fused results.
    """
    material = row[META_COL] if META_COL in row and pd.notnull(row[META_COL]) else "unknown material"
    power_str           = _get_val_with_unit(row, "Power",            "W")
    velocity_str        = _get_val_with_unit(row, "Velocity",         "mm/s")
    beam_diam_str       = _get_val_with_unit(row, "beam D",           "µm")
    layer_thickness_str = _get_val_with_unit(row, "layer thickness",  "µm")
    hatch_str           = _get_val_with_unit(row, "Hatch spacing",    "µm")

    f_belief_str = ", ".join([f'"{k}": {v:.2f}' for k,v in fused_belief.items()])

    none_p = ml_probs.get("none", 0.0)
    lof_p = ml_probs.get("lof", 0.0)
    ball_p = ml_probs.get("balling", 0.0)
    key_p = ml_probs.get("keyhole", 0.0)

    ml_output_str = (
        f"- \"none\": {none_p:.4f}\n- \"LoF\": {lof_p:.4f}\n- \"balling\": {ball_p:.4f}\n- \"keyhole\": {key_p:.4f}\n"
        f"- Prediction Entropy: {ml_entropy:.2f}\n"
        f"- Calculated Reliability: {ml_reliability:.2f}\n"
        f"Note: This model was trained on {exp_type} data."
    )

    prompt = (
        "You are a Senior AM Process Engineer (Supervisor). "
        "Your task is to assess in detail the potential imperfections for Laser Powder Bed Fusion printing "
        f"that arise in {material} manufactured at {power_str}, utilizing a {beam_diam_str} beam, "
        f"traveling at {velocity_str}, with a layer thickness of {layer_thickness_str} and hatch spacing of {hatch_str}. "
        f"Specifically, consider whether these parameters respect the typical process window for {material}. "
        "Review the automated Probabilistic Fusion analysis and provide the final decision.\n\n"
        "Agent 1 (Data-Driven ML Analyst) - DIRECT PREDICTIONS:\n"
        f"{ml_output_str}\n\n"
        "Agent 2 (Knowledge-Driven Analyst):\n"
        f"{rag_response}\n\n"
        "--- FUSION ENGINE OUTPUT ---\n"
        f"Computed Fused Belief: {{{f_belief_str}}}\n"
        f"Suggested Label: {fused_label}\n"
        "----------------------------\n\n"
        "Task:\n"
        "1. Start by outputting the [FUSED_BELIEF] exactly as computed above.\n"
        "2. Review the Suggested Label. Does it align with the Agent evidence?\n"
        "   - Use physical indicators (VED, Overlap) only as a sanity check for extreme outliers.\n"
        "   - Do NOT override the fused result unless parameters are physically impossible for the predicted defect (e.g. keyhole at zero power).\n"
        "   - Otherwise, respect the fusion.\n"
        "3. Provide a Defect Risk Profile and Safe Adjustment recommendation.\n"
        "4. Conclude with the final [LABEL].\n\n"
        "Return ONLY the schema below:\n"
        "[THINK] {validation of fusion and mechanistic check} [/THINK]\n"
        "[FUSED_BELIEF] {\"none\": 0.X, \"lof\": 0.X, \"balling\": 0.X, \"keyhole\": 0.X} [/FUSED_BELIEF]\n"
        "[DEFECT_RISK_PROFILE] {summary of risk} [/DEFECT_RISK_PROFILE]\n"
        "[SAFE_ADJUSTMENT] {recommendation} [/SAFE_ADJUSTMENT]\n"
        "[LABEL] {one of \"none\", \"lof\", \"balling\", \"keyhole\"} [/LABEL]"
    )
    return prompt


def _call_llm(prompt: str,
                        profile: Optional[str] = "default") -> str:
    client, model, default_kwargs = make_chat_client(profile=profile)
    system_msg = (
        "You are an LPBF process analysis assistant and act as an LPBF defect classification model."
    )
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
        
        full_response = "".join(final_text).strip()
        
        if not full_response:
             return "Error: Empty response from model (possibly thinking timeout or filter)."
        return full_response

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


def _deterministic_fusion(ml_probs_raw: Dict[str, float],
                          ml_reliability: float,
                          rag_resp: str,
                          is_ood: bool = False) -> Tuple[Dict[str, float], str, str]:
    """
    Performs Python-side deterministic fusion of agent outputs.
    Returns: (fused_belief_dict, fused_label, reasoning_summary)
    """
    default_belief = {l: 0.0 for l in LABEL_ORDER}
    
    ml_belief = ml_probs_raw.copy()
    ml_rel = ml_reliability
    
    rag_belief = _extract_json_block(rag_resp, "BELIEF") or default_belief
    rag_rel = _extract_float_block(rag_resp, "RELIABILITY")
    if rag_rel is None: rag_rel = 0.3 
    
    for b in [ml_belief, rag_belief]:
        total = sum(b.get(k, 0) for k in LABEL_ORDER)
        if total > 0:
            for k in LABEL_ORDER:
                b[k] = b.get(k, 0) / total
    
    if not is_ood:
        ml_rel *= 4.0

    fused_belief = {k: 0.0 for k in LABEL_ORDER}
    denom = ml_rel + rag_rel
    
    if denom > 0:
        for k in LABEL_ORDER:
            val = (ml_belief.get(k, 0.0) * ml_rel + rag_belief.get(k, 0.0) * rag_rel) / denom
            fused_belief[k] = val
    else:
        fused_belief = ml_belief.copy()

    fused_label = max(fused_belief, key=fused_belief.get)
    return fused_belief, fused_label


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

def _extract_json_block(text: str, tag: str) -> Optional[Dict[str, float]]:
    """Extracts a JSON dict from within [TAG]...[/TAG]."""
    regex = re.compile(f"\[{tag}\]\s*(.+?)\s*\[/{tag}\]", flags=re.IGNORECASE | re.DOTALL)
    m = regex.search(text)
    if not m:
        return None
    try:
        content = m.group(1).replace("'", '"')
        return json.loads(content)
    except Exception:
        return None

def _extract_float_block(text: str, tag: str) -> Optional[float]:
    """Extracts a float from within [TAG]...[/TAG]."""
    regex = re.compile(f"\[{tag}\]\s*([0-9\.]+)\s*\[/{tag}\]", flags=re.IGNORECASE | re.DOTALL)
    m = regex.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

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


def _get_output_paths(stem: str, model_tag: str) -> str:
    base_dir = f"./results_AM/AMagent_{model_tag}"
    os.makedirs(base_dir, exist_ok=True)
    fname = f"{model_tag}_raw_preds_{stem}.csv"
    return os.path.join(base_dir, fname)

def _load_partial_results(stem: str, model_tag: str) -> pd.DataFrame:
    out_path = _get_output_paths(stem, model_tag)
    cols = [
        "row_idx",
        "material",
        "Power",
        "Velocity",
        "beam D",
        "layer thickness",
        "Hatch spacing",
        "agent_rag_response",
        "agent_rag_label",
        "supervisor_raw_response",
        "supervisor_label",
        "prompts_debug",
        "ml_pred_label",
        "gt_label",
    ]
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        for c in cols:
            if c not in df.columns:
                df[c] = "" 
        df = df[cols]
    else:
        df = pd.DataFrame(columns=cols)
    return df


def _append_partial_result(stem: str, row_dict: dict, model_tag: str) -> None:
    out_path = _get_output_paths(stem, model_tag)
    fieldnames = [
        "row_idx",
        "material",
        "Power",
        "Velocity",
        "beam D",
        "layer thickness",
        "Hatch spacing",
        "agent_rag_response",
        "agent_rag_label",
        "supervisor_raw_response",
        "supervisor_label",
        "prompts_debug",
        "ml_pred_label",
        "gt_label",
    ]

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
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df_old.to_csv(out_path, index=False, encoding="utf-8") 
            write_header = False 

    mode = "a" if os.path.exists(out_path) else "w"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not os.path.exists(out_path) or write_header:
            writer.writeheader()
        writer.writerow(row_dict)


def _train_ml_and_predict_proba(stem: str, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """
    Returns: dict row_idx -> {label: prob} with labels in LABEL_ORDER.
    Uses features: Power, Velocity, beam D, layer thickness, material (one-hot).
    """
    model_name = BEST_MODELS.get(stem, "RF")
    num_feats = ["Power", "Velocity", "beam D", "layer thickness", "Hatch spacing"]
    cat_feats = ["material"]

    cols_needed = [c for c in num_feats + cat_feats if c in df_train.columns]
    if LABEL_COL not in df_train.columns:
        raise RuntimeError(f"{stem}: '{LABEL_COL}' not found in train set.")

    X_train = df_train[cols_needed].copy()
    y_train = df_train[LABEL_COL].copy()

    X_test = df_test.reindex(columns=cols_needed, fill_value=np.nan).copy()

    num_cols_present = [c for c in num_feats if c in cols_needed]
    cat_cols_present = [c for c in cat_feats if c in cols_needed]

    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if num_cols_present:
        transformers.append(("num", num_tf, num_cols_present))
    if cat_cols_present:
        transformers.append(("cat", cat_tf, cat_cols_present))
    pre = ColumnTransformer(transformers)

    from sklearn.utils.class_weight import compute_sample_weight

    if model_name == "GB":
        clf = GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=20,
        random_state=randomState,
        loss="log_loss",
    )
    else:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=randomState,
            n_jobs=-1,
            class_weight="balanced", 
        )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])

    y_train_norm = y_train.apply(_normalize_ground_truth_label)
    mask = y_train_norm.isin(LABEL_ORDER)
    X_train = X_train[mask]
    y_train_norm = y_train_norm[mask]

    if len(X_train) == 0:
        raise RuntimeError(f"{stem}: No valid training rows after label normalization.")

    if model_name == "GB":
        weights = compute_sample_weight(class_weight="balanced", y=y_train_norm)
        pipe.fit(X_train, y_train_norm, **{'clf__sample_weight': weights})
    else:
        pipe.fit(X_train, y_train_norm)

    proba = pipe.predict_proba(X_test)
    cls_order = list(pipe.classes_)

    proba_by_row: Dict[int, Dict[str, float]] = {}
    for ridx, idx in enumerate(df_test.index):
        row_map = {c: 0.0 for c in LABEL_ORDER}
        for c_idx, c_name in enumerate(cls_order):
            row_map[c_name] = float(proba[ridx][c_idx])
        row_map = {k: round(row_map[k], 4) for k in LABEL_ORDER}
        proba_by_row[int(idx)] = row_map

    return proba_by_row


def _argmax_label_from_probs(prob_map: Dict[str, float]) -> str:
    if not prob_map:
        return "unknown"
    best_lbl = "unknown"
    best_p = -1.0
    for lbl in LABEL_ORDER:
        p = float(prob_map.get(lbl, 0.0))
        if p > best_p:
            best_p = p
            best_lbl = lbl
    return best_lbl

def evaluate_resume(
    stem: str,
    supervisor_profile: str = "default",
    sub_agent_profile: str = "deepseek",
) -> Dict[str, float]:
    df_test = _load_exp_split(stem)
    if LABEL_COL not in df_test.columns:
        raise RuntimeError(f"{stem}: '{LABEL_COL}' not found in test set.")

    df_test = df_test.copy()
    if "row_idx" not in df_test.columns:
        df_test["row_idx"] = df_test.index

    df_train = _load_exp_train(stem)
    exp_type = _exp_type_from_stem(stem)
    ml_probs_by_row: Dict[int, Dict[str, float]] = {}
    if df_train is not None:
        try:
            ml_probs_by_row = _train_ml_and_predict_proba(stem, df_train, df_test)
        except Exception as e:
            print(f"[WARN] ML training failed for {stem}: {e}")

    df_partial = _load_partial_results(stem, model_tag=supervisor_profile)
    done_row_idxs = set(df_partial["row_idx"].tolist())

    rag_loader = get_rag_loader()

    for idx, row in df_test.iterrows():
        if idx in done_row_idxs:
            continue

        gt_label_str = _normalize_ground_truth_label(row[LABEL_COL])

        ml_probs = ml_probs_by_row.get(int(idx), {})
        ml_pred_label = _argmax_label_from_probs(ml_probs)

        material_name = row.get("material", "")
        seed = int(idx) + randomState
        rag_context = rag_loader.get_context(material_name, k=5, seed=seed)

        etype = exp_type if exp_type else "in-distribution"
        is_ood_model = (etype == "out-of-distribution")
        
        ml_entropy = _calculate_entropy(ml_probs, classes=tuple(LABEL_ORDER))
        ml_reliability = _ml_reliability_from_probs(ml_probs, is_ood_model, classes=tuple(LABEL_ORDER))

        
        resp_text_ml_clean = "" 
        agent_ml_label = ml_pred_label 

        prompt_rag = _build_kd_agent_prompt(row, rag_context=rag_context)
        try:
            resp_text_rag = _call_llm(prompt_rag, profile=sub_agent_profile)
            agent_rag_label = _extract_label_from_response(resp_text_rag)
            resp_text_rag_clean = resp_text_rag.replace("\n", " ").replace("\r", " ")
        except Exception as e:
            resp_text_rag_clean = f"[ERROR] {e}"
            agent_rag_label = "unknown"

        is_ood_exp = "OOD" in stem
        
        fused_belief, fused_label = _deterministic_fusion(
            ml_probs_raw=ml_probs,
            ml_reliability=ml_reliability,
            rag_resp=resp_text_rag_clean,
            is_ood=is_ood_exp
        )

        prompt_sup = _build_supervisor_prompt(
            row, 
            ml_probs=ml_probs,
            ml_entropy=ml_entropy,
            ml_reliability=ml_reliability,
            rag_response=resp_text_rag_clean,
            fused_belief=fused_belief,
            fused_label=fused_label,
            exp_type=etype
        )
        try:
            resp_text_sup = _call_llm(prompt_sup, profile=supervisor_profile)
            supervisor_label = _extract_label_from_response(resp_text_sup)
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
                "probs": ml_probs
            }
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

        _append_partial_result(stem, row_record, model_tag=supervisor_profile)

    df_results = _load_partial_results(stem, model_tag=supervisor_profile)
    df_results = df_results[df_results["row_idx"].isin(df_test.index)].copy()

    df_join = df_results.merge(
        df_test[[LABEL_COL, "material", "Power", "Velocity", "beam D", "layer thickness"]],
        left_on="row_idx",
        right_index=True,
        how="left",
        suffixes=("", "_true"),
    )

    df_join["gt_label_canon"]  = df_join[LABEL_COL].apply(_normalize_ground_truth_label)
    df_join["pred_label_canon"] = df_join["supervisor_label"].astype(str).str.lower()
    
    df_join["match_flag"] = df_join["gt_label_canon"] == df_join["pred_label_canon"]

    preds = df_join["pred_label_canon"].tolist()
    gts   = df_join["gt_label_canon"].tolist()
    valid_mask = np.array([(p in VALID_LABELS) and (g in VALID_LABELS) for p, g in zip(preds, gts)], dtype=bool)

    if not np.any(valid_mask):
        macro_f1 = float("nan")
        n_scored = 0
    else:
        preds_valid = [preds[i] for i in range(len(preds)) if valid_mask[i]]
        gts_valid   = [gts[i]   for i in range(len(gts))   if valid_mask[i]]
        macro_f1 = f1_score(gts_valid, preds_valid, average="macro")
        n_scored = len(preds_valid)

    result = {
        "experiment": stem,
        "macro_f1": float(macro_f1),
        "n_test_total": len(df_test),
        "n_scored": n_scored,
        "n_completed_rows": len(df_results),
    }
    return result


def main():
    SUPERVISOR_PROFILE = "gpt5"      
    SUB_AGENT_PROFILE = "gemini"     

    print(f"Running benchmark with supervisor={SUPERVISOR_PROFILE}, sub_agents={SUB_AGENT_PROFILE}")
    
    for stem in EXPERIMENTS:
        print("\n=======================================")
        print(f" evaluation on {stem} (resumable)")
        print("=======================================")

        try:
            res = evaluate_resume(
                stem,
                supervisor_profile=SUPERVISOR_PROFILE,
                sub_agent_profile=SUB_AGENT_PROFILE
            )
            print(
                f"{stem}: macro-F1 = {res.get('macro_f1', 0.0):.4f} | "
                f"scored {res.get('n_scored', 0)}/{res.get('n_test_total', 0)} rows | "
                f"completed rows stored: {res.get('n_completed_rows', 0)}"
            )
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
            continue

    print("\n--- Generating Metrics Summary ---")
    
    SEARCH_PATTERN = f"./results_AM/AMagent_{SUPERVISOR_PROFILE}/{SUPERVISOR_PROFILE}_raw_preds_*.csv"
    OUT_METRICS = f"./results_AM/AMagent_{SUPERVISOR_PROFILE}/metrics_summary.csv"
    
    files = glob.glob(SEARCH_PATTERN)
    
    summary_rows = []
    
    targets = [
        ("ml_pred", "ml_pred_label"),
        ("agent_rag", "agent_rag_label"),
        ("supervisor", "supervisor_label"),
    ]

    for f in sorted(files):
        basename = os.path.basename(f)
        prefix = f"{SUPERVISOR_PROFILE}_raw_preds_"
        if basename.startswith(prefix):
            exp_name = os.path.splitext(basename)[0].replace(prefix, "")
        else:
            exp_name = basename

        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] Failed to read {f}: {e}")
            continue
            
        if "gt_label" not in df.columns:
            print(f"[WARN] Missing 'gt_label' in {f}, skipping.")
            continue
        
        y_true_raw = df["gt_label"].tolist()
        y_true = [_normalize_ground_truth_label(l) for l in y_true_raw]
        
        row_metrics = {
            "experiment": exp_name, 
            "n_samples": len(df)
        }
        
        try:
            df_test = _load_exp_split(exp_name)
            row_metrics["n_total_test"] = len(df_test)
            if len(df) < len(df_test):
                print(f"[WARN] Experiment {exp_name} incomplete ({len(df)}/{len(df_test)} rows).")
        except:
            row_metrics["n_total_test"] = "unknown"

        for target_name, col_name in targets:
            resp_col = None
            if target_name == "agent_ml":
                resp_col = "agent_ml_response"
            elif target_name == "agent_rag":
                resp_col = "agent_rag_response"
            elif target_name == "supervisor":
                resp_col = "supervisor_raw_response"
            
            y_pred_derived = []
            
            if resp_col and resp_col in df.columns:
                responses = df[resp_col].fillna("").astype(str).tolist()
                for r in responses:
                    y_pred_derived.append(_extract_label_from_response(r))
            elif col_name in df.columns:
                raw_labels = df[col_name].fillna("unknown").astype(str).tolist()
                y_pred_derived = [_normalize_ground_truth_label(l) for l in raw_labels]
            else:
                print(f"[WARN] Column {col_name} missing in {f}")
                row_metrics[f"{target_name}_acc"] = None
                row_metrics[f"{target_name}_f1_macro"] = None
                row_metrics[f"{target_name}_f1_weighted"] = None
                continue
            
            y_pred = [_normalize_ground_truth_label(l) for l in y_pred_derived]
            
            acc = accuracy_score(y_true, y_pred)
            
            f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
            
            f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            
            row_metrics[f"{target_name}_acc"] = round(acc, 4)
            row_metrics[f"{target_name}_f1_macro"] = round(f1_macro, 4)
            row_metrics[f"{target_name}_f1_weighted"] = round(f1_weighted, 4)

        summary_rows.append(row_metrics)

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        base_cols = ["experiment", "n_samples", "n_total_test"]
        metric_cols = [c for c in df_summary.columns if c not in base_cols]
        metric_cols.sort()
        
        final_cols = base_cols + metric_cols
        df_summary = df_summary[final_cols]
        
        os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
        df_summary.to_csv(OUT_METRICS, index=False)
        print(f"Saved extended metrics summary to: {OUT_METRICS}")
        
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(df_summary)
    else:
        print("No valid results found to summarize.")

if __name__ == "__main__":
    main()
