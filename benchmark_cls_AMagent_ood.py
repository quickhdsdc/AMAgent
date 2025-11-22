import os
import re
import csv
import random
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from openai import OpenAI, AzureOpenAI
from app.config import config, LLMSettings
import glob
# ----------------------------------------------------
# Constants / Labels
# ----------------------------------------------------

LABEL_ORDER = ["none", "lof", "balling", "keyhole"]  # 0,1,2,3
VALID_LABELS = set(LABEL_ORDER)

# ----------------------------------------------------
# LLM config helpers
# ----------------------------------------------------

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
        client = OpenAI(api_key=llm.api_key)
        model = llm.model
        default_kwargs = {
            "model": model,
            "max_completion_tokens": llm.max_completion_tokens,
            "temperature": llm.temperature,
        }
        return client, model, default_kwargs

    else:
        raise ValueError(f"Unsupported api_type: {llm.api_type!r}")

# ----------------------------------------------------
# Experiment Config
# ----------------------------------------------------

EXP_DIR = "data_exp"
LABEL_COL = "defect_label"
META_COL = "material"
randomState = 42  # used for deterministic sampling

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

# Best model per experiment (from prior study)
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

# ----------------------------------------------------
# Domain knowledge per material (extend as needed)
# ----------------------------------------------------

DOMAIN_KNOWLEDGE: Dict[str, List[str]] = {
    "SS316L": [
        "[1] In LPBF of IN718 and Ti alloys, process windows typically involve beam powers from 100 to 370 W and scan speeds from 200 to 1400 mm/s. These produce melt pools with widths around 350 µm and lengths up to 860 µm, with preheating near 80 °C and beam diameters of about 100 µm. Stable processing occurs when hatch spacing (~500 µm conservative for 20 mm tracks) and layer thickness (~40–70 µm effective) are tuned to ensure sufficient overlap without overheating.",
        "[2] Across these parameters, melt pool depth-to-width ratios greater than 1 indicate keyhole-mode melting, but only limited porosity formation occurs, suggesting that deep melt pools do not necessarily lead to defects if other parameters remain balanced. Excessive energy density or focus misalignment can, however, increase porosity, surface roughness, or keyhole instability, while low power or high scan speed leads to lack-of-fusion.",
        "[3] Layer uniformity and powder handling critically influence consistency: a 70 ± 20 µm applied powder layer yields an effective 40 µm consolidated layer, with recoater control keeping thickness variation below 10 µm. Adjusting the laser focus changes spot size and local energy density—too large a spot reduces dimensional precision, while too tight a focus risks excessive penetration and local distortion.",
        "[4] For 316L stainless steel, most studies emphasize the relationship between laser energy input and densification rather than exact numeric windows. Reported powers reach 400 W with layer thicknesses near 250 µm, producing high densities (> 99%) when energy input is optimized. However, keyhole porosity appears above roughly 90 J/mm³ VED, while lack-of-fusion dominates below 50 J/mm³, consistent with general LPBF behavior.",
        "[5] Variations in laser power, hatch spacing, and layer thickness continue to affect microhardness, surface quality, and porosity even at constant energy density. Overlap ratio and scan strategy significantly influence residual stress and defect distribution. Stable builds require balancing power and speed to avoid the two extremes: insufficient melting causing LoF, and excessive power causing keyhole pores or balling.",
        "[6] Overall, both IN718 and 316L studies converge on the principle that optimal LPBF quality depends on maintaining a moderate volumetric energy density (~60–80 J/mm³) with sufficient overlap (≈50%) and melt-pool depth-to-layer ratio of 2–3×, ensuring consistent interlayer fusion and minimal porosity."
    ],
    "Ti-6Al-4V": [
        "[1] In SLM of Ti-6Al-4V, process parameters span wide ranges—laser power exceeding 180 W, scan speed up to 7 m/s, and essentially unrestricted powder layer thickness. Improper combinations within these ranges can cause low-density fusion and defects such as cracking, balling, and deformation, degrading mechanical properties. However, no explicit quantitative relationships are given linking specific laser power, scan speed, hatch spacing, beam diameter, or layer thickness to keyhole formation, lack-of-fusion, porosity, or melt-pool geometry.",
        "[2] For LPBF of Ti-6Al-4V on an EOS M290 (400 W continuous 1064 nm Yb-fiber laser, 100 µm spot size, 30 µm layer thickness), laser power, scan speed, and hatch spacing were varied across samples to fabricate cylinders for XCT porosity analysis and blocks for microstructural study. The available information includes no quantitative porosity data, melt-pool dimensions, or explicit defect observations, so no numerical process–defect relationships can be drawn from this content.",
        "[3] In LPBF of Ti-6Al-4V, key parameters—laser power, spot size, scan speed, hatch distance, layer thickness, and platform preheating—govern the input energy and hence defect formation. High power and low scan speed yield excessive energy, producing overheating, evaporation, and keyholes; conversely, low power and high speed result in insufficient energy density, causing incomplete melting and lack-of-fusion. Improper hatch distance or excessive layer thickness also promotes LoF. Even within acceptable energy density, high power and speed together can trigger balling due to intensified Marangoni flow and Plateau–Rayleigh instability. At an intermediate laser power of 175 W, gradually reducing scan speed transitions through incomplete melting, a stable processing window, and overheating. Gas porosity is also noted but generally remains small.",
        "[4] The combined effects of laser power, scan speed, hatch distance, and layer thickness can be represented by the volumetric energy density VED=P/(v⋅h⋅t). For Ti-6Al-4V, dense and stable melts typically occur around 60–80 J/mm³, while values above ~90 J/mm³ risk overheating and keyholing, and below ~50 J/mm³ cause lack-of-fusion. A representative process map marks transition regions at 98, 74, 49, 37, and 25 J/mm³, delineating keyhole, optimal, and incomplete-melting regimes.",
        "[5] Melt-pool depth and size in Ti-6Al-4V SLM are governed by absorbed laser energy, which depends on laser power, scan speed, hatch spacing, layer thickness, and scan strategy. Optimization of these parameters is essential to suppress defects—such as hydrogen porosity, lack-of-fusion pores, keyholes, cracks, and impurities—and to enhance density and build quality. Feature size additionally alters local thermal profiles, influencing resulting microstructure and mechanical properties."
    ],
    "IN718": [
        "[1] On an EOS M290 system, beam power between 100–370 W and scan velocity between 200–1400 mm/s produced melt pools with an estimated maximum width ≈ 350 µm and length ≈ 860 µm, suggesting a conservative hatch spacing ≈ 500 µm. Tests intentionally included regimes of keyholing and balling, confirming these defects at higher energy inputs.",
        "[2] Across the laser power–velocity space for IN718, linear relationships of constant melt pool width, depth, and area were observed. All cases had depth-to-half-width ratios > 1 (keyhole-mode melting), but only a small subset trapped porosity, implying a stable keyholing regime where deep melt pools can remain defect-free.",
        "[3] Melt pool morphology varied along individual tracks even with constant parameters, showing periodic depth fluctuations and humping-type variability. A large dataset of melt pool geometries for IN718 was established to support in-situ flaw detection.",
        "[4] Maintaining consistent powder layers required ≈ 70 µm ± 20 µm powder layers to achieve an effective nominal 40 µm layer after consolidation (~40% densification). The measured gap-control ensured <10 µm uncertainty, highlighting how layer thickness accuracy affects LPBF reproducibility.",
        "[5] Adjusting laser focus shift in SLM of IN718 changes spot size and local energy density, influencing surface quality and porosity. Proper focus tuning balances dimensional accuracy and sufficient energy to fuse roughly three powder layers; excessive or insufficient energy leads respectively to overheating or lack-of-fusion. Hot-isostatic pressing (HIP) is often required afterward to close internal pores that are not surface-connected.",
        "[6] Overall, studies of IN718 LPBF confirm that laser power, scan speed, and spot size govern melt-pool geometry (width up to ≈ 350 µm, length ≈ 860 µm), with a stable keyholing window existing between lack-of-fusion and excessive-energy regimes; precise control of layer thickness, hatch spacing, and focus position is essential for minimizing porosity and maintaining surface integrity."
    ],
    "17-4PH": [
        "[1] LPBF of 17-4PH stainless steel used process conditions comparable to standard laser parameters, where studies identified and characterized spatter particles that affected surface roughness, density, and mechanical response. Spatter formation, driven by excessive local energy or unstable melt pool dynamics, correlated with degraded surface finish and local density fluctuations, though no explicit numeric power, velocity, or temperature thresholds were reported.",
        "[2] In SLM of 17-4PH martensitic steel, the microstructure distribution and resulting mechanical anisotropy depended strongly on laser power and scan speed. Higher linear energy density (achieved through lower scan speeds or higher power) promoted deeper melt pools and martensitic laths, while excessive energy caused keyhole porosity. Although the extracted text lacks explicit numeric process values, prior quantitative frameworks for similar steels show transition to keyholing typically above ~90 J/mm³ volumetric energy density.",
        "[3] Surface post-processing of 17-4PH SLM parts demonstrated that as-built roughness and residual stress stem largely from incomplete melt track overlap and spatter redeposition. Optimizing overlap ratio and scanning pattern reduces trapped porosity and stress concentration. No exact scan speed or hatch spacing values are cited here, but trends indicate that smaller hatch spacing (≤100 µm) and moderate overlap (>30%) produce denser, smoother surfaces.",
        "[4] For 17-4PH fabricated using a 100 W fiber laser and inside-out hexagon hatching under nitrogen, powder particle size ranged 5–45 µm with up to eight recycling cycles. Literature cited reports that recycling SS17-4PH powder up to 20 cycles tends to reduce internal porosity and surface roughness, suggesting sufficient energy absorption and powder sphericity retention. In contrast, 316L recycled beyond 25 cycles exhibits increased porosity and inclusions, underscoring material-specific powder degradation thresholds.",
        "[5] Comparative 17-4PH studies note that mechanical and microstructural properties are highly sensitive to build orientation and energy input; insufficient laser power or excessive scan speed yield lack-of-fusion defects and poor interlayer bonding, whereas overly high power density promotes keyhole porosity. Experimental builds using moderate powers (100–200 W range) and optimized scan strategies achieve dense microstructures (>99% density) with reduced porosity.",
        "[6] Additional research on 17-4PH highlights the trade-off between heat accumulation and cooling rate: high energy input lowers porosity but risks retained austenite and keyhole formation, while low energy density (below ~50 J/mm³) leads to lack of fusion and high roughness. Appropriate process tuning—balancing power, speed, and hatch spacing—produces stable melt pools, uniform density, and minimal defect generation."
    ]
}

# Aliases to canonical material keys in DOMAIN_KNOWLEDGE
MATERIAL_ALIASES: Dict[str, str] = {
    # SS316L aliases
    "ss316l": "SS316L",
    "stainless steel 316l": "SS316L",
    "aisi 316l": "SS316L",
    "316l": "SS316L",
    "316l stainless steel": "SS316L",

    # Ti-6Al-4V aliases
    "ti-6al-4v": "Ti-6Al-4V",
    "ti6al4v": "Ti-6Al-4V",
    "ti 6al 4v": "Ti-6Al-4V",
    "ti64": "Ti-6Al-4V",
    "grade 5": "Ti-6Al-4V",

    # IN718 aliases
    "in718": "IN718",
    "inconel 718": "IN718",
    "alloy 718": "IN718",
    "nickel alloy 718": "IN718",
    "ni-based superalloy 718": "IN718",

    # 17-4PH aliases
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

def _pick_domain_passages(material: Optional[str], k: int = 5, seed: Optional[int] = None) -> List[str]:
    """
    Resolve material to a canonical key and sample up to k passages.
    If no knowledge is found, return [].
    """
    canonical = _canonicalize_material(material)
    if canonical is None:
        return []
    pool = DOMAIN_KNOWLEDGE.get(canonical, [])
    if not pool:
        return []
    rng = random.Random(seed)
    k = min(k, len(pool))
    return rng.sample(pool, k)

# ----------------------------------------------------
# Utilities: data loading and formatting
# ----------------------------------------------------

def _load_exp_split(stem: str) -> pd.DataFrame:
    test_path = os.path.join(EXP_DIR, f"{stem}_test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test split: {test_path}")
    df_test = pd.read_csv(test_path)
    return df_test

def _load_exp_train(stem: str) -> Optional[pd.DataFrame]:
    train_path = os.path.join(EXP_DIR, f"{stem}_train.csv")
    if not os.path.exists(train_path):
        return None
    return pd.read_csv(train_path)

def _get_val_with_unit(row: pd.Series, col: str, unit: str, fallback: str = "unknown") -> str:
    if col in row and pd.notnull(row[col]):
        return f"{row[col]} {unit}"
    else:
        return f"{fallback} {unit}"

# ----------------------------------------------------
# Prompt builder with RAG + ML block
# ----------------------------------------------------

def _build_prompt_for_row(row: pd.Series,
                          ml_probs: Optional[Dict[str, float]] = None,
                          exp_type: Optional[str] = None) -> str:
    """
    Build the classification prompt for one data sample.
    Includes: core task, domain knowledge (3 snippets), and optional ML probabilities.
    """
    material = row[META_COL] if META_COL in row and pd.notnull(row[META_COL]) else "unknown material"

    power_str           = _get_val_with_unit(row, "Power",            "W")
    velocity_str        = _get_val_with_unit(row, "Velocity",         "mm/s")
    beam_diam_str       = _get_val_with_unit(row, "beam D",           "µm")
    layer_thickness_str = _get_val_with_unit(row, "layer thickness",  "µm")

    prompt = (
        "Your task is to assess in detail the potential imperfections for Laser Powder Bed Fusion printing "
        f"that arise in {material} manufactured at {power_str}, utilizing a {beam_diam_str} beam, "
        f"traveling at {velocity_str}, with a layer thickness of {layer_thickness_str}.\n"
        "Return ONLY the schema below. No extra text, no commentary, no explanations\n"
        "[THINK] {concise justification} [/THINK]\n"
        "[LABEL] {one of \"none\", \"lof\", \"balling\", and \"keyhole\"} [/LABEL]"
    )

    # Domain knowledge block (deterministic sampling via row_idx + seed)
    seed = None
    if "row_idx" in row:
        try:
            seed = int(row["row_idx"]) + randomState
        except Exception:
            seed = randomState
    else:
        seed = randomState

    dk_snippets = _pick_domain_passages(material, k=3, seed=seed)
    if dk_snippets:
        prompt += "\nconsidering the following domain knowledge to predict the defect label:\n"
        for s in dk_snippets:
            prompt += f"- {s}\n"

    # ML classifier block
    if ml_probs is not None and isinstance(ml_probs, dict) and len(ml_probs) > 0:
        none_p = ml_probs.get("none", 0.0)
        lof_p = ml_probs.get("lof", 0.0)
        ball_p = ml_probs.get("balling", 0.0)
        key_p = ml_probs.get("keyhole", 0.0)
        etype = exp_type if exp_type else "in-distribution"

        prompt += (
            "\nPlease also consider the results predicted by a machine learning based classifier:\n"
            f"\"none\": {none_p}, \"LoF\": {lof_p}, \"balling\": {ball_p}, \"keyhole\": {key_p}\n"
            f"This model is trained on the {etype} datasets. If in-distribution, You can more trust the results\n"
        )

    return prompt

# ----------------------------------------------------
# LLM call + parsing
# ----------------------------------------------------

def _call_gpt_zero_shot(prompt: str,
                        profile: Optional[str] = "default") -> str:
    client, model, default_kwargs = make_chat_client(profile=profile)
    system_msg = (
        "You are an LPBF process analysis assistant and act as an LPBF defect classification model."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

_LABEL_REGEX = re.compile(
    r"\[LABEL\]\s*([^\[\]]+?)\s*\[/LABEL\]",
    flags=re.IGNORECASE | re.DOTALL,
)

def _extract_label_from_response(resp_text: str) -> str:
    m = _LABEL_REGEX.search(resp_text)
    if not m:
        return "unknown"

    raw = m.group(1).strip().lower()
    norm = raw.replace("-", " ").strip()
    norm = re.sub(r"\s+", " ", norm)

    if norm in ("none",):
        return "none"
    if norm in ("lof", "lack of fusion", "lack of fusion porosity", "lack of fusion defect"):
        return "lof"
    if norm in ("balling", "ball", "balling defect"):
        return "balling"
    if norm in ("keyhole", "keyhole porosity", "keyholing"):
        return "keyhole"
    return "unknown"

def _normalize_ground_truth_label(y) -> str:
    if y is None or (isinstance(y, float) and pd.isna(y)):
        return "unknown"
    try:
        cls_idx = int(float(y))
    except Exception:
        return "unknown"
    if 0 <= cls_idx < len(LABEL_ORDER):
        return LABEL_ORDER[cls_idx]
    return "unknown"

# ----------------------------------------------------
# Partial results persistence
# ----------------------------------------------------
def _load_partial_results(stem: str) -> pd.DataFrame:
    out_path = f"gpt5_raw_preds_{stem}.csv"
    cols = [
        "row_idx",
        "material",
        "Power",
        "Velocity",
        "beam D",
        "layer thickness",
        "gpt_raw_response",
        "ml_pred_label",
        "gpt_pred_label",
        "gt_label",
    ]
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        # Add any missing columns (e.g., old files) so downstream code can rely on them
        for c in cols:
            if c not in df.columns:
                df[c] = ""  # or np.nan
        # Reorder (and drop any legacy extras)
        df = df[cols]
    else:
        df = pd.DataFrame(columns=cols)
    return df


def _append_partial_result(stem: str, row_dict: dict) -> None:
    out_path = f"gpt5_raw_preds_{stem}.csv"
    fieldnames = [
        "row_idx",
        "material",
        "Power",
        "Velocity",
        "beam D",
        "layer thickness",
        "gpt_raw_response",
        "ml_pred_label",
        "gpt_pred_label",
        "gt_label",
    ]

    write_header = True
    if os.path.exists(out_path):
        # Check existing header
        with open(out_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
        # If header matches exactly, we can just append without header
        write_header = header != fieldnames
        if write_header:
            # Migrate: read whole file, rewrite with new header + aligned columns
            df_old = pd.read_csv(out_path)
            for c in fieldnames:
                if c not in df_old.columns:
                    df_old[c] = ""
            df_old = df_old[fieldnames]
            df_old.to_csv(out_path, index=False, encoding="utf-8")  # rewrites with new header
            write_header = False  # header already written

    mode = "a" if os.path.exists(out_path) else "w"
    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not os.path.exists(out_path) or write_header:
            writer.writeheader()
        writer.writerow(row_dict)

# ----------------------------------------------------
# Classical ML helper: train best model and get per-row probabilities
# ----------------------------------------------------

def _train_ml_and_predict_proba(stem: str, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """
    Returns: dict row_idx -> {label: prob} with labels in LABEL_ORDER.
    Uses features: Power, Velocity, beam D, layer thickness, material (one-hot).
    """
    model_name = BEST_MODELS.get(stem, "RF")
    num_feats = ["Power", "Velocity", "beam D", "layer thickness"]
    cat_feats = ["material"]

    # Build X/y with available columns (silently drop missing features if any)
    cols_needed = [c for c in num_feats + cat_feats if c in df_train.columns]
    if LABEL_COL not in df_train.columns:
        raise RuntimeError(f"{stem}: '{LABEL_COL}' not found in train set.")

    X_train = df_train[cols_needed].copy()
    y_train = df_train[LABEL_COL].copy()

    # Ensure same columns present in test (fill missing cols if needed)
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

    if model_name == "GB":
        clf = GradientBoostingClassifier(random_state=randomState)
    else:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=randomState,
            n_jobs=-1,
        )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])

    # Map integer y -> canonical labels
    y_train_norm = y_train.apply(_normalize_ground_truth_label)
    mask = y_train_norm.isin(LABEL_ORDER)
    X_train = X_train[mask]
    y_train_norm = y_train_norm[mask]

    if len(X_train) == 0:
        raise RuntimeError(f"{stem}: No valid training rows after label normalization.")

    pipe.fit(X_train, y_train_norm)

    # Predict probabilities
    proba = pipe.predict_proba(X_test)
    cls_order = list(pipe.classes_)

    proba_by_row: Dict[int, Dict[str, float]] = {}
    for ridx, idx in enumerate(df_test.index):
        row_map = {c: 0.0 for c in LABEL_ORDER}
        for c_idx, c_name in enumerate(cls_order):
            row_map[c_name] = float(proba[ridx][c_idx])
        # optional rounding for neatness
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
# ----------------------------------------------------
# Evaluation loop (RAG + ML-augmented prompting)
# ----------------------------------------------------
def evaluate_gpt5_on_experiment_with_resume(
    stem: str,
    gpt_profile: str = "gpt5",
) -> Dict[str, float]:
    # 1) Load test split
    df_test = _load_exp_split(stem)
    if LABEL_COL not in df_test.columns:
        raise RuntimeError(f"{stem}: '{LABEL_COL}' not found in test set.")

    df_test = df_test.copy()
    if "row_idx" not in df_test.columns:
        df_test["row_idx"] = df_test.index

    # 2) Train ML & proba
    df_train = _load_exp_train(stem)
    exp_type = _exp_type_from_stem(stem)
    ml_probs_by_row: Dict[int, Dict[str, float]] = {}
    if df_train is not None:
        try:
            ml_probs_by_row = _train_ml_and_predict_proba(stem, df_train, df_test)
        except Exception as e:
            print(f"[WARN] ML training failed for {stem}: {e}")

    # 3) Load partial
    df_partial = _load_partial_results(stem)
    done_row_idxs = set(df_partial["row_idx"].tolist())

    # 4) Query LLM for missing rows
    for idx, row in df_test.iterrows():
        if idx in done_row_idxs:
            continue

        # String GT label
        gt_label_str = _normalize_ground_truth_label(row[LABEL_COL])

        # ML predicted label (string)
        ml_probs = ml_probs_by_row.get(int(idx), {})
        ml_pred_label = _argmax_label_from_probs(ml_probs)

        # Build prompt
        try:
            prompt = _build_prompt_for_row(row, ml_probs=ml_probs, exp_type=exp_type)
            resp_text = _call_gpt_zero_shot(prompt, profile=gpt_profile)
            # sanitize response: remove newlines
            resp_text_clean = resp_text.replace("\n", " ").replace("\r", " ")
            gpt_label_str = _extract_label_from_response(resp_text)  # already normalized to strings
        except Exception as e:
            resp_text_clean = f"[ERROR] {e}"
            gpt_label_str = "unknown"

        # NEW schema with last three columns as requested
        row_record = {
            "row_idx": int(idx),
            "material": row.get("material", ""),
            "Power": row.get("Power", ""),
            "Velocity": row.get("Velocity", ""),
            "beam D": row.get("beam D", ""),
            "layer thickness": row.get("layer thickness", ""),
            "gpt_raw_response": resp_text_clean,
            "ml_pred_label": ml_pred_label,          # string
            "gpt_pred_label": gpt_label_str,         # string
            "gt_label": gt_label_str,                # string
        }

        _append_partial_result(stem, row_record)

    # 5) Reload predictions (full) and join with df_test for scoring
    df_results = _load_partial_results(stem)
    df_results = df_results[df_results["row_idx"].isin(df_test.index)].copy()

    df_join = df_results.merge(
        df_test[[LABEL_COL, "material", "Power", "Velocity", "beam D", "layer thickness"]],
        left_on="row_idx",
        right_index=True,
        how="left",
        suffixes=("", "_true"),
    )

    # Canonicalize GT from original df_test integers, for evaluation only
    df_join["gt_label_canon"]  = df_join[LABEL_COL].apply(_normalize_ground_truth_label)
    df_join["pred_label_canon"] = df_join["gpt_pred_label"].astype(str).str.lower()  # CHANGED: use new column

    # Derive match flag
    df_join["match_flag"] = df_join["gt_label_canon"] == df_join["pred_label_canon"]

    # 6) Macro-F1 computation
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


# ----------------------------------------------------
# Main
# ----------------------------------------------------

def main():
    results: List[Dict[str, float]] = []

    for stem in EXPERIMENTS:
        print("\n=======================================")
        print(f" Zero-shot GPT5 evaluation on {stem} (resumable)")
        print("=======================================")

        try:
            res = evaluate_gpt5_on_experiment_with_resume(
                stem,
                gpt_profile="default"
            )
        except FileNotFoundError as e:
            print(f"[ERROR] {stem}: {e}")
            continue
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
            continue

        results.append(res)
        print(
            f"{stem}: macro-F1 = {res['macro_f1']:.4f} | "
            f"scored {res['n_scored']}/{res['n_test_total']} rows | "
            f"completed rows stored: {res['n_completed_rows']}"
        )

    # Write summary CSV at the end
    out_csv = "gpt5_zero_shot_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "macro_F1",
            "n_scored",
            "n_test_total",
            "n_completed_rows",
        ])
        for r in results:
            writer.writerow([
                r["experiment"],
                f"{r['macro_f1']:.6f}",
                r["n_scored"],
                r["n_test_total"],
                r["n_completed_rows"],
            ])

    print(f"\nWrote GPT5 zero-shot summary: {out_csv}")

if __name__ == "__main__":
    main()

    SEARCH_PATTERNS = [
        "./results_AM/AMagent/gpt5_raw_preds_*.csv",
        "./gpt5_raw_preds_*.csv",  # optional fallback
    ]

    OUT_CSV = "./results_AM/AMagent/metrics_summary.csv"

    def macro_f1_and_accuracy(y_true: List[str], y_pred: List[str]) -> Tuple[float, float]:
        labels = sorted(set(y_true) | set(y_pred))
        tp = {l: 0 for l in labels};
        fp = {l: 0 for l in labels};
        fn = {l: 0 for l in labels}
        correct = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == yp:
                correct += 1
            for l in labels:
                if yp == l and yt == l:
                    tp[l] += 1
                elif yp == l and yt != l:
                    fp[l] += 1
                elif yp != l and yt == l:
                    fn[l] += 1
        f1s = []
        for l in labels:
            prec = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0
            rec = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        acc = correct / len(y_true) if y_true else 0.0
        return macro_f1, acc


    def find_files(patterns):
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
        return sorted(set(files))


    files = find_files(SEARCH_PATTERNS)

    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            rows.append({"experiment": os.path.basename(f), "n": 0,
                         "macro_f1": None, "accuracy": None, "file": f,
                         "error": f"read_failed: {e}"})
            continue

        if not {"gt_label", "gpt_pred_label"}.issubset(df.columns):
            rows.append({"experiment": os.path.basename(f), "n": 0,
                         "macro_f1": None, "accuracy": None, "file": f,
                         "error": "missing columns gt_label/gpt_pred_label"})
            continue

        y_true = df["gt_label"].astype(str).tolist()
        y_pred = df["gpt_pred_label"].astype(str).tolist()
        mf1, acc = macro_f1_and_accuracy(y_true, y_pred)
        exp = os.path.splitext(os.path.basename(f))[0].replace("gpt5_raw_preds_", "")
        rows.append({"experiment": exp, "n": len(y_true), "macro_f1": mf1, "accuracy": acc, "file": f})

    res = pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    # Pretty print
    with pd.option_context("display.max_colwidth", 60, "display.width", 120):
        print(res[["experiment", "n", "macro_f1", "accuracy"]])
    print(f"\nSaved: {OUT_CSV}")
