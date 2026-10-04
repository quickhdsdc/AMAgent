import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from amcore import (
    LABEL_COL, load_exp_split, load_exp_train, call_llm,
    get_rag_loader,
    retrieve_evidence_rows, get_rag_v2_loader, build_evidence_pack,
)
from amcore.parsing import extract_label_from_response
from amcore.prompts import (
    build_kd_agent_prompt_binary, build_supervisor_prompt_binary,
)
from amcore.fusion import (
    calculate_entropy,
    ml_reliability_from_probs_binary,
    deterministic_fusion_binary,
    BINARY_LABEL_ORDER,
)

RANDOM_STATE = 42
FEAT_NUM = ["Power", "Velocity", "beam D", "layer thickness", "Hatch spacing"]
FEAT_CAT = ["material"]


def normalize_binary_label(y) -> str:
    """0 / 'good' -> 'good'; 1 / 'defective' / 'bad' -> 'defective'."""
    if y is None or (isinstance(y, float) and pd.isna(y)):
        return "unknown"
    s = str(y).strip().lower()
    if s in ("good", "0", "0.0", "none", "desirable", "ok"):
        return "good"
    if s in ("defective", "1", "1.0", "bad", "defect", "lof", "balling", "keyhole"):
        return "defective"
    try:
        i = int(float(y))
        if i == 0:
            return "good"
        if i == 1:
            return "defective"
    except Exception:
        pass
    return "unknown"


_BIN_LABEL_RE = re.compile(
    r"\[LABEL\]\s*([^\[\]]+?)\s*\[/LABEL\]",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_binary_label(text: str) -> str:
    """Parse [LABEL]...[/LABEL] directly for a 2-class label."""
    if not isinstance(text, str):
        return "unknown"
    m = _BIN_LABEL_RE.search(text)
    if not m:
        return "unknown"
    raw = m.group(1).strip().lower()
    raw = re.sub(r"[\{\}\"\']", "", raw).strip()
    if ":" in raw:
        raw = raw.split(":")[-1].strip()
    return normalize_binary_label(raw)


def train_binary_dps(df_train: pd.DataFrame, df_test: pd.DataFrame
                     ) -> Dict[int, Dict[str, float]]:
    """Train a RF binary classifier; return row_idx -> {good, defective} probs."""
    cols_needed = [c for c in FEAT_NUM + FEAT_CAT if c in df_train.columns]
    X_train = df_train[cols_needed].copy()
    y_train = df_train[LABEL_COL].apply(normalize_binary_label)
    mask = y_train.isin(BINARY_LABEL_ORDER)
    X_train = X_train[mask]
    y_train = y_train[mask]
    if len(X_train) == 0:
        raise RuntimeError("no valid training rows after label normalization")

    num_present = [c for c in FEAT_NUM if c in cols_needed]
    cat_present = [c for c in FEAT_CAT if c in cols_needed]
    transformers = []
    if num_present:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median"))]), num_present))
    if cat_present:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_present))
    pre = ColumnTransformer(transformers)
    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                 n_jobs=-1, class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    X_test = df_test.reindex(columns=cols_needed, fill_value=np.nan).copy()
    proba = pipe.predict_proba(X_test)
    cls_order = list(pipe.classes_)
    out: Dict[int, Dict[str, float]] = {}
    for ridx, idx in enumerate(df_test.index):
        row_map = {c: 0.0 for c in BINARY_LABEL_ORDER}
        for c_idx, c_name in enumerate(cls_order):
            if c_name in row_map:
                row_map[c_name] = float(proba[ridx][c_idx])
        out[int(idx)] = {k: round(row_map[k], 4) for k in BINARY_LABEL_ORDER}
    return out


def argmax_binary(prob_map: Dict[str, float]) -> str:
    if not prob_map:
        return "unknown"
    return max(BINARY_LABEL_ORDER, key=lambda k: prob_map.get(k, 0.0))


FIELDNAMES = [
    "row_idx", "material", "Power", "Velocity",
    "beam D", "layer thickness", "Hatch spacing",
    "ml_prob_good", "ml_prob_defective", "ml_pred_label",
    "agent_rag_response", "agent_rag_label",
    "supervisor_raw_response", "supervisor_label",
    "fused_belief_good", "fused_belief_defective", "fused_label",
    "rag_rows", "rag_mode",
    "prompts_debug", "gt_label",
]


def _out_dir_for(supervisor_profile: str) -> str:
    d = os.path.join(ROOT, "results_AM", f"AMagent_binary_{supervisor_profile}")
    os.makedirs(d, exist_ok=True)
    return d


def _out_path(stem: str, supervisor_profile: str) -> str:
    return os.path.join(_out_dir_for(supervisor_profile),
                       f"{supervisor_profile}_raw_preds_{stem}.csv")


def _load_partial(path: str) -> set:
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
        return set(df["row_idx"].astype(int).tolist())
    except Exception:
        return set()


def _append_row(path: str, row: dict):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(row)


def _stratified_sample(df: pd.DataFrame, n: int, seed: int = 42,
                      strat_cols=("Power", "Velocity")) -> pd.DataFrame:
    """Deterministic stratified sample of ~n rows across `strat_cols` quantile
    bins. Falls back to random sample if any column is missing."""
    if any(c not in df.columns for c in strat_cols):
        return df.sample(n=min(n, len(df)), random_state=seed)
    df_ = df.copy()
    try:
        df_["_b0"] = pd.qcut(df_[strat_cols[0]].astype(float), q=5,
                             labels=False, duplicates="drop")
        df_["_b1"] = pd.qcut(df_[strat_cols[1]].astype(float), q=5,
                             labels=False, duplicates="drop")
    except Exception:
        return df.sample(n=min(n, len(df)), random_state=seed)
    df_["_bin"] = df_["_b0"].astype(str) + "_" + df_["_b1"].astype(str)
    bins = df_["_bin"].unique()
    per_bin = max(1, n // max(len(bins), 1))
    picked = []
    rng = np.random.RandomState(seed)
    for b in bins:
        sub = df_[df_["_bin"] == b]
        take = min(per_bin, len(sub))
        idx = rng.choice(sub.index, size=take, replace=False)
        picked.extend(idx.tolist())
    short = n - len(picked)
    if short > 0:
        remaining = df_.index.difference(picked)
        if len(remaining) > 0:
            picked.extend(rng.choice(remaining, size=min(short, len(remaining)),
                                    replace=False).tolist())
    picked = sorted(set(picked))[:n]
    out = df.loc[picked].copy()
    return out


def run_one_stem(stem: str, n_rows: int,
                supervisor_profile: str, sub_agent_profile: str,
                stratified: bool = False,
                use_legacy_rag: bool = False,
                rag_k: int = 6) -> dict:
    print(f"\n=== Running binary AM-Agent on {stem} ===")
    print(f"  n_rows={n_rows} stratified={stratified}")
    df_test_full = load_exp_split(stem)
    df_train = load_exp_train(stem)
    if df_train is None:
        raise RuntimeError(f"no train split for {stem}")
    if stratified:
        df_test = _stratified_sample(df_test_full, n_rows, seed=42)
    else:
        df_test = df_test_full.head(n_rows).copy()
    if "row_idx" not in df_test.columns:
        df_test["row_idx"] = df_test.index

    out_path = _out_path(stem, supervisor_profile)
    done = _load_partial(out_path)
    print(f"  out_path: {out_path}")
    print(f"  already done: {len(done)} rows")

    print(f"  training binary DPS on {len(df_train)} rows...")
    ml_probs_by_row = train_binary_dps(df_train, df_test)

    legacy_loader = get_rag_loader()
    rag_v2_loader = get_rag_v2_loader()
    is_ood = "OOD" in stem

    for ridx, (idx, row) in enumerate(df_test.iterrows(), 1):
        if int(idx) in done:
            continue
        gt_label = normalize_binary_label(row[LABEL_COL])
        ml_probs = ml_probs_by_row.get(int(idx), {})
        ml_pred = argmax_binary(ml_probs)

        material = row.get("material", "")
        proc_params = {c: row.get(c) for c in FEAT_NUM}

        rag_query = {"material": material, **proc_params}
        ev_rows: list = []
        if not use_legacy_rag:
            ev_rows = retrieve_evidence_rows(rag_query, k=rag_k,
                                            loader=rag_v2_loader)

        if ev_rows:
            evidence_pack = build_evidence_pack(ev_rows, rag_query)
            rag_context = None
            rag_mode = "rag_v2"
        else:
            evidence_pack = None
            rag_context = legacy_loader.get_context(material, k=5,
                                                  seed=int(idx) + RANDOM_STATE,
                                                  process_params=proc_params)
            rag_mode = "legacy_chunks" if rag_context else "empty"

        ml_entropy = calculate_entropy(ml_probs, classes=tuple(BINARY_LABEL_ORDER))
        ml_reliability = ml_reliability_from_probs_binary(ml_probs, is_ood)

        prompt_rag = build_kd_agent_prompt_binary(row,
                                                 rag_context=rag_context,
                                                 evidence_pack=evidence_pack)
        try:
            resp_text_rag = call_llm(prompt_rag, profile=sub_agent_profile)
            agent_rag_label = extract_binary_label(resp_text_rag)
        except Exception as e:
            resp_text_rag = f"[ERROR] {e}"
            agent_rag_label = "unknown"
        resp_text_rag_clean = resp_text_rag.replace("\n", " ").replace("\r", " ")

        fused_belief, fused_label = deterministic_fusion_binary(
            ml_probs_raw=ml_probs, ml_reliability=ml_reliability,
            rag_resp=resp_text_rag_clean, is_ood=is_ood,
        )

        prompt_sup = build_supervisor_prompt_binary(
            row, ml_probs=ml_probs, ml_entropy=ml_entropy,
            ml_reliability=ml_reliability, rag_response=resp_text_rag_clean,
            fused_belief=fused_belief, fused_label=fused_label,
            exp_type="out-of-distribution" if is_ood else "in-distribution",
        )
        try:
            resp_text_sup = call_llm(prompt_sup, profile=supervisor_profile)
            sup_label = extract_binary_label(resp_text_sup)
        except Exception as e:
            resp_text_sup = f"[ERROR] {e}"
            sup_label = "unknown"
        resp_text_sup_clean = resp_text_sup.replace("\n", " ").replace("\r", " ")

        rag_rows_audit = [
            {
                "rank":     r.get("rank"),
                "P_W":      r.get("P_W"),
                "v_mms":    r.get("v_mms"),
                "h_um":     r.get("h_um"),
                "t_um":     r.get("t_um"),
                "VED":      r.get("VED_Jmm3"),
                "defect":   r.get("defect"),
                "density":  r.get("density_pct"),
                "ev_type":  r.get("evidence_type"),
                "claim":    r.get("claim_strength"),
                "source":   r.get("source_file"),
                "audit":    r.get("audit", {}),
            }
            for r in ev_rows
        ]

        _append_row(out_path, {
            "row_idx": int(idx), "material": material,
            "Power": row.get("Power"), "Velocity": row.get("Velocity"),
            "beam D": row.get("beam D"),
            "layer thickness": row.get("layer thickness"),
            "Hatch spacing": row.get("Hatch spacing"),
            "ml_prob_good": ml_probs.get("good"),
            "ml_prob_defective": ml_probs.get("defective"),
            "ml_pred_label": ml_pred,
            "agent_rag_response": resp_text_rag_clean,
            "agent_rag_label": agent_rag_label,
            "supervisor_raw_response": resp_text_sup_clean,
            "supervisor_label": sup_label,
            "fused_belief_good": round(fused_belief.get("good", 0.0), 4),
            "fused_belief_defective": round(fused_belief.get("defective", 0.0), 4),
            "fused_label": fused_label,
            "rag_rows": json.dumps(rag_rows_audit, ensure_ascii=False)[:8000],
            "rag_mode": rag_mode,
            "prompts_debug": json.dumps({"ks": prompt_rag, "sup": prompt_sup,
                                        "ml": ml_probs,
                                        "ml_rel": ml_reliability,
                                        "ml_entropy": ml_entropy})[:8000],
            "gt_label": gt_label,
        })
        print(f"  [{ridx}/{len(df_test)}] idx={idx} mat={material} "
             f"GT={gt_label}  ML={ml_pred}  KS={agent_rag_label}  "
             f"Sup={sup_label}  rag={rag_mode}({len(ev_rows)})")

    df = pd.read_csv(out_path)
    valid = df[df["gt_label"].isin(BINARY_LABEL_ORDER) &
              df["supervisor_label"].isin(BINARY_LABEL_ORDER)]
    sup_f1 = ml_f1 = ks_f1 = float("nan")
    if len(valid):
        sup_f1 = f1_score(valid["gt_label"], valid["supervisor_label"],
                         average="macro", zero_division=0)
        ml_f1 = f1_score(valid["gt_label"], valid["ml_pred_label"],
                        average="macro", zero_division=0)
        ks_valid = valid[valid["agent_rag_label"].isin(BINARY_LABEL_ORDER)]
        if len(ks_valid):
            ks_f1 = f1_score(ks_valid["gt_label"], ks_valid["agent_rag_label"],
                            average="macro", zero_division=0)

    print(f"\n--- {stem} results ({len(valid)} valid rows) ---")
    print(f"  ML (DPS)     macro-F1 = {ml_f1:.3f}")
    print(f"  KS (LLM)     macro-F1 = {ks_f1:.3f}")
    print(f"  Supervisor   macro-F1 = {sup_f1:.3f}")
    return {"stem": stem, "n_valid": len(valid),
            "ml_f1": ml_f1, "ks_f1": ks_f1, "sup_f1": sup_f1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stem", default="Exp_BIN_NOVMAT_D0D1D2_OOD_AlSi10Mg")
    p.add_argument("--n-rows", type=int, default=46)
    p.add_argument("--supervisor", default="gpt5")
    p.add_argument("--sub-agent", default="gpt5")
    p.add_argument("--stratified", action="store_true",
                  help="use deterministic stratified sample instead of head()")
    p.add_argument("--use-legacy-rag", action="store_true",
                  help="bypass rag_v2 and use the legacy chunk RAG "
                       "(for ablation against the structured-row retrieval)")
    p.add_argument("--rag-k", type=int, default=6,
                  help="number of evidence rows to retrieve per query")
    args = p.parse_args()
    run_one_stem(args.stem, args.n_rows,
                args.supervisor, args.sub_agent,
                stratified=args.stratified,
                use_legacy_rag=args.use_legacy_rag,
                rag_k=args.rag_k)


if __name__ == "__main__":
    main()
