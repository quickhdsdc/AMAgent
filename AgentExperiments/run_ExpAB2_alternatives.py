import io
import json
import math
import os
import sys
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from amcore.labels import LABEL_ORDER, VALID_LABELS, normalize_ground_truth_label
from amcore.parsing import extract_json_block, extract_float_block

EXPS = [f"Exp_ID_{i}" for i in (1, 2, 3, 4)] + \
       [f"Exp_OOD_{i}" for i in (1, 2, 3, 4)]
CACHE_DIR = os.path.join(ROOT, "results_AM", "AMagent_gpt5")
OUT_DIR = os.path.join(ROOT, "results_AM", "stats")
os.makedirs(OUT_DIR, exist_ok=True)

R_MIN, R_MAX = 0.05, 0.9
BETA_ID, BETA_OOD = 4.0, 0.5
K = len(LABEL_ORDER)
EPS = 1e-9


def base_reliability(ml_probs: Dict[str, float]) -> Optional[float]:
    """Eq. (4): unclipped entropy-and-margin reliability R_cls."""
    vals = np.array([ml_probs.get(c, 0.0) for c in LABEL_ORDER], dtype=float)
    s = vals.sum()
    if s <= 0:
        return None
    vals = vals / s
    nz = vals[vals > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))
    h_norm = entropy / math.log2(K)
    p = np.sort(vals)[::-1]
    margin = float(p[0] - p[1])
    return (1.0 - h_norm) * (0.5 + 0.5 * margin)


def _norm_vec(b: Dict[str, float]) -> np.ndarray:
    v = np.array([b.get(c, 0.0) for c in LABEL_ORDER], dtype=float)
    s = v.sum()
    return v / s if s > 0 else np.ones(K) / K


def fuse_F1_uniform(p_dps: np.ndarray, p_ks: np.ndarray, *_) -> np.ndarray:
    """F1: Uniform Linear Opinion Pool. p = 0.5*p_DPS + 0.5*p_KS."""
    return 0.5 * p_dps + 0.5 * p_ks


def fuse_F2_bma_logpool(p_dps: np.ndarray, p_ks: np.ndarray,
                        r_dps: float, r_ks: float, *_) -> np.ndarray:
    """F2: Reliability-weighted logarithmic opinion pool (BMA realization).

    Standard BMA combines posteriors weighted by posterior model probability.
    In its log-linear (geometric) form,
        log p_fused(c) propto w_DPS * log p_DPS(c) + w_KS * log p_KS(c),
    which is the externally-Bayesian rule when (w_DPS, w_KS) are posterior
    model probabilities. We use the per-instance reliabilities (R_DPS, R_KS)
    as the weights, mirroring how a model evidence updates the posterior."""
    w_d = max(r_dps, EPS)
    w_k = max(r_ks, EPS)
    log_p = w_d * np.log(p_dps + EPS) + w_k * np.log(p_ks + EPS)
    log_p -= log_p.max()
    p = np.exp(log_p)
    return p / p.sum()


def fuse_F4_dempster_shafer(p_dps: np.ndarray, p_ks: np.ndarray,
                            r_dps: float, r_ks: float, *_) -> np.ndarray:
    """F4: Dempster-Shafer combination with reliability discounting.

    Each source treated as a mass function on singletons:
        m(c) = R * p(c)    for c in labels
        m(Theta) = 1 - R   (residual uncertainty on the full frame)
    Apply Dempster's rule of combination. For singletons c:
        m12(c) = [m1(c)*m2(c) + m1(c)*m2(Theta) + m1(Theta)*m2(c)] / (1 - K)
    where K is the conflict mass. Returns the normalized singleton beliefs
    (Theta mass redistributed proportionally via pignistic transformation)."""
    R1 = float(np.clip(r_dps, 0.0, 1.0))
    R2 = float(np.clip(r_ks, 0.0, 1.0))
    m1 = R1 * p_dps
    m2 = R2 * p_ks
    m1_theta = 1.0 - R1
    m2_theta = 1.0 - R2

    K_conf = float(m1.sum() * m2.sum() - (m1 * m2).sum())
    norm = 1.0 - K_conf
    if norm <= EPS:
        return 0.5 * p_dps + 0.5 * p_ks

    m12 = (m1 * m2 + m1 * m2_theta + m1_theta * m2) / norm
    m12_theta = (m1_theta * m2_theta) / norm
    m12 = m12 + m12_theta / K
    s = m12.sum()
    return m12 / s if s > 0 else np.ones(K) / K


def fuse_F5_proposed(p_dps: np.ndarray, p_ks: np.ndarray,
                     r_dps_base: float, r_ks: float,
                     is_ood: bool) -> np.ndarray:
    """F5 (proposed): reliability-weighted LOP with OOD modulation (M3_full).

    w_cls = beta(omega) * clip(R_cls)        (Eq. 3, 6)
    w_ks  = R_ks                              (LLM-reported reliability)
    p_fused = (w_cls * p_dps + w_ks * p_ks) / (w_cls + w_ks)        (Eq. 8)
    """
    beta = BETA_OOD if is_ood else BETA_ID
    w_cls = beta * float(np.clip(r_dps_base, R_MIN, R_MAX))
    w_ks = float(r_ks)
    den = w_cls + w_ks
    if den <= 0:
        return p_dps
    return (w_cls * p_dps + w_ks * p_ks) / den


def build_stacking_features(row: Dict) -> np.ndarray:
    """11-D feature: 4 p_DPS + 4 p_KS + R_DPS + R_KS + is_ood."""
    feats = list(row["p_dps"]) + list(row["p_ks"]) + \
            [row["r_dps_clipped"], row["r_ks"], float(row["is_ood"])]
    return np.array(feats, dtype=float)


def run_stacking_LOSO(records: List[Dict]) -> List[str]:
    """F3: multinomial logistic regression meta-learner with leave-one-stem-out.
    For each test stem, train on the other 7 stems' cached predictions."""
    n = len(records)
    preds = [""] * n
    label_to_idx = {l: i for i, l in enumerate(LABEL_ORDER)}

    stems = sorted({r["stem"] for r in records})
    by_stem = {s: [i for i, r in enumerate(records) if r["stem"] == s] for s in stems}

    for held_out in stems:
        train_idx = [i for s in stems if s != held_out for i in by_stem[s]]
        test_idx = by_stem[held_out]

        X_tr = np.vstack([build_stacking_features(records[i]) for i in train_idx])
        y_tr = np.array([label_to_idx[records[i]["gt"]] for i in train_idx])
        X_te = np.vstack([build_stacking_features(records[i]) for i in test_idx])

        clf = LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            max_iter=2000, C=1.0,
        )
        clf.fit(X_tr, y_tr)
        y_pr = clf.predict(X_te)
        for j, i in enumerate(test_idx):
            preds[i] = LABEL_ORDER[int(y_pr[j])]
    return preds


def load_records() -> List[Dict]:
    out = []
    for stem in EXPS:
        path = os.path.join(CACHE_DIR, f"gpt5_raw_preds_{stem}.csv")
        if not os.path.exists(path):
            print(f"  [skip] missing {path}")
            continue
        df = pd.read_csv(path)
        is_ood = "OOD" in stem
        for _, r in df.iterrows():
            gt = normalize_ground_truth_label(r.get("gt_label"))
            if gt not in VALID_LABELS:
                continue
            try:
                ml_probs = json.loads(r["prompts_debug"])["ml_stats"]["probs"]
            except Exception:
                continue
            rb = base_reliability(ml_probs)
            if rb is None:
                continue
            rag = r.get("agent_rag_response", "") or ""
            ks_belief = extract_json_block(rag, "BELIEF") or {l: 1.0 / K for l in LABEL_ORDER}
            ks_rel = extract_float_block(rag, "RELIABILITY")
            if ks_rel is None:
                ks_rel = 0.3

            p_dps = _norm_vec(ml_probs)
            p_ks = _norm_vec(ks_belief)
            out.append({
                "stem": stem,
                "is_ood": is_ood,
                "row_idx": int(r.get("row_idx", -1)),
                "gt": gt,
                "p_dps": p_dps,
                "p_ks": p_ks,
                "r_dps_base": float(rb),
                "r_dps_clipped": float(np.clip(rb, R_MIN, R_MAX)),
                "r_ks": float(ks_rel),
            })
    return out


def macro_f1_for(records: List[Dict], col: str, mask=None) -> float:
    if mask is None:
        sub = records
    else:
        sub = [r for r, m in zip(records, mask) if m]
    if not sub:
        return float("nan")
    y_true = [r["gt"] for r in sub]
    y_pred = [r[col] for r in sub]
    return f1_score(y_true, y_pred, labels=LABEL_ORDER,
                    average="macro", zero_division=0)


def mcnemar(correct_a: List[bool], correct_b: List[bool]) -> Tuple[int, int, float, str]:
    n10 = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)
    n01 = sum(1 for a, b in zip(correct_a, correct_b) if b and not a)
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0, "no_discordant"
    if n < 25:
        from math import comb
        k = min(n10, n01)
        p = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
        return n10, n01, min(2.0 * p, 1.0), "exact_binomial"
    from math import erfc, sqrt
    stat = (abs(n10 - n01) - 1) ** 2 / n
    return n10, n01, min(max(erfc(sqrt(stat) / sqrt(2)), 0.0), 1.0), "chi2_cc"


def main():
    print("Loading cached predictions...")
    records = load_records()
    print(f"  {len(records)} rows over {len({r['stem'] for r in records})} stems")

    for r in records:
        p_d, p_k = r["p_dps"], r["p_ks"]
        rb, rks, ood = r["r_dps_base"], r["r_ks"], r["is_ood"]
        r["F1_uniform"]    = LABEL_ORDER[int(np.argmax(fuse_F1_uniform(p_d, p_k)))]
        r["F2_bma_logpool"] = LABEL_ORDER[int(np.argmax(fuse_F2_bma_logpool(p_d, p_k, rb, rks)))]
        r["F4_dempster"]   = LABEL_ORDER[int(np.argmax(fuse_F4_dempster_shafer(p_d, p_k, rb, rks)))]
        r["F5_proposed"]   = LABEL_ORDER[int(np.argmax(fuse_F5_proposed(p_d, p_k, rb, rks, ood)))]
        r["dps_alone"]     = LABEL_ORDER[int(np.argmax(p_d))]
        r["ks_alone"]      = LABEL_ORDER[int(np.argmax(p_k))]

    print("Fitting F3 stacking (leave-one-stem-out)...")
    f3_preds = run_stacking_LOSO(records)
    for i, r in enumerate(records):
        r["F3_stacking"] = f3_preds[i]

    cols = ["dps_alone", "ks_alone", "F1_uniform", "F2_bma_logpool",
            "F3_stacking", "F4_dempster", "F5_proposed"]

    per_row_df = pd.DataFrame([
        {"stem": r["stem"], "is_ood": r["is_ood"], "row_idx": r["row_idx"],
         "gt": r["gt"], "r_dps_base": round(r["r_dps_base"], 4),
         "r_ks": round(r["r_ks"], 4),
         **{c: r[c] for c in cols}}
        for r in records
    ])
    per_row_df.to_csv(os.path.join(OUT_DIR, "expab2_alternatives_per_row.csv"),
                      index=False)

    rows = []
    for stem in EXPS:
        sub = [r for r in records if r["stem"] == stem]
        if not sub:
            continue
        rec = {"stem": stem, "n": len(sub)}
        for c in cols:
            rec[c] = round(macro_f1_for(sub, c), 4)
        rows.append(rec)

    id_stems = [r for r in rows if "ID" in r["stem"]]
    ood_stems = [r for r in rows if "OOD" in r["stem"]]
    for grp_name, grp in [("ID_mean", id_stems), ("OOD_mean", ood_stems),
                          ("ALL_mean", rows)]:
        if not grp:
            continue
        agg = {"stem": grp_name, "n": sum(r["n"] for r in grp)}
        for c in cols:
            agg[c] = round(float(np.mean([r[c] for r in grp])), 4)
        rows.append(agg)

    sm = pd.DataFrame(rows)
    sm.to_csv(os.path.join(OUT_DIR, "expab2_alternatives_summary.csv"), index=False)

    print("\n=== Macro-F1 — fusion alternatives (full cached test sets) ===")
    print(sm.to_string(index=False))

    print("\n=== McNemar — F5 (proposed) vs each alternative ===")
    mc_rows = []
    mask_id = [not r["is_ood"] for r in records]
    mask_ood = [r["is_ood"] for r in records]
    for grp_name, mask in [("ALL", [True] * len(records)),
                           ("ID", mask_id), ("OOD", mask_ood)]:
        sub = [r for r, m in zip(records, mask) if m]
        correct_f5 = [r["F5_proposed"] == r["gt"] for r in sub]
        for ref in ["F1_uniform", "F2_bma_logpool", "F3_stacking", "F4_dempster"]:
            correct_ref = [r[ref] == r["gt"] for r in sub]
            n10, n01, p, test = mcnemar(correct_f5, correct_ref)
            mc_rows.append({"subset": grp_name, "comparison": f"F5_vs_{ref}",
                            "n": len(sub), "n_F5_only_correct": n10,
                            "n_other_only_correct": n01, "test": test,
                            "pvalue": round(p, 5)})
            sig = " *" if p < 0.05 else ""
            print(f"  [{grp_name:3s}] F5 vs {ref:18s}: "
                  f"F5-only-correct={n10:4d}  other-only-correct={n01:4d}  "
                  f"p={p:.4g} ({test}){sig}")

    mc = pd.DataFrame(mc_rows)
    mc.to_csv(os.path.join(OUT_DIR, "expab2_alternatives_mcnemar.csv"), index=False)

    print(f"\n[OK] Summary       -> {OUT_DIR}/expab2_alternatives_summary.csv")
    print(f"[OK] Per-row       -> {OUT_DIR}/expab2_alternatives_per_row.csv")
    print(f"[OK] McNemar table -> {OUT_DIR}/expab2_alternatives_mcnemar.csv")


if __name__ == "__main__":
    main()
