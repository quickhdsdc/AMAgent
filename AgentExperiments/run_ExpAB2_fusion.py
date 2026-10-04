import io
import json
import math
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
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
K = len(LABEL_ORDER)


def base_reliability(ml_probs: Dict[str, float]) -> Optional[float]:
    """Unclipped entropy-and-margin reliability R_cls (Eq. 4)."""
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


def _norm(b: Dict[str, float]) -> Dict[str, float]:
    tot = sum(b.get(k, 0.0) for k in LABEL_ORDER)
    if tot <= 0:
        return {k: 1.0 / K for k in LABEL_ORDER}
    return {k: b.get(k, 0.0) / tot for k in LABEL_ORDER}


def fused_label(ml_probs, ks_belief, w_cls, w_ks) -> str:
    ml, ks = _norm(ml_probs), _norm(ks_belief)
    den = w_cls + w_ks
    if den <= 0:
        return max(ml, key=ml.get)
    fused = {k: (w_cls * ml[k] + w_ks * ks[k]) / den for k in LABEL_ORDER}
    return max(fused, key=fused.get)


def w_M1(r, ood):
    return 1.0 * np.clip(r, R_MIN, R_MAX)

def w_M2(r, ood):
    return (1.0 if ood else 4.0) * np.clip(r, R_MIN, R_MAX)

def w_M3(r, ood):
    return (0.5 if ood else 4.0) * np.clip(r, R_MIN, R_MAX)

def w_aux_penalty(r, ood):
    return (0.5 if ood else 1.0) * np.clip(r, R_MIN, R_MAX)

def w_aux_twostage(r, ood):
    if ood:
        return float(np.clip(0.5 * r, R_MIN, R_MAX))
    return float(np.clip(r, R_MIN, R_MAX)) * 4.0

SCHEMES = {
    "M1_reliability":  w_M1,
    "M2_id_boost":     w_M2,
    "M3_full":         w_M3,
    "aux_penaltyOnly": w_aux_penalty,
    "aux_twostage":    w_aux_twostage,
}


def process_stem(stem: str):
    path = os.path.join(CACHE_DIR, f"gpt5_raw_preds_{stem}.csv")
    if not os.path.exists(path):
        print(f"  [skip] missing {path}")
        return []
    df = pd.read_csv(path)
    is_ood = "OOD" in stem
    rows = []
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
        rec = {
            "stem": stem, "is_ood": is_ood, "row_idx": int(r.get("row_idx", -1)),
            "gt": gt,
            "ml_alone": max(_norm(ml_probs), key=_norm(ml_probs).get),
            "ks_alone": max(_norm(ks_belief), key=_norm(ks_belief).get),
            "M0_uniform": fused_label(ml_probs, ks_belief, 0.5, 0.5),
            "R_base": round(rb, 4), "ks_rel": round(float(ks_rel), 4),
        }
        for name, wfn in SCHEMES.items():
            rec[name] = fused_label(ml_probs, ks_belief,
                                    float(wfn(rb, is_ood)), float(ks_rel))
        rows.append(rec)
    return rows


def macro_f1(sub, col):
    return f1_score(sub["gt"].tolist(), sub[col].tolist(),
                    labels=LABEL_ORDER, average="macro", zero_division=0)


def mcnemar(correct_a, correct_b):
    """Paired McNemar test on two correct/incorrect vectors.

    Exact two-sided binomial when the discordant count is small (<25),
    continuity-corrected chi-square otherwise. Returns
    (n_a_only_correct, n_b_only_correct, pvalue, test_name).
    """
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
    allrows = []
    for stem in EXPS:
        allrows.extend(process_stem(stem))
    df = pd.DataFrame(allrows)
    df.to_csv(os.path.join(OUT_DIR, "beta_modulation_per_row.csv"), index=False)

    cols = ["ml_alone", "ks_alone", "M0_uniform"] + list(SCHEMES.keys())
    summary = []
    for stem in EXPS:
        sub = df[df["stem"] == stem]
        if sub.empty:
            continue
        rec = {"stem": stem, "n": len(sub)}
        for c in cols:
            rec[c] = round(macro_f1(sub, c), 4)
        summary.append(rec)
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUT_DIR, "beta_modulation_test.csv"), index=False)

    print("\n=== Macro-F1 of fused suggested-label, full cached test sets ===")
    print(sm.to_string(index=False))

    for grp, mask in [("ID  mean", ~df["is_ood"]), ("OOD mean", df["is_ood"]),
                      ("ALL mean", df["row_idx"] == df["row_idx"])]:
        g = df[mask]
        line = [f"{grp}: n={len(g):4d}"]
        for c in cols:
            f1s = [macro_f1(g[g["stem"] == s], c)
                   for s in g["stem"].unique()]
            line.append(f"{c}={np.mean(f1s):.4f}")
        print("  " + "  ".join(line))

    print("\n=== Label-flip counts (how many fused labels change) ===")
    base = "M3_full"
    for c in ["M0_uniform"] + list(SCHEMES.keys()):
        if c == base:
            continue
        flips = int((df[c] != df[base]).sum())
        flips_ood = int(((df[c] != df[base]) & df["is_ood"]).sum())
        print(f"  {base} vs {c:16s}: {flips:4d} flips total  "
              f"({flips_ood} on OOD)  of {len(df)} rows")

    same = int((df["M3_full"] == df["aux_twostage"]).sum())
    print(f"\n  M3_full vs aux_twostage (single- vs two-stage clip): "
          f"{same}/{len(df)} identical -> "
          f"{'BEHAVIOUR-PRESERVING' if same == len(df) else 'DIFFERS'}")

    print("\n=== McNemar paired test — M3 vs each simpler variant ===")
    mc_rows = []
    for grp, mask in [("ALL", df["row_idx"] == df["row_idx"]),
                      ("ID", ~df["is_ood"]), ("OOD", df["is_ood"])]:
        g = df[mask]
        correct = {c: (g[c] == g["gt"]).tolist()
                   for c in ["M0_uniform", "M1_reliability",
                             "M2_id_boost", "M3_full"]}
        for ref in ["M0_uniform", "M1_reliability", "M2_id_boost"]:
            n10, n01, p, test = mcnemar(correct["M3_full"], correct[ref])
            mc_rows.append({"subset": grp, "comparison": f"M3_vs_{ref}",
                            "n": len(g), "n_M3_only_correct": n10,
                            "n_other_only_correct": n01, "test": test,
                            "pvalue": round(p, 5)})
            sig = " *" if p < 0.05 else ""
            print(f"  [{grp:3s}] M3 vs {ref:15s}: "
                  f"M3-only-correct={n10:4d}  other-only-correct={n01:4d}  "
                  f"p={p:.4g} ({test}){sig}")
    mc = pd.DataFrame(mc_rows)
    mc.to_csv(os.path.join(OUT_DIR, "beta_modulation_mcnemar.csv"), index=False)
    print(f"[OK] McNemar table -> {OUT_DIR}/beta_modulation_mcnemar.csv")


if __name__ == "__main__":
    main()
