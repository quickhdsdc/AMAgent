import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from amcore.labels import canonicalize_material


KB_V3_FILENAMES: Dict[str, str] = {
    "SS316L":      "SS316L",
    "Ti-6Al-4V":   "Ti-6Al-4V",
    "IN718":       "IN718",
    "IN625":       "IN625",
    "17-4PH":      "SS17-4PH",
    "SS17-4PH":    "SS17-4PH",
    "AlSi10Mg":    "AlSi10Mg",
    "18Ni300":     "18Ni300",
    "Hastelloy X": "Hastelloy_X",
    "CuCrZr":      "CuCrZr",
}

_PARAM_SPECS: List[Tuple[str, str, float]] = [
    ("Power",            "P_W",       0.45),
    ("Velocity",         "v_mms",     0.45),
    ("Hatch spacing",    "h_um",      0.45),
    ("layer thickness",  "t_um",      0.45),
    ("beam D",           "beam_D_um", 0.55),
]

_VED_SIGMA = 0.35


@dataclass
class RetrievalConfig:
    w_num:  float = 1.0
    w_ved:  float = 1.5
    w_qual: float = 0.4
    w_evid: float = 0.3
    mmr_lambda: float = 0.75
    max_pool: int = 50
    ved_bucket: float = 20.0
    min_param_dims: int = 3


def _log_sim(x: Optional[float], q: Optional[float], sigma: float) -> float:
    if x is None or q is None:
        return 0.0
    if x <= 0 or q <= 0:
        return 0.0
    return math.exp(-abs(math.log(x / q)) / max(sigma, 1e-3))


def _derive_ved(P, v, h, t) -> Optional[float]:
    """VED [J/mm^3] = P / (v * h_mm * t_mm), with h,t given in um."""
    if not all(x is not None and x > 0 for x in (P, v, h, t)):
        return None
    denom = v * (h * 1e-3) * (t * 1e-3)
    if denom <= 1e-9:
        return None
    return P / denom


def _row_ved(row: Dict[str, Any]) -> Optional[float]:
    v = row.get("VED_Jmm3")
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    return _derive_ved(row.get("P_W"), row.get("v_mms"),
                      row.get("h_um"),  row.get("t_um"))


def _evidence_strength(row: Dict[str, Any]) -> float:
    """Map (evidence_type, claim_strength, gate) -> [0, 1] confidence weight."""
    et = (row.get("evidence_type") or "").lower()
    cs = (row.get("claim_strength") or "").lower()
    gate = (row.get("gate") or "").lower()

    base = {
        "experimental": 1.0,
        "simulation":   0.6,
        "review":       0.5,
    }.get(et, 0.7)
    claim_mul = {
        "measured":     1.0,
        "reported":     0.8,
        "hypothesized": 0.5,
    }.get(cs, 0.8)
    gate_bonus = 0.1 if gate in {"quant_2plus", "quant1_with_ved"} else 0.0
    return min(1.0, base * claim_mul + gate_bonus)


def _quality_score(info_score: float) -> float:
    """Squash info_score (~0..15) to a soft [0, 1] weight."""
    return min(1.0, max(0.0, math.log1p(max(info_score, 0.0)) / math.log(15.0)))


def _row_signature(row: Dict[str, Any], cfg: RetrievalConfig) -> Tuple[str, int]:
    """Categorical signature used for MMR diversity grouping."""
    defect = row.get("defect") or "?"
    ved = _row_ved(row)
    bucket = int(ved // cfg.ved_bucket) if ved else -1
    return (defect, bucket)


class EvidenceRowLoader:
    def __init__(self, kb_dir: str = "./bibliography/kb_v3"):
        self.kb_dir = kb_dir
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    def load(self, canonical: str) -> List[Dict[str, Any]]:
        if canonical in self._cache:
            return self._cache[canonical]
        stem = KB_V3_FILENAMES.get(canonical)
        if not stem:
            self._cache[canonical] = []
            return []
        path = os.path.join(self.kb_dir, f"{stem}_rows.jsonl")
        if not os.path.exists(path):
            self._cache[canonical] = []
            return []
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self._cache[canonical] = rows
        print(f"[rag_v2] loaded {len(rows)} evidence rows for {canonical}")
        return rows


_LOADER: Optional[EvidenceRowLoader] = None


def get_loader() -> EvidenceRowLoader:
    global _LOADER
    if _LOADER is None:
        _LOADER = EvidenceRowLoader()
    return _LOADER


def reset_loader() -> None:
    global _LOADER
    _LOADER = None


def score_row(row: Dict[str, Any], query: Dict[str, Any],
             cfg: RetrievalConfig = RetrievalConfig()
             ) -> Tuple[float, Dict[str, float]]:
    """Compute composite relevance for one row. Returns (rel, audit)."""
    per_dim: Dict[str, float] = {}
    present_sims: List[float] = []
    for q_key, r_key, sigma in _PARAM_SPECS:
        s = _log_sim(row.get(r_key), query.get(q_key), sigma)
        per_dim[q_key] = s
        if row.get(r_key) is not None and query.get(q_key) is not None:
            present_sims.append(s)

    num_sim = sum(present_sims) / len(present_sims) if present_sims else 0.0

    q_ved = query.get("VED_Jmm3") or _derive_ved(
        query.get("Power"),         query.get("Velocity"),
        query.get("Hatch spacing"), query.get("layer thickness"),
    )
    r_ved = _row_ved(row)
    ved_sim = _log_sim(r_ved, q_ved, _VED_SIGMA)

    qual_sim = _quality_score(float(row.get("info_score", 0.0)))
    evid_sim = _evidence_strength(row)

    rel = (cfg.w_num  * num_sim
         + cfg.w_ved  * ved_sim
         + cfg.w_qual * qual_sim
         + cfg.w_evid * evid_sim)

    audit = {
        "num_sim":  round(num_sim,  4),
        "ved_sim":  round(ved_sim,  4),
        "qual":     round(qual_sim, 4),
        "evid":     round(evid_sim, 4),
        "rel":      round(rel,      4),
        "per_dim":  {k: round(v, 3) for k, v in per_dim.items()},
        "q_VED":    None if q_ved is None else round(q_ved, 2),
        "r_VED":    None if r_ved is None else round(r_ved, 2),
    }
    return rel, audit


def _mmr_select(scored: List[Tuple[Dict[str, Any], Dict[str, float]]],
               k: int, cfg: RetrievalConfig
               ) -> List[Tuple[Dict[str, Any], Dict[str, float]]]:
    """Maximal Marginal Relevance over (defect, VED-bucket) signature.

    Diversity term penalizes a candidate if a row with the same signature is
    already selected. Within the same signature we still allow up to 2 rows
    (so a strong outcome category isn't represented by a single citation).
    """
    if not scored or k <= 0:
        return []
    pool = list(scored)
    selected: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
    sig_counts: Dict[Tuple[str, int], int] = {}

    lam = cfg.mmr_lambda
    while pool and len(selected) < k:
        best = None
        best_score = -1e9
        for i, (row, audit) in enumerate(pool):
            sig = _row_signature(row, cfg)
            penalty = sig_counts.get(sig, 0)
            if penalty >= 2:
                continue
            diversity_penalty = 0.35 * penalty
            mmr = lam * audit["rel"] - (1.0 - lam) * diversity_penalty
            if mmr > best_score:
                best_score = mmr
                best = i
        if best is None:
            break
        row, audit = pool.pop(best)
        sig_counts[_row_signature(row, cfg)] = sig_counts.get(
            _row_signature(row, cfg), 0) + 1
        selected.append((row, audit))
    return selected


def retrieve_evidence_rows(query: Dict[str, Any], k: int = 6,
                          cfg: Optional[RetrievalConfig] = None,
                          loader: Optional[EvidenceRowLoader] = None,
                          ) -> List[Dict[str, Any]]:
    """Return up to `k` evidence rows for the query, each annotated with an
    'audit' subdict tracing per-dimension similarities and weights.

    Args:
        query: dict with at least {'material'} and one or more of
               {'Power', 'Velocity', 'Hatch spacing', 'layer thickness',
                'beam D'}.  An explicit 'VED_Jmm3' overrides the derived VED.
        k: max rows to return after MMR.
        cfg: scoring + MMR config; defaults to RetrievalConfig().
        loader: optional EvidenceRowLoader (defaults to module singleton).

    Returns:
        List of dicts shaped as
            {**row, "audit": {<per-row scoring trace>}, "rank": <int>}.
    """
    cfg = cfg or RetrievalConfig()
    canonical = canonicalize_material(query.get("material"))
    if canonical is None:
        return []
    rows = (loader or get_loader()).load(canonical)
    if not rows:
        return []

    def _is_dense(r: Dict[str, Any]) -> bool:
        if r.get("VED_Jmm3") is not None:
            return True
        n = sum(1 for k in ("P_W", "v_mms", "h_um", "t_um", "beam_D_um")
                if r.get(k) is not None)
        return n >= cfg.min_param_dims

    rows = [r for r in rows if _is_dense(r)]
    if not rows:
        return []

    scored = [(r, score_row(r, query, cfg)[1]) for r in rows]
    scored.sort(key=lambda x: x[1]["rel"], reverse=True)
    pool = scored[: cfg.max_pool]
    chosen = _mmr_select(pool, k, cfg)

    out: List[Dict[str, Any]] = []
    for rank, (row, audit) in enumerate(chosen, 1):
        out.append({**row, "audit": audit, "rank": rank})
    return out


__all__ = [
    "RetrievalConfig", "EvidenceRowLoader", "score_row",
    "retrieve_evidence_rows", "get_loader", "reset_loader",
    "_derive_ved",
]
