from typing import Dict, Tuple

import numpy as np

from amcore.labels import LABEL_ORDER
from amcore.parsing import extract_json_block, extract_float_block

BETA_ID = 4.0
BETA_OOD = 0.5
R_MIN, R_MAX = 0.05, 0.9
RAG_RELIABILITY_DEFAULT = 0.3

_FOUR_CLASS = ("none", "lof", "balling", "keyhole")


def calculate_entropy(probs: Dict[str, float], classes=None) -> float:
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


def entropy_margin_reliability(ml_probs: Dict[str, float],
                               classes=_FOUR_CLASS) -> float:
    """Eq. (4): base DPS reliability R_cls from the normalized Shannon entropy
    and the top-1/top-2 margin. Unclipped, unmodulated; range [0, 1]."""
    values = np.array([ml_probs.get(c, 0.0) for c in classes], dtype=float)
    s = values.sum()
    if s <= 0:
        return 0.0
    values = values / s

    K = len(classes)
    h_norm = calculate_entropy(ml_probs, classes=classes) / np.log2(K) if K > 1 else 0.0
    p_sorted = np.sort(values)[::-1]
    margin = float(p_sorted[0] - p_sorted[1]) if len(p_sorted) >= 2 else 1.0
    return (1.0 - h_norm) * (0.5 + 0.5 * margin)


def dps_reliability(ml_probs: Dict[str, float], classes=_FOUR_CLASS) -> float:
    """Eq. (4)+(6): bounded DPS reliability R_cls in [R_MIN, R_MAX]."""
    return float(np.clip(entropy_margin_reliability(ml_probs, classes),
                         R_MIN, R_MAX))


def dps_fusion_weight(ml_probs: Dict[str, float], is_ood: bool,
                      classes=_FOUR_CLASS) -> float:
    """Eq. (3): w_cls = beta(omega) * R_cls.

    beta(omega) is the asymmetric distribution-shift modulation factor:
    BETA_ID in-distribution, BETA_OOD out-of-distribution.
    """
    beta = BETA_OOD if is_ood else BETA_ID
    return beta * dps_reliability(ml_probs, classes)


def ml_reliability_from_probs(ml_probs: Dict[str, float], is_ood: bool,
                              classes=_FOUR_CLASS) -> float:
    """LEGACY reliability score, retained for the supervisor-prompt display
    value only. Returns clip(beta_pre * R_cls) with beta_pre in {1.0 ID,
    0.5 OOD} — i.e. R_cls with the OOD penalty but *without* the ID boost.

    Not used for fusion any more: the fusion weight is `dps_fusion_weight()`.
    New code should call `dps_reliability()` (R_cls) or `dps_fusion_weight()`
    (w_cls) instead.
    """
    r = entropy_margin_reliability(ml_probs, classes)
    if r <= 0:
        return 0.1
    if is_ood:
        r *= 0.5
    return float(np.clip(r, R_MIN, R_MAX))


def _renormalize(belief: Dict[str, float], order) -> None:
    """In-place renormalization of a belief dict over `order` to sum 1."""
    total = sum(belief.get(k, 0.0) for k in order)
    if total > 0:
        for k in order:
            belief[k] = belief.get(k, 0.0) / total


def _linear_pool(ml_belief: Dict[str, float], w_cls: float,
                 rag_belief: Dict[str, float], w_ks: float,
                 order) -> Tuple[Dict[str, float], str]:
    """Eq. (8)-(9): reliability-weighted linear pool + argmax."""
    fused = {k: 0.0 for k in order}
    denom = w_cls + w_ks
    if denom > 0:
        for k in order:
            fused[k] = (ml_belief.get(k, 0.0) * w_cls
                        + rag_belief.get(k, 0.0) * w_ks) / denom
    else:
        fused = dict(ml_belief)
    return fused, max(fused, key=fused.get)


def deterministic_fusion(ml_probs_raw: Dict[str, float],
                         ml_reliability: float,
                         rag_resp: str,
                         is_ood: bool = False) -> Tuple[Dict[str, float], str]:
    """Reliability-weighted linear pool of the ML and KS belief distributions
    (Eq. 7-9). Returns (fused_belief_dict, fused_label).

    The DPS fusion weight w_cls is computed internally via `dps_fusion_weight`
    (Eq. 3). The `ml_reliability` argument is retained for backward
    compatibility with existing call sites and is ignored.
    """
    del ml_reliability

    default_belief = {l: 0.0 for l in LABEL_ORDER}
    ml_belief = dict(ml_probs_raw)
    rag_belief = extract_json_block(rag_resp, "BELIEF") or default_belief
    rag_rel = extract_float_block(rag_resp, "RELIABILITY")
    if rag_rel is None:
        rag_rel = RAG_RELIABILITY_DEFAULT

    _renormalize(ml_belief, LABEL_ORDER)
    _renormalize(rag_belief, LABEL_ORDER)

    w_cls = dps_fusion_weight(ml_probs_raw, is_ood, classes=tuple(LABEL_ORDER))
    w_ks = rag_rel
    return _linear_pool(ml_belief, w_cls, rag_belief, w_ks, LABEL_ORDER)


BINARY_LABEL_ORDER = ["good", "defective"]


def ml_reliability_from_probs_binary(ml_probs: Dict[str, float],
                                     is_ood: bool) -> float:
    """LEGACY binary reliability score (supervisor-prompt display value only).
    See `ml_reliability_from_probs`."""
    return ml_reliability_from_probs(ml_probs, is_ood,
                                     classes=tuple(BINARY_LABEL_ORDER))


def deterministic_fusion_binary(ml_probs_raw: Dict[str, float],
                                ml_reliability: float,
                                rag_resp: str,
                                is_ood: bool = False) -> Tuple[Dict[str, float], str]:
    """Reliability-weighted linear pool over {good, defective}. Eq. (3),(7-9).

    `ml_reliability` is retained for API compatibility and ignored; the DPS
    weight is computed internally via `dps_fusion_weight`.
    """
    del ml_reliability

    default_belief = {l: 0.0 for l in BINARY_LABEL_ORDER}
    ml_belief = dict(ml_probs_raw)
    rag_belief = extract_json_block(rag_resp, "BELIEF") or default_belief
    rag_rel = extract_float_block(rag_resp, "RELIABILITY")
    if rag_rel is None:
        rag_rel = RAG_RELIABILITY_DEFAULT

    _renormalize(ml_belief, BINARY_LABEL_ORDER)
    _renormalize(rag_belief, BINARY_LABEL_ORDER)

    w_cls = dps_fusion_weight(ml_probs_raw, is_ood,
                              classes=tuple(BINARY_LABEL_ORDER))
    w_ks = rag_rel
    return _linear_pool(ml_belief, w_cls, rag_belief, w_ks, BINARY_LABEL_ORDER)
