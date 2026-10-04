from __future__ import annotations

import math
from typing import Any, Dict

from .physics_loss import (
    VED_LOF_UPPER,
    VED_CONDUCTION_UPPER,
    VED_TRANSITION_UPPER,
)


def compute_ved(power_w, velocity_mm_s, hatch_um, layer_um) -> float:
    """VED = P / (V * h * t) in J/mm^3.

    Power in W, velocity in mm/s, hatch and layer thickness in micrometres
    (converted to mm internally). Returns NaN when any denominator term is
    zero or non-finite so callers can fall back gracefully.
    """
    try:
        p = float(power_w)
        v = float(velocity_mm_s)
        h = float(hatch_um) / 1000.0
        t = float(layer_um) / 1000.0
    except (TypeError, ValueError):
        return float("nan")

    denom = v * h * t
    if not math.isfinite(denom) or denom == 0.0:
        return float("nan")
    ved = p / denom
    return ved if math.isfinite(ved) else float("nan")


def ved_regime(ved: float) -> str:
    """Map a VED value to its process regime (human-readable string)."""
    if not math.isfinite(ved):
        return "unknown"
    if ved < VED_LOF_UPPER:
        return "lack-of-fusion"
    if ved <= VED_CONDUCTION_UPPER:
        return "conduction"
    if ved <= VED_TRANSITION_UPPER:
        return "transition"
    return "keyhole"


REGIME_ORDER = ["lack-of-fusion", "conduction", "transition", "keyhole"]
REGIME_TO_ID = {name: i for i, name in enumerate(REGIME_ORDER)}
NUM_REGIMES = len(REGIME_ORDER)


def ved_regime_id(ved: float) -> int:
    """Regime as an integer class id; -1 when VED is undefined."""
    return REGIME_TO_ID.get(ved_regime(ved), -1)


INCLUDE_MELTPOOL = False

T0_K = 298.0

_MATERIAL_PROPS: Dict[str, Dict[str, float]] = {
    "SS316L":    {"rho": 7.98, "Cp": 500.0, "k": 15.0, "Tm": 1700.0},
    "SS17-4PH":  {"rho": 7.75, "Cp": 460.0, "k": 18.0, "Tm": 1720.0},
    "Ti-6Al-4V": {"rho": 4.43, "Cp": 560.0, "k":  7.0, "Tm": 1928.0},
}


def _material_props(material: str):
    """Look up thermophysical constants; tolerant to case/whitespace."""
    if material is None:
        return None
    key = str(material).strip()
    if key in _MATERIAL_PROPS:
        return _MATERIAL_PROPS[key]
    for k, v in _MATERIAL_PROPS.items():
        if k.lower() == key.lower():
            return v
    return None


def predict_melt_pool(power_w, velocity_mm_s, material: str):
    """Predicted (depth_um, width_um) via Liu et al. 2025 Eqs. (4)-(5).

    Returns (nan, nan) when the material is unknown or inputs are invalid.
    See the unit caveats above before trusting absolute values.
    """
    props = _material_props(material)
    if props is None:
        return float("nan"), float("nan")
    try:
        p = float(power_w)
        v = float(velocity_mm_s)
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    if not (math.isfinite(p) and math.isfinite(v)) or p <= 0 or v <= 0:
        return float("nan"), float("nan")

    rho, cp, k = props["rho"], props["Cp"], props["k"]
    dt = props["Tm"] - T0_K

    d = (0.37e6 * p**0.51 * v**-0.46 * rho**-0.46
         * cp**-0.46 * k**-0.06 * dt**-0.51)
    w = (0.44e6 * p**0.59 * v**-0.36 * rho**-0.39
         * cp**-0.39 * k**-0.21 * dt**-0.60)
    return d, w


def build_physics_snippet(row: Dict[str, Any]) -> str:
    """Physics feature string appended to the classifier's feature text.

    Always emits VED and the VED regime. Emits melt-pool depth/width only
    when `INCLUDE_MELTPOOL` is True.
    """
    ved = compute_ved(
        row.get("Power"),
        row.get("Velocity"),
        row.get("Hatch spacing"),
        row.get("layer thickness"),
    )

    if math.isfinite(ved):
        snippet = (
            f"; volumetric energy density {ved:.1f} J/mm³"
            f"; process regime {ved_regime(ved)}"
        )
    else:
        snippet = "; volumetric energy density unknown; process regime unknown"

    if INCLUDE_MELTPOOL:
        d, w = predict_melt_pool(
            row.get("Power"), row.get("Velocity"), row.get("material")
        )
        if math.isfinite(d) and math.isfinite(w) and w > 0:
            snippet += (
                f"; predicted melt-pool depth {d:.0f} µm"
                f"; predicted melt-pool width {w:.0f} µm"
                f"; depth/width ratio {d / w:.2f}"
            )

    return snippet
