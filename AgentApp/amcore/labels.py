from typing import Dict, Optional
import pandas as pd

LABEL_ORDER = ["none", "lof", "balling", "keyhole"]
VALID_LABELS = set(LABEL_ORDER)

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

    "alsi10mg": "AlSi10Mg",
    "al-si-10mg": "AlSi10Mg",
    "alsi10": "AlSi10Mg",
    "aluminum-silicon-magnesium": "AlSi10Mg",

    "18ni300": "18Ni300",
    "18 ni 300": "18Ni300",
    "18-ni-300": "18Ni300",
    "maraging steel 300": "18Ni300",
    "maraging steel": "18Ni300",
    "ms1": "18Ni300",
    "maraging": "18Ni300",

    "hastelloy x": "Hastelloy X",
    "hastelloyx": "Hastelloy X",
    "hastelloy-x": "Hastelloy X",

    "in625": "IN625",
    "inconel 625": "IN625",
    "alloy 625": "IN625",

    "cucrzr": "CuCrZr",
    "cu-cr-zr": "CuCrZr",
}


def canonicalize_material(name: Optional[str]) -> Optional[str]:
    if not name or (isinstance(name, float) and pd.isna(name)):
        return None
    key = str(name).strip().lower()
    return MATERIAL_ALIASES.get(key, None)


def normalize_ground_truth_label(y) -> str:
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
