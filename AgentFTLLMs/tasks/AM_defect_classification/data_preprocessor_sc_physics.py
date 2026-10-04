from __future__ import annotations

import math
import pandas as pd
from typing import Dict, Any
from datasets import Dataset

from .physics_features import build_physics_snippet, compute_ved, ved_regime_id
from .task_labels import LABEL_ORDER


def _canon_label_text_from_int(x):
    try:
        xi = int(x)
    except Exception:
        return "none"
    if 0 <= xi < len(LABEL_ORDER):
        return LABEL_ORDER[xi]
    return "none"


def _safe_val(x):
    if pd.isna(x):
        return "unknown"
    return str(x)


def _safe_float(x, default=0.0):
    """Convert to float; return `default` for NaN / non-numeric / None."""
    if x is None:
        return float(default)
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(fx) or math.isinf(fx):
        return float(default)
    return fx


def _build_feature_text(row: Dict[str, Any], use_physics_features: bool = True) -> str:
    """Compact, deterministic feature string.

    Adds hatch spacing vs C7. When `use_physics_features` is True it also
    appends physics-informed features (VED + process regime) via
    `physics_features.build_physics_snippet`. Set False to reproduce the
    loss-only ablation variant (no physics in the prompt).
    """
    mat = row.get("material", "unknown")
    pwr = _safe_val(row.get("Power"))
    vel = _safe_val(row.get("Velocity"))
    bd = _safe_val(row.get("beam D"))
    lt = _safe_val(row.get("layer thickness"))
    hs = _safe_val(row.get("Hatch spacing"))

    base = (
        f"material {mat}; "
        f"laser power {pwr} W; "
        f"scan speed {vel} mm/s; "
        f"beam diameter {bd} µm; "
        f"layer thickness {lt} µm; "
        f"hatch spacing {hs} µm"
    )
    if use_physics_features:
        return base + build_physics_snippet(row)
    return base


class DataPreprocessor:
    """Sequence-classification preprocessor with raw-parameter passthrough."""

    def __init__(self, use_physics_features: bool = True) -> None:
        self.use_physics_features = use_physics_features
        mode = "with" if use_physics_features else "without"
        print(
            f"Preprocessing the data for sequence classification "
            f"(physics variant, {mode} physics features)..."
        )

    def _row_to_seqcls(
        self,
        sample: Dict[str, Any],
        tokenizer,
        max_length: int,
    ) -> Dict[str, Any]:
        sample["label"] = int(sample["label"])
        sample["label_text"] = _canon_label_text_from_int(sample["label"])

        text = _build_feature_text(sample, self.use_physics_features)
        sample["text"] = text

        enc = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        sample["input_ids_text"] = enc["input_ids"]
        sample["attention_mask_text"] = enc["attention_mask"]

        sample["raw_params"] = [
            _safe_float(sample.get("Power")),
            _safe_float(sample.get("Velocity")),
            _safe_float(sample.get("Hatch spacing")),
            _safe_float(sample.get("layer thickness")),
        ]

        ved = compute_ved(
            sample.get("Power"),
            sample.get("Velocity"),
            sample.get("Hatch spacing"),
            sample.get("layer thickness"),
        )
        sample["regime_label"] = ved_regime_id(ved)
        return sample

    def preprocess_data(
        self,
        tokenizer,
        dataset: Dataset,
        max_length: int = 256,
        shuffle: bool = False,
    ) -> Dataset:
        print("Preprocessing dataset... (sequence classification + physics passthrough)")

        def map_fn(ex):
            return self._row_to_seqcls(ex, tokenizer, max_length)

        keep_cols = {
            "label",
            "material",
            "Power",
            "Velocity",
            "beam D",
            "layer thickness",
            "Hatch spacing",
            "raw_params",
            "regime_label",
            "text",
            "label_text",
            "input_ids_text",
            "attention_mask_text",
        }

        processed = dataset.map(
            map_fn,
            remove_columns=[c for c in dataset.column_names if c not in keep_cols],
            keep_in_memory=True,
        )

        if shuffle:
            processed = processed.shuffle(seed=42)
        return processed
