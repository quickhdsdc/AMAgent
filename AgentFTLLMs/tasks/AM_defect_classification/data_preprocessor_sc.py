# data_preprocessor.py
import pandas as pd
from typing import Dict, Any
from datasets import Dataset

LABEL_ORDER = ["none", "lof", "balling", "keyhole"]  # class ids 0,1,2,3

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

def _build_feature_text(row: Dict[str, Any]) -> str:
    """
    Build a compact, deterministic feature string for sequence classification.
    """
    mat = row.get("material", "unknown")
    pwr = _safe_val(row.get("Power"))
    vel = _safe_val(row.get("Velocity"))
    bd  = _safe_val(row.get("beam D"))
    lt  = _safe_val(row.get("layer thickness"))

    # NOTE: keep exact units/ordering stable for the classifier
    return (
        f"material {mat}; "
        f"laser power {pwr} W; "
        f"scan speed {vel} mm/s; "
        f"beam diameter {bd} µm; "
        f"layer thickness {lt} µm"
    )


class DataPreprocessor:
    """
    AM LPBF defect classification preprocessor for SEQUENCE CLASSIFICATION.

    - No chat template, no assistant messages.
    - Produces:
        text                (string features)
        label               (int 0..3)
        label_text          (canonical string)
        input_ids           (tokenized)
        attention_mask      (tokenized)
    """

    def __init__(self) -> None:
        print("Preprocessing the data for sequence classification...")

    def _row_to_seqcls(
        self,
        sample: Dict[str, Any],
        tokenizer,
        max_length: int,
    ) -> Dict[str, Any]:
        # normalize / attach label text
        sample["label"] = int(sample["label"])
        sample["label_text"] = _canon_label_text_from_int(sample["label"])

        # build plain feature string
        text = _build_feature_text(sample)
        sample["text"] = text

        # tokenize
        enc = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        sample["input_ids_text"] = enc["input_ids"]
        sample["attention_mask_text"] = enc["attention_mask"]

        return sample

    def preprocess_data(
        self,
        tokenizer,
        dataset: Dataset,
        max_length: int = 256,
        shuffle: bool = False,
    ) -> Dataset:
        """
        Map every row to a plain feature string + tokenization for sequence classification.
        Returns a Dataset with: text, label, label_text, input_ids, attention_mask
        """

        print("Preprocessing dataset... (sequence classification)")
        def map_fn(ex):
            return self._row_to_seqcls(ex, tokenizer, max_length)

        processed = dataset.map(
            map_fn,
            remove_columns=[
                c for c in dataset.column_names
                if c not in [
                    "label",
                    "material",
                    "Power",
                    "Velocity",
                    "beam D",
                    "layer thickness",
                ]
            ],
            keep_in_memory=True,
        )

        if shuffle:
            processed = processed.shuffle(seed=42)

        return processed

