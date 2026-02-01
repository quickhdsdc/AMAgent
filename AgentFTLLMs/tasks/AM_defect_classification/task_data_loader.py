import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import load_dataset


LABEL_ORDER = ["none", "lof", "balling", "keyhole"]  # 0,1,2,3


def _canon_label_text_from_int(x: int) -> str:
    """
    Map integer class id (0-3) to its canonical defect string.
    Falls back to 'unknown' if out of range.
    """
    try:
        xi = int(x)
    except Exception:
        return "unknown"
    if 0 <= xi < len(LABEL_ORDER):
        return LABEL_ORDER[xi]
    return "unknown"


class TaskDataLoader:
    """
    TaskDataLoader for AM QC defect classification.

    This loader expects data in:
        data/_am_qc/<experiment_name>/
            train_df.csv
            valid_df.csv
            test_df.csv

    If valid_df.csv does not exist yet, we will:
        - read <experiment_name>_train.csv from data_exp/
        - make a stratified train/valid split
        - read <experiment_name>_test.csv from data_exp/
        - write all 3 CSVs into data/_am_qc/<experiment_name>/
    Then we load those CSVs into a HuggingFace `DatasetDict`.

    Columns in each df:
        material
        Power
        Velocity
        beam D
        layer thickness
        defect_label      (int 0..3)
        label             (duplicate of defect_label, explicitly named 'label')
        label_text        (string, e.g. "keyhole")

    We expose:
        - dataset splits via load_train/load_val/load_test
        - label vocab via get_labels()
    """

    def __init__(
        self,
        experiment_name: str,
        cache_root: str = "data/_am_qc",
        source_root: str = "data/_am_qc/data_exp",
        val_frac: float = 0.1,
        random_state: int = 42,
    ) -> None:

        self.experiment_name = experiment_name
        self.cache_root = Path(cache_root)
        self.source_root = Path(source_root)
        self.task_data_dir = self.cache_root / experiment_name
        self.task_data_dir.mkdir(parents=True, exist_ok=True)

        # file paths we want to end up with
        train_cached = self.task_data_dir / "train_df.csv"
        valid_cached = self.task_data_dir / "valid_df.csv"
        test_cached  = self.task_data_dir / "test_df.csv"

        # If we don't yet have cached train/valid/test, build them
        if not (train_cached.exists() and valid_cached.exists() and test_cached.exists()):
            # 1. load experiment splits from data_exp
            train_src = self.source_root / f"{experiment_name}_train.csv"
            test_src  = self.source_root / f"{experiment_name}_test.csv"

            if not train_src.exists():
                raise FileNotFoundError(f"Missing source train split {train_src}")
            if not test_src.exists():
                raise FileNotFoundError(f"Missing source test split {test_src}")

            df_train_full = pd.read_csv(train_src)
            df_test_full  = pd.read_csv(test_src)

            # Sanity: require 'defect_label' exists
            if "defect_label" not in df_train_full.columns:
                raise RuntimeError(f"{experiment_name}: 'defect_label' not in train split")
            if "defect_label" not in df_test_full.columns:
                raise RuntimeError(f"{experiment_name}: 'defect_label' not in test split")

            # Add label_text + label (int) columns
            df_train_full["label"] = df_train_full["defect_label"].astype(int)
            df_train_full["label_text"] = df_train_full["label"].apply(_canon_label_text_from_int)

            df_test_full["label"] = df_test_full["defect_label"].astype(int)
            df_test_full["label_text"] = df_test_full["label"].apply(_canon_label_text_from_int)

            # 2. make stratified train/valid split from df_train_full
            # we'll hold out val_frac (e.g. 10%) as validation
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_frac,
                random_state=random_state,
            )

            y = df_train_full["label"].values
            train_idx, val_idx = next(splitter.split(df_train_full, y))

            df_train = df_train_full.iloc[train_idx].reset_index(drop=True)
            df_valid = df_train_full.iloc[val_idx].reset_index(drop=True)

            # 3. save canonicalized CSVs
            df_train.to_csv(train_cached, index=False)
            df_valid.to_csv(valid_cached, index=False)
            df_test_full.to_csv(test_cached, index=False)

        # At this point, we should have all three cached CSVs
        data_files = {
            "train": str(train_cached),
            "valid": str(valid_cached),
            "test":  str(test_cached),
        }
        self.data_files = data_files

        # Build HuggingFace DatasetDict
        # keep_default_na=False so empty strings stay "" instead of NaN
        self.dataset = load_dataset(
            "csv",
            data_files=data_files,
            keep_default_na=False,
            cache_dir=str(self.task_data_dir),
        )

        # build labels / id maps from the train split
        # self.dataset["train"] is a Dataset with columns including "label" (ints) and "label_text"
        label_ids = sorted(set(int(x) for x in self.dataset["train"]["label"]))
        # we assume 0..3, but let's infer anyway:
        id2label = {i: LABEL_ORDER[i] for i in label_ids}
        label2id = {v: k for k, v in id2label.items()}
        labels_ordered = [id2label[i] for i in sorted(id2label.keys())]

        self.id2label = id2label      # e.g. {0:"none",1:"lof",2:"balling",3:"keyhole"}
        self.label2id = label2id      # e.g. {"none":0, ...}
        self.labels = labels_ordered  # e.g. ["none","lof","balling","keyhole"]

    # ------------- API -------------
    def load_train(self):
        return self.dataset["train"]

    def load_val(self):
        return self.dataset["valid"]

    def load_test(self):
        return self.dataset["test"]

    def load_data(self):
        return self.load_train(), self.load_val(), self.load_test()

    def get_labels(self) -> Tuple[List[str], Dict[str,int], Dict[int,str]]:
        return self.labels, self.label2id, self.id2label
