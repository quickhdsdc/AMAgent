import os
from typing import List, Optional

import pandas as pd

EXP_DIR = "data_exp"
EXP_DIR_BINARY = "data_exp_binary"
LABEL_COL = "defect_label"
META_COL = "material"

EXPERIMENTS = [
    "Exp_ID_1",  "Exp_OOD_1",
    "Exp_ID_2",  "Exp_OOD_2",
    "Exp_ID_3",  "Exp_OOD_3",
    "Exp_ID_4",  "Exp_OOD_4",
]

BINARY_EXPERIMENTS_HINT = [
    "Exp_BIN_MAT_ID_1", "Exp_BIN_MAT_OOD_1",
    "Exp_BIN_MAT_ID_2", "Exp_BIN_MAT_OOD_2",
    "Exp_BIN_MAT_ID_3", "Exp_BIN_MAT_OOD_3",
    "Exp_BIN_MAT_ID_4", "Exp_BIN_MAT_OOD_4",
    "Exp_BIN_SRC_D0toD1D2", "Exp_BIN_SRC_D0D1toD2", "Exp_BIN_SRC_AllMixed",
]


def exp_type_from_stem(stem: str) -> str:
    """Categorize a stem as in-distribution or out-of-distribution.

    Recognized in-distribution markers: "_ID_", "_AllMixed", trailing "_ID".
    Everything else (incl. OOD, DEV, GEO, SRC_X-to-Y) is out-of-distribution.
    """
    if "_ID_" in stem or stem.endswith("_ID") or "AllMixed" in stem:
        return "in-distribution"
    return "out-of-distribution"


def is_binary_stem(stem: str) -> bool:
    return "_BIN_" in stem


def exp_base_dir(stem: str) -> str:
    """Pick the right experiment directory for a stem."""
    return EXP_DIR_BINARY if is_binary_stem(stem) else EXP_DIR


def load_exp_split(stem: str, base_dir: Optional[str] = None) -> pd.DataFrame:
    if base_dir is None:
        base_dir = exp_base_dir(stem)
    test_path = os.path.join(base_dir, f"{stem}_test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test split: {test_path}")
    return pd.read_csv(test_path)


def load_exp_train(stem: str, base_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    if base_dir is None:
        base_dir = exp_base_dir(stem)
    train_path = os.path.join(base_dir, f"{stem}_train.csv")
    if not os.path.exists(train_path):
        return None
    return pd.read_csv(train_path)


def list_binary_experiments(base_dir: str = EXP_DIR_BINARY) -> List[str]:
    """Discover all binary experiment stems present on disk
    (those with both _train.csv and _test.csv files)."""
    if not os.path.isdir(base_dir):
        return []
    stems_with_train = {
        f[: -len("_train.csv")] for f in os.listdir(base_dir)
        if f.endswith("_train.csv")
    }
    stems_with_test = {
        f[: -len("_test.csv")] for f in os.listdir(base_dir)
        if f.endswith("_test.csv")
    }
    return sorted(stems_with_train & stems_with_test)
