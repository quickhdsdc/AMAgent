from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

from amcore.data_io import LABEL_COL
from amcore.labels import LABEL_ORDER, normalize_ground_truth_label

BEST_MODELS = {
    "Exp_ID_1":  "RF",
    "Exp_OOD_1": "RF",
    "Exp_ID_2":  "RF",
    "Exp_OOD_2": "GB",
    "Exp_ID_3":  "RF",
    "Exp_OOD_3": "RF",
    "Exp_ID_4":  "RF",
    "Exp_OOD_4": "GB",
}

DEFAULT_NUM_FEATS = ["Power", "Velocity", "beam D", "layer thickness", "Hatch spacing"]
DEFAULT_CAT_FEATS = ["material"]

RANDOM_STATE = 42


def _make_clf(model_name: str):
    if model_name == "GB":
        return GradientBoostingClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=20,
            random_state=RANDOM_STATE,
            loss="log_loss",
        )
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )


def build_ml_pipeline(stem: str,
                     model_name: Optional[str] = None) -> Tuple[Pipeline, str]:
    """Return (sklearn Pipeline, model_name_used)."""
    if model_name is None:
        model_name = BEST_MODELS.get(stem, "RF")

    num_tf = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer([
        ("num", num_tf, DEFAULT_NUM_FEATS),
        ("cat", cat_tf, DEFAULT_CAT_FEATS),
    ])

    pipe = Pipeline([
        ("pre", pre),
        ("clf", _make_clf(model_name)),
    ])
    return pipe, model_name


def train_and_predict_proba(stem: str,
                            df_train: pd.DataFrame,
                            df_test: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """Train the best model for `stem` and return row_idx -> label-prob dict."""
    cols_needed = [c for c in DEFAULT_NUM_FEATS + DEFAULT_CAT_FEATS
                   if c in df_train.columns]
    if LABEL_COL not in df_train.columns:
        raise RuntimeError(f"{stem}: '{LABEL_COL}' not found in train set.")

    X_train = df_train[cols_needed].copy()
    y_train = df_train[LABEL_COL].copy()
    X_test = df_test.reindex(columns=cols_needed, fill_value=np.nan).copy()

    num_present = [c for c in DEFAULT_NUM_FEATS if c in cols_needed]
    cat_present = [c for c in DEFAULT_CAT_FEATS if c in cols_needed]

    num_tf = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if num_present:
        transformers.append(("num", num_tf, num_present))
    if cat_present:
        transformers.append(("cat", cat_tf, cat_present))
    pre = ColumnTransformer(transformers)

    model_name = BEST_MODELS.get(stem, "RF")
    pipe = Pipeline([("pre", pre), ("clf", _make_clf(model_name))])

    y_train_norm = y_train.apply(normalize_ground_truth_label)
    mask = y_train_norm.isin(LABEL_ORDER)
    X_train = X_train[mask]
    y_train_norm = y_train_norm[mask]
    if len(X_train) == 0:
        raise RuntimeError(f"{stem}: No valid training rows after label normalization.")

    if model_name == "GB":
        weights = compute_sample_weight(class_weight="balanced", y=y_train_norm)
        pipe.fit(X_train, y_train_norm, **{"clf__sample_weight": weights})
    else:
        pipe.fit(X_train, y_train_norm)

    proba = pipe.predict_proba(X_test)
    cls_order = list(pipe.classes_)

    out: Dict[int, Dict[str, float]] = {}
    for ridx, idx in enumerate(df_test.index):
        row_map = {c: 0.0 for c in LABEL_ORDER}
        for c_idx, c_name in enumerate(cls_order):
            row_map[c_name] = float(proba[ridx][c_idx])
        out[int(idx)] = {k: round(row_map[k], 4) for k in LABEL_ORDER}
    return out


def argmax_label_from_probs(prob_map: Dict[str, float]) -> str:
    if not prob_map:
        return "unknown"
    best_lbl, best_p = "unknown", -1.0
    for lbl in LABEL_ORDER:
        p = float(prob_map.get(lbl, 0.0))
        if p > best_p:
            best_p = p
            best_lbl = lbl
    return best_lbl
