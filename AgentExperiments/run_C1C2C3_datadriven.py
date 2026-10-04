import os
import sys

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.getcwd())

from amcore import EXPERIMENTS, LABEL_COL
from amcore.data_io import EXP_DIR
from amcore.ml_zoo import BEST_MODELS, _make_clf, RANDOM_STATE

META_COLS = ["material"]


class CORALTransformer(BaseEstimator, TransformerMixin):
    """Correlation Alignment (CORAL) for feature alignment (Sun et al.)."""

    def __init__(self, reg: float = 1e-5):
        self.reg = reg
        self.whitening = None
        self.coloring = None

    def fit(self, X_source, X_target):
        src_cov = np.cov(X_source, rowvar=False) + self.reg * np.eye(X_source.shape[1])
        tgt_cov = np.cov(X_target, rowvar=False) + self.reg * np.eye(X_target.shape[1])
        self.whitening = linalg.fractional_matrix_power(src_cov, -0.5)
        self.coloring = linalg.fractional_matrix_power(tgt_cov, 0.5)
        return self

    def transform(self, X):
        if self.whitening is None or self.coloring is None:
            return X
        return np.dot(np.dot(X, self.whitening), self.coloring).real


def importance_weighting(X_source, X_target, r_clf=None):
    """Discriminator-based importance weights for covariate shift."""
    if r_clf is None:
        r_clf = LogisticRegression(solver="liblinear", random_state=RANDOM_STATE)
    X_all = np.vstack([X_source, X_target])
    y_domain = np.concatenate([np.zeros(len(X_source)), np.ones(len(X_target))])
    r_clf.fit(X_all, y_domain)
    probs = np.clip(r_clf.predict_proba(X_source)[:, 1], 0.05, 0.95)
    weights = probs / (1 - probs)
    return weights / weights.sum() * len(X_source)


def mixup_augmentation(X, y, alpha: float = 0.2, num_new=None):
    """Manifold-mixup variant for tabular trees (mix features, keep dominant label)."""
    if num_new is None:
        num_new = len(X) // 2
    n = len(X)
    X_mix, y_mix = [], []
    for _ in range(num_new):
        i, j = np.random.randint(0, n), np.random.randint(0, n)
        lam = np.random.beta(alpha, alpha)
        X_mix.append(lam * X[i] + (1 - lam) * X[j])
        y_mix.append(y[i] if lam >= 0.5 else y[j])
    return np.vstack([X, np.array(X_mix)]), np.concatenate([y, np.array(y_mix)])


def _load_split(stem: str):
    train_path = os.path.join(EXP_DIR, f"{stem}_train.csv")
    test_path = os.path.join(EXP_DIR, f"{stem}_test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train split: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test split: {test_path}")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def _split_features_labels(df: pd.DataFrame):
    if LABEL_COL not in df.columns:
        raise RuntimeError(f"Expected label col '{LABEL_COL}' not in dataframe.")
    drop_cols = [LABEL_COL] + [c for c in META_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols].copy(), df[LABEL_COL].copy(), feature_cols


def _eval_strategies(df_train, df_test, best_model_proto, label_encoder: LabelEncoder):
    X_train_df, y_train_series, _ = _split_features_labels(df_train)
    X_test_df, y_test_series, _ = _split_features_labels(df_test)
    X_train, X_test = X_train_df.values, X_test_df.values
    y_train_enc = label_encoder.transform(y_train_series.values)

    seen = set(label_encoder.classes_)
    mask = np.array([lbl in seen for lbl in y_test_series.values], dtype=bool)
    if not np.any(mask):
        return {}
    y_test_enc = label_encoder.transform(y_test_series[mask].values)
    X_test_known = X_test[mask]

    results = {}

    model = clone(best_model_proto)
    model.fit(X_train, y_train_enc)
    results["Baseline"] = f1_score(y_test_enc, model.predict(X_test_known), average="macro")

    try:
        w = importance_weighting(X_train, X_test)
        m = clone(best_model_proto)
        m.fit(X_train, y_train_enc, sample_weight=w)
        results["ImpWeight"] = f1_score(y_test_enc, m.predict(X_test_known), average="macro")
    except Exception as e:
        print(f"ImpWeight failed: {e}")
        results["ImpWeight"] = 0.0

    try:
        coral = CORALTransformer().fit(X_train, X_test)
        m = clone(best_model_proto)
        m.fit(coral.transform(X_train), y_train_enc)
        results["CORAL"] = f1_score(y_test_enc, m.predict(X_test_known), average="macro")
    except Exception as e:
        print(f"CORAL failed: {e}")
        results["CORAL"] = 0.0

    try:
        X_mix, y_mix = mixup_augmentation(X_train, y_train_enc, alpha=0.2)
        m = clone(best_model_proto)
        m.fit(X_mix, y_mix)
        results["Mixup"] = f1_score(y_test_enc, m.predict(X_test_known), average="macro")
    except Exception as e:
        print(f"Mixup failed: {e}")
        results["Mixup"] = 0.0

    return results


def run_experiment(stem: str) -> dict:
    df_train, df_test = _load_split(stem)
    model_name = BEST_MODELS.get(stem, "RF")
    best_model_proto = _make_clf(model_name)
    _, y_train_series, _ = _split_features_labels(df_train)
    le = LabelEncoder().fit(y_train_series.values)
    strategies_res = _eval_strategies(df_train, df_test, best_model_proto, le)
    return {
        "experiment": stem,
        "best_model_name": model_name,
        "results": strategies_res,
        "n_test": len(df_test),
    }


def main():
    summary_rows = []
    for stem in EXPERIMENTS:
        print(f"Running {stem}...")
        try:
            res = run_experiment(stem)
            r = res["results"]
            print(f"  [{stem}] Baseline: {r.get('Baseline',0):.4f}, "
                  f"IW: {r.get('ImpWeight',0):.4f}, "
                  f"CORAL: {r.get('CORAL',0):.4f}, "
                  f"Mixup: {r.get('Mixup',0):.4f}")
            summary_rows.append({
                "experiment": stem,
                "model": res["best_model_name"],
                "Baseline_F1": r.get("Baseline", 0),
                "ImpWeight_F1": r.get("ImpWeight", 0),
                "CORAL_F1": r.get("CORAL", 0),
                "Mixup_F1": r.get("Mixup", 0),
            })
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        out_csv = "./results_AM/experiment_results_tf_summary.csv"
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved summary to {out_csv}")
        print(df)


if __name__ == "__main__":
    main()
