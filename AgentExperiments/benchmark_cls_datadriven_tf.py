import os
import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_sample_weight
from scipy import linalg
import random

# ----------------------------------------------------
# Config
# ----------------------------------------------------

EXP_DIR = "data_exp"   # where Exp_ID_1_train.csv etc. live
LABEL_COL = "defect_label"
META_COLS = ["material"]  # columns we should NOT treat as numeric features
randomState = 11

# ----------------------------------------------------
# Model zoo and fixed selection
# ----------------------------------------------------

# Kernel reused for GaussianProcessClassifier
ckernel = (
    (ConstantKernel() * RBF() + WhiteKernel())
    + (ConstantKernel() * RBF() + WhiteKernel())
)

MODEL_ZOO = {
    "RF": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=randomState,
            n_jobs=-1,
            class_weight="balanced", # Handle imbalance directly
        ),
    "GP": GaussianProcessClassifier(
        kernel=ckernel,
        multi_class="one_vs_rest",
        n_restarts_optimizer=10,
        max_iter_predict=500,
        random_state=randomState,
    ),
    "SVM": SVC(
        kernel="rbf",
        gamma="scale",
        C=12,
        random_state=randomState,
    ),
    "GB": GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=20,
        random_state=randomState,
        loss="log_loss",
    ),
    "NN": MLPClassifier(
        max_iter=1000,
        random_state=randomState,
    ),
    "Bootstrap": BaggingClassifier(
        random_state=randomState,
    ),
}

EXPERIMENTS = [
    {"stem": "Exp_ID_1"},
    {"stem": "Exp_OOD_1"},
    {"stem": "Exp_ID_2"},
    {"stem": "Exp_OOD_2"},
    {"stem": "Exp_ID_3"},
    {"stem": "Exp_OOD_3"},
    {"stem": "Exp_ID_4"},
    {"stem": "Exp_OOD_4"},
]

# Fixed best model per experiment (no CV/selection)
BEST_MODELS = {
    "Exp_ID_1": "RF",
    "Exp_OOD_1": "RF",
    "Exp_ID_2": "RF",
    "Exp_OOD_2": "GB",
    "Exp_ID_3": "RF",
    "Exp_OOD_3": "RF",
    "Exp_ID_4": "RF",
    "Exp_OOD_4": "GB",
}

def _get_fixed_best_model_proto(stem: str):
    name = BEST_MODELS.get(stem)
    if name is None:
        raise KeyError(f"No fixed best model specified for experiment '{stem}'.")
    if name not in MODEL_ZOO:
        raise KeyError(f"Unknown model name '{name}' in BEST_MODELS for '{stem}'.")
    return name, MODEL_ZOO[name]

# ----------------------------------------------------
# DG / TF Algorithms
# ----------------------------------------------------

class CORALTransformer(BaseEstimator, TransformerMixin):
    """
    Correlation Alignment (CORAL) for feature alignment.
    Aligns covariance of Source to Target.
    """
    def __init__(self, reg=1e-5):
        self.reg = reg
        self.src_cov = None
        self.target_cov = None
        self.whitening = None
        self.coloring = None

    def fit(self, X_source, X_target):
        # Calculate source and target covariances
        # Add regularization for stability
        self.src_cov = np.cov(X_source, rowvar=False) + self.reg * np.eye(X_source.shape[1])
        self.target_cov = np.cov(X_target, rowvar=False) + self.reg * np.eye(X_target.shape[1])
        
        # Whitening matrix (Source^-0.5)
        # scipy.linalg.fractional_matrix_power can calculate A^0.5
        # We need A^-0.5
        self.whitening = linalg.fractional_matrix_power(self.src_cov, -0.5)
        
        # Coloring matrix (Target^0.5)
        self.coloring = linalg.fractional_matrix_power(self.target_cov, 0.5)
        return self

    def transform(self, X):
        if self.whitening is None or self.coloring is None:
            return X
        # X_aligned = X * whitening * coloring
        # Check shapes: X is (N, D), Matrices are (D, D)
        return np.dot(np.dot(X, self.whitening), self.coloring).real

def importance_weighting(X_source, X_target, r_clf=None):
    """
    Discriminator-based Importance Weighting (Covariate Shift).
    Trains a classifier to distinguish source (LABEL=0) from target (LABEL=1).
    Weights = P(Target|x) / P(Source|x)
    """
    if r_clf is None:
        r_clf = LogisticRegression(solver="liblinear", random_state=randomState)

    # Create dataset identifying source vs target
    X_all = np.vstack([X_source, X_target])
    y_domain = np.concatenate([np.zeros(len(X_source)), np.ones(len(X_target))])
    
    r_clf.fit(X_all, y_domain)
    
    # Probabilities of being Target
    probs = r_clf.predict_proba(X_source)[:, 1]
    
    # Avoid division by zero warnings
    probs = np.clip(probs, 0.05, 0.95)
    
    # Density ratio w(x) = p_target(x) / p_source(x) ~ P(T|x)/P(S|x) * P(S)/P(T)
    # We ignore the constant P(S)/P(T) as it scales all weights equally
    weights = probs / (1 - probs)
    
    # Normalize weights to sum to n_source
    weights = weights / weights.sum() * len(X_source)
    return weights

def mixup_augmentation(X, y, alpha=0.2, num_new=None):
    """
    Perform mixup augmentation on training data.
    Linear interpolation between pairs of examples.
    """
    if num_new is None:
        num_new = len(X) // 2
        
    n_samples = len(X)
    X_mix = []
    y_mix = []
    
    # If y is categorical/encoded integers, mixup on labels is tricky for RF/GB
    # (they expect discrete classes). 
    # For Tree ensembles, we typically only mix features and keep the label of the dominant parent,
    # or use sample weights. 
    # Simpler "Manifold Mixup" for Tabular: 
    # Create new sample x' = lam*xi + (1-lam)*xj
    # Label y' = yi if lam > 0.5 else yj
    
    for _ in range(num_new):
        i = np.random.randint(0, n_samples)
        j = np.random.randint(0, n_samples)
        
        lam = np.random.beta(alpha, alpha)
        
        x_new = lam * X[i] + (1 - lam) * X[j]
        y_new = y[i] if lam >= 0.5 else y[j]
        
        X_mix.append(x_new)
        y_mix.append(y_new)
        
    X_aug = np.vstack([X, np.array(X_mix)])
    y_aug = np.concatenate([y, np.array(y_mix)])
    
    return X_aug, y_aug


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------

def _load_split(stem: str):
    train_path = os.path.join(EXP_DIR, f"{stem}_train.csv")
    test_path  = os.path.join(EXP_DIR, f"{stem}_test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train split: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test split: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    return df_train, df_test


def _split_features_labels(df: pd.DataFrame):
    if LABEL_COL not in df.columns:
        raise RuntimeError(f"Expected label col '{LABEL_COL}' not in dataframe.")

    drop_cols = [LABEL_COL] + [c for c in META_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_df = df[feature_cols].copy()
    y_series = df[LABEL_COL].copy()

    return X_df, y_series, feature_cols


def _final_train_and_eval_strategies(
    experiment_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    best_model_proto,
    label_encoder: LabelEncoder,
):
    """
    Evaluates 3 DG/TF strategies + Baseline.
    """
    # TRAIN data
    X_train_df, y_train_series, feature_cols_train = _split_features_labels(df_train)
    X_train = X_train_df.values
    y_train_enc = label_encoder.transform(y_train_series.values)

    # TEST data
    X_test_df, y_test_series, feature_cols_test = _split_features_labels(df_test)
    X_test = X_test_df.values
    
    # Test Labels (filter known)
    y_test = y_test_series.values
    seen_classes = set(label_encoder.classes_)
    mask_known = np.array([lbl in seen_classes for lbl in y_test], dtype=bool)
    
    if not np.any(mask_known):
        return {}

    y_test_known = y_test_series[mask_known].values
    y_test_enc = label_encoder.transform(y_test_known)
    X_test_known = X_test[mask_known]

    results = {}
    
    # ------------------------------------------------
    # 0. Baseline (No DG)
    # ------------------------------------------------
    model = clone(best_model_proto)
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test_known)
    results["Baseline"] = f1_score(y_test_enc, y_pred, average="macro")

    # ------------------------------------------------
    # 1. Importance Weighting (Covariate Shift)
    # ------------------------------------------------
    # Weights for Source samples to match Target distribution
    try:
        sample_weights = importance_weighting(X_train, X_test) # Transductive use of X_test
        model_iw = clone(best_model_proto)
        # GB and RF support sample_weight in fit
        model_iw.fit(X_train, y_train_enc, sample_weight=sample_weights)
        y_pred = model_iw.predict(X_test_known)
        results["ImpWeight"] = f1_score(y_test_enc, y_pred, average="macro")
    except Exception as e:
        print(f"ImpWeight failed: {e}")
        results["ImpWeight"] = 0.0

    # ------------------------------------------------
    # 2. CORAL (Feature Alignment)
    # ------------------------------------------------
    try:
        coral = CORALTransformer()
        # Align Source to Target
        coral.fit(X_train, X_test)
        X_train_coral = coral.transform(X_train)
        X_test_coral = coral.transform(X_test) # Also transform test? 
        # Wait, CORAL usually aligns Source to match Target. So we transform Source. 
        # Target is considered the "Anchor". So Target features stay as they are (or identity transform).
        # Actually, in standard CORAL (Sun et al.), we map source to target. 
        # So we train on f(X_source) and test on X_target.
        
        # However, checking implementation: transform(X) applies whitening(Src)*coloring(Tgt).
        # So transform(X_train) makes it look like Target.
        # X_test should remain X_test.
        
        model_coral = clone(best_model_proto)
        model_coral.fit(X_train_coral, y_train_enc)
        y_pred = model_coral.predict(X_test_known) # Predict on original test features (Target domain)
        results["CORAL"] = f1_score(y_test_enc, y_pred, average="macro")
    except Exception as e:
        print(f"CORAL failed: {e}")
        results["CORAL"] = 0.0

    # ------------------------------------------------
    # 3. Mixup (Data Augmentation)
    # ------------------------------------------------
    try:
        X_mix, y_mix = mixup_augmentation(X_train, y_train_enc, alpha=0.2)
        model_mix = clone(best_model_proto)
        model_mix.fit(X_mix, y_mix)
        y_pred = model_mix.predict(X_test_known)
        results["Mixup"] = f1_score(y_test_enc, y_pred, average="macro")
    except Exception as e:
        print(f"Mixup failed: {e}")
        results["Mixup"] = 0.0

    return results


def run_experiment(stem: str) -> dict:
    df_train, df_test = _load_split(stem)

    best_model_name, best_model_proto = _get_fixed_best_model_proto(stem)

    _, y_train_series, _ = _split_features_labels(df_train)
    le = LabelEncoder()
    le.fit(y_train_series.values)

    strategies_res = _final_train_and_eval_strategies(
        experiment_name=stem,
        df_train=df_train,
        df_test=df_test,
        best_model_proto=best_model_proto,
        label_encoder=le,
    )

    result = {
        "experiment": stem,
        "best_model_name": best_model_name,
        "results": strategies_res,
        "n_test": len(df_test)
    }
    return result


def main():
    summary_rows = []

    for exp in EXPERIMENTS:
        stem = exp["stem"]
        print(f"Running {stem}...")
        try:
            res = run_experiment(stem)
            
            # Print row
            r_dict = res["results"]
            print(f"  [{stem}] Baseline: {r_dict.get('Baseline',0):.4f}, "
                  f"IW: {r_dict.get('ImpWeight',0):.4f}, "
                  f"CORAL: {r_dict.get('CORAL',0):.4f}, "
                  f"Mixup: {r_dict.get('Mixup',0):.4f}")
            
            row = {
                "experiment": stem,
                "model": res["best_model_name"],
                "Baseline_F1": r_dict.get('Baseline',0),
                "ImpWeight_F1": r_dict.get('ImpWeight',0),
                "CORAL_F1": r_dict.get('CORAL',0),
                "Mixup_F1": r_dict.get('Mixup',0),
            }
            summary_rows.append(row)
            
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")

    # Save summary
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        out_csv = "experiment_results_tf_summary.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved summary to {out_csv}")
        print(df)

if __name__ == "__main__":
    main()
