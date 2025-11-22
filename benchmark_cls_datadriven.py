import os
import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.base import clone

# ----------------------------------------------------
# Config
# ----------------------------------------------------

EXP_DIR = "data_exp"   # where Exp_ID_1_train.csv etc. live
LABEL_COL = "defect_label"
META_COLS = ["material"]  # columns we should NOT treat as numeric features
randomState = 42

# ----------------------------------------------------
# Model zoo and fixed selection
# ----------------------------------------------------

# Kernel reused for GaussianProcessClassifier
ckernel = (
    (ConstantKernel() * RBF() + WhiteKernel())
    + (ConstantKernel() * RBF() + WhiteKernel())
)

MODEL_ZOO = {
    "RF": RandomForestClassifier(random_state=randomState),
    "GP": GaussianProcessClassifier(
        kernel=ckernel,
        multi_class="one_vs_rest",
        n_restarts_optimizer=10,
        max_iter_predict=500,
        random_state=21,
    ),
    "SVM": SVC(
        kernel="rbf",
        gamma="scale",
        C=12,
        random_state=21,
    ),
    "GB": GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=20,
        random_state=21,
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

# Experiments definition (train/test file stems)
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
    "Exp_ID_2": "GB",
    "Exp_OOD_2": "GB",
    "Exp_ID_3": "Bootstrap",
    "Exp_OOD_3": "Bootstrap",
    "Exp_ID_4": "RF",
    "Exp_OOD_4": "RF",
}

def _get_fixed_best_model_proto(stem: str):
    name = BEST_MODELS.get(stem)
    if name is None:
        raise KeyError(f"No fixed best model specified for experiment '{stem}'.")
    if name not in MODEL_ZOO:
        raise KeyError(f"Unknown model name '{name}' in BEST_MODELS for '{stem}'.")
    return name, MODEL_ZOO[name]

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------

def _load_split(stem: str):
    """
    Load train/test CSVs for one experiment stem.
    Returns (df_train, df_test).
    """
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
    """
    Given a dataframe containing features, LABEL_COL, and maybe META_COLS,
    return:
        X_df (features only),
        y_series (labels only),
        feature_cols (column order used).
    We REMOVE LABEL_COL and META_COLS from X.
    """
    if LABEL_COL not in df.columns:
        raise RuntimeError(f"Expected label col '{LABEL_COL}' not in dataframe.")

    drop_cols = [LABEL_COL] + [c for c in META_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_df = df[feature_cols].copy()
    y_series = df[LABEL_COL].copy()

    return X_df, y_series, feature_cols


def _final_train_and_eval(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    best_model_proto,
    label_encoder: LabelEncoder,
):
    """
    Retrain the chosen model on FULL train split (after label encoding),
    then evaluate on the provided test split.
    Returns dict with macro-F1, accuracy, and evaluation metadata.
    """
    # TRAIN data
    X_train_df, y_train_series, feature_cols_train = _split_features_labels(df_train)
    X_train = X_train_df.values
    y_train_enc = label_encoder.transform(y_train_series.values)

    # TEST data
    X_test_df, y_test_series, feature_cols_test = _split_features_labels(df_test)

    # Sanity check: feature columns must match exactly (order & names)
    if feature_cols_test != feature_cols_train:
        raise RuntimeError(
            f"Train/test feature column mismatch:\n"
            f"  train: {feature_cols_train}\n"
            f"  test : {feature_cols_test}"
        )

    X_test = X_test_df.values

    # Score only rows whose label is in label_encoder.classes_
    y_test = y_test_series.values
    seen_classes = set(label_encoder.classes_)
    mask_known = np.array([lbl in seen_classes for lbl in y_test], dtype=bool)

    if not np.any(mask_known):
        return {
            "macro_f1_test": float("nan"),
            "accuracy_test": float("nan"),
            "n_test_total": len(df_test),
            "n_test_scored": 0,
            "all_classes_train": list(label_encoder.classes_),
            "all_classes_test": sorted(list(pd.unique(y_test))),
        }

    y_test_known = y_test_series[mask_known].values
    y_test_enc = label_encoder.transform(y_test_known)
    X_test_known = X_test[mask_known]

    # Retrain chosen model on ALL train data
    best_model = clone(best_model_proto)
    best_model.fit(X_train, y_train_enc)

    # Predict on filtered test portion
    y_pred_enc = best_model.predict(X_test_known)
    macro_f1 = f1_score(y_test_enc, y_pred_enc, average="macro")
    acc = accuracy_score(y_test_enc, y_pred_enc)

    return {
        "macro_f1_test": float(macro_f1),
        "accuracy_test": float(acc),
        "n_test_total": len(df_test),
        "n_test_scored": int(np.sum(mask_known)),
        "all_classes_train": list(label_encoder.classes_),
        "all_classes_test": sorted(list(pd.unique(y_test))),
    }


def run_experiment(stem: str) -> dict:
    """
    For a given experiment stem (e.g. 'Exp_ID_1'):
      1. Load train/test splits.
      2. Pick the fixed best model from BEST_MODELS.
      3. Fit label encoder on train labels.
      4. Train the chosen model on full train.
      5. Evaluate on test.
    """
    df_train, df_test = _load_split(stem)

    # Choose fixed model
    best_model_name, best_model_proto = _get_fixed_best_model_proto(stem)

    # Fit label encoder on TRAIN labels only
    _, y_train_series, _ = _split_features_labels(df_train)
    le = LabelEncoder()
    le.fit(y_train_series.values)

    # Final train + test eval
    test_eval = _final_train_and_eval(
        df_train=df_train,
        df_test=df_test,
        best_model_proto=best_model_proto,
        label_encoder=le,
    )

    # Collect basic metadata
    _, _, feature_cols = _split_features_labels(df_train)

    result = {
        "experiment": stem,
        "best_model_name": best_model_name,
        "feature_cols": feature_cols,
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "test_eval": test_eval,
        "cv_report": None,  # not used in fixed-model mode
    }
    return result


def main():
    all_results = []

    for exp in EXPERIMENTS:
        stem = exp["stem"]
        print("\n=======================================")
        print(f" Running {stem}")
        print("=======================================")

        try:
            res = run_experiment(stem)
        except FileNotFoundError as e:
            print(f"[ERROR] {stem}: {e}")
            continue
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
            continue

        all_results.append(res)

        # Per-experiment summary (no CV)
        best_model = res["best_model_name"]
        macro_f1_test = res["test_eval"]["macro_f1_test"]
        accuracy_test = res["test_eval"]["accuracy_test"]
        n_test_total = res["test_eval"]["n_test_total"]
        n_test_scored = res["test_eval"]["n_test_scored"]

        print(f"Train samples: {res['train_samples']}, Test samples: {res['test_samples']}")
        print(f"Selected model (fixed): {best_model}")
        print(f"  Test macro-F1 = {macro_f1_test:.4f} | Accuracy = {accuracy_test:.4f} "
              f"(scored {n_test_scored}/{n_test_total} rows)")

    # ------------------------------------------------
    # Global summary table (CSV + stdout)
    # ------------------------------------------------
    summary_csv_path = "experiment_results_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "best_model",
            "test_macro_F1",
            "test_accuracy",
            "test_rows_scored",
            "test_rows_total",
            "train_samples",
            "test_samples",
        ])

        print("\n=== FINAL SUMMARY ===")
        print(f"{'experiment':<12} {'best_model':>12} {'Test_F1':>10} {'Acc':>8} {'scored/total':>15}")

        for res in all_results:
            best_model = res["best_model_name"]
            macro_f1_test = res["test_eval"]["macro_f1_test"]
            accuracy_test = res["test_eval"]["accuracy_test"]
            n_test_total  = res["test_eval"]["n_test_total"]
            n_test_scored = res["test_eval"]["n_test_scored"]

            print(f"{res['experiment']:<12} {best_model:>12} "
                  f"{macro_f1_test:>10.4f} {accuracy_test:>8.4f} {n_test_scored:>5d}/{n_test_total:<5d}")

            writer.writerow([
                res["experiment"],
                best_model,
                f"{macro_f1_test:.6f}",
                f"{accuracy_test:.6f}",
                n_test_scored,
                n_test_total,
                res["train_samples"],
                res["test_samples"],
            ])

    print(f"\nWrote experiment summary: {summary_csv_path}")


if __name__ == "__main__":
    main()



