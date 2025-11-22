import os
import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.base import clone

# ----------------------------------------------------
# Config
# ----------------------------------------------------

EXP_DIR = "data_exp"   # where Exp_ID_1_train.csv etc. live
LABEL_COL = "defect_label"
META_COLS = ["material"]  # columns we should NOT treat as numeric features
randomState = 42

# Kernel reused from before (GaussianProcessClassifier)
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
# Each dict will produce:
#   <stem>_train.csv and <stem>_test.csv from EXP_DIR
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


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------

def _safe_num_splits(y_encoded: np.ndarray, max_splits: int = 5) -> int:
    """
    Pick a safe StratifiedKFold n_splits based on the smallest class.
    Ensures every fold sees at least 1 sample of each class.
    """
    _, counts = np.unique(y_encoded, return_counts=True)
    min_class_count = counts.min()
    n_splits = min(max_splits, int(min_class_count))
    if n_splits < 2:
        n_splits = 2
    return n_splits


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
        y_series (labels only).
    We REMOVE LABEL_COL and META_COLS from X.
    """
    if LABEL_COL not in df.columns:
        raise RuntimeError(f"Expected label col '{LABEL_COL}' not in dataframe.")

    drop_cols = [LABEL_COL] + [c for c in META_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_df = df[feature_cols].copy()
    y_series = df[LABEL_COL].copy()

    return X_df, y_series, feature_cols


def _cv_score_model(model, X, y, n_splits: int, random_state: int) -> list[float]:
    """
    Stratified K-fold CV for one model.
    Returns a list of macro-F1 scores across folds.
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        est = clone(model)
        try:
            est.fit(X_tr, y_tr)
            y_pred = est.predict(X_val)
            f1_macro = f1_score(y_val, y_pred, average="macro")
            fold_scores.append(float(f1_macro))
        except Exception as e:
            print(f"[WARN] CV fold failed for {type(model).__name__}: {e}")
            fold_scores.append(-1.0)

    return fold_scores


def _select_best_model(df_train: pd.DataFrame):
    """
    1. Extract X_train, y_train.
    2. Encode labels.
    3. Cross-validate each model in MODEL_ZOO.
    4. Return (best_model_name, trained_best_model, label_encoder, cv_report).

    trained_best_model here is NOT yet refit; we'll refit after selection
    on the entire training set.
    """
    # Split out features + label
    X_train_df, y_train_series, feature_cols = _split_features_labels(df_train)

    # Encode labels from TRAIN ONLY
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_series.values)

    # To numpy
    X_train = X_train_df.values
    y_train = y_train_enc

    # Pick stratified CV folds safely
    n_splits = _safe_num_splits(y_train, max_splits=5)

    # Evaluate each model in MODEL_ZOO
    per_model_stats = {}
    for model_name, model in MODEL_ZOO.items():
        scores = _cv_score_model(
            model=model,
            X=X_train,
            y=y_train,
            n_splits=n_splits,
            random_state=randomState,
        )

        scores_arr = np.array(scores, dtype=float)
        mean_f1 = float(np.mean(scores_arr))
        std_f1 = float(np.std(scores_arr))

        per_model_stats[model_name] = {
            "fold_scores": scores,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
            "n_splits": n_splits,
            "n_train_samples": len(df_train),
            "n_features": len(feature_cols),
        }

    # Pick best by highest mean macro-F1
    best_model_name = max(
        per_model_stats,
        key=lambda m: per_model_stats[m]["mean_macro_f1"]
    )
    best_model_proto = MODEL_ZOO[best_model_name]

    return best_model_name, best_model_proto, le, per_model_stats, feature_cols


def _final_train_and_eval(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    best_model_proto,
    label_encoder: LabelEncoder,
):
    """
    Retrain best model on FULL train split (after label encoding),
    then evaluate on the provided test split.
    Returns dict with macro-F1 and predictions.
    """
    # TRAIN data
    X_train_df, y_train_series, feature_cols_train = _split_features_labels(df_train)
    X_train = X_train_df.values
    y_train_enc = label_encoder.transform(y_train_series.values)

    # TEST data
    X_test_df, y_test_series, feature_cols_test = _split_features_labels(df_test)

    # sanity check: feature columns must match exactly
    if feature_cols_test != feature_cols_train:
        raise RuntimeError(
            f"Train/test feature column mismatch:\n"
            f"  train: {feature_cols_train}\n"
            f"  test : {feature_cols_test}"
        )

    X_test = X_test_df.values
    # Note: test labels may include classes unseen in train.
    # We'll handle that: any unseen class can't be encoded by LabelEncoder.
    # We'll map unseen to -1 and ignore them in F1 calc (macro-F1 with unknown label would break).
    # So we only score rows whose label is in label_encoder.classes_.
    y_test = y_test_series.values
    seen_classes = set(label_encoder.classes_)

    mask_known = [lbl in seen_classes for lbl in y_test]
    mask_known = np.array(mask_known, dtype=bool)

    if not np.any(mask_known):
        # Edge case: test is entirely unseen classes
        # We'll just report NaN macro-F1.
        return {
            "macro_f1_test": float("nan"),
            "n_test_total": len(df_test),
            "n_test_scored": 0,
            "all_classes_train": list(label_encoder.classes_),
            "all_classes_test": sorted(list(pd.unique(y_test))),
        }

    y_test_known = y_test_series[mask_known].values
    y_test_enc = label_encoder.transform(y_test_known)

    X_test_known = X_test[mask_known]

    # Retrain best model on ALL train data
    best_model = clone(best_model_proto)
    best_model.fit(X_train, y_train_enc)

    # Predict on filtered test portion
    y_pred_enc = best_model.predict(X_test_known)

    macro_f1 = f1_score(y_test_enc, y_pred_enc, average="macro")

    return {
        "macro_f1_test": float(macro_f1),
        "n_test_total": len(df_test),
        "n_test_scored": int(np.sum(mask_known)),
        "all_classes_train": list(label_encoder.classes_),
        "all_classes_test": sorted(list(pd.unique(y_test))),
    }


def run_experiment(stem: str) -> dict:
    """
    For a given experiment stem (e.g. 'Exp_ID_1'):
      1. Load train/test splits.
      2. Do CV model selection on train.
      3. Retrain best model on full train.
      4. Evaluate on test.
    Returns dict with all metadata/results.
    """
    df_train, df_test = _load_split(stem)

    # 1-2. Model selection via CV on train
    (
        best_model_name,
        best_model_proto,
        label_encoder,
        cv_report,
        feature_cols,
    ) = _select_best_model(df_train)

    # 3-4. Final train on full train and evaluate on test
    test_eval = _final_train_and_eval(
        df_train=df_train,
        df_test=df_test,
        best_model_proto=best_model_proto,
        label_encoder=label_encoder,
    )

    result = {
        "experiment": stem,
        "best_model_name": best_model_name,
        "cv_report": cv_report,
        "feature_cols": feature_cols,
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "test_eval": test_eval,
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

        # Pretty print per-experiment summary
        best_model = res["best_model_name"]
        cv_best_mean = res["cv_report"][best_model]["mean_macro_f1"]
        cv_best_std = res["cv_report"][best_model]["std_macro_f1"]
        folds = res["cv_report"][best_model]["n_splits"]

        macro_f1_test = res["test_eval"]["macro_f1_test"]
        n_test_total = res["test_eval"]["n_test_total"]
        n_test_scored = res["test_eval"]["n_test_scored"]

        print(f"Train samples: {res['train_samples']}, Test samples: {res['test_samples']}")
        print(f"Selected model via CV: {best_model}")
        print(f"  CV mean macro-F1 = {cv_best_mean:.4f} ± {cv_best_std:.4f} ({folds}-fold)")
        print(f"  Test macro-F1    = {macro_f1_test:.4f} "
              f"(scored {n_test_scored}/{n_test_total} test rows)")

        # Optional: print other models' CV scores for context
        print("\n  CV scores by model (mean_macro_F1):")
        for mname, stats in res["cv_report"].items():
            print(f"    {mname:<12} {stats['mean_macro_f1']:.4f}")

    # ------------------------------------------------
    # Global summary table (CSV + stdout)
    # ------------------------------------------------
    summary_csv_path = "experiment_results_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "best_model",
            "cv_mean_macro_F1",
            "cv_std_macro_F1",
            "cv_folds",
            "test_macro_F1",
            "test_rows_scored",
            "test_rows_total",
            "train_samples",
            "test_samples",
        ])

        print("\n=== FINAL SUMMARY ===")
        print(f"{'experiment':<12} {'best_model':>12} "
              f"{'CV_meanF1':>12} {'Test_F1':>10} {'scored/total':>15}")

        for res in all_results:
            best_model = res["best_model_name"]
            cv_best_mean = res["cv_report"][best_model]["mean_macro_f1"]
            cv_best_std = res["cv_report"][best_model]["std_macro_f1"]
            folds = res["cv_report"][best_model]["n_splits"]

            macro_f1_test = res["test_eval"]["macro_f1_test"]
            n_test_total  = res["test_eval"]["n_test_total"]
            n_test_scored = res["test_eval"]["n_test_scored"]

            print(f"{res['experiment']:<12} {best_model:>12} "
                  f"{cv_best_mean:>12.4f} {macro_f1_test:>10.4f} "
                  f"{n_test_scored:>5d}/{n_test_total:<5d}")

            writer.writerow([
                res["experiment"],
                best_model,
                f"{cv_best_mean:.6f}",
                f"{cv_best_std:.6f}",
                folds,
                f"{macro_f1_test:.6f}",
                n_test_scored,
                n_test_total,
                res["train_samples"],
                res["test_samples"],
            ])

    print(f"\nWrote experiment summary: {summary_csv_path}")


if __name__ == "__main__":
    main()

