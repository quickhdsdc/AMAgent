import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------
# Config (same as in your experiment script)
# ----------------------------------------------------
EXP_DIR = "data_exp"   # folder containing Exp_ID_1_train.csv
STEM = "Exp_ID_1"
LABEL_COL = "defect_label"
META_COLS = ["material"]
RANDOM_STATE = 42

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def _load_train(stem: str) -> pd.DataFrame:
    train_path = os.path.join(EXP_DIR, f"{stem}_train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train split: {train_path}")
    df_train = pd.read_csv(train_path)
    return df_train


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


def main():
    # 1) Load training data
    df_train = _load_train(STEM)

    # 2) Split into features + labels
    X_train_df, y_train_series, feature_cols = _split_features_labels(df_train)
    X_train = X_train_df.values

    # 3) Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_series.values)

    # 4) Define and train RandomForest (same as "RF" in your MODEL_ZOO)
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(X_train, y_train_enc)

    # 5) Build two inference samples
    # IMPORTANT: keys here must match your actual feature column names!
    sample_params = [
        {
            "Power": 350,   # W
            "Velocity": 800,    # mm/s
            "beam D": 100, # µm
            "layer thickness": 30,  # µm
            "Hatch spacing": 120 # µm
        },
        {
            "Power": 190,   # W
            "Velocity": 1200,   # mm/s
            "beam D": 100, # µm
            "layer thickness": 30,  # µm
            "Hatch spacing": 120 # µm
        },
        {
            "Power": 300,  # W
            "Velocity": 1100,  # mm/s
            "beam D": 100,  # µm
            "layer thickness": 30,  # µm
            "Hatch spacing": 100  # µm
        },
        {
            "Power": 200,  # W
            "Velocity": 900,  # mm/s
            "beam D": 100,  # µm
            "layer thickness": 30,  # µm
            "Hatch spacing": 100  # µm
        },
    ]

    # 6) Create a feature dataframe for these samples
    # To handle any extra features present in training, start from the mean of train features
    base_row = X_train_df.mean()
    X_new_df = pd.DataFrame([base_row.copy() for _ in sample_params])

    for i, params in enumerate(sample_params):
        for key, value in params.items():
            if key not in X_new_df.columns:
                raise KeyError(
                    f"Feature '{key}' not found in training features. "
                    f"Available features: {list(X_new_df.columns)}"
                )
            X_new_df.loc[i, key] = value

    # Ensure correct column order
    X_new_df = X_new_df[feature_cols]
    X_new = X_new_df.values

    # 7) Predict labels (and optionally probabilities)
    y_pred_enc = rf.predict(X_new)
    y_pred_labels = le.inverse_transform(y_pred_enc)

    # If RF supports predict_proba, get class probabilities
    if hasattr(rf, "predict_proba"):
        y_proba = rf.predict_proba(X_new)
        class_labels = le.inverse_transform(np.arange(y_proba.shape[1]))
    else:
        y_proba = None
        class_labels = None

    # 8) Print results
    print("Trained on Exp_ID_1 with RandomForestClassifier.")
    print(f"Feature columns used: {feature_cols}\n")

    for i, params in enumerate(sample_params):
        print(f"Sample #{i+1} params: {params}")
        print(f"  Predicted label: {y_pred_labels[i]}")
        if y_proba is not None:
            print("  Class probabilities:")
            for cls, p in zip(class_labels, y_proba[i]):
                print(f"    {cls}: {p:.4f}")
        print()

if __name__ == "__main__":
    main()

