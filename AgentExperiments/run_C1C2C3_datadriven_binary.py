import os
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_STATE = 42
FEAT = ["Power", "Velocity", "beam D", "layer thickness", "Hatch spacing"]
EXP = {"AlSi10Mg": 5, "18Ni300": 6, "HastelloyX": 7}
SPLIT = os.path.join(ROOT, "data_exp", "Exp_OOD_{}_{}.csv")
OUT = os.path.join(ROOT, "results_AM", "stats",
                   "datadriven_binary_novelmat.csv")


class CORALTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reg=1e-5):
        self.reg = reg
        self.whitening = None
        self.coloring = None

    def fit(self, Xs, Xt):
        sc = np.cov(Xs, rowvar=False) + self.reg * np.eye(Xs.shape[1])
        tc = np.cov(Xt, rowvar=False) + self.reg * np.eye(Xt.shape[1])
        self.whitening = linalg.fractional_matrix_power(sc, -0.5)
        self.coloring = linalg.fractional_matrix_power(tc, 0.5)
        return self

    def transform(self, X):
        if self.whitening is None:
            return X
        return np.dot(np.dot(X, self.whitening), self.coloring).real


def importance_weighting(Xs, Xt):
    clf = LogisticRegression(solver="liblinear", random_state=RANDOM_STATE)
    Xall = np.vstack([Xs, Xt])
    yd = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xt))])
    clf.fit(Xall, yd)
    p = np.clip(clf.predict_proba(Xs)[:, 1], 0.05, 0.95)
    w = p / (1 - p)
    return w / w.sum() * len(Xs)


def mixup(X, y, alpha=0.2):
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(X)
    Xm, ym = [], []
    for _ in range(n // 2):
        i, j = rng.randint(0, n), rng.randint(0, n)
        lam = rng.beta(alpha, alpha)
        Xm.append(lam * X[i] + (1 - lam) * X[j])
        ym.append(y[i] if lam >= 0.5 else y[j])
    return np.vstack([X, np.array(Xm)]), np.concatenate([y, np.array(ym)])


def _rf():
    return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                  n_jobs=-1, class_weight="balanced")


def _norm(v):
    s = str(v).strip().lower()
    if s in ("good", "0", "0.0"):
        return 0
    if s in ("defective", "1", "1.0"):
        return 1
    return -1


def run_stem(alloy):
    n = EXP[alloy]
    tr = pd.read_csv(SPLIT.format(n, "train"))
    te = pd.read_csv(SPLIT.format(n, "test"))
    Xtr = tr[FEAT].astype(float).values
    ytr = tr["defect_label"].apply(_norm).values
    m = ytr >= 0
    Xtr, ytr = Xtr[m], ytr[m]
    Xte = te[FEAT].astype(float).values
    yte = te["defect_label"].apply(_norm).values
    m = yte >= 0
    Xte, yte = Xte[m], yte[m]

    res = {"exp": f"Exp_OOD_{n}", "alloy": alloy, "n_test": int(len(yte))}
    res["C1_datadriven"] = f1_score(yte, _rf().fit(Xtr, ytr).predict(Xte),
                                    average="macro")
    Xm, ym = mixup(Xtr, ytr)
    res["C2_mixup"] = f1_score(yte, _rf().fit(Xm, ym).predict(Xte),
                               average="macro")
    w = importance_weighting(Xtr, Xte)
    res["C3_impweight"] = f1_score(
        yte, _rf().fit(Xtr, ytr, sample_weight=w).predict(Xte), average="macro")
    coral = CORALTransformer().fit(Xtr, Xte)
    res["CORAL"] = f1_score(
        yte, _rf().fit(coral.transform(Xtr), ytr).predict(coral.transform(Xte)),
        average="macro")
    return res


def main():
    rows = [run_stem(a) for a in EXP]
    df = pd.DataFrame(rows)
    mean = {"exp": "mean", "alloy": "", "n_test": int(df.n_test.sum())}
    for c in ["C1_datadriven", "C2_mixup", "C3_impweight", "CORAL"]:
        mean[c] = df[c].mean()
    df = pd.concat([df, pd.DataFrame([mean])], ignore_index=True)
    df.to_csv(OUT, index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
