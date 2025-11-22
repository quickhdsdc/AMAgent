import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ---------------------------
# Embedded results (Table 8: SOTA comparison)
# ---------------------------
table8_rows = [
    # experiment, method, f1, acc
    ("Exp_ID_1",  "AM-Agent",      0.842, 0.841),
    ("Exp_OOD_1", "AM-Agent",      0.561, 0.760),
    ("Exp_ID_2",  "AM-Agent",      0.765, 0.852),
    ("Exp_OOD_2", "AM-Agent",      0.459, 0.496),
    ("Exp_ID_3",  "AM-Agent",      0.836, 0.870),
    ("Exp_OOD_3", "AM-Agent",      0.384, 0.497),
    ("Exp_ID_4",  "AM-Agent",      0.868, 0.866),
    ("Exp_OOD_4", "AM-Agent",      0.641, 0.814),

    ("Exp_ID_1",  "Data-driven",   0.883, 0.889),
    ("Exp_OOD_1", "Data-driven",   0.383, 0.563),
    ("Exp_ID_2",  "Data-driven",   0.799, 0.884),
    ("Exp_OOD_2", "Data-driven",   0.251, 0.265),
    ("Exp_ID_3",  "Data-driven",   0.878, 0.889),
    ("Exp_OOD_3", "Data-driven",   0.193, 0.298),
    ("Exp_ID_4",  "Data-driven",   0.887, 0.884),
    ("Exp_OOD_4", "Data-driven",   0.434, 0.667),

    ("Exp_ID_1",  "Llama-CL-FT",   0.186, 0.234),
    ("Exp_OOD_1", "Llama-CL-FT",   0.179, 0.560),
    ("Exp_ID_2",  "Llama-CL-FT",   0.125, 0.244),
    ("Exp_OOD_2", "Llama-CL-FT",   0.161, 0.249),
    ("Exp_ID_3",  "Llama-CL-FT",   0.236, 0.285),
    ("Exp_OOD_3", "Llama-CL-FT",   0.195, 0.397),
    ("Exp_ID_4",  "Llama-CL-FT",   0.220, 0.338),
    ("Exp_OOD_4", "Llama-CL-FT",   0.132, 0.173),

    ("Exp_ID_1",  "Llama-SC-FT",   0.730, 0.736),
    ("Exp_OOD_1", "Llama-SC-FT",   0.439, 0.623),
    ("Exp_ID_2",  "Llama-SC-FT",   0.615, 0.772),
    ("Exp_OOD_2", "Llama-SC-FT",   0.346, 0.368),
    ("Exp_ID_3",  "Llama-SC-FT",   0.715, 0.773),
    ("Exp_OOD_3", "Llama-SC-FT",   0.235, 0.399),
    ("Exp_ID_4",  "Llama-SC-FT",   0.562, 0.573),
    ("Exp_OOD_4", "Llama-SC-FT",   0.394, 0.660),
]
df8 = pd.DataFrame(table8_rows, columns=["experiment","method","f1","acc"])
df8["split"] = np.where(df8["experiment"].str.contains("OOD"), "OOD", "ID")

# ---------------------------
# Embedded results (Table 9: Ablation)
# ---------------------------
table9_rows = [
    ("Exp_ID_1",  "AM-Agent", 0.842, 0.841),
    ("Exp_OOD_1", "AM-Agent", 0.561, 0.760),
    ("Exp_ID_2",  "AM-Agent", 0.765, 0.852),
    ("Exp_OOD_2", "AM-Agent", 0.459, 0.496),
    ("Exp_ID_3",  "AM-Agent", 0.836, 0.870),
    ("Exp_OOD_3", "AM-Agent", 0.384, 0.497),
    ("Exp_ID_4",  "AM-Agent", 0.868, 0.866),
    ("Exp_OOD_4", "AM-Agent", 0.641, 0.814),

    ("Exp_ID_1",  "GPT5-ZS",  0.461, 0.533),
    ("Exp_OOD_1", "GPT5-ZS",  0.499, 0.656),
    ("Exp_ID_2",  "GPT5-ZS",  0.430, 0.498),
    ("Exp_OOD_2", "GPT5-ZS",  0.447, 0.467),
    ("Exp_ID_3",  "GPT5-ZS",  0.510, 0.627),
    ("Exp_OOD_3", "GPT5-ZS",  0.393, 0.507),
    ("Exp_ID_4",  "GPT5-ZS",  0.531, 0.524),
    ("Exp_OOD_4", "GPT5-ZS",  0.483, 0.691),

    ("Exp_ID_1",  "GPT5-RAG", 0.497, 0.586),
    ("Exp_OOD_1", "GPT5-RAG", 0.525, 0.698),
    ("Exp_ID_2",  "GPT5-RAG", 0.513, 0.605),
    ("Exp_OOD_2", "GPT5-RAG", 0.428, 0.449),
    ("Exp_ID_3",  "GPT5-RAG", 0.598, 0.652),
    ("Exp_OOD_3", "GPT5-RAG", 0.427, 0.561),
    ("Exp_ID_4",  "GPT5-RAG", 0.519, 0.513),
    ("Exp_OOD_4", "GPT5-RAG", 0.462, 0.658),
]
df9 = pd.DataFrame(table9_rows, columns=["experiment","method","f1","acc"])
df9["split"] = np.where(df9["experiment"].str.contains("OOD"), "OOD", "ID")

# ---------------------------
# Helpers
# ---------------------------
def by_split(df, metric):
    return (df.groupby(["method","split"])[metric]
              .agg(mean="mean", std="std", n="count")
              .reset_index()
              .pivot(index="method", columns="split", values=["mean","std","n"])
              .sort_values(("mean","ID"), ascending=False)
              .round(3))

def generalization_gap(df, metric="f1"):
    g = (df.pivot_table(index=["method","split"], values=metric, aggfunc="mean")
           .reset_index()
           .pivot(index="method", columns="split", values=metric))
    g["gap_ID_minus_OOD"] = g["ID"] - g["OOD"]
    return g.sort_values("gap_ID_minus_OOD", ascending=False).round(3)

def cohens_d(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx<2 or ny<2: return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2)) if (nx+ny-2) > 0 else 0.0
    return (x.mean() - y.mean()) / sp if sp > 0 else np.nan

def cliffs_delta(x, y):
    """
    Cliff's delta (Δ): probability that a value from x is greater than a value from y
    minus the probability that it is smaller.
    Returns Δ in [-1, 1]. Positive means x > y overall.
    """
    x = list(x); y = list(y)
    n = len(x) * len(y)
    if n == 0:
        return np.nan
    greater = sum(1 for a in x for b in y if a > b)
    less    = sum(1 for a in x for b in y if a < b)  # <-- 'in y' was missing
    return (greater - less) / n

# optional helper for interpretation (Romano et al. thresholds)
def cliffs_delta_label(delta):
    if np.isnan(delta):
        return "NA"
    ad = abs(delta)
    if ad < 0.147: return "negligible"
    if ad < 0.33:  return "small"
    if ad < 0.474: return "medium"
    return "large"

def paired_tests(df, baseline="AM-Agent", metric="f1", methods=None):
    piv = df.pivot_table(index="experiment", columns="method", values=metric, aggfunc="first")
    comps = [m for m in piv.columns if m != baseline]
    if methods: comps = [m for m in comps if m in methods]
    rows = []
    for m in comps:
        x = piv[baseline].dropna()
        y = piv[m].reindex_like(x).dropna()
        idx = x.index.intersection(y.index)
        x, y = x.loc[idx], y.loc[idx]
        t_p = stats.ttest_rel(x, y, alternative="greater").pvalue if len(x)>=2 else np.nan
        w_p = stats.wilcoxon(x, y, alternative="greater").pvalue if len(x)>=5 else np.nan
        rows.append({
            "compare": f"{baseline} vs {m}",
            "n_pairs": len(x),
            "mean_diff": float((x - y).mean()),
            "t_p(one-sided)": t_p,
            "wilcoxon_p(one-sided)": w_p,
            "cohens_d": cohens_d(x, y),
            "cliffs_delta": cliffs_delta(x, y),
        })
    res = pd.DataFrame(rows)
    # Holm–Bonferroni adjust
    for col in ["t_p(one-sided)", "wilcoxon_p(one-sided)"]:
        p = res[col].fillna(1.0).values
        order = np.argsort(p); m = len(p); adj = np.empty_like(p, dtype=float)
        for rank, idx in enumerate(order, start=1):
            adj[idx] = min(1.0, (m - rank + 1) * p[idx])
        res[col + " (Holm)"] = adj
    return res.sort_values("mean_diff", ascending=False).round(4)

# ---------------------------
# Outputs you asked for
# ---------------------------

print("\n=== Table 8: macro-F1 by split (ID/OOD) ===")
print(by_split(df8, "f1"))

print("\n=== Table 8: Accuracy by split (ID/OOD) ===")
print(by_split(df8, "acc"))

print("\n=== Table 8: generalization gap (ID − OOD) on F1 ===")
print(generalization_gap(df8, "f1"))

print("\n=== Table 8: Paired tests vs AM-Agent (macro-F1, pooled ID+OOD) ===")
print(paired_tests(df8, baseline="AM-Agent", metric="f1"))

print("\n=== Table 9: macro-F1 by split (ID/OOD) ===")
print(by_split(df9, "f1"))

print("\n=== Table 9: Accuracy by split (ID/OOD) ===")
print(by_split(df9, "acc"))

print("\n=== Table 9: generalization gap (ID − OOD) on F1 ===")
print(generalization_gap(df9, "f1"))

print("\n=== Table 9: Paired tests vs AM-Agent (macro-F1) ===")
print(paired_tests(df9, baseline="AM-Agent", metric="f1",
                   methods=["GPT5-ZS","GPT5-RAG"]))

# ---------------------------
# Histogram (Ablation, macro-F1)
# ---------------------------
methods = ["AM-Agent", "GPT5-ZS", "GPT5-RAG"]
data = [df9[df9["method"] == m]["f1"].values for m in methods]
#
# plt.figure(figsize=(8,5))
# bins = np.linspace(0.35, 0.90, 15)
# for arr, label in zip(data, methods):
#     plt.hist(arr, bins=bins, alpha=0.5, label=label, edgecolor='black')
# plt.xlabel("Macro-F1")
# plt.ylabel("Count (experiments)")
# plt.title("Ablation (Table 9): Distribution of Macro-F1 by Method")
# plt.legend()
# plt.tight_layout()
# plt.savefig("./results_AM/ablation_histogram_f1.png", dpi=200)

import matplotlib.pyplot as plt
import numpy as np

# Methods and their average OOD macro-F1 scores
methods = ['AM-Agent-ID', 'AM-Agent-OOD', 'GPT5-ZS', 'GPT5-RAG', 'Data-driven']
f1_scores = [0.876, 0.437, 0.456, 0.465, 0.311]

# Wider figure, larger fonts
fig, ax = plt.subplots(figsize=(12, 5))

bar_width = 0.5
index = np.arange(len(methods))

colors = ['#f7afb9', '#faaa89', '#ffdc7e', '#7ed0f8', '#8dc0b3']

bars = ax.bar(index, f1_scores, bar_width, alpha=0.9, color=colors)

ax.set_ylabel('Average OOD macro-F1', fontsize=16)
ax.set_xticks(index)
ax.set_xticklabels(methods, fontsize=14, rotation=15)

ax.set_ylim([0, 1.0])

# Thicker axes and larger tick labels
ax.tick_params(axis='y', labelsize=14)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# Value labels
for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.015,
        f"{yval:.3f}",
        ha='center',
        va='bottom',
        fontsize=14
    )

plt.tight_layout()
plt.show()

plt.savefig("./results_AM/ablation_histogram_f1.png", dpi=300)