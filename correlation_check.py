"""Task 35 - Basic Correlation Check (Data Analytics track)

Checks relationships between numeric variables of the Iris dataset using
Pandas, and saves a correlation table plus scatterplots.

Run:  python correlation_check.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

# 1. Load data -----------------------------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame.drop(columns="target")
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
print("Shape:", df.shape)
print(df.describe().round(2), "\n")

# 2. Correlation table (Pearson) -----------------------------------------
corr = df.corr(method="pearson").round(3)
print("Pearson correlation table:\n", corr, "\n")
corr.to_csv(OUT / "correlation_table.csv")

# Strongest pairs, ranked by absolute correlation
cols = list(corr.columns)
pairs = pd.DataFrame(
    [
        (cols[i], cols[j], corr.iloc[i, j])
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
    ],
    columns=["var_1", "var_2", "pearson_r"],
)
pairs = pairs.reindex(pairs.pearson_r.abs().sort_values(ascending=False).index)
print("Pairs ranked by |r|:\n", pairs.to_string(index=False), "\n")
pairs.to_csv(OUT / "ranked_pairs.csv", index=False)

# 3. Heatmap -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
fig.colorbar(im, label="Pearson r")
ax.set_title("Iris - Correlation Heatmap")
fig.tight_layout()
fig.savefig(OUT / "correlation_heatmap.png", dpi=150)
plt.close(fig)

# 4. Scatterplots (inspect before trusting r) ----------------------------
selected = [
    ("petal_length", "petal_width"),    # expected strong positive
    ("sepal_length", "petal_length"),   # expected strong positive
    ("sepal_length", "sepal_width"),    # expected weak / near zero
]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (x, y) in zip(axes, selected):
    ax.scatter(df[x], df[y], alpha=0.7, edgecolor="k", linewidth=0.3)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}\nr = {corr.loc[x, y]:.2f}")
fig.tight_layout()
fig.savefig(OUT / "scatterplots.png", dpi=150)
plt.close(fig)

# 5. Extra check: does the sign flip within species? ---------------------
by_species = iris.frame.copy()
by_species.columns = [*df.columns, "target"]
within = (
    by_species.groupby("target")[["sepal_length", "sepal_width"]]
    .corr().xs("sepal_width", level=1)["sepal_length"].round(3)
)
within.index = iris.target_names[within.index]
print("sepal_length vs sepal_width within each species:\n", within)
within.to_csv(OUT / "sepal_corr_by_species.csv", header=["pearson_r"])
