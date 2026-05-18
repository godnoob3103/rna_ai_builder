import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pycombat import Combat

COUNTS_DIR   = Path("A:/data sci/data/counts")
MATRIX_FILE  = COUNTS_DIR / "combined_exon_matrix.csv"
LABELS_FILE  = COUNTS_DIR / "combined_labels.csv"
OUT_MATRIX   = COUNTS_DIR / "combined_matrix_combat.csv"
OUT_LABELS   = COUNTS_DIR / "combined_labels_103.csv"
OUT_PCA      = COUNTS_DIR / "pca_batch_effect.png"

MIN_SAMPLE_FRAC = 0.10
TOP_N_EXONS     = 5000


# ── helpers ──────────────────────────────────────────────────────────────────

def pca_plot(ax, X, batch, labels, title):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(StandardScaler().fit_transform(X))
    colors = {0: "steelblue", 1: "darkorange"}
    batch_names = {0: "ERR (batch 0)", 1: "SRR (batch 1)"}
    for b in [0, 1]:
        mask = batch == b
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[b], label=batch_names[b],
                   alpha=0.75, edgecolors="none", s=50)
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(fontsize=9)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Load matrix (skip 6 metadata rows) ──────────────────────────────────
    print("Loading combined_exon_matrix.csv ...")
    df = pd.read_csv(MATRIX_FILE, index_col=0, skiprows=range(1, 7))
    print(f"  Loaded: {df.shape[0]} samples × {df.shape[1]:,} exons")

    labels = pd.read_csv(LABELS_FILE, index_col=0)
    labels = labels.loc[df.index]          # align to matrix row order

    batch = np.array([0 if s.startswith("ERR") else 1 for s in df.index])
    print(f"  ERR (batch 0): {(batch==0).sum()}  |  SRR (batch 1): {(batch==1).sum()}")

    # 2. Expression filter ────────────────────────────────────────────────────
    n_samples    = df.shape[0]
    min_samples  = max(1, int(np.ceil(n_samples * MIN_SAMPLE_FRAC)))
    keep         = (df > 0).sum(axis=0) >= min_samples
    df           = df.loc[:, keep]
    print(f"\nAfter expression filter (>0 in >={MIN_SAMPLE_FRAC*100:.0f}% of {n_samples} samples):")
    print(f"  {df.shape[1]:,} exons kept")

    # 3. log1p normalization ──────────────────────────────────────────────────
    df = np.log1p(df)
    print(f"\nAfter log1p normalization: {df.shape[1]:,} exons")

    # 4. Top-5000 by variance ─────────────────────────────────────────────────
    top_exons = df.var(axis=0).nlargest(TOP_N_EXONS).index
    df        = df[top_exons]
    print(f"\nAfter top-{TOP_N_EXONS} variance filter: {df.shape[1]:,} exons")

    X      = df.values.astype(np.float64)
    sample_names = df.index.tolist()
    exon_names   = df.columns.tolist()

    # 5. ComBat batch correction ──────────────────────────────────────────────
    print("\nRunning ComBat batch correction ...")
    combat   = Combat()
    X_combat = combat.fit_transform(X, batch)
    print("  Done.")

    # 6. Save outputs ─────────────────────────────────────────────────────────
    df_combat = pd.DataFrame(X_combat, index=sample_names, columns=exon_names)
    df_combat.to_csv(OUT_MATRIX)
    print(f"\nSaved: {OUT_MATRIX}")
    print(f"  {df_combat.shape[0]} samples × {df_combat.shape[1]:,} exons")

    labels.to_csv(OUT_LABELS)
    print(f"Saved: {OUT_LABELS}")
    dist = labels["label"].value_counts().to_dict()
    print(f"  {len(labels)} samples  |  label distribution: {dist}")

    # 7. PCA before / after ComBat ────────────────────────────────────────────
    print("\nPlotting PCA before/after ComBat ...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pca_plot(axes[0], X,        batch, labels, "Before ComBat")
    pca_plot(axes[1], X_combat, batch, labels, "After ComBat")
    plt.suptitle("PCA — Batch Effect (ERR vs SRR)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_PCA, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_PCA}")

    print("\nDone.")


if __name__ == "__main__":
    main()
