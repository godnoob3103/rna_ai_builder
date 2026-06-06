import numpy as np
import pandas as pd
from pathlib import Path

COUNTS_DIR  = Path("A:/data sci/data/counts")
MATRIX_FILE = COUNTS_DIR / "matrices" / "combined_exon_matrix.csv"
LABELS_FILE = COUNTS_DIR / "matrices" / "combined_labels.csv"
OUT_MATRIX  = COUNTS_DIR / "exp1_srr_only" / "features.csv"
OUT_LABELS  = COUNTS_DIR / "exp1_srr_only" / "labels.csv"

MIN_SAMPLE_FRAC = 0.10
TOP_N_EXONS = 5000


def main():
    # --- 1. Load matrix (skip 6 metadata rows) ---
    print("Loading combined_exon_matrix.csv ...")
    df = pd.read_csv(MATRIX_FILE, index_col=0, skiprows=range(1, 7))
    print(f"  Loaded: {df.shape[0]} samples × {df.shape[1]:,} exons")

    # --- 2. Split SRR samples ---
    srr_mask = df.index.str.startswith("SRR")
    srr_matrix = df.loc[srr_mask].copy()
    print(f"\nSRR samples : {srr_matrix.shape[0]}")
    print(f"ERR samples : {(~srr_mask).sum()}")

    # --- 3. Filter exons: count > 0 in >= 10% of SRR samples ---
    n_samples = srr_matrix.shape[0]
    min_samples = max(1, int(np.ceil(n_samples * MIN_SAMPLE_FRAC)))
    nonzero_counts = (srr_matrix > 0).sum(axis=0)
    keep_exons = nonzero_counts >= min_samples
    srr_matrix = srr_matrix.loc[:, keep_exons]
    print(f"\nAfter expression filter (>0 in >={MIN_SAMPLE_FRAC*100:.0f}% of {n_samples} samples):")
    print(f"  {keep_exons.sum():,} exons kept")

    # --- 4. Normalize with log1p ---
    srr_matrix = np.log1p(srr_matrix)
    print(f"\nAfter log1p normalization: {srr_matrix.shape[1]:,} exons")

    # --- 5. Filter top 5000 exons by variance ---
    variances = srr_matrix.var(axis=0)
    top_exons = variances.nlargest(TOP_N_EXONS).index
    srr_matrix = srr_matrix[top_exons]
    print(f"\nAfter top-{TOP_N_EXONS} variance filter: {srr_matrix.shape[1]:,} exons")

    # --- 6. Save outputs ---
    print(f"\nSaving {OUT_MATRIX} ...")
    srr_matrix.to_csv(OUT_MATRIX)
    print(f"  {srr_matrix.shape[0]} samples × {srr_matrix.shape[1]:,} exons")

    labels = pd.read_csv(LABELS_FILE, index_col=0)
    srr_labels = labels.loc[labels.index.str.startswith("SRR")]
    srr_labels.to_csv(OUT_LABELS)
    print(f"\nSaving {OUT_LABELS} ...")
    print(f"  {srr_labels.shape[0]} samples  |  label distribution: {srr_labels['label'].value_counts().to_dict()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
