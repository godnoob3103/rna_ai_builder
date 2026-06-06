"""
Step 2 — Count 4-mer frequencies directly from trimmed FASTQ files.

For each sample:
  1. Find matching _1.fastq.gz and _2.fastq.gz across all 4 input folders
  2. Read all reads from both files (every 2nd line starting from line 2)
  3. Count all 4-mers (sliding window, skip N-containing k-mers)
  4. Normalize to frequency (sum = 1)

Output: /mnt/a/data sci/data/counts/kmer_matrix.csv  (102 samples × 256 features)
"""
import gzip
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

COUNTS_DIR  = Path("/mnt/a/data sci/data/counts")
LABELS_FILE = COUNTS_DIR / "exp2_combined_combat/labels.csv"
OUT_MATRIX  = COUNTS_DIR / "kmer_matrix.csv"

FASTQ_DIRS = [
    Path("/mnt/a/data sci/data/trimmed/cancer"),
    Path("/mnt/a/data sci/data/trimmed/non_cancer"),
    Path("/mnt/a/data/lung_cancer/trimmed/tumor"),
    Path("/mnt/a/data/lung_cancer/trimmed/normal"),
]

BASES     = "ACGT"
ALL_4MERS = ["".join(k) for k in itertools.product(BASES, repeat=4)]  # 256 k-mers
K         = 4


def find_fastq_pair(sample_id: str) -> tuple[Path, Path] | None:
    """Return (_1, _2) gz paths for sample_id, searching all FASTQ_DIRS."""
    for d in FASTQ_DIRS:
        f1 = d / f"{sample_id}_1.fastq.gz"
        f2 = d / f"{sample_id}_2.fastq.gz"
        if f1.exists() and f2.exists():
            return f1, f2
        # some datasets omit paired suffix — try bare name
        f1b = d / f"{sample_id}.fastq.gz"
        if f1b.exists():
            return f1b, f1b
    return None


def count_kmers_from_fastq(paths: tuple[Path, ...], k: int = K) -> np.ndarray:
    """Count k-mer frequencies from one or more gzipped FASTQ files."""
    counts = {km: 0 for km in ALL_4MERS}
    total  = 0

    for path in paths:
        with gzip.open(path, "rt", errors="replace") as fh:
            line_num = 0
            for line in fh:
                line_num += 1
                # FASTQ: record = 4 lines; line 2 is the sequence
                if line_num % 4 != 2:
                    continue
                seq = line.strip().upper()
                for i in range(len(seq) - k + 1):
                    kmer = seq[i:i + k]
                    if "N" in kmer:
                        continue
                    if kmer in counts:
                        counts[kmer] += 1
                        total += 1

    if total == 0:
        return np.zeros(len(ALL_4MERS), dtype=np.float32)

    return np.array([counts[km] / total for km in ALL_4MERS], dtype=np.float32)


def main():
    labels  = pd.read_csv(LABELS_FILE, index_col=0)
    samples = labels.index.tolist()
    n       = len(samples)
    print(f"Samples: {n}  |  cancer={labels['label'].sum()}  normal={(labels['label']==0).sum()}")
    print()

    rows    = {}
    missing = []

    for i, sample_id in enumerate(samples, 1):
        pair = find_fastq_pair(sample_id)

        if pair is None:
            print(f"  [{i:3d}/{n}] {sample_id} — FASTQ NOT FOUND")
            missing.append(sample_id)
            continue

        print(f"  [{i:3d}/{n}] {sample_id} ...", end=" ", flush=True)
        freq = count_kmers_from_fastq(pair)
        rows[sample_id] = freq
        print("done")

    matrix = pd.DataFrame.from_dict(rows, orient="index", columns=ALL_4MERS)
    matrix.index.name = "sample_id"
    matrix.to_csv(OUT_MATRIX)

    print(f"\nMatrix shape: {matrix.shape}")
    print(f"Saved:        {OUT_MATRIX}")
    if missing:
        print(f"Skipped ({len(missing)}): {', '.join(missing)}")


if __name__ == "__main__":
    main()
