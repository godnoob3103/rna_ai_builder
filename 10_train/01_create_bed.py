"""
Step 1 — Parse top 5000 exon columns from features.csv and create top5000_exons.bed.

Column names in features.csv are pandas-deduplicated ENSG IDs (e.g. ENSG00000067048.45).
Chr/Start/End are recovered from the first 5 metadata rows of combined_exon_matrix.csv,
which pandas reads with the same deduplication applied to column names.

Output: /mnt/a/data sci/data/counts/top5000_exons.bed  (BED3+name, 0-based)
"""
import pandas as pd
from pathlib import Path

COUNTS_DIR   = Path("/mnt/a/data sci/data/counts")
FULL_MATRIX  = COUNTS_DIR / "matrices/combined_exon_matrix.csv"
FEAT_MATRIX  = COUNTS_DIR / "exp2_combined_combat/features.csv"
OUT_BED      = COUNTS_DIR / "top5000_exons.bed"


def main():
    print("Reading top-5000 exon column names from features.csv ...")
    feat_cols = pd.read_csv(FEAT_MATRIX, index_col=0, nrows=0).columns
    print(f"  {len(feat_cols)} columns found")

    print("Reading metadata rows from combined_exon_matrix.csv ...")
    # rows 1-5 after the header are: Chr, Start, End, Strand, Length
    # pandas applies the same deduplication to column names as when features.csv was built
    meta = pd.read_csv(FULL_MATRIX, header=0, nrows=5, index_col=0)
    # meta.index = ['Chr', 'Start', 'End', 'Strand', 'Length']

    print("Writing BED file ...")
    written = 0
    skipped = 0
    with open(OUT_BED, "w") as bed:
        for col in feat_cols:
            if col not in meta.columns:
                skipped += 1
                continue
            chrom  = str(meta.loc["Chr",    col])
            start  = int(meta.loc["Start",  col]) - 1   # featureCounts is 1-based → BED is 0-based
            end    = int(meta.loc["End",    col])
            strand = str(meta.loc["Strand", col])
            # BED6: chrom, start, end, name, score, strand
            bed.write(f"{chrom}\t{start}\t{end}\t{col}\t0\t{strand}\n")
            written += 1

    print(f"  Written : {written}")
    print(f"  Skipped : {skipped}")
    print(f"Saved: {OUT_BED}")


if __name__ == "__main__":
    main()
