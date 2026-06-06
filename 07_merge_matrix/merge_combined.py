"""
Merge all featureCounts exon-level count files into combined_exon_matrix.csv.

Sources:
  ERR samples: A:/data sci/data/counts/raw_counts/ERR*_exon_counts.txt   (20 samples)
  SRR samples: A:/data/lung_cancer/counts/SRR*_exon_counts.txt            (83 samples)

Output:
  matrices/combined_exon_matrix.csv  — rows = samples, cols = exons (2,164,410)
  Row order: 6 metadata rows (Geneid/Chr/Start/End/Strand/Length), then one row per sample.
"""

from pathlib import Path

COUNTS_DIR = Path("A:/data sci/data/counts")
ERR_DIR    = COUNTS_DIR / "raw_counts"
SRR_DIR    = Path("A:/data/lung_cancer/counts")
OUTPUT     = COUNTS_DIR / "matrices" / "combined_exon_matrix.csv"


def iter_count_file(filepath: Path):
    """Yield (meta_6_list, count_str) for each exon, skipping comment/header lines."""
    with open(filepath, "r") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("Geneid"):
                continue
            parts = line.rstrip("\n").split("\t")
            yield parts[:6], parts[6]


def sample_name(filepath: Path) -> str:
    return filepath.name.replace("_exon_counts.txt", "")


def main():
    err_files = sorted(ERR_DIR.glob("ERR*_exon_counts.txt"))
    srr_files = sorted(SRR_DIR.glob("SRR*_exon_counts.txt"))
    all_files = err_files + srr_files

    print(f"ERR files : {len(err_files)}")
    print(f"SRR files : {len(srr_files)}")
    print(f"Total     : {len(all_files)}")

    # --- read metadata (Geneid/Chr/Start/End/Strand/Length) from first file ---
    print("\nReading metadata from first file...")
    meta: list[list[str]] = [[] for _ in range(6)]
    first_counts: list[str] = []

    for m, c in iter_count_file(all_files[0]):
        for i, v in enumerate(m):
            meta[i].append(v)
        first_counts.append(c)

    n_exons = len(first_counts)
    print(f"Exons per sample: {n_exons:,}")

    # --- write combined matrix ---
    print(f"\nWriting {OUTPUT} ...")
    header_names = ["Geneid", "Chr", "Start", "End", "Strand", "Length"]

    with open(OUTPUT, "w", newline="", buffering=8 * 1024 * 1024) as out:
        # 6 metadata rows
        for name, vals in zip(header_names, meta):
            out.write(name + "," + ",".join(vals) + "\n")

        # first sample
        out.write(sample_name(all_files[0]) + "," + ",".join(first_counts) + "\n")
        print(f"  [ 1/{len(all_files)}] {sample_name(all_files[0])}")

        # remaining samples
        for idx, filepath in enumerate(all_files[1:], 2):
            sname = sample_name(filepath)
            counts = [c for _, c in iter_count_file(filepath)]
            if len(counts) != n_exons:
                print(f"  WARNING: {sname} has {len(counts)} exons, expected {n_exons}. Skipping.")
                continue
            out.write(sname + "," + ",".join(counts) + "\n")
            print(f"  [{idx:3d}/{len(all_files)}] {sname}")

    print(f"\nDone. Wrote {len(all_files)} samples x {n_exons:,} exons -> {OUTPUT}")


if __name__ == "__main__":
    main()
