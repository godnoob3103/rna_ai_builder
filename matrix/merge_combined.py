import pandas as pd
import os

OLD_MATRIX  = "/mnt/a/data sci/data/counts/exon_matrix.csv"
OLD_LABELS  = "/mnt/a/data sci/data/counts/labels.csv"
NEW_COUNTS  = "/mnt/a/data/lung_cancer/counts"
OUT_MATRIX  = "/mnt/a/data sci/data/counts/combined_exon_matrix.csv"
OUT_LABELS  = "/mnt/a/data sci/data/counts/combined_labels.csv"

# Tumor / Normal sample IDs (GSE229705)
tumor_samples = [
    "SRR24166342","SRR24166340","SRR24166338","SRR24166336","SRR24166334",
    "SRR24166332","SRR24166330","SRR24166328","SRR24166278","SRR24166276",
    "SRR24166274","SRR24166272","SRR24166270","SRR24166268","SRR24166266",
    "SRR24166264","SRR24166208","SRR24166206","SRR24166204","SRR24166202",
    "SRR24166200","SRR24166198","SRR24166194","SRR24166262","SRR24166260",
    "SRR24166258","SRR24166256","SRR24166254","SRR24166252","SRR24166250",
    "SRR24166248","SRR24166144","SRR24166142","SRR24166140","SRR24166138",
    "SRR24166136","SRR24166134","SRR24166132","SRR24166130","SRR24166326",
    "SRR24166324","SRR24166322","SRR24166320","SRR24166318","SRR24166316",
    "SRR24166314","SRR24166312","SRR24166192","SRR24166190","SRR24166188",
]

normal_samples = [
    "SRR24166343","SRR24166341","SRR24166339","SRR24166337","SRR24166335",
    "SRR24166333","SRR24166331","SRR24166329","SRR24166279","SRR24166277",
    "SRR24166275","SRR24166273","SRR24166271","SRR24166269","SRR24166267",
    "SRR24166265","SRR24166209","SRR24166207","SRR24166205","SRR24166203",
    "SRR24166201","SRR24166199","SRR24166195","SRR24166263","SRR24166261",
    "SRR24166259","SRR24166257","SRR24166255","SRR24166253","SRR24166251",
    "SRR24166249","SRR24166145","SRR24166143","SRR24166141","SRR24166139",
    "SRR24166137","SRR24166135","SRR24166133","SRR24166131","SRR24166327",
    "SRR24166325","SRR24166323","SRR24166321","SRR24166319","SRR24166317",
    "SRR24166315","SRR24166313","SRR24166193","SRR24166191","SRR24166189",
]

def load_count_file(sample, counts_dir):
    path = os.path.join(counts_dir, f"{sample}_exon_counts.txt")
    df = pd.read_csv(path, sep='\t', comment='#')
    df = df.rename(columns={df.columns[-1]: sample})
    return df.set_index(['Geneid', 'Chr', 'Start', 'End', 'Strand', 'Length'])[[sample]]

# --- Load existing ERR matrix ---
print("Loading existing ERR matrix...")
old_matrix = pd.read_csv(OLD_MATRIX, index_col=0, header=[0, 1, 2, 3, 4, 5])
old_labels = pd.read_csv(OLD_LABELS, index_col=0)
print(f"  ERR matrix: {old_matrix.shape}  (samples x exons)")

# --- Load new SRR count files ---
print("\nLoading new SRR count files...")
new_dfs = []
new_labels_dict = {}

for s in tumor_samples:
    path = os.path.join(NEW_COUNTS, f"{s}_exon_counts.txt")
    if os.path.exists(path):
        new_dfs.append(load_count_file(s, NEW_COUNTS))
        new_labels_dict[s] = 1
        print(f"  {s} (tumor) ✓")
    else:
        print(f"  {s} (tumor) MISSING — skip")

for s in normal_samples:
    path = os.path.join(NEW_COUNTS, f"{s}_exon_counts.txt")
    if os.path.exists(path):
        new_dfs.append(load_count_file(s, NEW_COUNTS))
        new_labels_dict[s] = 0
        print(f"  {s} (normal) ✓")
    else:
        print(f"  {s} (normal) MISSING — skip")

if not new_dfs:
    print("\nNo new count files found. Run lung_cancer/run_pipeline.sh first.")
    exit(1)

# Build new SRR matrix (samples x exons), matching column order of old matrix
print("\nBuilding new SRR matrix...")
new_combined = pd.concat(new_dfs, axis=1)   # exons x samples
new_matrix = new_combined.T                  # samples x exons
new_matrix.index.name = old_matrix.index.name

# Normalise MultiIndex column types to strings so old (CSV-read) and new match
new_matrix.columns = pd.MultiIndex.from_tuples(
    [tuple(str(v) for v in col) for col in new_matrix.columns],
    names=new_matrix.columns.names
)

# Align columns: keep only exons present in both
shared_exons = old_matrix.columns.intersection(new_matrix.columns)
print(f"  Shared exons: {len(shared_exons):,} / {len(old_matrix.columns):,} (old) / {len(new_matrix.columns):,} (new)")

old_aligned  = old_matrix[shared_exons]
new_aligned  = new_matrix[shared_exons]

# --- Combine ---
print("\nCombining...")
combined_matrix = pd.concat([old_aligned, new_aligned], axis=0)

new_label_df = pd.DataFrame.from_dict(new_labels_dict, orient='index', columns=['label'])
combined_labels = pd.concat([old_labels, new_label_df], axis=0)

# --- Save ---
combined_matrix.to_csv(OUT_MATRIX)
combined_labels.to_csv(OUT_LABELS)

n_cancer = (combined_labels['label'] == 1).sum()
n_normal = (combined_labels['label'] == 0).sum()

print(f"\nCombined matrix: {combined_matrix.shape}  (samples x exons)")
print(f"  Cancer/Tumor : {n_cancer}")
print(f"  Normal       : {n_normal}")
print(f"\nSaved:")
print(f"  {OUT_MATRIX}")
print(f"  {OUT_LABELS}")
