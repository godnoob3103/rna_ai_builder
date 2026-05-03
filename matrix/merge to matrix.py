import pandas as pd
import os
import glob

counts_dir = "/mnt/a/data sci/data/counts"

# Cancer samples
cancer = ["ERR164550","ERR164557","ERR164562","ERR164564",
          "ERR164569","ERR164570","ERR164572","ERR164574","ERR164575","ERR164578"]

# Non-cancer samples
non_cancer = ["ERR164476","ERR164479","ERR164480","ERR164484","ERR164487",
              "ERR164491","ERR164497","ERR164498","ERR164502","ERR164503","ERR164504","ERR164507"]

def load_counts(sample, counts_dir):
    path = f"{counts_dir}/{sample}_exon_counts.txt"
    df = pd.read_csv(path, sep='\t', comment='#')
    df = df[['Geneid','Chr','Start','End','Strand','Length', df.columns[-1]]]
    df = df.rename(columns={df.columns[-1]: sample})
    return df.set_index(['Geneid','Chr','Start','End','Strand','Length'])

# Load ทุก sample
print("Loading cancer samples...")
dfs = []
for s in cancer:
    path = f"{counts_dir}/{s}_exon_counts.txt"
    if os.path.exists(path):
        dfs.append(load_counts(s, counts_dir))
        print(f"  {s} ✓")

print("Loading non-cancer samples...")
for s in non_cancer:
    path = f"{counts_dir}/{s}_exon_counts.txt"
    if os.path.exists(path):
        dfs.append(load_counts(s, counts_dir))
        print(f"  {s} ✓")

# Merge
print("Merging...")
matrix = pd.concat(dfs, axis=1)

# Labels
labels = {}
for s in cancer:
    if s in matrix.columns: labels[s] = 1
for s in non_cancer:
    if s in matrix.columns: labels[s] = 0

label_df = pd.DataFrame.from_dict(labels, orient='index', columns=['label'])

# Save
matrix.T.to_csv(f"{counts_dir}/exon_matrix.csv")
label_df.to_csv(f"{counts_dir}/labels.csv")

print(f"\nMatrix shape: {matrix.T.shape}")
print(f"Cancer samples: {sum(v==1 for v in labels.values())}")
print(f"Non-cancer samples: {sum(v==0 for v in labels.values())}")
print("Saved: exon_matrix.csv and labels.csv")
EOF
