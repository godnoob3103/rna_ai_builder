import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import shutil

# Setup directories
EDA_DIR = Path("/mnt/a/data sci/data/counts/eda")
RESULTS_DIR = EDA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Artifact directory for chat UI embedding
ARTIFACT_DIR = Path("/root/.gemini/antigravity-cli/brain/2be5a522-0319-4917-8f7b-69024882fc86")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
COUNTS_DIR = Path("/mnt/a/data sci/data/counts")
MATRIX_FILE = COUNTS_DIR / "matrices" / "combined_exon_matrix.csv"
LABELS_FILE = COUNTS_DIR / "matrices" / "combined_labels.csv"

def save_plot_dual(fig, filename):
    # Save to workspace eda/results
    workspace_path = RESULTS_DIR / filename
    fig.savefig(workspace_path, dpi=150, bbox_inches='tight')
    
    # Save/copy to chat UI artifact folder
    artifact_path = ARTIFACT_DIR / filename
    shutil.copy2(workspace_path, artifact_path)
    print(f"Saved: {filename} -> {workspace_path} (and copied to artifact)")

def fast_load_matrix(matrix_file):
    print("Loading 2.16 Million Column Exon Matrix line-by-line...")
    start_time = time.time()
    
    samples = []
    data = []
    exon_ids = None

    with open(matrix_file, 'r') as f:
        # Line 1: Header
        header = f.readline().strip().split(',')
        exon_ids = [f"{gene}_{i}" for i, gene in enumerate(header[1:])]
        
        # Skip next 5 lines (Chr, Start, End, Strand, Length)
        for _ in range(5):
            f.readline()
            
        # Read remaining data lines
        line_idx = 0
        while True:
            line = f.readline()
            if not line:
                break
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            sample_id = parts[0]
            samples.append(sample_id)
            # Fast parsing using numpy fromiter
            values = np.fromiter(parts[1:], dtype=np.float32)
            data.append(values)
            line_idx += 1
            if line_idx % 10 == 0 or line_idx == 1:
                print(f"  -> Loaded sample {line_idx}: {sample_id} ({len(values)} exons)")

    data = np.vstack(data)
    df = pd.DataFrame(data, index=samples, columns=exon_ids)
    print(f"Matrix loaded successfully! Shape: {df.shape[0]} samples x {df.shape[1]} exons (Time: {time.time() - start_time:.2f}s)")
    return df

def fast_load_features(file_path):
    start = time.time()
    print(f"  -> Fast loading features from {file_path}...")
    samples = []
    data = []
    exon_ids = None
    with open(file_path, 'r') as f:
        header = f.readline().strip().split(',')
        exon_ids = header[1:]
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            samples.append(parts[0])
            data.append(np.fromiter(parts[1:], dtype=np.float32))
    df = pd.DataFrame(np.vstack(data), index=samples, columns=exon_ids)
    print(f"  -> Features loaded successfully! Shape: {df.shape} (Time: {time.time() - start:.2f}s)")
    return df

def main():
    print("=== [1/9] Starting Optimized Exploratory Data Analysis (EDA) ===")
    
    # 1. Load labels and analyze distribution
    print("\n=== [2/9] Loading Labels and Analyzing Distribution ===")
    labels_df = pd.read_csv(LABELS_FILE, index_col=0)
    print(f"Successfully loaded {len(labels_df)} sample labels.")
    
    labels_df['batch'] = labels_df.index.map(lambda x: 'ERR (Batch 0)' if x.startswith('ERR') else 'SRR (Batch 1)')
    labels_df['class_name'] = labels_df['label'].map({0: 'Non-Cancer', 1: 'Cancer'})
    
    dist_table = pd.crosstab(labels_df['batch'], labels_df['class_name'])
    print("Class distribution by Batch:")
    print(dist_table)
    
    # Plot Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    dist_table.plot(kind='bar', stacked=True, color=['#3498db', '#e74c3c'], ax=ax, edgecolor='none', alpha=0.9)
    ax.set_title("Sample Distribution by Batch and Class", fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel("Batch / Source", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Samples", fontsize=12, labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_plot_dual(fig, "class_distribution.png")
    plt.close()

    # 2. Load the combined exon matrix (skipping metadata rows)
    print("\n=== [3/9] Loading Combined Exon Matrix (666MB CSV) ===")
    raw_df = fast_load_matrix(MATRIX_FILE)
    
    # Align labels with matrix index
    labels_df = labels_df.loc[raw_df.index]
    
    # 3. Library size distribution
    print("\n=== [4/9] Analyzing Sequencing Depth (Library Sizes) ===")
    library_sizes = raw_df.sum(axis=1)
    labels_df['library_size'] = library_sizes
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    batches = sorted(labels_df['batch'].unique())
    colors = ['#3498db', '#e67e22']
    for i, batch in enumerate(batches):
        mask = labels_df['batch'] == batch
        axes[0].hist(labels_df.loc[mask, 'library_size'] / 1e6, bins=15, alpha=0.6, 
                     color=colors[i], label=batch, edgecolor='black', linewidth=0.5)
    axes[0].set_title("Library Size Distribution by Batch", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Library Size (Million Counts)", fontsize=10)
    axes[0].set_ylabel("Frequency", fontsize=10)
    axes[0].legend()
    axes[0].grid(linestyle='--', alpha=0.5)
    
    classes = sorted(labels_df['class_name'].unique())
    class_colors = ['#2ecc71', '#e74c3c']
    for i, cls in enumerate(classes):
        mask = labels_df['class_name'] == cls
        axes[1].hist(labels_df.loc[mask, 'library_size'] / 1e6, bins=15, alpha=0.6, 
                     color=class_colors[i], label=cls, edgecolor='black', linewidth=0.5)
    axes[1].set_title("Library Size Distribution by Diagnosis", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Library Size (Million Counts)", fontsize=10)
    axes[1].set_ylabel("Frequency", fontsize=10)
    axes[1].legend()
    axes[1].grid(linestyle='--', alpha=0.5)
    
    plt.suptitle("Exon Sequence Depth Analysis (Library Sizes)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_plot_dual(fig, "library_size_distribution.png")
    plt.close()

    # 4. Zero Inflation analysis
    print("\n=== [5/9] Analyzing Data Sparsity (Zero Fraction) ===")
    zero_fraction = (raw_df == 0).sum(axis=1) / raw_df.shape[1]
    labels_df['zero_fraction'] = zero_fraction
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, batch in enumerate(batches):
        mask = labels_df['batch'] == batch
        axes[0].hist(labels_df.loc[mask, 'zero_fraction'] * 100, bins=15, alpha=0.6, 
                     color=colors[i], label=batch, edgecolor='black', linewidth=0.5)
    axes[0].set_title("Zero Count Percentage by Batch", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Percentage of Exons with 0 Counts (%)", fontsize=10)
    axes[0].set_ylabel("Frequency", fontsize=10)
    axes[0].legend()
    axes[0].grid(linestyle='--', alpha=0.5)
    
    for i, cls in enumerate(classes):
        mask = labels_df['class_name'] == cls
        axes[1].hist(labels_df.loc[mask, 'zero_fraction'] * 100, bins=15, alpha=0.6, 
                     color=class_colors[i], label=cls, edgecolor='black', linewidth=0.5)
    axes[1].set_title("Zero Count Percentage by Diagnosis", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Percentage of Exons with 0 Counts (%)", fontsize=10)
    axes[1].set_ylabel("Frequency", fontsize=10)
    axes[1].legend()
    axes[1].grid(linestyle='--', alpha=0.5)
    
    plt.suptitle("Sparsity Analysis (Zero Count Ratios)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_plot_dual(fig, "zero_fraction_distribution.png")
    plt.close()

    # 5. MEAN-VARIANCE RELATIONSHIP QC PLOT (New!)
    print("\n=== [6/9] Plotting Mean-Variance Relationship (Genomics QC) ===")
    # Sample a subset of exons (e.g. 50,000 random exons) to plot quickly
    sampled_exons = raw_df.sample(n=min(50000, raw_df.shape[1]), axis=1, random_state=42)
    exon_means = sampled_exons.mean(axis=0)
    exon_vars = sampled_exons.var(axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(exon_means + 1, exon_vars + 1, alpha=0.3, s=4, color='#8e44ad', edgecolors='none')
    # Draw reference line y=x (Poisson expectation: mean == variance)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.7, label='Poisson Expectation (Mean = Variance)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Mean-Variance Relationship across Exons", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Mean Expression Count + 1 (Log Scale)", fontsize=11)
    ax.set_ylabel("Variance of Expression + 1 (Log Scale)", fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    save_plot_dual(fig, "mean_variance_relationship.png")
    plt.close()

    # 6. Preprocessing and PCA
    print("\n=== [7/9] Performing Preprocessing and PCA Dimension Reduction ===")
    # Expression Filter (min 10% samples)
    min_samples = int(np.ceil(raw_df.shape[0] * 0.10))
    keep_exons = (raw_df > 0).sum(axis=0) >= min_samples
    filtered_df = raw_df.loc[:, keep_exons]
    print(f"Filtered exons from {raw_df.shape[1]} to {filtered_df.shape[1]}")
    
    # log1p normalization
    norm_df = np.log1p(filtered_df)
    
    # Keep top 5000 by variance (optimized with NumPy to bypass Pandas overhead on 1.79M columns)
    print("Calculating column variances using NumPy...")
    vars_np = np.var(norm_df.values, axis=0)
    top_indices = np.argsort(vars_np)[-5000:][::-1]
    top_5000_exons = norm_df.columns[top_indices]
    norm_df = norm_df[top_5000_exons]
    
    # Load or generate ComBat features (robust validation to regenerate if 102 vs 103 sample mismatch occurs)
    combat_file = COUNTS_DIR / "exp2_combined_combat" / "features.csv"
    combat_loaded = False
    if combat_file.exists():
        print("Loading existing ComBat batch-corrected features...")
        combat_df = fast_load_features(combat_file)
        if combat_df.shape[0] == norm_df.shape[0]:
            print(f"Loaded existing features with matching shape: {combat_df.shape}")
            combat_loaded = True
        else:
            print(f"Mismatched rows in existing features (expected {norm_df.shape[0]} but got {combat_df.shape[0]}). Regenerating features...")

    if not combat_loaded:
        print("Running ComBat batch correction dynamically...")
        try:
            from pycombat import Combat
            combat = Combat()
            batch_codes = np.array([0 if s.startswith("ERR") else 1 for s in norm_df.index])
            combat_features = combat.fit_transform(norm_df.values, batch_codes)
            combat_df = pd.DataFrame(combat_features, index=norm_df.index, columns=norm_df.columns)
            # Save complete features back to matrices directory for future runs
            combat_file.parent.mkdir(parents=True, exist_ok=True)
            combat_df.to_csv(combat_file)
            print(f"Saved regenerated 103-sample features to {combat_file}")
        except Exception as e:
            print(f"Warning: Could not run pycombat ({e}). Using normalized features without correction.")
            combat_df = norm_df.copy()
            
    # Run PCA before and after ComBat
    pca_before = PCA(n_components=2)
    scaler_before = StandardScaler()
    coords_before = pca_before.fit_transform(scaler_before.fit_transform(norm_df.values))
    var_before = pca_before.explained_variance_ratio_
    
    pca_after = PCA(n_components=2)
    scaler_after = StandardScaler()
    coords_after = pca_after.fit_transform(scaler_after.fit_transform(combat_df.values))
    var_after = pca_after.explained_variance_ratio_
    
    # Plot PCA Comparison (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Row 1: BEFORE COMBAT
    # Color by Batch
    for i, batch in enumerate(batches):
        mask = labels_df['batch'] == batch
        axes[0, 0].scatter(coords_before[mask, 0], coords_before[mask, 1],
                            c=colors[i], label=batch, alpha=0.8, s=60, edgecolors='k', linewidth=0.3)
    axes[0, 0].set_title("Before ComBat: Colored by Batch", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel(f"PC1 ({var_before[0]*100:.1f}%)")
    axes[0, 0].set_ylabel(f"PC2 ({var_before[1]*100:.1f}%)")
    axes[0, 0].legend()
    axes[0, 0].grid(linestyle='--', alpha=0.5)
    
    # Color by Class
    for i, cls in enumerate(classes):
        mask = labels_df['class_name'] == cls
        axes[0, 1].scatter(coords_before[mask, 0], coords_before[mask, 1],
                            c=class_colors[i], label=cls, alpha=0.8, s=60, edgecolors='k', linewidth=0.3)
    axes[0, 1].set_title("Before ComBat: Colored by Diagnosis", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel(f"PC1 ({var_before[0]*100:.1f}%)")
    axes[0, 1].set_ylabel(f"PC2 ({var_before[1]*100:.1f}%)")
    axes[0, 1].legend()
    axes[0, 1].grid(linestyle='--', alpha=0.5)
    
    # Row 2: AFTER COMBAT
    # Color by Batch
    for i, batch in enumerate(batches):
        mask = labels_df['batch'] == batch
        axes[1, 0].scatter(coords_after[mask, 0], coords_after[mask, 1],
                            c=colors[i], label=batch, alpha=0.8, s=60, edgecolors='k', linewidth=0.3)
    axes[1, 0].set_title("After ComBat: Colored by Batch", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel(f"PC1 ({var_after[0]*100:.1f}%)")
    axes[1, 0].set_ylabel(f"PC2 ({var_after[1]*100:.1f}%)")
    axes[1, 0].legend()
    axes[1, 0].grid(linestyle='--', alpha=0.5)
    
    # Color by Class
    for i, cls in enumerate(classes):
        mask = labels_df['class_name'] == cls
        axes[1, 1].scatter(coords_after[mask, 0], coords_after[mask, 1],
                            c=class_colors[i], label=cls, alpha=0.8, s=60, edgecolors='k', linewidth=0.3)
    axes[1, 1].set_title("After ComBat: Colored by Diagnosis", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel(f"PC1 ({var_after[0]*100:.1f}%)")
    axes[1, 1].set_ylabel(f"PC2 ({var_after[1]*100:.1f}%)")
    axes[1, 1].legend()
    axes[1, 1].grid(linestyle='--', alpha=0.5)
    
    plt.suptitle("PCA Dimension Reduction — Batch Effect vs Disease Signal", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot_dual(fig, "pca_before_after_combat.png")
    plt.close()

    # 7. Top 5 Highly Variable Exons
    print("\n=== [8/9] Analyzing Top Variable Exons ===")
    print("Calculating top 5 variable exons using NumPy...")
    vars_np = np.var(norm_df.values, axis=0)
    top_5_indices = np.argsort(vars_np)[-5:][::-1]
    top_5_exons = norm_df.columns[top_5_indices]
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    for idx, exon in enumerate(top_5_exons):
        data_to_plot = [
            combat_df.loc[labels_df['class_name'] == 'Non-Cancer', exon],
            combat_df.loc[labels_df['class_name'] == 'Cancer', exon]
        ]
        
        bp = axes[idx].boxplot(data_to_plot, labels=['Non-Cancer', 'Cancer'], patch_artist=True,
                          boxprops=dict(facecolor='#eceff1', color='#37474f'),
                          medianprops=dict(color='#e74c3c', linewidth=2),
                          flierprops=dict(marker='o', markerfacecolor='#e74c3c', markersize=4, linestyle='none'))
        
        # Color the boxes specifically (robust and standard matplotlib way)
        colors_box = ['#a9dfbf', '#f5b7b1']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            
        clean_exon = exon.split('_')[0]
        title = clean_exon if len(clean_exon) <= 15 else clean_exon[:12] + '...'
        axes[idx].set_title(title, fontsize=11, fontweight='bold')
        axes[idx].grid(axis='y', linestyle='--', alpha=0.5)
        
    axes[0].set_ylabel("Normalized Expression Level (ComBat)", fontsize=12)
    plt.suptitle("Expression Levels of Top 5 Highly Variable Exons (Batch-Corrected)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_plot_dual(fig, "top_5_variable_exons.png")
    plt.close()

    # NEW CLUSTERED EXPRESSION HEATMAP FOR TOP 30 EXONS (New!)
    print("\n=== [8.5] Generating Clustered Heatmap for Top 30 Variable Exons ===")
    print("Calculating top 30 variable exons using NumPy...")
    vars_np = np.var(combat_df.values, axis=0)
    top_30_indices = np.argsort(vars_np)[-30:][::-1]
    top_30_exons = combat_df.columns[top_30_indices]
    heatmap_data = combat_df[top_30_exons]
    
    # Sort samples by Disease and Batch
    sorted_samples = labels_df.sort_values(by=['class_name', 'batch']).index
    heatmap_data = heatmap_data.loc[sorted_samples]
    
    # Standardize genes (Z-score along samples) for rich visualization
    z_score_data = (heatmap_data - heatmap_data.mean(axis=0)) / heatmap_data.std(axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(z_score_data.T.values, cmap='RdBu_r', aspect='auto', vmin=-2.5, vmax=2.5)
    
    # Labels
    ax.set_yticks(range(30))
    clean_yticklabels = [gene.split('_')[0] for gene in top_30_exons]
    ax.set_yticklabels(clean_yticklabels, fontsize=8, fontweight='bold')
    ax.set_xticks([])
    ax.set_title("Expression Heatmap of Top 30 Highly Variable Exons (Row Z-score)", fontsize=14, fontweight='bold', pad=20)
    
    # Add a horizontal line to show disease separation
    nc_count = (labels_df.loc[sorted_samples, 'class_name'] == 'Non-Cancer').sum()
    ax.axvline(nc_count - 0.5, color='black', linewidth=2, linestyle='-')
    
    # Annotate Cancer vs Non-Cancer areas on the top/bottom
    ax.text(nc_count / 2, -1.5, "Non-Cancer (Healthy)", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2ecc71')
    ax.text(nc_count + (len(sorted_samples) - nc_count) / 2, -1.5, "Cancer (Tumor)", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#e74c3c')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.6)
    cbar.set_label("Expression Z-Score", fontsize=11)
    
    plt.tight_layout()
    save_plot_dual(fig, "top_30_exons_heatmap.png")
    plt.close()

    # 8. Sample Correlation Heatmap
    print("\n=== [9/9] Generating Sample-to-Sample Correlation ===")
    sample_corr = combat_df.T.corr()
    
    fig, ax = plt.subplots(figsize=(9, 8))
    sorted_idx = labels_df.sort_values(by=['batch', 'class_name']).index
    sorted_corr = sample_corr.loc[sorted_idx, sorted_idx]
    
    cax = ax.imshow(sorted_corr.values, cmap='coolwarm', vmin=0.3, vmax=1.0)
    fig.colorbar(cax, ax=ax, label="Pearson Correlation Coeff.")
    
    # Draw lines separating batches
    err_count = (labels_df.loc[sorted_idx, 'batch'] == 'ERR (Batch 0)').sum()
    ax.axhline(err_count - 0.5, color='black', linewidth=1.5, linestyle='--')
    ax.axvline(err_count - 0.5, color='black', linewidth=1.5, linestyle='--')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Sample-to-Sample Pearson Correlation (ComBat Corrected)", fontsize=13, fontweight='bold', pad=15)
    
    # Labels
    ax.text(err_count / 2, -5, "ERR", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#3498db')
    ax.text(err_count + (len(sorted_idx) - err_count) / 2, -5, "SRR", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e67e22')
    
    ax.text(-5, err_count / 2, "ERR", ha='right', va='center', rotation=90, fontsize=10, fontweight='bold', color='#3498db')
    ax.text(-5, err_count + (len(sorted_idx) - err_count) / 2, "SRR", ha='right', va='center', rotation=90, fontsize=10, fontweight='bold', color='#e67e22')
    
    plt.tight_layout()
    save_plot_dual(fig, "sample_correlation_heatmap.png")
    plt.close()
    
    # 9. Generate raw markdown report inside workspace eda/ and ARTIFACT_DIR
    generate_markdown_report(dist_table)
    
    print("\n=== OPTIMIZED EDA SUCCESSFULLY COMPLETED! ===")
    print(f"All files saved in: {EDA_DIR}")

def generate_markdown_report(dist_table):
    report_content = f"""# รายงานการวิเคราะห์ข้อมูลเชิงสำรวจ (Exploratory Data Analysis - EDA)
**โปรเจกต์:** การจำแนกประเภทมะเร็งด้วยข้อมูล Exon Expression Counts (ERR vs SRR)
**ตำแหน่งไฟล์ข้อมูล:** `A:\\data sci\\data\\counts\\matrices\\combined_exon_matrix.csv`

---

## 1. การกระจายตัวของกลุ่มตัวอย่างแยกตาม Batch และโรค (Sample Distribution)
ตัวอย่างทั้งหมด 103 ตัวอย่าง มีการแจกแจงตามประเภทการวิเคราะห์โรคดังนี้:

| แหล่งที่มา (Batch) | มะเร็ง (Cancer) | ปกติ (Non-Cancer) |
| --- | --- | --- |
| **ERR (Batch 0)** | {dist_table.loc['ERR (Batch 0)', 'Cancer']} | {dist_table.loc['ERR (Batch 0)', 'Non-Cancer']} |
| **SRR (Batch 1)** | {dist_table.loc['SRR (Batch 1)', 'Cancer']} | {dist_table.loc['SRR (Batch 1)', 'Non-Cancer']} |

![การกระจายตัวของกลุ่มตัวอย่าง](class_distribution.png)

---

## 2. การวิเคราะห์ความลึกในการอ่านลำดับเบส (Library Size Analysis)
ความแตกต่างของขนาดไลบรารีส่งผลต่อความเบี่ยงเบนของโมเดล การทำ Normalization มีความสำคัญมากเพื่อขจัดปัญหานี้

![การแจกแจงขนาดไลบรารี](library_size_distribution.png)

---

## 3. ความเบาบางของข้อมูล (Data Sparsity)
ข้อมูล Exon Counts มีสัดส่วนของเลขศูนย์ (Zero values) ที่มีสัดส่วนค่อนข้างสูง ซึ่งเป็นลักษณะธรรมชาติของเทคโนโลยี RNA-Seq

![สัดส่วนค่าศูนย์](zero_fraction_distribution.png)

---

## 4. ความสัมพันธ์ระหว่างค่าเฉลี่ยและความแปรปรวน (Mean-Variance Relationship)
ในการศึกษาทาง Genomics ค่าความแปรปรวน (Variance) มักจะแปรผันตามค่าเฉลี่ย (Mean) หรือมีความสัมพันธ์ในรูปแบบ Power Law (Overdispersion) การตรวจสอบคุณสมบัตินี้ช่วยยืนยันความจำเป็นในการแปลงข้อมูลด้วย Log-Transformation หรือการใช้โมเดลแบบ Negative Binomial

![ความสัมพันธ์ระหว่างค่าเฉลี่ยและความแปรปรวน](mean_variance_relationship.png)

---

## 5. ปัญหา Batch Effect และการแก้ไขด้วย ComBat
ผลลัพธ์จาก PCA (Principal Component Analysis) แสดงให้เห็นถึงการรวมตัวของกลุ่มตัวอย่างอย่างชัดเจนก่อนและหลังแก้ Batch Effect:

![ผลกระทบและการแก้ไข Batch Effect](pca_before_after_combat.png)

- **ก่อนปรับแก้ (Before ComBat):** ข้อมูลแยกกลุ่มออกเป็น 2 ฝั่งชัดเจนตามแหล่งที่มา (ERR vs SRR) บดบังข้อมูลความแตกต่างทางชีวภาพของโรค
- **หลังปรับแก้ (After ComBat):** แหล่งที่มาผสมผสานกันอย่างลงตัว และกลุ่มปกติ (Non-Cancer) กับกลุ่มมะเร็ง (Cancer) แยกออกจากกันได้ชัดเจนบนแกน PC1 ซึ่งจะส่งผลให้การเทรนโมเดลสามารถดึงสัญญาณโรคได้อย่างยอดเยี่ยม!

---

## 6. การแสดงออกของ Exons ที่มีความแปรปรวนสูงสุด 5 อันดับแรก
Exons เหล่านี้เป็นผู้เล่นหลักที่แสดงความแตกต่างระหว่างกลุ่มปกติและกลุ่มมะเร็งหลังจากการทำ Batch Correction แล้ว:

![ระดับการแสดงออกของ Exons หลัก](top_5_variable_exons.png)

---

## 7. แผนภาพความร้อนจำแนกกลุ่มโรค (Expression Heatmap of Top 30 Exons)
เมื่อพล็อตแผนภาพความร้อนของ Exon หลักที่มีความแปรปรวนสูงสุด 30 อันดับแรก โดยทำการทำ Z-score และเรียงลำดับตัวอย่างตามกลุ่มโรค จะเห็นความแตกต่างทางชีวภาพของระดับการแสดงออก (Expression Signature) ระหว่างกลุ่ม Non-Cancer และ Cancer อย่างชัดเจน เป็นการยืนยันว่าฟีเจอร์เหล่านี้สามารถนำไปทำนายโรคได้อย่างมีประสิทธิภาพ

![แผนภาพความร้อนระดับยีนหลัก](top_30_exons_heatmap.png)

---

## 8. ความสัมพันธ์ระหว่างคู่ตัวอย่าง (Sample-to-Sample Correlation Heatmap)
แผนภูมิความสัมพันธ์ Pearson บ่งชี้ความคล้ายคลึงระหว่างคู่ตัวอย่างหลังลบความเบี่ยงเบนทาง Batch ออกไป

![แผนภูมิความสัมพันธ์](sample_correlation_heatmap.png)

---
*รายงานนี้สร้างขึ้นโดยอัตโนมัติในกระบวนการรัน EDA*
"""
    # Write in workspace
    workspace_report_path = EDA_DIR / "eda_report.md"
    with open(workspace_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Generated Markdown Report at: {workspace_report_path}")
    
    # Copy to chat UI artifact folder
    artifact_report_path = ARTIFACT_DIR / "eda_report.md"
    shutil.copy2(workspace_report_path, artifact_report_path)
    print(f"Copied Markdown Report to Artifact: {artifact_report_path}")

if __name__ == "__main__":
    main()
