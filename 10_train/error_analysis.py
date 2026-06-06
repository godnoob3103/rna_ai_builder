import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

COUNTS_DIR = Path("/mnt/a/data sci/data/counts")
MATRIX_PATH = COUNTS_DIR / "exp2_combined_combat" / "features.csv"
LABELS_PATH = COUNTS_DIR / "exp2_combined_combat" / "labels.csv"

def load_data(matrix_path, labels_path):
    X_df = pd.read_csv(matrix_path, index_col=0)
    y_df = pd.read_csv(labels_path, index_col=0)
    
    if y_df.index.name is None or y_df.index.name == '':
        y_df.index.name = X_df.index.name
    
    common_idx = X_df.index.intersection(y_df.index)
    X_df = X_df.loc[common_idx]
    y = y_df.loc[common_idx, "label"]
    return X_df, y

def main():
    print("=== Loading Data ===")
    X_df, y_df = load_data(MATRIX_PATH, LABELS_PATH)
    sample_ids = X_df.index.tolist()
    X = X_df.values.astype(np.float32)
    y = y_df.values.astype(int)
    
    print(f"Loaded {len(sample_ids)} samples.")
    print(f"Cancer: {y.sum()}, Normal: {len(y) - y.sum()}")
    
    # Check exons
    exon_1 = "ENSG00000164266_620977"
    exon_2 = "ENSG00000019169_316124"
    exon_3 = "ENSG00000204305_741180"
    
    for exon in [exon_1, exon_2, exon_3]:
        if exon not in X_df.columns:
            print(f"Warning: {exon} not in columns!")
            
    # Apply rules
    x_ex1 = X_df[exon_1].values
    x_ex2 = X_df[exon_2].values
    x_ex3 = X_df[exon_3].values
    
    # Rule 1
    preds_r1 = np.where(x_ex1 > 2.75, 1, 0)
    
    # Rule 2
    preds_r2 = []
    for i in range(len(y)):
        val1 = x_ex1[i]
        val2 = x_ex2[i]
        val3 = x_ex3[i]
        if val1 <= 2.75:
            if val2 <= 2.71:
                preds_r2.append(1)
            else:
                preds_r2.append(0)
        else:
            if val3 <= 5.77:
                preds_r2.append(1)
            else:
                preds_r2.append(0)
    preds_r2 = np.array(preds_r2)
    
    # ML Models LOO-CV
    models = [
        ("Random Forest", lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("XGBoost",       lambda: XGBClassifier(n_estimators=200, eval_metric="logloss",
                                                 random_state=42, verbosity=0, n_jobs=-1)),
        ("SVM (RBF)",     lambda: SVC(kernel="rbf", probability=True, random_state=42)),
    ]
    
    loo_preds = {}
    loo = LeaveOneOut()
    for model_name, model_fn in models:
        print(f"Running LOO-CV: {model_name}...")
        preds = np.zeros(len(y), dtype=int)
        for train_idx, test_idx in loo.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr = y[train_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            clf = model_fn()
            clf.fit(X_tr, y_tr)
            preds[test_idx[0]] = clf.predict(X_te)[0]
        loo_preds[model_name] = preds

    all_preds = {
        "Rule 1": preds_r1,
        "Rule 2": preds_r2,
        **loo_preds
    }
    
    print("\n=== Error Analysis ===")
    error_summary = []
    fps_dict = {}
    fns_dict = {}
    misclassified_dict = {}
    
    for name, preds in all_preds.items():
        fps = [sample_ids[i] for i in range(len(y)) if y[i] == 0 and preds[i] == 1]
        fns = [sample_ids[i] for i in range(len(y)) if y[i] == 1 and preds[i] == 0]
        tps = [sample_ids[i] for i in range(len(y)) if y[i] == 1 and preds[i] == 1]
        tns = [sample_ids[i] for i in range(len(y)) if y[i] == 0 and preds[i] == 0]
        
        fps_dict[name] = set(fps)
        fns_dict[name] = set(fns)
        misclassified_dict[name] = set(fps + fns)
        
        acc = (len(tps) + len(tns)) / len(y)
        print(f"{name}:")
        print(f"  Accuracy: {acc*100:.2f}%")
        print(f"  False Positives (FP) ({len(fps)}): {fps}")
        print(f"  False Negatives (FN) ({len(fns)}): {fns}")
        
        error_summary.append({
            "Model": name,
            "Accuracy": acc,
            "FP_Count": len(fps),
            "FN_Count": len(fns),
            "Total_Errors": len(fps) + len(fns)
        })
        
    df_summary = pd.DataFrame(error_summary)
    df_summary.to_csv(COUNTS_DIR / "results" / "error_analysis_summary.csv", index=False)
    
    # Overlap analysis
    print("\n=== Error Overlap ===")
    model_names = list(all_preds.keys())
    fp_overlap = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    fn_overlap = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    total_overlap = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    
    for m1 in model_names:
        for m2 in model_names:
            fp_intersect = len(fps_dict[m1].intersection(fps_dict[m2]))
            fn_intersect = len(fns_dict[m1].intersection(fns_dict[m2]))
            total_intersect = len(misclassified_dict[m1].intersection(misclassified_dict[m2]))
            
            fp_overlap.loc[m1, m2] = fp_intersect
            fn_overlap.loc[m1, m2] = fn_intersect
            total_overlap.loc[m1, m2] = total_intersect
            
    print("FP Overlap (Count of common FPs):")
    print(fp_overlap)
    print("\nFN Overlap (Count of common FNs):")
    print(fn_overlap)
    print("\nTotal Error Overlap:")
    print(total_overlap)
    
    fp_overlap.to_csv(COUNTS_DIR / "results" / "fp_overlap.csv")
    fn_overlap.to_csv(COUNTS_DIR / "results" / "fn_overlap.csv")
    total_overlap.to_csv(COUNTS_DIR / "results" / "total_overlap.csv")
    
    # Misclassified by ALL
    all_misclassified = set(sample_ids)
    for name in model_names:
        all_misclassified = all_misclassified.intersection(misclassified_dict[name])
    print(f"\nSamples misclassified by ALL models ({len(all_misclassified)}): {list(all_misclassified)}")
    
    # Misclassified by Rule 2 but NOT by any ML models
    ml_models = ["Random Forest", "XGBoost", "SVM (RBF)"]
    r2_only = misclassified_dict["Rule 2"].copy()
    for ml in ml_models:
        r2_only = r2_only.difference(misclassified_dict[ml])
    print(f"Samples misclassified by Rule 2 but NOT by any ML model ({len(r2_only)}): {list(r2_only)}")
    
    # Misclassified by ML but NOT by Rule 2
    ml_only = set()
    for ml in ml_models:
        ml_only = ml_only.union(misclassified_dict[ml])
    ml_only = ml_only.difference(misclassified_dict["Rule 2"])
    print(f"Samples misclassified by at least one ML model but NOT by Rule 2 ({len(ml_only)}): {list(ml_only)}")
    
    # Matplotlib plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmaps = ["Oranges", "Blues", "Purples"]
    titles = ["FP Shared Error Counts", "FN Shared Error Counts", "Total Shared Error Counts"]
    matrices = [fp_overlap, fn_overlap, total_overlap]
    
    for ax, matrix, title, cmap_name in zip(axes, matrices, titles, cmaps):
        im = ax.imshow(matrix.values.astype(float), cmap=cmap_name)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(len(model_names)))
        ax.set_yticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_yticklabels(model_names)
        
        # Max value for text color threshold
        max_val = matrix.values.max()
        for i in range(len(model_names)):
            for j in range(len(model_names)):
                val = int(matrix.values[i, j])
                threshold = max_val / 2 if max_val > 0 else 1
                ax.text(j, i, str(val), ha="center", va="center", 
                        color="white" if val > threshold else "black",
                        fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plot_path = COUNTS_DIR / "results" / "error_overlap_heatmap.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved overlap heatmap to {plot_path}")

    # Plot sample-specific errors
    all_errors = set()
    for name in model_names:
        all_errors = all_errors.union(misclassified_dict[name])
    
    all_errors_sorted = sorted(list(all_errors))
    if len(all_errors_sorted) > 0:
        error_matrix = pd.DataFrame(index=all_errors_sorted, columns=model_names)
        for sample in all_errors_sorted:
            for name in model_names:
                if sample in fps_dict[name]:
                    error_matrix.loc[sample, name] = 1
                elif sample in fns_dict[name]:
                    error_matrix.loc[sample, name] = -1
                else:
                    error_matrix.loc[sample, name] = 0
                    
        plt.figure(figsize=(10, max(5, len(all_errors_sorted) * 0.25 + 2)))
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#1f77b4', '#f7f7f7', '#ff7f0e'])
        
        display_index = []
        for sample in all_errors_sorted:
            true_label = y[sample_ids.index(sample)]
            label_str = "Cancer" if true_label == 1 else "Normal"
            display_index.append(f"{sample} ({label_str})")
            
        data_plot = error_matrix.values.astype(float)
        plt.imshow(data_plot, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.yticks(range(len(all_errors_sorted)), display_index)
        
        # Add grid lines
        plt.gca().set_xticks(np.arange(-.5, len(model_names), 1), minor=True)
        plt.gca().set_yticks(np.arange(-.5, len(all_errors_sorted), 1), minor=True)
        plt.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.5)
        
        cbar = plt.colorbar(ticks=[-0.67, 0, 0.67])
        cbar.ax.set_yticklabels(['FN (Underpredict)', 'Correct', 'FP (Overpredict)'])
        
        plt.title("Error Analysis: Model Predictions per Sample")
        plt.xlabel("Model")
        plt.ylabel("Sample ID (True Class)")
        plt.tight_layout()
        
        matrix_plot_path = COUNTS_DIR / "results" / "sample_error_matrix.png"
        plt.savefig(matrix_plot_path, dpi=150)
        plt.close()
        print(f"Saved sample error matrix plot to {matrix_plot_path}")

if __name__ == "__main__":
    main()
