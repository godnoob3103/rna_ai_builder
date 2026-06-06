import numpy as np
import pandas as pd
import time
import subprocess
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneOut
from pycombat import Combat
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

COUNTS_DIR = Path("/mnt/a/data sci/data/counts")
COMBAT_FEATS_PATH = COUNTS_DIR / "exp2_combined_combat" / "features.csv"
COMBAT_LABELS_PATH = COUNTS_DIR / "exp2_combined_combat" / "labels.csv"
MATRIX_PATH = COUNTS_DIR / "matrices" / "combined_exon_matrix.csv"
LABELS_PATH = COUNTS_DIR / "matrices" / "combined_labels.csv"
RESULTS_DIR = COUNTS_DIR / "results"

def load_raw_features(top_exons):
    subset_path = Path("/tmp/subset_raw_features.csv")
    if not subset_path.exists():
        print("Extracting subset features from raw matrix...")
        with open(MATRIX_PATH, "r") as f:
            header_line = f.readline().strip()
        headers = header_line.split(",")
        
        # Sort top_exons by index to align with cut output
        sorted_top_exons = sorted(top_exons, key=lambda x: int(x.split("_")[-1]))
        fields = [1] + [int(col.split("_")[-1]) + 1 for col in sorted_top_exons]
        fields_str = ",".join(map(str, fields))
        
        cut_cmd = f"cut -d \",\" -f {fields_str} \"{MATRIX_PATH}\" > \"{subset_path}\""
        subprocess.run(cut_cmd, shell=True, check=True)
    
    df = pd.read_csv(subset_path, index_col=0)
    df = df.iloc[5:] # Drop metadata rows
    df = df.astype(np.float64)
    # Rename columns to their coordinate-names
    sorted_top_exons = sorted(top_exons, key=lambda x: int(x.split("_")[-1]))
    df.columns = sorted_top_exons
    return df

def get_error_metrics(y_true, y_pred, sample_ids):
    fps = [sample_ids[i] for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1]
    fns = [sample_ids[i] for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0]
    tps = sum((y_true == 1) & (y_pred == 1))
    tns = sum((y_true == 0) & (y_pred == 0))
    
    acc = (tps + tns) / len(y_true)
    sens = tps / sum(y_true) if sum(y_true) > 0 else 0.0
    spec = tns / sum(y_true == 0) if sum(y_true == 0) > 0 else 0.0
    
    return acc, sens, spec, set(fps), set(fns)

def plot_overlap_heatmap(overlap_matrix, model_names, title, save_path, cmap_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(overlap_matrix.values.astype(float), cmap=cmap_name)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(model_names)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(model_names, fontsize=9)
    
    max_val = overlap_matrix.values.max()
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            val = int(overlap_matrix.values[i, j])
            threshold = max_val / 2 if max_val > 0 else 1
            ax.text(j, i, str(val), ha="center", va="center", 
                    color="white" if val > threshold else "black",
                    fontweight="bold", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    t_start = time.time()
    print("=== Loading Config and Meta ===")
    feats_df = pd.read_csv(COMBAT_FEATS_PATH, nrows=2, index_col=0)
    top_exons = list(feats_df.columns)
    
    exon_1 = "ENSG00000164266_620977"
    exon_2 = "ENSG00000019169_316124"
    exon_3 = "ENSG00000204305_741180"
    exons_3 = [exon_1, exon_2, exon_3]
    
    # -------------------------------------------------------------------------
    # SCENARIO A: Leaky Cross-Validation (ComBat Global)
    # -------------------------------------------------------------------------
    print("\n=== Running SCENARIO A: Leaky Cross-Validation ===")
    X_a_df = pd.read_csv(COMBAT_FEATS_PATH, index_col=0)
    y_a_df = pd.read_csv(COMBAT_LABELS_PATH, index_col=0)
    common_a = X_a_df.index.intersection(y_a_df.index)
    X_a_df = X_a_df.loc[common_a]
    y_a = y_a_df.loc[common_a, "label"].values.astype(int)
    sample_ids_a = X_a_df.index.tolist()
    
    # Exons subset
    X_a_all = X_a_df.values.astype(np.float32)
    X_a_3 = X_a_df[exons_3].values.astype(np.float32)
    
    # Models to evaluate
    ml_models_all = [
        ("SVM (RBF) - 5k Exons", lambda: SVC(kernel="rbf", random_state=42)),
        ("XGBoost - 5k Exons",   lambda: XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)),
        ("Random Forest - 5k Exons", lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ]
    ml_models_3 = [
        ("SVM (RBF) - 3 Exons", lambda: SVC(kernel="rbf", random_state=42)),
        ("XGBoost - 3 Exons",   lambda: XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)),
        ("Random Forest - 3 Exons", lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("Decision Tree - 3 Exons", lambda: DecisionTreeClassifier(max_depth=2, random_state=42)),
    ]
    
    loo = LeaveOneOut()
    preds_a = {}
    
    # Rules
    x_ex1_a = X_a_df[exon_1].values
    x_ex2_a = X_a_df[exon_2].values
    x_ex3_a = X_a_df[exon_3].values
    
    preds_a["Rule 1 - 1 Exon"] = np.where(x_ex1_a > 2.75, 1, 0)
    
    r2_preds = []
    for i in range(len(y_a)):
        r2_preds.append(1 if (x_ex1_a[i] <= 2.75 and x_ex2_a[i] <= 2.71) or (x_ex1_a[i] > 2.75 and x_ex3_a[i] <= 5.77) else 0)
    preds_a["Rule 2 - 3 Exons"] = np.array(r2_preds)
    
    # ML Models - 5k exons
    for name, model_fn in ml_models_all:
        preds = np.zeros(len(y_a), dtype=int)
        for train_idx, test_idx in loo.split(X_a_all):
            X_tr, X_te = X_a_all[train_idx], X_a_all[test_idx]
            y_tr = y_a[train_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            clf = model_fn()
            clf.fit(X_tr, y_tr)
            preds[test_idx[0]] = clf.predict(X_te)[0]
        preds_a[name] = preds
        
    # ML Models - 3 exons
    for name, model_fn in ml_models_3:
        preds = np.zeros(len(y_a), dtype=int)
        for train_idx, test_idx in loo.split(X_a_3):
            X_tr, X_te = X_a_3[train_idx], X_a_3[test_idx]
            y_tr = y_a[train_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            clf = model_fn()
            clf.fit(X_tr, y_tr)
            preds[test_idx[0]] = clf.predict(X_te)[0]
        preds_a[name] = preds
        
    # -------------------------------------------------------------------------
    # SCENARIO B: Leakage-Free Cross-Validation (ComBat Inside Loop)
    # -------------------------------------------------------------------------
    print("\n=== Running SCENARIO B: Leakage-Free Cross-Validation ===")
    df_raw = load_raw_features(top_exons)
    df_raw = np.log1p(df_raw) # Log normalization
    
    y_raw_df = pd.read_csv(LABELS_PATH, index_col=0)
    if y_raw_df.index.name is None or y_raw_df.index.name == '':
        y_raw_df.index.name = df_raw.index.name
    common_b = df_raw.index.intersection(y_raw_df.index)
    df_raw = df_raw.loc[common_b]
    y_b = y_raw_df.loc[common_b, "label"].values.astype(int)
    sample_ids_b = df_raw.index.tolist()
    
    X_b = df_raw.values.astype(np.float64)
    batch = np.array([0 if s.startswith("ERR") else 1 for s in sample_ids_b])
    
    df_cols = list(df_raw.columns)
    ex_idx1 = df_cols.index(exon_1)
    ex_idx2 = df_cols.index(exon_2)
    ex_idx3 = df_cols.index(exon_3)
    ex_indices_3 = [ex_idx1, ex_idx2, ex_idx3]
    
    preds_b = {
        "Rule 1 - 1 Exon": [],
        "Rule 2 - 3 Exons": [],
        "SVM (RBF) - 5k Exons": [],
        "XGBoost - 5k Exons": [],
        "Random Forest - 5k Exons": [],
        "SVM (RBF) - 3 Exons": [],
        "XGBoost - 3 Exons": [],
        "Random Forest - 3 Exons": [],
        "Decision Tree - 3 Exons": [],
    }
    
    for idx, (train_idx, test_idx) in enumerate(loo.split(X_b)):
        X_tr, X_te = X_b[train_idx], X_b[test_idx]
        y_tr = y_b[train_idx]
        batch_tr, batch_te = batch[train_idx], batch[test_idx]
        
        # Remove zero variance features dynamically to prevent NaNs in ComBat
        variances = np.var(X_tr, axis=0)
        non_zero_var_indices = np.where(variances > 1e-9)[0]
        
        X_tr_f = X_tr[:, non_zero_var_indices]
        X_te_f = X_te[:, non_zero_var_indices]
        
        filtered_cols = [df_cols[i] for i in non_zero_var_indices]
        f_idx1 = filtered_cols.index(exon_1)
        f_idx2 = filtered_cols.index(exon_2)
        f_idx3 = filtered_cols.index(exon_3)
        
        # Fit ComBat on training set ONLY
        combat = Combat()
        combat.fit(X_tr_f, batch_tr)
        
        # Workaround for pycombat validation bug: append dummy sample
        other_batch = 1 - batch_te[0]
        dummy_idx = np.where(batch_tr == other_batch)[0][0]
        X_dummy = X_tr_f[dummy_idx:dummy_idx+1]
        
        X_te_dummy = np.vstack([X_te_f, X_dummy])
        batch_te_dummy = np.array([batch_te[0], other_batch])
        
        # Correct train and test sets using ComBat
        X_tr_c = combat.transform(X_tr_f, batch_tr)
        X_te_c_dummy = combat.transform(X_te_dummy, batch_te_dummy)
        X_te_c = X_te_c_dummy[0:1]
        
        # Rules (using ComBat corrected features)
        val1 = X_te_c[0, f_idx1]
        val2 = X_te_c[0, f_idx2]
        val3 = X_te_c[0, f_idx3]
        
        preds_b["Rule 1 - 1 Exon"].append(1 if val1 > 2.75 else 0)
        preds_b["Rule 2 - 3 Exons"].append(1 if (val1 <= 2.75 and val2 <= 2.71) or (val1 > 2.75 and val3 <= 5.77) else 0)
        
        # 5k Exons Model evaluations
        scaler = StandardScaler()
        X_tr_cs = scaler.fit_transform(X_tr_c)
        X_te_cs = scaler.transform(X_te_c)
        
        # SVM 5k
        clf = SVC(kernel="rbf", random_state=42)
        clf.fit(X_tr_cs, y_tr)
        preds_b["SVM (RBF) - 5k Exons"].append(clf.predict(X_te_cs)[0])
        
        # XGBoost 5k
        clf = XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
        clf.fit(X_tr_cs, y_tr)
        preds_b["XGBoost - 5k Exons"].append(clf.predict(X_te_cs)[0])
        
        # RF 5k
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_tr_cs, y_tr)
        preds_b["Random Forest - 5k Exons"].append(clf.predict(X_te_cs)[0])
        
        # 3 Exons Model evaluations (exons only)
        f_ex_indices_3 = [f_idx1, f_idx2, f_idx3]
        X_tr_c_3 = X_tr_c[:, f_ex_indices_3]
        X_te_c_3 = X_te_c[:, f_ex_indices_3]
        
        scaler_3 = StandardScaler()
        X_tr_cs_3 = scaler_3.fit_transform(X_tr_c_3)
        X_te_cs_3 = scaler_3.transform(X_te_c_3)
        
        # SVM 3
        clf = SVC(kernel="rbf", random_state=42)
        clf.fit(X_tr_cs_3, y_tr)
        preds_b["SVM (RBF) - 3 Exons"].append(clf.predict(X_te_cs_3)[0])
        
        # XGBoost 3
        clf = XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
        clf.fit(X_tr_cs_3, y_tr)
        preds_b["XGBoost - 3 Exons"].append(clf.predict(X_te_cs_3)[0])
        
        # RF 3
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_tr_cs_3, y_tr)
        preds_b["Random Forest - 3 Exons"].append(clf.predict(X_te_cs_3)[0])
        
        # DT 3
        clf = DecisionTreeClassifier(max_depth=2, random_state=42)
        clf.fit(X_tr_cs_3, y_tr)
        preds_b["Decision Tree - 3 Exons"].append(clf.predict(X_te_cs_3)[0])
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx+1}/{len(y_b)} folds in Scenario B...")
            
    # Convert lists to arrays
    for key in preds_b:
        preds_b[key] = np.array(preds_b[key])
        
    # -------------------------------------------------------------------------
    # ANALYZE AND SAVE RESULTS
    # -------------------------------------------------------------------------
    print("\n=== Compiling Results and Generating Tables ===")
    summary_rows = []
    
    fps_dict_a, fns_dict_a = {}, {}
    fps_dict_b, fns_dict_b = {}, {}
    misclassified_dict_b = {}
    
    model_keys = [
        "Rule 1 - 1 Exon", "Rule 2 - 3 Exons",
        "SVM (RBF) - 5k Exons", "XGBoost - 5k Exons", "Random Forest - 5k Exons",
        "SVM (RBF) - 3 Exons", "XGBoost - 3 Exons", "Random Forest - 3 Exons", "Decision Tree - 3 Exons"
    ]
    
    for key in model_keys:
        # Scenario A
        acc_a, sens_a, spec_a, fps_a, fns_a = get_error_metrics(y_a, preds_a[key], sample_ids_a)
        fps_dict_a[key] = fps_a
        fns_dict_a[key] = fns_a
        
        # Scenario B
        acc_b, sens_b, spec_b, fps_b, fns_b = get_error_metrics(y_b, preds_b[key], sample_ids_b)
        fps_dict_b[key] = fps_b
        fns_dict_b[key] = fns_b
        misclassified_dict_b[key] = fps_b.union(fns_b)
        
        summary_rows.append({
            "Configuration": key,
            "Leaky_Accuracy": acc_a,
            "Leaky_Sensitivity": sens_a,
            "Leaky_Specificity": spec_a,
            "Leaky_Total_Errors": len(fps_a) + len(fns_a),
            "LeakFree_Accuracy": acc_b,
            "LeakFree_Sensitivity": sens_b,
            "LeakFree_Specificity": spec_b,
            "LeakFree_Total_Errors": len(fps_b) + len(fns_b)
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS_DIR / "complete_error_analysis_summary.csv", index=False)
    print("Saved CSV: complete_error_analysis_summary.csv")
    
    # Pairwise overlap matrices for Leaky and Leak-free Scenarios
    print("Generating overlap heatmaps...")
    fp_overlap_a = pd.DataFrame(index=model_keys, columns=model_keys, dtype=int)
    fn_overlap_a = pd.DataFrame(index=model_keys, columns=model_keys, dtype=int)
    total_overlap_a = pd.DataFrame(index=model_keys, columns=model_keys, dtype=int)
    
    fp_overlap_b = pd.DataFrame(index=model_keys, columns=model_keys, dtype=int)
    fn_overlap_b = pd.DataFrame(index=model_keys, columns=model_keys, dtype=int)
    total_overlap_b = pd.DataFrame(index=model_keys, columns=model_keys, dtype=int)
    
    for m1 in model_keys:
        for m2 in model_keys:
            # Scenario A
            fp_overlap_a.loc[m1, m2] = len(fps_dict_a[m1].intersection(fps_dict_a[m2]))
            fn_overlap_a.loc[m1, m2] = len(fns_dict_a[m1].intersection(fns_dict_a[m2]))
            total_overlap_a.loc[m1, m2] = len((fps_dict_a[m1].union(fns_dict_a[m1])).intersection(fps_dict_a[m2].union(fns_dict_a[m2])))
            
            # Scenario B
            fp_overlap_b.loc[m1, m2] = len(fps_dict_b[m1].intersection(fps_dict_b[m2]))
            fn_overlap_b.loc[m1, m2] = len(fns_dict_b[m1].intersection(fns_dict_b[m2]))
            total_overlap_b.loc[m1, m2] = len(misclassified_dict_b[m1].intersection(misclassified_dict_b[m2]))
            
    # Save CSVs
    fp_overlap_b.to_csv(RESULTS_DIR / "fp_overlap_leakfree.csv")
    fn_overlap_b.to_csv(RESULTS_DIR / "fn_overlap_leakfree.csv")
    
    # Plotting Heatmaps
    plot_overlap_heatmap(total_overlap_a, model_keys, "Leaky Scenario: Total Error Overlaps", RESULTS_DIR / "error_overlap_heatmap_leaky.png", "Oranges")
    plot_overlap_heatmap(total_overlap_b, model_keys, "Leakage-Free Scenario: Total Error Overlaps", RESULTS_DIR / "error_overlap_heatmap_leakfree.png", "Blues")
    
    # Sample-level complete error matrix plot for Scenario B (Leak-free)
    print("Generating complete sample error matrix plot...")
    all_errors_b = set()
    for key in model_keys:
        all_errors_b = all_errors_b.union(misclassified_dict_b[key])
    all_errors_b_sorted = sorted(list(all_errors_b))
    
    if len(all_errors_b_sorted) > 0:
        error_matrix = pd.DataFrame(index=all_errors_b_sorted, columns=model_keys)
        for sample in all_errors_b_sorted:
            for key in model_keys:
                if sample in fps_dict_b[key]:
                    error_matrix.loc[sample, key] = 1 # FP
                elif sample in fns_dict_b[key]:
                    error_matrix.loc[sample, key] = -1 # FN
                else:
                    error_matrix.loc[sample, key] = 0 # Correct
                    
        plt.figure(figsize=(12, max(6, len(all_errors_b_sorted) * 0.25 + 2)))
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#1f77b4', '#f7f7f7', '#ff7f0e'])
        
        display_index = []
        for sample in all_errors_b_sorted:
            true_label = y_b[sample_ids_b.index(sample)]
            label_str = "Cancer" if true_label == 1 else "Normal"
            display_index.append(f"{sample} ({label_str})")
            
        data_plot = error_matrix.values.astype(float)
        plt.imshow(data_plot, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        
        plt.xticks(range(len(model_keys)), model_keys, rotation=45, ha='right', fontsize=9)
        plt.yticks(range(len(all_errors_b_sorted)), display_index, fontsize=8)
        
        # Grid lines
        plt.gca().set_xticks(np.arange(-.5, len(model_keys), 1), minor=True)
        plt.gca().set_yticks(np.arange(-.5, len(all_errors_b_sorted), 1), minor=True)
        plt.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.5)
        
        cbar = plt.colorbar(ticks=[-0.67, 0, 0.67])
        cbar.ax.set_yticklabels(['FN (Underpredict)', 'Correct', 'FP (Overpredict)'], fontsize=9)
        
        plt.title("Sample-Level Error Matrix (Leakage-Free Scenario)", fontsize=12, fontweight="bold")
        plt.xlabel("Model Configuration", fontsize=10)
        plt.ylabel("Sample ID (True Class)", fontsize=10)
        plt.tight_layout()
        
        plt.savefig(RESULTS_DIR / "sample_error_matrix_complete.png", dpi=150)
        plt.close()
        print("Saved sample error matrix plot to sample_error_matrix_complete.png")
        
    print(f"\nDone in {time.time() - t_start:.2f} seconds!")

if __name__ == "__main__":
    main()
