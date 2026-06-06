import numpy as np
import pandas as pd
import time
import subprocess
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneOut
from pycombat import Combat
from sklearn.metrics import accuracy_score

COUNTS_DIR = Path("/mnt/a/data sci/data/counts")
COMBAT_FEATS_PATH = COUNTS_DIR / "exp2_combined_combat" / "features.csv"
MATRIX_PATH = COUNTS_DIR / "matrices" / "combined_exon_matrix.csv"
LABELS_PATH = COUNTS_DIR / "matrices" / "combined_labels.csv"

def main():
    t0 = time.time()
    print("=== Step 1: Loading Column Names and Preparing Indices ===")
    feats_df = pd.read_csv(COMBAT_FEATS_PATH, nrows=2, index_col=0)
    top_exons = list(feats_df.columns)
    
    # Sort top_exons by their index suffixes so they match the order of fields sorted by cut
    sorted_top_exons = sorted(top_exons, key=lambda x: int(x.split("_")[-1]))
    fields = [1] + [int(col.split("_")[-1]) + 1 for col in sorted_top_exons]
    fields_str = ",".join(map(str, fields))
    
    print(f"Prepared fields: {len(fields)} in {time.time() - t0:.2f}s")
    
    print("=== Step 2: Extracting Column Subset using Bash Cut ===")
    subset_path = "/tmp/subset_raw_features.csv"
    cut_cmd = f"cut -d \",\" -f {fields_str} \"{MATRIX_PATH}\" > \"{subset_path}\""
    subprocess.run(cut_cmd, shell=True, check=True)
    print(f"Cut command finished in {time.time() - t0:.2f}s")
    
    print("=== Step 3: Loading and Normalizing Raw Exon Data ===")
    df = pd.read_csv(subset_path, index_col=0)
    df = df.iloc[5:] # Drop the Chr, Start, End, Strand, Length rows
    
    # Convert dtype to float
    df = df.astype(np.float64)
    
    # Rename columns to their actual names with coordinates
    df.columns = sorted_top_exons
    
    # Normalization (log1p)
    df = np.log1p(df)
    
    # Align labels
    y_df = pd.read_csv(LABELS_PATH, index_col=0)
    if y_df.index.name is None or y_df.index.name == '':
        y_df.index.name = df.index.name
        
    common_idx = df.index.intersection(y_df.index)
    df = df.loc[common_idx]
    y = y_df.loc[common_idx, "label"].values.astype(int)
    
    sample_ids = df.index.tolist()
    X = df.values.astype(np.float64)
    batch = np.array([0 if s.startswith("ERR") else 1 for s in sample_ids])
    
    print(f"Loaded and aligned dataset: {X.shape[0]} samples x {X.shape[1]} exons")
    print(f"Cancer: {y.sum()}, Normal: {len(y) - y.sum()}")
    print(f"ERR batch: {(batch==0).sum()}, SRR batch: {(batch==1).sum()}")
    print(f"Data prep finished in {time.time() - t0:.2f}s")
    
    # Exon column names inside df
    df_cols = list(df.columns)
    exon_1 = "ENSG00000164266_620977"
    exon_2 = "ENSG00000019169_316124"
    exon_3 = "ENSG00000204305_741180"
    
    # Initialize lists to store predictions
    preds_r2 = []
    preds_svm = []
    preds_xgb = []
    preds_rf = []
    
    loo = LeaveOneOut()
    
    print("\n=== Step 4: Running Leakage-Free LOO-CV (ComBat Inside Loop) ===")
    t_loop = time.time()
    for idx, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        batch_tr, batch_te = batch[train_idx], batch[test_idx]
        
        # Filter out zero-variance features in X_tr to prevent division by zero in ComBat
        variances = np.var(X_tr, axis=0)
        non_zero_var_indices = np.where(variances > 1e-9)[0]
        
        X_tr_f = X_tr[:, non_zero_var_indices]
        X_te_f = X_te[:, non_zero_var_indices]
        
        filtered_cols = [df_cols[i] for i in non_zero_var_indices]
        ex_idx1 = filtered_cols.index(exon_1)
        ex_idx2 = filtered_cols.index(exon_2)
        ex_idx3 = filtered_cols.index(exon_3)
        
        # Fit ComBat on training set ONLY
        combat = Combat()
        combat.fit(X_tr_f, batch_tr)
        
        # Workaround for pycombat validation bug: append a dummy sample from the other batch
        other_batch = 1 - batch_te[0]
        dummy_idx = np.where(batch_tr == other_batch)[0][0]
        X_dummy = X_tr_f[dummy_idx:dummy_idx+1]
        
        X_te_dummy = np.vstack([X_te_f, X_dummy])
        batch_te_dummy = np.array([batch_te[0], other_batch])
        
        # Correct train and test set using fitted ComBat parameters
        X_tr_c = combat.transform(X_tr_f, batch_tr)
        X_te_c_dummy = combat.transform(X_te_dummy, batch_te_dummy)
        X_te_c = X_te_c_dummy[0:1] # Extract the real test sample corrected values
        
        # --- Rule 2 (using ComBat corrected values for Exons 1, 2, and 3) ---
        val1 = X_te_c[0, ex_idx1]
        val2 = X_te_c[0, ex_idx2]
        val3 = X_te_c[0, ex_idx3]
        
        if val1 <= 2.75:
            pred_r2 = 1 if val2 <= 2.71 else 0
        else:
            pred_r2 = 1 if val3 <= 5.77 else 0
        preds_r2.append(pred_r2)
        
        # --- ML Models ---
        # Scale corrected features
        scaler = StandardScaler()
        X_tr_cs = scaler.fit_transform(X_tr_c)
        X_te_cs = scaler.transform(X_te_c)
        
        # SVM (RBF)
        clf_svm = SVC(kernel="rbf", random_state=42)
        clf_svm.fit(X_tr_cs, y_tr)
        preds_svm.append(clf_svm.predict(X_te_cs)[0])
        
        # XGBoost
        clf_xgb = XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
        clf_xgb.fit(X_tr_cs, y_tr)
        preds_xgb.append(clf_xgb.predict(X_te_cs)[0])
        
        # Random Forest
        clf_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf_rf.fit(X_tr_cs, y_tr)
        preds_rf.append(clf_rf.predict(X_te_cs)[0])
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(y)} folds in {time.time() - t_loop:.2f}s", flush=True)
            
    # Calculate un-leaked accuracies
    print("\n=== FINAL RESULTS (LEAKAGE-FREE LOO-CV) ===")
    print(f"Rule 2 (3-Exon):    {accuracy_score(y, preds_r2)*100:.2f}%")
    print(f"SVM (RBF):          {accuracy_score(y, preds_svm)*100:.2f}%")
    print(f"XGBoost:            {accuracy_score(y, preds_xgb)*100:.2f}%")
    print(f"Random Forest:      {accuracy_score(y, preds_rf)*100:.2f}%")
    print(f"\nTotal script execution time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
