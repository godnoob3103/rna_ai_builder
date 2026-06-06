import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path

# Directories
COUNTS_DIR = Path("a:/data sci/data/counts")
MATRIX_PATH = COUNTS_DIR / "exp2_combined_combat" / "features.csv"
LABELS_PATH = COUNTS_DIR / "matrices" / "combined_labels.csv"

def fast_load_features(file_path):
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
    return pd.DataFrame(np.vstack(data), index=samples, columns=exon_ids)

def main():
    print("=== Loading Features and Labels ===")
    df_features = fast_load_features(MATRIX_PATH)
    df_labels = pd.read_csv(LABELS_PATH, index_col=0)
    
    # Align labels with features
    y_true = df_labels.loc[df_features.index, "label"].values.astype(int)
    
    # Extract the relevant exons for the rules
    exon_1 = "ENSG00000164266_620977"
    exon_2 = "ENSG00000019169_316124"
    exon_3 = "ENSG00000204305_741180"
    
    # Ensure they exist in features
    for exon in [exon_1, exon_2, exon_3]:
        if exon not in df_features.columns:
            print(f"Error: Exon {exon} not found in features!")
            return
            
    x_ex1 = df_features[exon_1].values
    x_ex2 = df_features[exon_2].values
    x_ex3 = df_features[exon_3].values
    
    print(f"Loaded {len(y_true)} samples successfully.")
    print(f"Class distribution: Cancer = {y_true.sum()}, Healthy = {len(y_true) - y_true.sum()}")
    
    # --- RULE 1: 1-Level If-Else Rule ---
    print("\n==============================================")
    print("RULE 1: 1-Level Single-Gene If-Else Rule")
    print("----------------------------------------------")
    print(f"If {exon_1} > 2.75  => CANCER")
    print("Else                  => HEALTHY")
    
    # Apply rule
    y_pred_r1 = np.where(x_ex1 > 2.75, 1, 0)
    
    # Calculate metrics
    acc_r1 = accuracy_score(y_true, y_pred_r1)
    tn_r1, fp_r1, fn_r1, tp_r1 = confusion_matrix(y_true, y_pred_r1).ravel()
    sens_r1 = tp_r1 / (tp_r1 + fn_r1)
    spec_r1 = tn_r1 / (tn_r1 + fp_r1)
    
    print(f"\nResults for Rule 1:")
    print(f"  - Accuracy:    {acc_r1*100:.2f}% ({tn_r1+tp_r1}/{len(y_true)} correct)")
    print(f"  - Sensitivity (Sensitivity): {sens_r1*100:.2f}% (Cancer correctly predicted: {tp_r1}/{tp_r1+fn_r1})")
    print(f"  - Specificity (Specificity): {spec_r1*100:.2f}% (Healthy correctly predicted: {tn_r1}/{tn_r1+fp_r1})")
    print(f"  - Confusion Matrix:")
    print(f"      Predicted Healthy  Predicted Cancer")
    print(f"Actual Healthy   {tn_r1:<16}  {fp_r1:<16}")
    print(f"Actual Cancer    {fn_r1:<16}  {tp_r1:<16}")
    
    # --- RULE 2: 2-Level Nested If-Else Rule ---
    print("\n==============================================")
    print("RULE 2: 2-Level Nested 3-Gene If-Else Rule")
    print("----------------------------------------------")
    print(f"If {exon_1} <= 2.75:")
    print(f"    If {exon_2} <= 2.71 => CANCER")
    print(f"    Else                => HEALTHY")
    print(f"Else (i.e. > 2.75):")
    print(f"    If {exon_3} <= 5.77 => CANCER")
    print(f"    Else                => HEALTHY")
    
    # Apply rule
    y_pred_r2 = []
    for i in range(len(y_true)):
        val1 = x_ex1[i]
        val2 = x_ex2[i]
        val3 = x_ex3[i]
        
        if val1 <= 2.75:
            if val2 <= 2.71:
                y_pred_r2.append(1)
            else:
                y_pred_r2.append(0)
        else:
            if val3 <= 5.77:
                y_pred_r2.append(1)
            else:
                y_pred_r2.append(0)
                
    y_pred_r2 = np.array(y_pred_r2)
    
    # Calculate metrics
    acc_r2 = accuracy_score(y_true, y_pred_r2)
    tn_r2, fp_r2, fn_r2, tp_r2 = confusion_matrix(y_true, y_pred_r2).ravel()
    sens_r2 = tp_r2 / (tp_r2 + fn_r2)
    spec_r2 = tn_r2 / (tn_r2 + fp_r2)
    
    print(f"\nResults for Rule 2:")
    print(f"  - Accuracy:    {acc_r2*100:.2f}% ({tn_r2+tp_r2}/{len(y_true)} correct)")
    print(f"  - Sensitivity (Sensitivity): {sens_r2*100:.2f}% (Cancer correctly predicted: {tp_r2}/{tp_r2+fn_r2})")
    print(f"  - Specificity (Specificity): {spec_r2*100:.2f}% (Healthy correctly predicted: {tn_r2}/{tn_r2+fp_r2})")
    print(f"  - Confusion Matrix:")
    print(f"      Predicted Healthy  Predicted Cancer")
    print(f"Actual Healthy   {tn_r2:<16}  {fp_r2:<16}")
    print(f"Actual Cancer    {fn_r2:<16}  {tp_r2:<16}")
    
if __name__ == "__main__":
    main()
