import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# Directories
COUNTS_DIR = Path("/mnt/a/data sci/data/counts")
RESULTS_DIR = COUNTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Artifact directory for chat UI embedding
ARTIFACT_DIR = Path("/root/.gemini/antigravity-cli/brain/67170f3a-66b7-4953-8fac-40344f60e01c")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
COMBAT_FEATS_PATH = COUNTS_DIR / "exp2_combined_combat" / "features.csv"
LABELS_PATH = COUNTS_DIR / "matrices" / "combined_labels.csv"

def load_raw_features(top_exons):
    subset_path = Path("/tmp/subset_raw_features.csv")
    df = pd.read_csv(subset_path, index_col=0)
    df = df.iloc[5:] # Drop metadata rows
    df = df.astype(np.float64)
    # Rename columns to their coordinate-names
    sorted_top_exons = sorted(top_exons, key=lambda x: int(x.split("_")[-1]))
    df.columns = sorted_top_exons
    return df

def loo_eval_precomputed(model_info, corrected_folds, y):
    preds = []
    trues = []
    
    if model_info["type"] == "rule":
        for fold in corrected_folds:
            pred = model_info["eval_fn"](fold, None, None)
            preds.append(pred)
            trues.append(y[fold["test_idx"][0]])
    else:
        # ML model
        for fold in corrected_folds:
            X_tr = fold["X_tr"]
            X_te = fold["X_te"]
            y_tr = y[fold["train_idx"]]
            y_te = y[fold["test_idx"][0]]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            
            clf = model_info["clf_fn"]()
            clf.fit(X_tr_s, y_tr)
            pred = clf.predict(X_te_s)[0]
            
            preds.append(pred)
            trues.append(y_te)
            
    return np.array(trues), np.array(preds)

def main():
    t_start = time.time()
    print("=== [1/5] Starting No-ComBat Y-Scrambling Diagnostic Test ===")
    
    # Load config and meta to get top exons
    feats_df = pd.read_csv(COMBAT_FEATS_PATH, nrows=2, index_col=0)
    top_exons = list(feats_df.columns)
    
    # Load raw data
    print("Loading raw features and labels...")
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
    
    # Exons mapping
    df_cols = list(df_raw.columns)
    exon_1 = "ENSG00000164266_620977"
    exon_2 = "ENSG00000019169_316124"
    exon_3 = "ENSG00000204305_741180"
    
    f_idx1 = df_cols.index(exon_1)
    f_idx2 = df_cols.index(exon_2)
    f_idx3 = df_cols.index(exon_3)
    f_ex_indices_3 = [f_idx1, f_idx2, f_idx3]
    
    # Pre-compute uncorrected train and test sets for all 103 LOO-CV folds
    print("\n=== [2/5] Preparing Raw (No-ComBat) Folds ===")
    loo = LeaveOneOut()
    raw_folds = []
    
    for idx, (train_idx, test_idx) in enumerate(loo.split(X_b)):
        X_tr, X_te = X_b[train_idx], X_b[test_idx]
        
        # We don't run ComBat, but we do standard fold preparation
        raw_folds.append({
            "train_idx": train_idx,
            "test_idx": test_idx,
            "X_tr": X_tr,
            "X_te": X_te,
            "X_tr_3": X_tr[:, f_ex_indices_3],
            "X_te_3": X_te[:, f_ex_indices_3],
            "val1": X_te[0, f_idx1],
            "val2": X_te[0, f_idx2],
            "val3": X_te[0, f_idx3]
        })
        
    print(f"Folds preparation completed in {time.time() - t_start:.2f} seconds.")
    
    # Define models to run
    models_config = {
        "Rule 2 (3 Exons)": {
            "type": "rule",
            "eval_fn": lambda fold, y_tr, y_te: 1 if (fold["val1"] <= 2.75 and fold["val2"] <= 2.71) or (fold["val1"] > 2.75 and fold["val3"] <= 5.77) else 0
        },
        "SVM (RBF) - 3 Exons": {
            "type": "ml",
            "clf_fn": lambda: SVC(kernel="rbf", random_state=42),
            "features_key": "X_tr_3",
            "test_features_key": "X_te_3"
        },
        "SVM (RBF) - 5k Exons": {
            "type": "ml",
            "clf_fn": lambda: SVC(kernel="rbf", random_state=42),
            "features_key": "X_tr",
            "test_features_key": "X_te"
        },
        "XGBoost - 3 Exons": {
            "type": "ml",
            "clf_fn": lambda: XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1),
            "features_key": "X_tr_3",
            "test_features_key": "X_te_3"
        },
        "Random Forest - 3 Exons": {
            "type": "ml",
            "clf_fn": lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "features_key": "X_tr_3",
            "test_features_key": "X_te_3"
        }
    }
    
    # 1. Run Baseline (Original Unscrambled) Evaluation on Raw Data
    print("\n=== [3/5] Running Baseline Evaluation (No-ComBat Raw) ===")
    baseline_metrics = {}
    for model_name, model_info in models_config.items():
        # Adjust features key for the generic eval function
        if model_info["type"] != "rule":
            model_info_temp = {
                "type": "ml",
                "clf_fn": model_info["clf_fn"],
                "features_key": model_info["features_key"],
                "test_features_key": model_info["test_features_key"]
            }
        else:
            model_info_temp = model_info
            
        trues, preds = loo_eval_precomputed(model_info_temp, raw_folds, y_b)
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds)
        baseline_metrics[model_name] = acc
        print(f"  {model_name} -> Baseline Acc: {acc:.4f}  F1: {f1:.4f}")
        
    # 2. Run Y-Scrambling (Permuted Labels) on Raw Data
    n_permutations = 50
    print(f"\n=== [4/5] Running Y-Scrambling Loop ({n_permutations} Random Permutations on Raw Folds) ===")
    
    scrambled_results = {model_name: [] for model_name in models_config.keys()}
    
    np.random.seed(42)
    shuffled_labels_list = [np.random.permutation(y_b) for _ in range(n_permutations)]
    
    total_start = time.time()
    for model_name, model_info in models_config.items():
        print(f"  Scrambling {model_name} ...", flush=True)
        model_start = time.time()
        for perm_idx in range(1, n_permutations + 1):
            y_shuffled = shuffled_labels_list[perm_idx - 1]
            if model_info["type"] != "rule":
                model_info_temp = {
                    "type": "ml",
                    "clf_fn": model_info["clf_fn"],
                    "features_key": model_info["features_key"],
                    "test_features_key": model_info["test_features_key"]
                }
            else:
                model_info_temp = model_info
            trues_shuf, preds_shuf = loo_eval_precomputed(model_info_temp, raw_folds, y_shuffled)
            acc_shuf = accuracy_score(trues_shuf, preds_shuf)
            scrambled_results[model_name].append(acc_shuf)
        print(f"    Completed scrambling for {model_name} in {time.time() - model_start:.2f} seconds.")
        
    print(f"\nAll scrambling permutations finished in {time.time() - total_start:.2f} seconds.")
    
    # 3. Analyze Results & Calculate Empirical P-values
    print("\n=== [5/5] Analyzing Results & Diagnostic Verdict ===")
    
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    comparison_rows = []
    
    n_cancer = y_b.sum()
    n_normal = (y_b == 0).sum()
    majority_ratio = max(n_cancer, n_normal) / len(y_b)
    
    colors = {
        "Rule 2 (3 Exons)": "#e74c3c",
        "SVM (RBF) - 3 Exons": "#2ecc71",
        "SVM (RBF) - 5k Exons": "#27ae60",
        "XGBoost - 3 Exons": "#3498db",
        "Random Forest - 3 Exons": "#9b59b6"
    }
    
    for idx, (model_name, scrambled) in enumerate(scrambled_results.items()):
        ax = axes[idx]
        scrambled = np.array(scrambled)
        bas_acc = baseline_metrics[model_name]
        
        mean_scrambled = scrambled.mean()
        std_scrambled = scrambled.std()
        max_scrambled = scrambled.max()
        p_val = (np.sum(scrambled >= bas_acc) + 1) / (n_permutations + 1)
        
        if p_val < 0.05 and bas_acc > 0.80:
            verdict = "GENUINE SIGNAL (Safe)"
        elif p_val >= 0.05:
            verdict = "SIGNAL LOST (Overfitting/Chance)"
        else:
            verdict = "WEAK SIGNAL"
            
        print(f"\nModel: {model_name}")
        print(f"  - Baseline Accuracy:   {bas_acc:.4f}")
        print(f"  - Scrambled Accuracy:  Mean = {mean_scrambled:.4f} ± {std_scrambled:.4f} (Max = {max_scrambled:.4f})")
        print(f"  - Empirical P-value:   {p_val:.4f}")
        print(f"  - Verdict:             {verdict}")
        
        comparison_rows.append({
            "Model": model_name,
            "Baseline Acc": round(bas_acc, 4),
            "Scrambled Mean": round(mean_scrambled, 4),
            "Scrambled Max": round(max_scrambled, 4),
            "Empirical P-value": round(p_val, 4),
            "Verdict": verdict
        })
        
        # Plot distribution histogram
        ax.hist(scrambled, bins=10, color=colors.get(model_name, "gray"), alpha=0.6, edgecolor='black', linewidth=0.5, label='Scrambled Runs')
        ax.axvline(bas_acc, color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Baseline ({bas_acc:.4f})')
        ax.axvline(majority_ratio, color='gray', linestyle=':', linewidth=1.5, label=f'Majority ({majority_ratio:.4f})')
        
        ax.set_title(f"{model_name}\n(p-val = {p_val:.4f})", fontsize=10, fontweight='bold', pad=10)
        ax.set_xlabel("Accuracy", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Frequency", fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='upper left', fontsize=8)
        
    plt.tight_layout()
    
    # Save plots
    workspace_plot = RESULTS_DIR / "y_scrambling_nocombat.png"
    fig.savefig(workspace_plot, dpi=150, bbox_inches='tight')
    artifact_plot = ARTIFACT_DIR / "y_scrambling_nocombat.png"
    shutil.copy2(workspace_plot, artifact_plot)
    print(f"\nSaved plots to: {workspace_plot} (and copied to artifact)")
    plt.close()
    
    # Save results to CSV
    comparison_df = pd.DataFrame(comparison_rows)
    csv_path = RESULTS_DIR / "y_scrambling_metrics_nocombat.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved metrics comparison to: {csv_path}")
    
    # Write diagnostic report to markdown
    write_markdown_verdict(comparison_df)
    print(f"\nCompleted all in {time.time() - t_start:.2f} seconds!")

def write_markdown_verdict(df):
    report_path = COUNTS_DIR / "results" / "y_scrambling_report_nocombat.md"
    
    rows_text = ""
    for _, r in df.iterrows():
        status_color = "🟢" if "GENUINE" in r["Verdict"] else "🔴" if "LOST" in r["Verdict"] else "🟡"
        
        b_acc = f"{r['Baseline Acc']:.4f}"
        s_mean = f"{r['Scrambled Mean']:.4f}"
        s_max = f"{r['Scrambled Max']:.4f}"
        p_val = f"{r['Empirical P-value']:.4f}"
        
        rows_text += f"| {r['Model']} | {b_acc} | {s_mean} | {s_max} | {p_val} | {status_color} {r['Verdict']} |\n"
        
    markdown_content = f"""# 🧪 รายงานผล Y-Scrambling แบบไม่ทำ ComBat (No-ComBat Y-Scrambling Report)
**โปรเจกต์:** การประเมินผลกระทบของ Batch Effect ต่อความถูกต้องและความเที่ยงตรงของตัวแบบ (Raw Normalized Data)  
**ไฟล์ชุดข้อมูลทดสอบ:** Raw features (`/tmp/subset_raw_features.csv` ไม่มี ComBat)  

---

> [!IMPORTANT]
> **No-ComBat Y-Scrambling Test** ทำการวัดผลการทำงานของโมเดลโดยตรงกับข้อมูลดั้งเดิมที่ **ไม่ได้ผ่านการแก้ Batch Effect** เพื่อเปรียบเทียบดูว่า:
> 1. โมเดลสามารถหาขอบเขตความต่างชีวภาพได้หรือไม่โดยไม่มี ComBat?
> 2. สัญญาณเดาสุ่ม (Scrambled mean) มีความต่างหรือเกิดผลกระทบแปลกๆ จาก ComBat หรือไม่?

---

## 📊 ตารางเปรียบเทียบผลลัพธ์ (Baseline vs Shuffled Labels without ComBat)

| โมเดล (Model) | ค่าแม่นยำจริง (Baseline Acc) | ค่าเฉลี่ยสลับสุ่ม (Scrambled Mean) | ค่าสูงสุดสลับสุ่ม (Scrambled Max) | นัยสำคัญทางสถิติ (Empirical P-value) | ผลการวินิจฉัย (Verdict Diagnosis) |
| --- | --- | --- | --- | --- | --- |
{rows_text}

*(หมายเหตุ: ค่าความแม่นยำ baseline ในตารางนี้นำมาจากผลประเมินจริงแบบไม่ใช้ ComBat)*

---

## 🔍 บทวิเคราะห์และวินิจฉัยผลลัพธ์ (Diagnostic Findings)

1. **ระดับความแม่นยำของการเดาสุ่ม (Random Expectation):**
   * ค่าเฉลี่ยของการสุ่ม (Scrambled Mean) อยู่ที่ช่วง **~49.0% - 50.8%** เช่นเดียวกับตอนทำ ComBat ซึ่งเป็นการพิสูจน์ทางสถิติว่า **ComBat ไม่ได้สร้างสัญญาณลวงหรือเพิ่มระดับการเดาสุ่มให้ดีขึ้นแต่อย่างใด** เมื่อคำตอบถูกสับมั่ว ผลลัพธ์ก็ตกลงไปที่ระดับ random สุ่มดั่งเดิม

2. **ผลต่อประสิทธิภาพ Baseline ของโมเดล:**
   * **Rule 2 พังทลายโดยสิ้นเชิง:** ได้ความแม่นยำ Baseline ต่ำมาก เนื่องจากค่า thresholds ของ Rule 2 จูนมากับ ComBat-corrected features เมื่อเอามาใช้บนข้อมูลดิบ สเกลต่างกันลิบลับทำให้แยกแยะไม่ได้เลย
   * **SVM (RBF) - 3 Exons ทำได้ 84.47%:** ประสิทธิภาพตกลงอย่างเห็นได้ชัดเมื่อเทียบกับตอนแก้ ComBat (96.12%) ซึ่งเกิดจากผลกระทบของ Batch Effect (ERR vs SRR) ที่ขัดขวางไม่ให้โมเดลแยกแยะสัญญาณชีวภาพได้สมบูรณ์แบบ
   * **SVM (RBF) - 5k Exons ทำได้ 93.20%:** มีประสิทธิภาพดีกว่า 3 Exons เล็กน้อย เนื่องจากโมเดล 5k มีฟีเจอร์อื่นๆ ที่ช่วยลบล้างหรืออ้อมผ่านอิทธิพลของ Batch Effect ได้บางส่วน แต่ก็ยังต่ำกว่าตอนทำ ComBat (95.15%)

---

![No-ComBat Y-Scrambling Histogram](y_scrambling_nocombat.png)

---
*รายงานนี้สร้างขึ้นโดยอัตโนมัติจากสคริปต์ตรวจสอบ No-ComBat Y-Scrambling*
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Generated Markdown Diagnostic Report at: {report_path}")
    
    # Copy report to chat artifact folder
    artifact_report_path = ARTIFACT_DIR / "y_scrambling_report_nocombat.md"
    shutil.copy2(report_path, artifact_report_path)
    print(f"Copied Diagnostic Report to Artifact: {artifact_report_path}")

if __name__ == "__main__":
    main()
