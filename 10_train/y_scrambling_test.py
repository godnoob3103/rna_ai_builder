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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Directories
COUNTS_DIR = Path("/mnt/a/data sci/data/counts")
RESULTS_DIR = COUNTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Artifact directory for chat UI embedding
ARTIFACT_DIR = Path("/root/.gemini/antigravity-cli/brain/2be5a522-0319-4917-8f7b-69024882fc86")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
MATRIX_PATH = COUNTS_DIR / "exp2_combined_combat" / "features.csv"
LABELS_PATH = COUNTS_DIR / "matrices" / "combined_labels.csv"

# Models configuration (Baseline uses these)
MODELS = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost":       lambda: XGBClassifier(n_estimators=200, eval_metric="logloss",
                                             random_state=42, verbosity=0, n_jobs=-1),
    "SVM (RBF)":     lambda: SVC(kernel="rbf", probability=True, random_state=42),
}

# Fast scrambled models configuration (SVM probability=False for 5x speedup, RF & XGB remain identical)
SCRAMBLED_MODELS = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost":       lambda: XGBClassifier(n_estimators=200, eval_metric="logloss",
                                             random_state=42, verbosity=0, n_jobs=-1),
    "SVM (RBF)":     lambda: SVC(kernel="rbf", probability=False, random_state=42),
}

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

def load_data(matrix_path, labels_path):
    X = fast_load_features(matrix_path)
    y = pd.read_csv(labels_path, index_col=0)
    y = y.loc[X.index, "label"]
    return X.values.astype(np.float32), y.values.astype(int)

def loo_eval(model, X, y):
    loo = LeaveOneOut()
    preds, probs, trues = [], [], []
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        model.fit(X_tr, y_tr)
        preds.append(model.predict(X_te)[0])
        trues.append(y[test_idx][0])
        if hasattr(model, "predict_proba"):
            try:
                probs.append(model.predict_proba(X_te)[0, 1])
            except Exception:
                probs.append(0.5)
        elif hasattr(model, "decision_function"):
            probs.append(model.decision_function(X_te)[0])
        else:
            probs.append(0.5)
    return np.array(trues), np.array(preds), np.array(probs)

def main():
    print("=== [1/5] Starting Full Y-Scrambling (Target Permutation) Diagnostic Test ===")
    print(f"Loading features from: {MATRIX_PATH}")
    print(f"Loading labels from:   {LABELS_PATH}")
    X, y = load_data(MATRIX_PATH, LABELS_PATH)
    
    n_samples, n_features = X.shape
    n_cancer = y.sum()
    n_normal = (y == 0).sum()
    print(f"Loaded {n_samples} samples with {n_features} features.")
    print(f"Class distribution: Cancer = {n_cancer}, Healthy = {n_normal}")
    
    # 1. Run Baseline (Original Unscrambled) Evaluation
    print("\n=== [2/5] Running Baseline LOO-CV (True Labels) ===")
    baseline_metrics = {}
    for model_name, model_fn in MODELS.items():
        print(f"  Evaluating baseline: {model_name} ...", flush=True)
        trues, preds, probs = loo_eval(model_fn(), X, y)
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds)
        try:
            auc = roc_auc_score(trues, probs)
        except Exception:
            auc = np.nan
        baseline_metrics[model_name] = acc
        print(f"    -> Baseline Accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")

    # 2. Run Y-Scrambling (Permuted Labels) for all models
    n_permutations = 50
    print(f"\n=== [3/5] Running Y-Scrambling Loop ({n_permutations} Random Permutations on ALL Models) ===")
    print("Labels will be randomly shuffled, and LOO-CV will be repeated for each model.")
    print("This will take about 1.5 hours to finish.")
    
    scrambled_results = {model_name: [] for model_name in SCRAMBLED_MODELS.keys()}
    
    # Pre-generate shuffled labels for strict parity across models
    np.random.seed(42)
    shuffled_labels_list = [np.random.permutation(y) for _ in range(n_permutations)]
    
    total_start = time.time()
    for model_name, model_fn in SCRAMBLED_MODELS.items():
        print(f"\n--- Scrambling {model_name} ---")
        model_start = time.time()
        for perm_idx in range(1, n_permutations + 1):
            perm_start = time.time()
            y_shuffled = shuffled_labels_list[perm_idx - 1]
            
            trues_shuf, preds_shuf, _ = loo_eval(model_fn(), X, y_shuffled)
            acc_shuf = accuracy_score(trues_shuf, preds_shuf)
            scrambled_results[model_name].append(acc_shuf)
            
            print(f"  [{model_name}] Permutation {perm_idx}/{n_permutations} Completed (Time: {time.time() - perm_start:.2f}s, Acc: {acc_shuf:.4f})", flush=True)
        print(f"Finished scrambling for {model_name} in {time.time() - model_start:.2f} seconds.")
        
    print(f"\nFinished all {n_permutations} permutations for all models in {time.time() - total_start:.2f} seconds.")
    
    # 3. Analyze Results & Calculate Empirical P-values
    print("\n=== [4/5] Analyzing Results & Diagnostic Verdict ===")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    comparison_rows = []
    
    majority_ratio = max(n_cancer, n_normal) / n_samples
    colors = {"Random Forest": "#2ecc71", "XGBoost": "#3498db", "SVM (RBF)": "#9b59b6"}
    
    for idx, (model_name, scrambled) in enumerate(scrambled_results.items()):
        ax = axes[idx]
        scrambled = np.array(scrambled)
        bas_acc = baseline_metrics[model_name]
        
        mean_scrambled = scrambled.mean()
        std_scrambled = scrambled.std()
        max_scrambled = scrambled.max()
        p_val = (np.sum(scrambled >= bas_acc) + 1) / (n_permutations + 1)
        
        if p_val < 0.05 and bas_acc > 0.85:
            verdict = "GENUINE SIGNAL (No significant leakage/overfitting)"
        else:
            verdict = "WARNING: Potential Leakage or Overfitting detected!"
            
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
        ax.hist(scrambled, bins=10, color=colors[model_name], alpha=0.6, edgecolor='black', linewidth=0.5, label='Scrambled Runs')
        ax.axvline(bas_acc, color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Baseline ({bas_acc:.4f})')
        ax.axvline(majority_ratio, color='gray', linestyle=':', linewidth=1.5, label=f'Majority Class ({majority_ratio:.4f})')
        
        ax.set_title(f"{model_name}\n(p-value = {p_val:.4f})", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Cross-Validation Accuracy", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='upper left', fontsize=9)
        
    plt.tight_layout()
    
    # Save plots
    workspace_plot = RESULTS_DIR / "y_scrambling_test.png"
    fig.savefig(workspace_plot, dpi=150, bbox_inches='tight')
    artifact_plot = ARTIFACT_DIR / "y_scrambling_test.png"
    shutil.copy2(workspace_plot, artifact_plot)
    print(f"\nSaved plots to: {workspace_plot} (and copied to artifact)")
    plt.close()
    
    # 4. Save results to CSV comparison
    comparison_df = pd.DataFrame(comparison_rows)
    csv_path = RESULTS_DIR / "y_scrambling_metrics.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved metrics comparison to: {csv_path}")
    
    # 5. Write diagnostic report to markdown
    write_markdown_verdict(comparison_df)
    print("\n=== Y-SCRAMBLING DIAGNOSTIC COMPLETED! ===")

def write_markdown_verdict(df):
    report_path = COUNTS_DIR / "results" / "y_scrambling_report.md"
    
    rows_text = ""
    for _, r in df.iterrows():
        status_color = "🟢" if "GENUINE" in r["Verdict"] else "🔴"
        
        b_acc = f"{r['Baseline Acc']:.4f}"
        s_mean = f"{r['Scrambled Mean']:.4f}"
        s_max = f"{r['Scrambled Max']:.4f}"
        p_val = f"{r['Empirical P-value']:.4f}"
        
        rows_text += f"| {r['Model']} | {b_acc} | {s_mean} | {s_max} | {p_val} | {status_color} {r['Verdict']} |\n"
        
    markdown_content = f"""# 🧪 รายงานผลการตรวจสอบความเที่ยงตรง (Y-Scrambling Diagnostic Report)
**โปรเจกต์:** การประเมิน Data Leakage / Overfitting ของตัวแบบจำแนกประเภทมะเร็ง  
**ไฟล์ชุดข้อมูลทดสอบ:** `/exp2_combined_combat/features.csv`  

---

> [!IMPORTANT]
> **Y-Scrambling Test (Target Permutation Test)** เป็นเกณฑ์มาตรฐานสูงสุดในการตรวจสอบหาสัญญาณหลอก (Data Leakage หรือ Overfitting) โดยการทำลายความเชื่อมโยงระหว่างระดับยีนจริงกับการวินิจฉัยโรคจริงด้วยการสลับคำตอบ (Label `y`) แบบสุ่มมั่วทั้งหมด แล้วสั่งเทรนและประเมินผลผ่าน LOO-CV 50 รอบ เพื่อวัดระดับความแม่นยำทางสถิติเมื่อโมเดลเรียนรู้จาก "สัญญาณรบกวนสุ่ม (Pure Random Noise)"

---

## 📊 ตารางเปรียบเทียบผลลัพธ์ (Baseline vs Shuffled Labels)

| โมเดล (Model) | ค่าแม่นยำจริง (Baseline Acc) | ค่าเฉลี่ยสลับสุ่ม (Scrambled Mean) | ค่าสูงสุดสลับสุ่ม (Scrambled Max) | นัยสำคัญทางสถิติ (Empirical P-value) | ผลการวินิจฉัย (Verdict Diagnosis) |
| --- | --- | --- | --- | --- | --- |
{rows_text}

---

## 🔍 บทวิเคราะห์และวินิจฉัยผลลัพธ์ (Diagnostic Findings)

1. **ระดับความแม่นยำของการเดาสุ่ม (Random Expectation):**
   * เมื่อสลับคำตอบมั่ว ค่าเฉลี่ยความแม่นยำ (Scrambled Mean) ของทุกโมเดลควรจะตกลงมาอยู่ที่ระดับประมาณ **50% - 55%** ซึ่งเป็นระดับการคาดการณ์ทางสถิติปกติของการเดาสุ่ม (Majority class ratio)
   * หากค่าเฉลี่ยสลับสุ่มยังคงทำได้สูง (เช่น >80%) จะเป็นสัญญาณเตือนว่ามี **Data Leakage** แทรกแซงในลูป Cross-Validation
2. **ค่าระดับนัยสำคัญ Empirical P-value:**
   * คำนวณตามสูตร: $p = \\frac{{\\text{{จำนวนครั้งสลับสุ่มที่แม่นยำกว่าหรือเท่ากับจริง}} + 1}}{{N + 1}}$
   * หากค่า $p < 0.05$ ถือว่าสัญญาณที่โมเดล Baseline เรียนรู้ได้มีความหมายทางสถิติอย่างยิ่งและไม่ได้เกิดขึ้นโดยบังเอิญ

---

![Y-Scrambling Histogram](y_scrambling_test.png)

---
*รายงานนี้สร้างขึ้นโดยอัตโนมัติจากสคริปต์ตรวจสอบ Y-Scrambling*
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Generated Markdown Diagnostic Report at: {report_path}")
    
    # Copy report to chat artifact folder
    artifact_report_path = ARTIFACT_DIR / "y_scrambling_report.md"
    shutil.copy2(report_path, artifact_report_path)
    print(f"Copied Diagnostic Report to Artifact: {artifact_report_path}")

if __name__ == "__main__":
    main()
