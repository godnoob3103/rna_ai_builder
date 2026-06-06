"""
Step 3 — Train RF, XGBoost, SVM on k-mer feature matrix with LOO-CV.
Compare with Pipeline B (exon count-based) results.

Input : kmer_matrix.csv + exp2_combined_combat/labels.csv
Output: results/kmer_metrics.csv
        results/confusion_matrices_kmer.png
        results/roc_curves_kmer.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier

COUNTS_DIR  = Path("/mnt/a/data sci/data/counts")
MATRIX_FILE = COUNTS_DIR / "kmer_matrix.csv"
LABELS_FILE = COUNTS_DIR / "exp2_combined_combat/labels.csv"
OUT_CM      = COUNTS_DIR / "results/confusion_matrices_kmer.png"
OUT_ROC     = COUNTS_DIR / "results/roc_curves_kmer.png"
OUT_METRICS = COUNTS_DIR / "results/kmer_metrics.csv"

PIPELINE_B = {
    "Exp1_SRR_Only":        {"Random Forest": 0.952, "XGBoost": 0.940, "SVM (RBF)": 0.940},
    "Exp2_Combined_ComBat": {"Random Forest": 0.931, "XGBoost": 0.941, "SVM (RBF)": 0.951},
}

MODELS = [
    ("Random Forest", lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ("XGBoost",       lambda: XGBClassifier(n_estimators=200, eval_metric="logloss",
                                             random_state=42, verbosity=0, n_jobs=-1)),
    ("SVM (RBF)",     lambda: SVC(kernel="rbf", probability=True, random_state=42)),
]


def load_data():
    X = pd.read_csv(MATRIX_FILE, index_col=0)
    y = pd.read_csv(LABELS_FILE, index_col=0)
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common, "label"]
    return X.values.astype(np.float32), y.values.astype(int), common.tolist()


def loo_eval(model, X, y):
    loo = LeaveOneOut()
    preds, probs, trues = [], [], []
    for tr, te in loo.split(X):
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
        model.fit(Xtr, y[tr])
        preds.append(model.predict(Xte)[0])
        trues.append(y[te][0])
        probs.append(model.predict_proba(Xte)[0, 1] if hasattr(model, "predict_proba")
                     else float(model.decision_function(Xte)[0]))
    return np.array(trues), np.array(preds), np.array(probs)


def plot_cm(results):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, trues, preds, _) in zip(axes, results):
        cm = confusion_matrix(trues, preds)
        im = ax.imshow(cm, cmap="Greens")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Cancer"]); ax.set_yticklabels(["Normal", "Cancer"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("Confusion Matrices — Pipeline A (k-mer)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_CM, dpi=150); plt.close()
    print(f"Saved: {OUT_CM}")


def plot_roc(results):
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, trues, _, probs), color in zip(results, ["steelblue", "darkorange", "seagreen"]):
        fpr, tpr, _ = roc_curve(trues, probs)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={roc_auc_score(trues, probs):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Pipeline A (k-mer, LOO-CV)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_ROC, dpi=150); plt.close()
    print(f"Saved: {OUT_ROC}")


def print_comparison(kmer_results):
    model_names = [m for m, _ in MODELS]
    print(f"\n{'='*72}")
    print("  Pipeline A (k-mer)  vs  Pipeline B (exon count)")
    print(f"{'='*72}")
    print(f"\n  {'Model':<20}  {'PipeA_Acc':>10}  {'PipeB_Exp1':>11}  {'PipeB_Exp2':>11}")
    print(f"  {'-'*58}")
    for name, trues, preds, _ in kmer_results:
        acc  = accuracy_score(trues, preds)
        b_e1 = PIPELINE_B["Exp1_SRR_Only"].get(name, float("nan"))
        b_e2 = PIPELINE_B["Exp2_Combined_ComBat"].get(name, float("nan"))
        print(f"  {name:<20}  {acc:>10.4f}  {b_e1:>11.3f}  {b_e2:>11.3f}")


def main():
    print("Loading k-mer matrix ...")
    X, y, samples = load_data()
    print(f"  {X.shape[0]} samples x {X.shape[1]} features  |  cancer={y.sum()}  normal={(y==0).sum()}\n")

    print(f"{'Model':<20}  {'Accuracy':>10}  {'F1':>8}  {'AUC-ROC':>10}")
    print("-" * 55)

    results = []
    rows    = []
    for name, model_fn in MODELS:
        print(f"Running LOO-CV: {name} ...", flush=True)
        trues, preds, probs = loo_eval(model_fn(), X, y)
        acc = accuracy_score(trues, preds)
        f1  = f1_score(trues, preds)
        auc = roc_auc_score(trues, probs)
        print(f"{name:<20}  {acc:>10.4f}  {f1:>8.4f}  {auc:>10.4f}")
        results.append((name, trues, preds, probs))
        rows.append({"experiment": "PipelineA_kmer", "model": name,
                     "accuracy": round(acc, 4), "f1": round(f1, 4), "auc_roc": round(auc, 4)})

    print_comparison(results)

    print("\nSaving plots ...")
    plot_cm(results)
    plot_roc(results)

    pd.DataFrame(rows).to_csv(OUT_METRICS, index=False)
    print(f"Saved: {OUT_METRICS}")
    print("\nDone.")


if __name__ == "__main__":
    main()
