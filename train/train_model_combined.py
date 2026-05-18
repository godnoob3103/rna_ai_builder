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

COUNTS_DIR = Path("A:/data sci/data/counts")

DATASETS = {
    "Step1_SRR":       (COUNTS_DIR / "srr_matrix_filtered.csv",    COUNTS_DIR / "srr_labels_filtered.csv"),
    "Step2_Combined":  (COUNTS_DIR / "combined_matrix_combat.csv", COUNTS_DIR / "combined_labels_103.csv"),
}

MODELS = [
    ("Random Forest", lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ("XGBoost",       lambda: XGBClassifier(n_estimators=200, eval_metric="logloss",
                                             random_state=42, verbosity=0, n_jobs=-1)),
    ("SVM (RBF)",     lambda: SVC(kernel="rbf", probability=True, random_state=42)),
]


def load_data(matrix_path, labels_path):
    X = pd.read_csv(matrix_path, index_col=0)
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
            probs.append(model.predict_proba(X_te)[0, 1])
        else:
            probs.append(model.decision_function(X_te)[0])
    return np.array(trues), np.array(preds), np.array(probs)


def run_dataset(tag, matrix_path, labels_path):
    X, y = load_data(matrix_path, labels_path)
    print(f"\n{'='*60}")
    print(f"  {tag}  —  {X.shape[0]} samples × {X.shape[1]:,} features")
    print(f"  cancer={y.sum()}  normal={(y==0).sum()}")
    print(f"{'='*60}")
    print(f"  {'Model':<20}  {'Accuracy':>10}  {'F1':>8}  {'AUC-ROC':>10}")
    print(f"  {'-'*53}")

    results = {}
    for model_name, model_fn in MODELS:
        print(f"  Running LOO-CV: {model_name} ...", flush=True)
        trues, preds, probs = loo_eval(model_fn(), X, y)
        acc = accuracy_score(trues, preds)
        f1  = f1_score(trues, preds)
        auc = roc_auc_score(trues, probs)
        print(f"  {model_name:<20}  {acc:>10.4f}  {f1:>8.4f}  {auc:>10.4f}")
        results[model_name] = dict(acc=acc, f1=f1, auc=auc, trues=trues, preds=preds, probs=probs)

    return results


def plot_all(all_results):
    tags      = list(all_results.keys())
    tag_short = {"Step1_SRR": "Step1\n(SRR only)", "Step2_Combined": "Step2\n(Combined+ComBat)"}
    model_names = [m for m, _ in MODELS]
    colors      = ["steelblue", "darkorange", "seagreen"]

    # ── confusion matrices (2 rows × 3 cols) ─────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for row, tag in enumerate(tags):
        for col, model_name in enumerate(model_names):
            r   = all_results[tag][model_name]
            cm  = confusion_matrix(r["trues"], r["preds"])
            ax  = axes[row, col]
            im  = ax.imshow(cm, cmap="Blues")
            ax.set_title(f"{model_name}\n{tag_short[tag]}", fontsize=10)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Normal", "Cancer"])
            ax.set_yticklabels(["Normal", "Cancer"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            fontsize=13, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("Confusion Matrices — Step1 vs Step2", fontsize=13)
    plt.tight_layout()
    path = COUNTS_DIR / "confusion_matrices_combined.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

    # ── ROC curves ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, tag in zip(axes, tags):
        for (model_name, _), color in zip(MODELS, colors):
            r = all_results[tag][model_name]
            fpr, tpr, _ = roc_curve(r["trues"], r["probs"])
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{model_name} (AUC={r['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {tag_short[tag].replace(chr(10), ' ')}")
        ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = COUNTS_DIR / "roc_curves_combined.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def print_comparison(all_results):
    model_names = [m for m, _ in MODELS]
    metrics     = [("Accuracy", "acc"), ("F1", "f1"), ("AUC-ROC", "auc")]

    print(f"\n{'='*70}")
    print("  COMPARISON: Step1 (SRR only)  vs  Step2 (Combined + ComBat)")
    print(f"{'='*70}")

    for metric_label, key in metrics:
        print(f"\n  {metric_label}")
        print(f"  {'Model':<20}  {'Step1_SRR':>12}  {'Step2_Combined':>16}  {'diff':>8}")
        print(f"  {'-'*62}")
        for model_name in model_names:
            v1   = all_results["Step1_SRR"][model_name][key]
            v2   = all_results["Step2_Combined"][model_name][key]
            diff = v2 - v1
            sign = "+" if diff >= 0 else ""
            print(f"  {model_name:<20}  {v1:>12.4f}  {v2:>16.4f}  {sign}{diff:>7.4f}")


def main():
    all_results = {}
    for tag, (matrix_path, labels_path) in DATASETS.items():
        all_results[tag] = run_dataset(tag, matrix_path, labels_path)

    print_comparison(all_results)

    print("\nSaving plots ...")
    plot_all(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
