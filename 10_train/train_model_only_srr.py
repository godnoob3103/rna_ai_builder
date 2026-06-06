import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier

COUNTS_DIR  = Path("A:/data sci/data/counts")
MATRIX_FILE = COUNTS_DIR / "exp1_srr_only" / "features.csv"
LABELS_FILE = COUNTS_DIR / "exp1_srr_only" / "labels.csv"
OUT_CM      = COUNTS_DIR / "results" / "confusion_matrices.png"
OUT_ROC     = COUNTS_DIR / "results" / "roc_curves.png"
OUT_METRICS = COUNTS_DIR / "results" / "metrics.csv"


def load_data():
    X = pd.read_csv(MATRIX_FILE, index_col=0)
    y = pd.read_csv(LABELS_FILE, index_col=0)
    y = y.loc[X.index, "label"]
    return X.values.astype(np.float32), y.values.astype(int), X.index.tolist()


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


def report(name, trues, preds, probs):
    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds)
    auc = roc_auc_score(trues, probs)
    print(f"{name:<20}  Accuracy={acc:.4f}  F1={f1:.4f}  AUC-ROC={auc:.4f}")
    return acc, f1, auc


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, trues, preds, _) in zip(axes, results):
        cm = confusion_matrix(trues, preds)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Cancer"])
        ax.set_yticklabels(["Normal", "Cancer"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(OUT_CM, dpi=150)
    plt.close()
    print(f"Saved: {OUT_CM}")


def plot_roc_curves(results):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["steelblue", "darkorange", "seagreen"]
    for (name, trues, _, probs), color in zip(results, colors):
        fpr, tpr, _ = roc_curve(trues, probs)
        auc = roc_auc_score(trues, probs)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Leave-One-Out CV")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_ROC, dpi=150)
    plt.close()
    print(f"Saved: {OUT_ROC}")


def main():
    print("Loading data...")
    X, y, samples = load_data()
    print(f"  {X.shape[0]} samples × {X.shape[1]:,} features  |  "
          f"cancer={y.sum()}  normal={(y==0).sum()}\n")

    models = [
        ("Random Forest",  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("XGBoost",        XGBClassifier(n_estimators=200, use_label_encoder=False,
                                         eval_metric="logloss", random_state=42,
                                         verbosity=0, n_jobs=-1)),
        ("SVM (RBF)",      SVC(kernel="rbf", probability=True, random_state=42)),
    ]

    print(f"{'Model':<20}  {'Accuracy':>10}  {'F1':>8}  {'AUC-ROC':>10}")
    print("-" * 55)

    results = []
    for name, model in models:
        print(f"Running LOO-CV: {name} ...", flush=True)
        trues, preds, probs = loo_eval(model, X, y)
        report(name, trues, preds, probs)
        results.append((name, trues, preds, probs))

    print("\nSaving plots...")
    plot_confusion_matrices(results)
    plot_roc_curves(results)

    rows = []
    for name, trues, preds, probs in results:
        rows.append({
            "experiment": "Exp1_SRR_Only",
            "model":      name,
            "accuracy":   round(accuracy_score(trues, preds), 4),
            "f1":         round(f1_score(trues, preds), 4),
            "auc_roc":    round(roc_auc_score(trues, probs), 4),
        })
    pd.DataFrame(rows).to_csv(OUT_METRICS, index=False)
    print(f"Saved: {OUT_METRICS}")
    print("\nDone.")


if __name__ == "__main__":
    main()
