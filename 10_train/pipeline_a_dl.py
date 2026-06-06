#!/usr/bin/env python3
"""
Pipeline A — Deep Learning on Raw RNA-seq Reads (FASTQ)

Strategy:
  1. For each sample, reservoir-sample 10,000 reads from _1 + _2 FASTQ files.
  2. One-hot encode each read (A=0001 T=0010 G=0100 C=1000 N=0000), pad/trim to 100bp.
  3. Keep each read separate → shape per sample: (10000, 100, 4).
  4. Process reads independently inside CNN / LSTM / Transformer models using a memory-efficient
     chunked forward pass, and mean-pool the features inside the model prior to fully connected layers.
  5. Train with 102-fold LOO-CV on GPU, using PyTorch DataLoaders with mini-batch size = 16.
  6. Save encoded data to /mnt/a/data sci/data/counts/encoded_reads.pt after the first run.

Output (results/):
  dl_metrics.csv, roc_curves_dl.png, confusion_matrices_dl.png

Run with:
  nohup python3 pipeline_a_dl.py > pipeline_a_dl.log 2>&1 &
"""

import gzip
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
COUNTS_DIR  = Path("/mnt/a/data sci/data/counts")
LABELS_FILE = COUNTS_DIR / "exp2_combined_combat/labels.csv"
ENCODED_FILE = COUNTS_DIR / "encoded_reads.pt"
OUT_DIR     = COUNTS_DIR / "results"

FASTQ_DIRS = [
    Path("/mnt/a/data sci/data/trimmed/cancer"),
    Path("/mnt/a/data sci/data/trimmed/non_cancer"),
    Path("/mnt/a/data/lung_cancer/trimmed/tumor"),
    Path("/mnt/a/data/lung_cancer/trimmed/normal"),
]

# ── Hyper-parameters ───────────────────────────────────────────────────────────
N_READS  = 10000  # max reads sampled per sample (both files combined)
READ_LEN = 100    # bp — reads are padded (N) or trimmed to this length
EPOCHS   = 200
LR       = 1e-3
SEED     = 42
BATCH_SIZE = 16

# Pipeline B baselines (Exp2 Combined ComBat, LOO-CV AUC)
PIPELINE_B = {"Random Forest": 0.931, "XGBoost": 0.941, "SVM (RBF)": 0.951}

# ── One-hot encoding ───────────────────────────────────────────────────────────
# A=0001  T=0010  G=0100  C=1000  N=0000
_BASE_MAP: dict[str, list[int]] = {
    "A": [0, 0, 0, 1],
    "T": [0, 0, 1, 0],
    "G": [0, 1, 0, 0],
    "C": [1, 0, 0, 0],
}


def one_hot_read(seq: str, length: int = READ_LEN) -> np.ndarray:
    seq = seq[:length].upper()
    arr = np.zeros((length, 4), dtype=np.float32)
    for i, b in enumerate(seq):
        enc = _BASE_MAP.get(b)
        if enc:
            arr[i] = enc
    return arr  # (length, 4) — missing positions stay 0000 (N)


# ── FASTQ helpers ──────────────────────────────────────────────────────────────
def find_fastq_pair(sample_id: str) -> list[Path] | None:
    for d in FASTQ_DIRS:
        f1 = d / f"{sample_id}_1.fastq.gz"
        f2 = d / f"{sample_id}_2.fastq.gz"
        if f1.exists() and f2.exists():
            return [f1, f2]
    return None


def reservoir_sample(paths: list[Path], n: int, seed: int) -> list[str]:
    """Reservoir-sample n sequence lines across one or more FASTQ.gz files."""
    rng = random.Random(seed)
    reservoir: list[str] = []
    count = 0
    max_scan_reads = 50000  # Scan at most 50,000 reads per file to avoid hours of slow gzip reading
    for path in paths:
        with gzip.open(path, "rt", errors="replace") as fh:
            line_num = 0
            reads_scanned = 0
            for line in fh:
                line_num += 1
                if line_num % 4 != 2:   # FASTQ line 2 of every 4 is the sequence
                    continue
                count += 1
                seq = line.strip()
                if len(reservoir) < n:
                    reservoir.append(seq)
                else:
                    j = rng.randint(0, count - 1)
                    if j < n:
                        reservoir[j] = seq
                reads_scanned += 1
                if reads_scanned >= max_scan_reads:
                    break
    return reservoir


def encode_sample(sample_id: str) -> np.ndarray | None:
    """Return one-hot representation shape (N_READS, READ_LEN, 4) as uint8, or None."""
    pair = find_fastq_pair(sample_id)
    if pair is None:
        return None
    reads = reservoir_sample(pair, N_READS, seed=SEED)
    if not reads:
        return None
    encoded_list = []
    for r in reads:
        encoded_list.append(one_hot_read(r))
    # Pad with all-zeros reads if fewer reads than N_READS are available
    while len(encoded_list) < N_READS:
        encoded_list.append(np.zeros((READ_LEN, 4), dtype=np.float32))
    encoded = np.stack(encoded_list).astype(np.uint8)  # Save as uint8 to save disk & RAM space
    return encoded


# ── Chunked Feature Aggregation Helper ─────────────────────────────────────────
def extract_read_features(feature_extractor, x: torch.Tensor, chunk_size: int = 1000) -> torch.Tensor:
    """
    x: shape (B, N_reads, L, 4)
    Processes reads in chunks of size chunk_size to avoid GPU OOM. Uses PyTorch gradient
    checkpointing during training to dramatically reduce peak GPU memory and avoid swapping.
    Returns: mean pooled features of shape (B, D)
    """
    B, N_reads, L, C = x.shape
    sum_features = None

    for i in range(0, N_reads, chunk_size):
        chunk = x[:, i : i + chunk_size]  # (B, actual_chunk_size, L, C)
        actual_chunk_size = chunk.shape[1]

        # Flatten batch and chunk dimensions
        chunk_flat = chunk.reshape(-1, L, C)  # (B * actual_chunk_size, L, C)

        # Use gradient checkpointing if tracking gradients to avoid storing activation history
        if chunk_flat.requires_grad:
            feats = checkpoint.checkpoint(feature_extractor, chunk_flat, use_reentrant=False)
        else:
            feats = feature_extractor(chunk_flat)  # (B * actual_chunk_size, D)

        # Reshape features back to (B, actual_chunk_size, D)
        feats = feats.view(B, actual_chunk_size, -1)

        # Sum features across the reads in this chunk
        chunk_sum = feats.sum(dim=1)  # (B, D)

        if sum_features is None:
            sum_features = chunk_sum
        else:
            sum_features = sum_features + chunk_sum

    return sum_features / N_reads


# ── Models ─────────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=8, padding="same")
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, padding="same")
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(64, 1)

    def extract_features(self, x):  # x: (N, L, 4)
        x = x.permute(0, 2, 1)      # (N, 4, L)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1) # (N, 64)
        return x

    def forward(self, x):  # x: (B, N_reads, L, 4)
        feats = extract_read_features(self.extract_features, x, chunk_size=1000)
        return self.fc(self.drop(feats)).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4, hidden_size=64,
            num_layers=2, batch_first=True,
            dropout=0.3, bidirectional=True,
        )
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(128, 1)  # 64 fwd + 64 bwd

    def extract_features(self, x):  # x: (N, L, 4)
        _, (h, _) = self.lstm(x)
        # h: (num_layers*2, N, 64) — take last-layer fwd + bwd
        h = torch.cat([h[-2], h[-1]], dim=-1)  # (N, 128)
        return h

    def forward(self, x):  # x: (B, N_reads, L, 4)
        feats = extract_read_features(self.extract_features, x, chunk_size=1000)
        return self.fc(self.drop(feats)).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj    = nn.Linear(4, 32)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.fc      = nn.Linear(32, 1)

    def extract_features(self, x):  # x: (N, L, 4)
        x = self.proj(x)             # (N, L, 32)
        x = self.encoder(x)
        x = x.mean(dim=1)            # global avg pool over sequence: (N, 32)
        return x

    def forward(self, x):  # x: (B, N_reads, L, 4)
        feats = extract_read_features(self.extract_features, x, chunk_size=1000)
        return self.fc(feats).squeeze(-1)


# ── Training helpers ───────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LR,
) -> None:
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Calculate pos_weight for BCEWithLogitsLoss to handle imbalance
    neg_count = (y_train == 0).sum().item()
    pos_count = (y_train == 1).sum().item()
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            # Move to device and cast uint8 to float32 on the GPU
            batch_X = batch_X.to(device=device, dtype=torch.float32)
            batch_X.requires_grad = True  # Enable gradient tracking on input to trigger checkpointing
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()


@torch.no_grad()
def predict_prob(model: nn.Module, X: torch.Tensor, device: torch.device) -> float:
    model.eval()
    # Move to device and cast to float32
    X = X.to(device=device, dtype=torch.float32)
    logit = model(X)
    return float(torch.sigmoid(logit).item())


# ── LOO-CV ─────────────────────────────────────────────────────────────────────
def loo_cv(
    model_cls,
    X: torch.Tensor,
    y: torch.Tensor,
    model_name: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    n      = len(y)
    y_true = np.empty(n, dtype=np.float32)
    y_prob = np.empty(n, dtype=np.float32)
    t0     = time.time()

    for i in range(n):
        mask   = torch.arange(n) != i
        X_tr   = X[mask]
        y_tr   = y[mask]

        torch.manual_seed(SEED)
        model = model_cls()
        train_model(model, X_tr, y_tr, device)
        y_prob[i] = predict_prob(model, X[i:i+1], device)
        y_true[i] = y[i].item()

        if (i + 1) % 10 == 0 or i == n - 1:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            print(
                f"    [{i+1:3d}/{n}]  elapsed {elapsed:5.0f}s  ETA {eta:5.0f}s",
                flush=True,
            )

    return y_true, y_prob


# ── Plotting ───────────────────────────────────────────────────────────────────
_COLORS = {"CNN": "#e6194b", "LSTM": "#3cb44b", "Transformer": "#4363d8"}


def plot_roc_curves(results: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_prob) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, color=_COLORS[name], label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Pipeline A — ROC Curves (LOO-CV)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_confusion_matrices(results: dict, out_path: Path) -> None:
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    for ax, name in zip(axes, names):
        y_true, y_prob = results[name]
        y_pred = (y_prob >= 0.5).astype(int)
        cm     = confusion_matrix(y_true, y_pred)
        im     = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Cancer"])
        ax.set_yticklabels(["Normal", "Cancer"])
        for r in range(2):
            for c in range(2):
                ax.text(c, r, cm[r, c], ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[r, c] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Confusion Matrices — Pipeline A (DL, LOO-CV)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Detect GPU / CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load labels ──
    labels  = pd.read_csv(LABELS_FILE, index_col=0)
    samples = labels.index.tolist()
    n_all   = len(samples)
    print(f"Samples: {n_all}  |  cancer={int(labels['label'].sum())}  "
          f"normal={int((labels['label']==0).sum())}")
    print(f"Reads per sample: {N_READS}  |  Read length: {READ_LEN}bp\n")

    # ── Phase 1: Check cache or encode all samples ──
    if ENCODED_FILE.exists():
        print("=" * 60)
        print(f"Loading cached encoded dataset from {ENCODED_FILE}")
        print("=" * 60)
        cached_data = torch.load(ENCODED_FILE)
        X = cached_data["X"]
        y = cached_data["y"]
        valid_ids = cached_data["valid_ids"]
        M = len(valid_ids)
        print(f"Loaded {M} samples → X shape: {X.shape}, y shape: {y.shape}")
    else:
        print("=" * 60)
        print("Phase 1: Encoding FASTQ samples")
        print("=" * 60)
        X_list, y_list, valid_ids, missing = [], [], [], []
        for i, sid in enumerate(samples, 1):
            print(f"  [{i:3d}/{n_all}] {sid} ...", end=" ", flush=True)
            enc = encode_sample(sid)
            if enc is None:
                print("FASTQ NOT FOUND — skipped")
                missing.append(sid)
                continue
            X_list.append(enc)
            y_list.append(float(labels.loc[sid, "label"]))
            valid_ids.append(sid)
            print("done")

        if not X_list:
            sys.exit("No samples encoded — check FASTQ paths.")

        X = torch.tensor(np.stack(X_list), dtype=torch.uint8)
        y = torch.tensor(y_list, dtype=torch.float32)
        M = len(valid_ids)
        print(f"\nEncoded {M}/{n_all} samples → X shape: {X.shape}")
        if missing:
            print(f"Skipped ({len(missing)}): {', '.join(missing)}")

        print(f"Saving encoded dataset to {ENCODED_FILE} ...", flush=True)
        torch.save({"X": X, "y": y, "valid_ids": valid_ids}, ENCODED_FILE)
        print("Saved cache successfully.")

    # ── Phase 2: LOO-CV for each model ──
    model_registry = {
        "CNN":         CNN,
        "LSTM":        LSTMModel,
        "Transformer": TransformerModel,
    }
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for model_name, model_cls in model_registry.items():
        print(f"\n{'='*60}")
        print(f"Phase 2: LOO-CV — {model_name}  ({M} folds × {EPOCHS} epochs)")
        print("=" * 60)
        y_true, y_prob = loo_cv(model_cls, X, y, model_name, device)
        results[model_name] = (y_true, y_prob)

        acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
        f1  = f1_score(y_true, (y_prob >= 0.5).astype(int))
        auc = roc_auc_score(y_true, y_prob)
        print(f"  → Accuracy={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    # ── Save metrics CSV ──
    rows = []
    for name, (y_true, y_prob) in results.items():
        y_pred = (y_prob >= 0.5).astype(int)
        rows.append({
            "pipeline": "A (DL)",
            "model":    name,
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1":       round(float(f1_score(y_true, y_pred)), 4),
            "auc":      round(float(roc_auc_score(y_true, y_prob)), 4),
        })
    for name, auc in PIPELINE_B.items():
        rows.append({"pipeline": "B (ML)", "model": name,
                     "accuracy": None, "f1": None, "auc": auc})

    metrics_df  = pd.DataFrame(rows)
    metrics_csv = OUT_DIR / "dl_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved: {metrics_csv}")

    # ── Print comparison table ──
    print(f"\n{'='*62}")
    print("  Pipeline A (Deep Learning) vs Pipeline B (ML) — LOO-CV AUC")
    print(f"{'='*62}")
    print(f"  {'Model':<16} {'Pipeline':<12} {'Accuracy':>9} {'F1':>7} {'AUC':>7}")
    print(f"  {'-'*56}")
    for _, row in metrics_df.iterrows():
        acc = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "   —  "
        f1  = f"{row['f1']:.4f}"       if row["f1"]       is not None else "  —  "
        print(f"  {row['model']:<16} {row['pipeline']:<12} {acc:>9} {f1:>7} {row['auc']:>7.4f}")
    print("=" * 62)

    # ── Plots ──
    plot_roc_curves(results, OUT_DIR / "roc_curves_dl.png")
    plot_confusion_matrices(results, OUT_DIR / "confusion_matrices_dl.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
