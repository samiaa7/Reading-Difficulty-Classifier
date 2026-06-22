from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import build_dataset, FEATURE_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("nn_model.pth")
PLOTS_DIR  = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmotionalLoadNet(nn.Module):


    def __init__(self, input_dim: int = 15, dropout1: float = 0.3, dropout2: float = 0.2):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout1),

            # Layer 2
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout2),

            # Layer 3
            nn.Linear(32, 16),
            nn.ReLU(),

            # Output
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Weight initialisation — He init for ReLU layers
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

def make_loaders(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
    ye_train: np.ndarray,
    ye_val:   np.ndarray,
    ye_test:  np.ndarray,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    def to_ds(X, y):
        return TensorDataset(
            torch.tensor(X,  dtype=torch.float32),
            torch.tensor(y,  dtype=torch.float32),
        )

    train_loader = DataLoader(to_ds(X_train, ye_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(to_ds(X_val,   ye_val),   batch_size=batch_size)
    test_loader  = DataLoader(to_ds(X_test,  ye_test),  batch_size=batch_size)
    return train_loader, val_loader, test_loader


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def train_epoch(
    model: EmotionalLoadNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        # Gradient clipping — prevents exploding gradients on small dataset
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(
    model: EmotionalLoadNet,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            all_preds.append(preds.cpu().numpy())
            all_true.append(y_batch.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(all_preds),
        np.concatenate(all_true),
    )


def train(
    epochs:     int   = 100,
    lr:         float = 1e-3,
    batch_size: int   = 64,
    patience:   int   = 15,
) -> None:
    logger.info("=" * 55)
    logger.info("PYTORCH — Emotional Load Regression")
    logger.info("Device: %s", DEVICE)
    logger.info("=" * 55)

    # ── Data ──
    (X_train, X_val, X_test,
     yd_train, yd_val, yd_test,
     ye_train, ye_val, ye_test,
     scaler) = build_dataset(wikilarge_path="data/wikilarge.txt.src", save=False)   # scaler already saved by sklearn trainer

    train_loader, val_loader, test_loader = make_loaders(
        X_train, X_val, X_test,
        ye_train, ye_val, ye_test,
        batch_size=batch_size,
    )

    logger.info("Train: %d  Val: %d  Test: %d", len(X_train), len(X_val), len(X_test))

    # ── Model ──
    model     = EmotionalLoadNet(input_dim=len(FEATURE_NAMES)).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # ── Training loop ──
    best_val_loss = float("inf")
    patience_ctr  = 0
    train_losses, val_losses = [], []
    best_state = None

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, val_preds, val_true = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if epoch % 10 == 0 or epoch == 1:
            val_mae  = mae(val_true, val_preds)
            val_r2   = r_squared(val_true, val_preds)
            logger.info(
                "Epoch %3d/%d  train_loss=%.5f  val_loss=%.5f  "
                "val_mae=%.4f  val_r2=%.4f  lr=%.6f",
                epoch, epochs, tr_loss, vl_loss, val_mae, val_r2,
                scheduler.get_last_lr()[0]
            )

        # Early stopping
        if vl_loss < best_val_loss - 1e-5:
            best_val_loss = vl_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # ── Restore best weights ──
    if best_state:
        model.load_state_dict(best_state)

    # ── Test evaluation ──
    logger.info("\n--- Test Set Evaluation ---")
    _, test_preds, test_true = eval_epoch(model, test_loader, criterion)

    test_mae  = mae(test_true, test_preds)
    test_rmse = rmse(test_true, test_preds)
    test_r2   = r_squared(test_true, test_preds)

    logger.info("Test MAE  : %.4f", test_mae)
    logger.info("Test RMSE : %.4f", test_rmse)
    logger.info("Test R²   : %.4f", test_r2)

    # Bucket accuracy (how often does the model predict the right 0.25-wide bin?)
    bucket_true = (test_true * 4).astype(int).clip(0, 3)
    bucket_pred = (test_preds * 4).astype(int).clip(0, 3)
    bucket_acc  = float(np.mean(bucket_true == bucket_pred))
    logger.info("Bucket accuracy (±0.25): %.4f", bucket_acc)

    # ── Save ──
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    # ── Plots ──
    _plot_loss_curves(train_losses, val_losses)
    _plot_predictions(test_true, test_preds)
    _plot_residuals(test_true, test_preds)

    logger.info("\nAll plots saved to %s/", PLOTS_DIR)
    logger.info("Training complete.")

def _plot_loss_curves(train_losses: list, val_losses: list) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train loss", color="#534AB7", linewidth=1.5)
    ax.plot(val_losses,   label="Val loss",   color="#E24B4A", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, color="#534AB7", s=20)
    lims = [0, 1]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True emotional load"); ax.set_ylabel("Predicted emotional load")
    ax.set_title("Predicted vs True Emotional Load", fontsize=12, fontweight="bold")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "prediction_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.4, color="#0F6E56", s=20)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted"); axes[0].spines[["top","right"]].set_visible(False)
    axes[1].hist(residuals, bins=30, color="#534AB7", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution"); axes[1].spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residuals.png", dpi=150, bbox_inches="tight")
    plt.close()


def load_nn_model(input_dim: int = 15) -> EmotionalLoadNet:
    model = EmotionalLoadNet(input_dim=input_dim)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_emotional_load(
    model: EmotionalLoadNet,
    features: np.ndarray,
) -> float:
    """
    Returns emotional load score ∈ [0, 1].
    features: shape (1, 15), already scaled.
    """
    tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        score = model(tensor).item()
    return float(np.clip(score, 0.0, 1.0))


if __name__ == "__main__":
    train()
