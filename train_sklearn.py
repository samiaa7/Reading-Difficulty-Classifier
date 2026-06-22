from __future__ import annotations

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline

from data_pipeline import build_dataset, FEATURE_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIFFICULTY_LABELS = ["Easy", "Moderate", "Hard", "Very Hard"]
MODEL_PATH = Path("rf_model.pkl")
PLOTS_DIR  = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def train() -> None:
    logger.info("=" * 55)
    logger.info("SKLEARN — Reading Difficulty Classifier")
    logger.info("=" * 55)

    # ── Data ──
    (X_train, X_val, X_test,
     yd_train, yd_val, yd_test,
     ye_train, ye_val, ye_test,
     scaler)  = build_dataset(wikilarge_path="data/wikilarge.txt.src", save=True)

    # Combine train + val for final GridSearch (CV handles splitting internally)
    X_tv   = np.vstack([X_train, X_val])
    yd_tv  = np.concatenate([yd_train, yd_val])

    # ── Baseline: default Random Forest ──
    logger.info("\n--- Baseline Random Forest ---")
    baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    baseline.fit(X_train, yd_train)
    base_acc = accuracy_score(yd_val, baseline.predict(X_val))
    logger.info("Validation accuracy (baseline): %.4f", base_acc)

    # ── GridSearchCV ──
    logger.info("\n--- GridSearchCV ---")
    param_grid = {
        "n_estimators":     [100, 200],
        "max_depth":        [None, 15, 25],
        "min_samples_split":[2, 5],
        "min_samples_leaf": [1, 2],
        "max_features":     ["sqrt", "log2"],
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid,
        cv=5, scoring="f1_macro",
        n_jobs=-1, verbose=1,
    )
    grid_search.fit(X_tv, yd_tv)
    logger.info("Best params: %s", grid_search.best_params_)
    logger.info("Best CV F1 (macro): %.4f", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    # ── Test evaluation ──
    logger.info("\n--- Test Set Evaluation ---")
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(yd_test, y_pred)
    f1  = f1_score(yd_test, y_pred, average="macro")
    logger.info("Test accuracy : %.4f", acc)
    logger.info("Test F1 macro : %.4f", f1)
    logger.info("\nClassification Report:\n%s",
        classification_report(yd_test, y_pred, target_names=DIFFICULTY_LABELS))

    # ── Cross-validation ──
    cv_scores = cross_val_score(best_model, X_tv, yd_tv, cv=5, scoring="accuracy")
    logger.info("5-fold CV accuracy: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # ── Save model ──
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    logger.info("Model saved to %s", MODEL_PATH)

    # ── Plots ──
    _plot_confusion_matrix(yd_test, y_pred)
    _plot_feature_importance(best_model)
    _plot_cv_scores(cv_scores)

    logger.info("\nAll plots saved to %s/", PLOTS_DIR)
    logger.info("Training complete.")

def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d", ".2f"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Purples",
            xticklabels=DIFFICULTY_LABELS,
            yticklabels=DIFFICULTY_LABELS,
            ax=ax, linewidths=0.5
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_feature_importance(model: RandomForestClassifier) -> None:
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    names_sorted = [FEATURE_NAMES[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#534AB7" if i < 5 else "#0F6E56" if i < 10 else "#9CA3AF"
              for i in range(len(names_sorted))]
    bars = ax.barh(names_sorted[::-1], importances[indices][::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate top 3
    for i, (bar, name) in enumerate(zip(bars[::-1][:3], names_sorted[:3])):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{importances[indices[i]]:.3f}", va="center", fontsize=9, color="#534AB7")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_cv_scores(scores: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    folds = [f"Fold {i+1}" for i in range(len(scores))]
    ax.bar(folds, scores, color="#534AB7", alpha=0.8, edgecolor="white")
    ax.axhline(scores.mean(), color="#E24B4A", linestyle="--",
               linewidth=1.5, label=f"Mean = {scores.mean():.4f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cv_scores.png", dpi=150, bbox_inches="tight")
    plt.close()


def load_rf_model() -> RandomForestClassifier:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_difficulty(
    model: RandomForestClassifier,
    features: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    """
    Returns (predicted_class, confidence, class_probabilities).
    features: shape (1, 15), already scaled.
    """
    probs = model.predict_proba(features)[0]
    pred  = int(np.argmax(probs))
    conf  = float(probs[pred])
    return pred, conf, probs


if __name__ == "__main__":
    train()
