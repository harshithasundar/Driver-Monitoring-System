# =============================================================================
# src/model/trainer.py — Random Forest training pipeline
#
# Loads features.csv, preprocesses the data, trains a Random Forest
# classifier, evaluates it, and saves the model + scaler to disk.
#
# Run directly:
#   python -m src.model.trainer
#   python src/model/trainer.py
# =============================================================================

from __future__ import annotations

import os
import sys
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    DATASET_PATH,
    MODEL_PATH,
    SCALER_PATH,
    MODELS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LABEL_NAMES,
)


# =============================================================================
# Data loading
# =============================================================================

def load_dataset(path: str = DATASET_PATH) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and validate the feature CSV.

    Returns
    -------
    X : (N, 7) float32 array of features
    y : (N,)   int array of labels {0=AWAKE, 1=DROWSY}

    Raises
    ------
    FileNotFoundError if the CSV does not exist.
    ValueError        if class distribution is too imbalanced (< 10 % minority).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Run data_collector.py first:\n"
            "  python data_collector.py --label awake  --samples 600\n"
            "  python data_collector.py --label drowsy --samples 600"
        )

    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows from {path}")
    print(f"[INFO] Class distribution:\n{df[TARGET_COLUMN].value_counts().to_string()}")

    # Validate all expected columns are present
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Warn on severe class imbalance
    counts = df[TARGET_COLUMN].value_counts()
    minority_pct = counts.min() / counts.sum()
    if minority_pct < 0.10:
        print(
            f"[WARN] Severe class imbalance detected "
            f"({minority_pct*100:.1f}% minority). "
            "Consider collecting more data for the minority class."
        )

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(int)

    return X, y


# =============================================================================
# Model training
# =============================================================================

def build_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline: StandardScaler → RandomForestClassifier.

    Why a Pipeline?
    ---------------
    Bundling the scaler inside the pipeline ensures the scaler is always
    fitted on training data and applied correctly to test/live data.
    This prevents data leakage and makes deployment a single .pkl file.

    Why StandardScaler with Random Forest?
    ---------------------------------------
    Strictly speaking, tree-based models are scale-invariant.  We include
    the scaler here so the same pipeline can be swapped for a scale-sensitive
    model (SVM, Logistic Regression) without any other code changes.

    Random Forest hyperparameters:
    n_estimators = 200   — more trees → lower variance, diminishing returns
    max_depth    = None  — fully grown trees; forest bagging prevents overfit
    min_samples_split = 4 — prevents splitting on tiny groups (slight regularisation)
    class_weight = 'balanced' — adjusts for any remaining class imbalance
    random_state = 42    — reproducibility
    n_jobs = -1          — use all CPU cores
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",       # standard for classification
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train(X: np.ndarray, y: np.ndarray) -> tuple[Pipeline, dict]:
    """
    Split data, run cross-validation, train final model, return metrics.

    Parameters
    ----------
    X : feature matrix
    y : labels

    Returns
    -------
    pipeline : fitted Pipeline (scaler + RF)
    metrics  : dict of evaluation results
    """
    # 80 / 20 stratified split — stratified ensures both classes appear in test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    pipeline = build_pipeline()

    # ── Cross-validation on the training set ─────────────────────────────────
    # 5-fold stratified CV gives an unbiased estimate of generalisation error
    # before we ever touch the held-out test set.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    print(f"[INFO] 5-fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Final fit on full training data ──────────────────────────────────────
    pipeline.fit(X_train, y_train)

    # ── Test-set evaluation ───────────────────────────────────────────────────
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "cv_f1_mean":  float(cv_scores.mean()),
        "cv_f1_std":   float(cv_scores.std()),
        "test_report": classification_report(
            y_test, y_pred,
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1]]
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc":     roc_auc_score(y_test, y_pred_prob),
        "X_test":      X_test,
        "y_test":      y_test,
        "y_pred_prob": y_pred_prob,
    }

    print("\n[RESULTS] Classification Report:")
    print(metrics["test_report"])
    print(f"[RESULTS] ROC-AUC: {metrics['roc_auc']:.4f}")

    return pipeline, metrics


# =============================================================================
# Feature importance
# =============================================================================

def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Extract and rank feature importances from the trained RF."""
    rf = pipeline.named_steps["clf"]
    importance_df = pd.DataFrame({
        "feature":   FEATURE_COLUMNS,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df


# =============================================================================
# Saving
# =============================================================================

def save_model(pipeline: Pipeline) -> None:
    """Persist the fitted pipeline to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    # We save the full pipeline (scaler + model) as one file.
    # At inference time, pipeline.predict() automatically scales input.
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[INFO] Model saved → {MODEL_PATH}")


# =============================================================================
# Evaluation plots
# =============================================================================

def save_evaluation_plots(metrics: dict, importance_df: pd.DataFrame) -> None:
    """Save confusion matrix, ROC curve, and feature importance chart."""
    plot_dir = os.path.join(MODELS_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics["confusion_matrix"],
        display_labels=[LABEL_NAMES[0], LABEL_NAMES[1]],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(metrics["y_test"], metrics["y_pred_prob"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)

    # ── Feature importance ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1],
        color="steelblue",
    )
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title("Random Forest — Feature Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)

    print(f"[INFO] Plots saved → {plot_dir}/")


# =============================================================================
# Entry point
# =============================================================================

def run_training() -> Pipeline:
    """Full training pipeline. Returns the fitted model."""
    print("=" * 55)
    print(" Drowsiness Detection — Random Forest Training")
    print("=" * 55)

    X, y = load_dataset()
    pipeline, metrics = train(X, y)

    importance_df = get_feature_importance(pipeline)
    print("\n[INFO] Feature importances:")
    print(importance_df.to_string(index=False))

    save_model(pipeline)
    save_evaluation_plots(metrics, importance_df)

    print("\n[DONE] Training complete.")
    return pipeline


if __name__ == "__main__":
    run_training()
