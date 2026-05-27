"""
modelling.py
Melatih model Machine Learning menggunakan Scikit-Learn dengan
MLflow autolog + manual artifact logging untuk memenuhi
Kriteria 2 (Basic, Skilled, Advance) submission Dicoding.
"""

import os
import sys
import tempfile
import logging

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Konfigurasi ---
TRACKING_URI = "http://127.0.0.1:5000"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "target"
TARGET_NAMES = ["class_0", "class_1", "class_2"]


def load_data() -> pd.DataFrame:
    """Memuat dataset Wine dari Scikit-Learn."""
    wine = load_wine(as_frame=True)
    df = wine.data.copy()
    df[TARGET_COL] = wine.target
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_data(df: pd.DataFrame, target_col: str):
    """Memisahkan fitur dan target, lalu split train/test."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Melatih RandomForestClassifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    logger.info("Model training selesai.")
    return model


def evaluate_model(model, X_test, y_test, target_names=None):
    """Evaluasi model pada data testing."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {acc:.4f}")
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    return acc, y_pred


def log_confusion_matrix(model, X_test, y_test, run_id):
    """Generate dan log confusion matrix sebagai artifact."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TARGET_NAMES)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - Run {run_id[:8]}")
    plt.tight_layout()
    path = os.path.join(tempfile.gettempdir(), "confusion_matrix.png")
    plt.savefig(path, dpi=100)
    plt.close()
    mlflow.log_artifact(path, "model_analysis")
    logger.info("Logged confusion_matrix.png")


def log_classification_report(y_test, y_pred):
    """Generate dan log classification report sebagai artifact."""
    report = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
    content = "Classification Report - Wine Dataset\n"
    content += "=" * 50 + "\n"
    content += report
    path = os.path.join(tempfile.gettempdir(), "classification_report.txt")
    with open(path, "w") as f:
        f.write(content)
    mlflow.log_artifact(path, "model_analysis")
    logger.info("Logged classification_report.txt")


def log_feature_importance(model, feature_names):
    """Generate dan log feature importance plot sebagai artifact."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices], color="steelblue")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    path = os.path.join(tempfile.gettempdir(), "feature_importance.png")
    plt.savefig(path, dpi=100)
    plt.close()
    mlflow.log_artifact(path, "model_analysis")
    logger.info("Logged feature_importance.png")


# --- MAIN ---
if __name__ == "__main__":
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment("Wine_Classification_Basic")

        logger.info("=" * 50)
        logger.info("MEMULAI MODELLING (Dengan autolog + manual artifacts)")
        logger.info("=" * 50)

        # Load & split data
        df = load_data()
        X_train, X_test, y_train, y_test = split_data(df, TARGET_COL)
        feature_names = [c for c in df.columns if c != TARGET_COL]

        # Aktifkan autolog SEBELUM training
        mlflow.sklearn.autolog()

        # Jalankan dalam session MLflow agar manual logging tercatat dalam run yang sama
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}")

            # Train (autolog otomatis catat params, metrics, model)
            model = train_model(X_train, y_train)

            # Evaluate (autolog otomatis catat accuracy)
            acc, y_pred = evaluate_model(model, X_test, y_test, TARGET_NAMES)

            # --- SKILLED / ADVANCE: Manual artifact logging ---
            log_confusion_matrix(model, X_test, y_test, run_id)
            log_classification_report(y_test, y_pred)
            log_feature_importance(model, feature_names)

            # Manual model logging
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="Wine_RF_Basic"
            )

        print(f"\n{'=' * 50}")
        print(f"Run ID: {run_id}")
        print(f"Accuracy: {acc:.4f}")
        print(f"{'=' * 50}")
        logger.info("Modelling selesai. Cek MLflow UI untuk hasil.")
        print(f"\nMLflow UI: {TRACKING_URI}")

    except Exception as e:
        logger.exception(f"Terjadi error: {e}")
        sys.exit(1)
