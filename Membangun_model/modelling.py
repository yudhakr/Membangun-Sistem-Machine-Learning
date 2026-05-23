"""
modelling.py
Melatih model Machine Learning menggunakan Scikit-Learn dengan
MLflow autolog untuk tracking parameter, metrics, dan artifact.

Level: BASIC
"""

import os
import sys
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Konfigurasi ---
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset_preprocessing.csv")
TRACKING_URI = "http://127.0.0.1:5000"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "target"


def load_data(path: str) -> pd.DataFrame:
    """
    Memuat dataset preprocessing dari CSV.

    Parameters:
        path (str): Path ke file CSV.

    Returns:
        pd.DataFrame: DataFrame berisi fitur dan target.
    """
    if not os.path.exists(path):
        logger.error(f"File tidak ditemukan: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_data(df: pd.DataFrame, target_col: str):
    """
    Memisahkan fitur dan target, lalu split train/test.

    Parameters:
        df (pd.DataFrame): DataFrame lengkap.
        target_col (str): Nama kolom target.

    Returns:
        X_train, X_test, y_train, y_test
    """
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
    """
    Melatih RandomForestClassifier.

    Parameters:
        X_train: Feature training.
        y_train: Target training.

    Returns:
        Trained model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    logger.info("Model training selesai.")
    return model


def evaluate_model(model, X_test, y_test, target_names=None):
    """
    Evaluasi model pada data testing.

    Parameters:
        model: Trained model.
        X_test: Feature testing.
        y_test: Target testing.
        target_names: List nama kelas target.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {acc:.4f}")
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    return acc


# --- MAIN ---
if __name__ == "__main__":
    try:
        # Simpan target names untuk display
        target_names = ["class_0", "class_1", "class_2"]

        # Set tracking URI
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment("Wine_Classification_Basic")

        logger.info("=" * 50)
        logger.info("MEMULAI MODELLING (BASIC - autolog)")
        logger.info("=" * 50)

        # Load data
        df = load_data(DATA_PATH)

        # Split data
        X_train, X_test, y_train, y_test = split_data(df, TARGET_COL)

        # Autolog MLflow
        mlflow.autolog()

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate
        evaluate_model(model, X_test, y_test, target_names)

        logger.info("Modelling selesai. Cek MLflow UI untuk hasil.")
        print(f"\nMLflow UI: {TRACKING_URI}")

    except Exception as e:
        logger.exception(f"Terjadi error: {e}")
        sys.exit(1)
