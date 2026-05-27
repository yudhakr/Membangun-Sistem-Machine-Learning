"""
modelling.py — BASIC VERSION
Melatih model RandomForestClassifier dengan MLflow autolog.
Tidak ada manual logging — murni autolog.
"""

import logging

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TRACKING_URI = "http://127.0.0.1:5000"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "target"
TARGET_NAMES = ["class_0", "class_1", "class_2"]


def load_data() -> pd.DataFrame:
    wine = load_wine(as_frame=True)
    df = wine.data.copy()
    df[TARGET_COL] = wine.target
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


if __name__ == "__main__":
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("Wine_Classification_Basic")

    logger.info("=" * 50)
    logger.info("MEMULAI MODELLING BASIC — autolog only")
    logger.info("=" * 50)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Autolog aktif — semua dicatat otomatis oleh MLflow
    mlflow.sklearn.autolog()

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {acc:.4f}")
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))

    print(f"\nMLflow UI: {TRACKING_URI}")
