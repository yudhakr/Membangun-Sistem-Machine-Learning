"""
modelling_tuning.py
Hyperparameter tuning dengan GridSearchCV dan manual logging MLflow.
Tidak menggunakan mlflow.autolog() - semua logging dilakukan manual.

Level: SKILLED (+ ADVANCE dengan DagsHub)
"""

import os
import sys
import json
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import mlflow
import mlflow.sklearn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Konfigurasi ---
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset_preprocessing.csv")
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Wine_Classification_Tuning"
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "target"
TARGET_NAMES = ["class_0", "class_1", "class_2"]

# Hyperparameter grid untuk GridSearchCV
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}

# ADVANCE: DagsHub configuration
USE_DAGSHUB = False
DAGSHUB_USERNAME = "<USERNAME>"
DAGSHUB_REPO = "<REPO_NAME>"


# ============================================================
# FUNGSI-FUNGSI MODULAR
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Memuat dataset preprocessing dari CSV.

    Parameters:
        path (str): Path ke file CSV.

    Returns:
        pd.DataFrame: DataFrame fitur dan target.
    """
    if not os.path.exists(path):
        logger.error(f"File tidak ditemukan: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_data(df: pd.DataFrame, target_col: str):
    """
    Split data menjadi train dan test set.

    Parameters:
        df (pd.DataFrame): DataFrame lengkap.
        target_col (str): Nama kolom target.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def perform_tuning(X_train, y_train):
    """
    Melakukan hyperparameter tuning dengan GridSearchCV.

    Parameters:
        X_train: Feature training.
        y_train: Target training.

    Returns:
        Tuple (best_model, best_params, cv_results).
    """
    logger.info("Memulai GridSearchCV...")
    base_model = RandomForestClassifier(random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def evaluate_model(model, X_test, y_test):
    """
    Evaluasi model pada data testing.

    Parameters:
        model: Trained model.
        X_test: Feature testing.
        y_test: Target testing.

    Returns:
        Dict berisi metrics evaluasi.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    }

    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"Recall (macro): {metrics['recall_macro']:.4f}")
    logger.info(f"F1-Score (macro): {metrics['f1_macro']:.4f}")

    return metrics, y_pred, y_proba


def save_confusion_matrix(y_test, y_pred, save_path: str):
    """
    Menyimpan gambar confusion matrix.

    Parameters:
        y_test: Target actual.
        y_pred: Target prediksi.
        save_path (str): Path untuk menyimpan gambar.
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=TARGET_NAMES
    )
    disp.plot(cmap="Blues", ax=plt.gca())
    plt.title("Confusion Matrix - Wine Classification", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")


def save_classification_report(y_test, y_pred, save_path: str):
    """
    Menyimpan classification report ke file teks.

    Parameters:
        y_test: Target actual.
        y_pred: Target prediksi.
        save_path (str): Path file output.
    """
    report = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Classification Report - Wine Dataset\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    logger.info(f"Classification report saved: {save_path}")


def save_feature_importance(model, feature_names, save_path: str):
    """
    Menyimpan gambar feature importance.

    Parameters:
        model: Trained RandomForest model.
        feature_names: Nama-nama fitur.
        save_path (str): Path untuk menyimpan gambar.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance - Random Forest", fontsize=14)
    plt.bar(range(len(importances)), importances[indices], color="steelblue", align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Feature importance saved: {save_path}")


def save_model_summary(best_params, metrics, save_path: str):
    """
    Menyimpan ringkasan model ke file teks.

    Parameters:
        best_params (dict): Best hyperparameters.
        metrics (dict): Metrics evaluasi.
        save_path (str): Path file output.
    """
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("MODEL SUMMARY - Wine Classification\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Algorithm: RandomForestClassifier\n\n")
        f.write("Best Hyperparameters:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nEvaluation Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\nClasses: {TARGET_NAMES}\n")
        f.write("=" * 50 + "\n")
    logger.info(f"Model summary saved: {save_path}")


def save_training_log(best_params, metrics, save_path: str):
    """
    Menyimpan training log ke file teks.

    Parameters:
        best_params (dict): Best hyperparameters.
        metrics (dict): Metrics evaluasi.
        save_path (str): Path file output.
    """
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("TRAINING LOG\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Tracking URI: {TRACKING_URI}\n")
        f.write(f"Experiment: {EXPERIMENT_NAME}\n\n")
        f.write("Best Params:\n")
        f.write(json.dumps(best_params, indent=2) + "\n\n")
        f.write("Metrics:\n")
        f.write(json.dumps(metrics, indent=2) + "\n")
    logger.info(f"Training log saved: {save_path}")


def save_roc_curve(y_test, y_proba, n_classes, save_path: str):
    """
    Menyimpan gambar ROC Curve (One-vs-Rest).

    Parameters:
        y_test: Target actual.
        y_proba: Probability predictions.
        n_classes (int): Jumlah kelas.
        save_path (str): Path untuk menyimpan gambar.
    """
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    plt.figure(figsize=(8, 6))
    colors = ["blue", "red", "green"]

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr, color=colors[i], lw=2,
            label=f"{TARGET_NAMES[i]} (AUC = {roc_auc:.2f})"
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Wine Classification", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved: {save_path}")


def init_dagshub():
    """
    Inisialisasi DagsHub untuk tracking eksperimen.
    """
    global USE_DAGSHUB
    try:
        import dagshub
        dagshub.init(
            repo_owner=DAGSHUB_USERNAME,
            repo_name=DAGSHUB_REPO,
            mlflow=True
        )
        USE_DAGSHUB = True
        logger.info("DagsHub initialized successfully.")
    except ImportError:
        logger.warning("dagshub not installed. Skipping DagsHub integration.")
    except Exception as e:
        logger.warning(f"DagsHub init failed: {e}. Skipping.")


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Fungsi utama: orchestrate tuning + logging workflow.
    """
    try:
        # Buat direktori artifacts
        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        # Inisialisasi DagsHub (ADVANCE)
        init_dagshub()

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        logger.info("=" * 50)
        logger.info("MEMULAI MODELLING TUNING (SKILLED + ADVANCE)")
        logger.info("=" * 50)

        # --- Load data ---
        df = load_data(DATA_PATH)

        # --- Split data ---
        X_train, X_test, y_train, y_test = split_data(df, TARGET_COL)
        feature_names = list(X_train.columns)

        # --- Start MLflow run ---
        with mlflow.start_run(run_name=f"RF_Tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log dataset info
            mlflow.log_param("dataset_shape", str(df.shape))
            mlflow.log_param("feature_count", X_train.shape[1])
            mlflow.log_param("class_count", len(TARGET_NAMES))
            mlflow.log_param("target_names", str(TARGET_NAMES))
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("cv_folds", 5)

            # --- Hyperparameter Tuning ---
            best_model, best_params, cv_results = perform_tuning(X_train, y_train)

            # Log best parameters
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_param("best_cv_accuracy", cv_results["mean_test_score"][cv_results["rank_test_score"] == 1][0])

            # --- Evaluasi ---
            metrics, y_pred, y_proba = evaluate_model(best_model, X_test, y_test)

            # Log metrics
            mlflow.log_metrics(metrics)

            # --- Save dan log confusion matrix image ---
            cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
            save_confusion_matrix(y_test, y_pred, cm_path)
            mlflow.log_artifact(cm_path, "evaluation_plots")

            # --- Save dan log classification report ---
            cr_path = os.path.join(ARTIFACT_DIR, "classification_report.txt")
            save_classification_report(y_test, y_pred, cr_path)
            mlflow.log_artifact(cr_path, "evaluation_reports")

            # --- ADVANCE: Additional artifacts ---
            # 1. Feature importance
            fi_path = os.path.join(ARTIFACT_DIR, "feature_importance.png")
            save_feature_importance(best_model, feature_names, fi_path)
            mlflow.log_artifact(fi_path, "model_analysis")

            # 2. Model summary
            ms_path = os.path.join(ARTIFACT_DIR, "model_summary.txt")
            save_model_summary(best_params, metrics, ms_path)
            mlflow.log_artifact(ms_path, "model_analysis")

            # 3. Training log
            tl_path = os.path.join(ARTIFACT_DIR, "training_log.txt")
            save_training_log(best_params, metrics, tl_path)
            mlflow.log_artifact(tl_path, "model_analysis")

            # 4. ROC Curve
            roc_path = os.path.join(ARTIFACT_DIR, "roc_curve.png")
            save_roc_curve(y_test, y_proba, len(TARGET_NAMES), roc_path)
            mlflow.log_artifact(roc_path, "evaluation_plots")

            # --- Log model ---
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="random_forest_model",
                registered_model_name="Wine_RF_Tuned"
            )

            # Log params as catatan tambahan
            mlflow.log_param("algorithm", "RandomForestClassifier")
            mlflow.log_param("tuning_method", "GridSearchCV")
            mlflow.log_param("param_grid", str(PARAM_GRID))

            # Print hasil
            run_id = mlflow.active_run().info.run_id
            print(f"\n{'=' * 50}")
            print(f"RUN ID       : {run_id}")
            print(f"EXPERIMENT   : {EXPERIMENT_NAME}")
            print(f"BEST PARAMS  : {best_params}")
            print(f"ACCURACY     : {metrics['accuracy']:.4f}")
            print(f"F1 MACRO     : {metrics['f1_macro']:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))
            print(f"{'=' * 50}")
            print(f"MLflow UI: {TRACKING_URI}")
            if USE_DAGSHUB:
                print(f"DagsHub : https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}")

        logger.info("Modelling tuning selesai!")

    except Exception as e:
        logger.exception(f"Terjadi error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
