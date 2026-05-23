"""
automate_Ayudha.py
Preprocessing otomatis untuk Wine Classification Dataset.
Workflow: load raw data -> clean -> preprocess -> split -> save.

Author: Ayudha
"""

import os
import sys
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Konfigurasi ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "dataset_raw", "wine_raw.csv")
PREPROC_DIR = os.path.join(BASE_DIR, "preprocessing")
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "target"
FEATURE_COUNT = 10


def load_data(path: str) -> pd.DataFrame:
    """
    Memuat dataset Wine dari file CSV.

    Parameters:
        path (str): Path ke file CSV dataset raw.

    Returns:
        pd.DataFrame: DataFrame mentah.
    """
    if not os.path.exists(path):
        logger.error(f"File tidak ditemukan: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan dataset: handling missing values, duplicate, dan encoding.

    Tahapan:
    1. Handling missing values (isi dengan median jika ada)
    2. Menghapus data duplikat
    3. Encoding fitur kategorikal (jika ada)

    Parameters:
        df (pd.DataFrame): DataFrame mentah.

    Returns:
        pd.DataFrame: DataFrame bersih.
    """
    logger.info("=" * 50)
    logger.info("DATA CLEANING")
    logger.info("=" * 50)

    # --- 1. Handling Missing Values ---
    logger.info("[1] Handling Missing Values...")
    if df.isnull().sum().sum() > 0:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"  - {col}: filled with median={median_val:.4f}")
        logger.info("  Missing values handled.")
    else:
        logger.info("  No missing values found.")

    # --- 2. Handling Duplicate ---
    logger.info("[2] Handling Duplicate Data...")
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df.drop_duplicates(keep="first", inplace=True)
        logger.info(f"  Removed {duplicate_count} duplicates.")
        logger.info(f"  New shape: {df.shape}")
    else:
        logger.info("  No duplicates found.")

    # --- 3. Encoding Categorical ---
    logger.info("[3] Encoding Categorical Features...")
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])
            logger.info(f"  - {col}: encoded")
    else:
        logger.info("  No categorical features found. Skipping encoding.")

    logger.info("Data cleaning completed.")
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Melakukan preprocessing: feature selection dan scaling.

    Tahapan:
    1. Feature selection dengan SelectKBest
    2. Scaling dengan StandardScaler

    Parameters:
        df (pd.DataFrame): DataFrame bersih.

    Returns:
        Tuple (X, y, selected_features, scaler).
    """
    logger.info("=" * 50)
    logger.info("DATA PREPROCESSING")
    logger.info("=" * 50)

    # --- 1. Feature Selection ---
    logger.info("[1] Feature Selection...")
    feature_cols = [col for col in df.columns if col != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    k = min(FEATURE_COUNT, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    selected_features = np.array(feature_cols)[selector.get_support()]

    feature_scores = pd.DataFrame({
        "Fitur": feature_cols,
        "Skor": selector.scores_
    }).sort_values("Skor", ascending=False)

    logger.info(f"  Top-{k} features selected:")
    for _, row in feature_scores.head(k).iterrows():
        logger.info(f"    - {row['Fitur']}: {row['Skor']:.2f}")

    X = pd.DataFrame(X_selected, columns=selected_features)
    logger.info(f"  X shape after selection: {X.shape}")

    # --- 2. Scaling ---
    logger.info("[2] Standard Scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=selected_features)
    logger.info("  Scaling completed.")

    logger.info("Preprocessing completed.")
    return X, y, selected_features, scaler


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Membagi data menjadi training dan testing set.

    Parameters:
        X (pd.DataFrame): Fitur.
        y (pd.Series): Target.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    logger.info("=" * 50)
    logger.info("TRAIN-TEST SPLIT")
    logger.info("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_test : {X_test.shape}")
    logger.info(f"  y_train: {y_train.shape}")
    logger.info(f"  y_test : {y_test.shape}")
    logger.info(f"  Test size: {TEST_SIZE * 100:.0f}%")

    return X_train, X_test, y_train, y_test


def save_data(X_train: pd.DataFrame, X_test: pd.DataFrame,
              y_train: pd.Series, y_test: pd.Series,
              output_dir: str) -> None:
    """
    Menyimpan dataset hasil preprocessing ke file CSV.

    Parameters:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
        output_dir (str): Direktori output.
    """
    logger.info("=" * 50)
    logger.info("SAVING DATASET")
    logger.info("=" * 50)

    os.makedirs(output_dir, exist_ok=True)

    files = {
        "X_train.csv": X_train,
        "X_test.csv": X_test,
        "y_train.csv": pd.DataFrame(y_train, columns=[TARGET_COL]),
        "y_test.csv": pd.DataFrame(y_test, columns=[TARGET_COL]),
    }

    for filename, data in files.items():
        path = os.path.join(output_dir, filename)
        data.to_csv(path, index=False)
        logger.info(f"  Saved: {filename} ({data.shape})")

    logger.info(f"All files saved to: {output_dir}")


def main():
    """
    Fungsi utama: orchestrate seluruh pipeline preprocessing.
    """
    try:
        logger.info("=" * 60)
        logger.info("  AUTOMATE PREPROCESSING - Wine Dataset")
        logger.info("=" * 60)

        # 1. Load
        df = load_data(RAW_DATA_PATH)

        # 2. EDA summary
        logger.info(f"\nDataset info:\n  Rows: {df.shape[0]}\n  Cols: {df.shape[1]}")

        # 3. Clean
        df = clean_data(df)

        # 4. Preprocess
        X, y, selected_features, scaler = preprocess_data(df)

        # 5. Split
        X_train, X_test, y_train, y_test = split_data(X, y)

        # 6. Save
        save_data(X_train, X_test, y_train, y_test, PREPROC_DIR)

        logger.info("=" * 60)
        logger.info("  PREPROCESSING COMPLETED SUCCESSFULLY!")
        logger.info(f"  Output: {PREPROC_DIR}")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
