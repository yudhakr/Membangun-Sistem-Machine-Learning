"""
automate_Ayudha.py
Script otomatis untuk preprocessing dataset Diabetes.
Alur: load data -> preprocessing -> save data siap training.

Author: Ayudha
"""

import os
import sys

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


# --- Konfigurasi ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "dataset_raw", "diabetes_raw.csv")
PREPROCESSED_DATA_PATH = os.path.join(BASE_DIR, "preprocessing", "dataset_preprocessing.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2
FEATURE_COUNT = 8


def load_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Memuat dataset Diabetes.

    Parameters:
        filepath (str): Path ke file CSV dataset raw.

    Returns:
        pd.DataFrame: DataFrame berisi data lengkap (fitur + target).
    """
    try:
        # Coba load dari CSV jika file tersedia
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"[OK] Data berhasil dimuat dari: {filepath}")
        else:
            # Fallback: load langsung dari sklearn
            print(f"[INFO] File {filepath} tidak ditemukan. Memuat dari Scikit-Learn...")
            diabetes = load_diabetes()
            df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
            df["target"] = diabetes.target

            # Simpan sebagai raw dataset
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            print(f"[OK] Data raw disimpan ke: {filepath}")

        print(f"[INFO] Shape dataset: {df.shape}")
        return df

    except Exception as e:
        print(f"[ERROR] Gagal memuat data: {e}")
        sys.exit(1)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan preprocessing pada dataset.

    Tahapan:
    1. Handling missing values
    2. Feature selection (SelectKBest)
    3. Train-test split
    4. Standard scaling

    Parameters:
        df (pd.DataFrame): DataFrame mentah.

    Returns:
        pd.DataFrame: DataFrame hasil preprocessing.
    """
    print("\n" + "=" * 50)
    print("MEMULAI PREPROCESSING")
    print("=" * 50)

    # --- 1. Handling Missing Values ---
    print("\n[1] Handling Missing Values...")
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  - Kolom '{col}': {df[col].isnull().sum()} missing diisi median")
        print(f"  Total missing values setelah handling: {df.isnull().sum().sum()}")
    else:
        print("  Tidak ada missing values. Data sudah bersih.")

    # --- 2. Feature Selection ---
    print("\n[2] Feature Selection (SelectKBest)...")
    feature_names = [col for col in df.columns if col != "target"]
    X = df[feature_names]
    y = df["target"]

    k = min(FEATURE_COUNT, X.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)

    selected_features = np.array(feature_names)[selector.get_support()]
    feature_scores = pd.DataFrame({
        "Fitur": feature_names,
        "Skor": selector.scores_
    }).sort_values("Skor", ascending=False)

    print(f"  10 fitur asli:")
    print(feature_scores.to_string(index=False))
    print(f"  Fitur terpilih ({k}): {list(selected_features)}")

    # --- 3. Train-Test Split ---
    print("\n[3] Train-Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X[selected_features], y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"  Training set: {X_train.shape}")
    print(f"  Testing set : {X_test.shape}")

    # --- 4. Standard Scaling ---
    print("\n[4] Standard Scaling...")
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)

    print("  Scaling selesai.")

    # Gabungkan data training dan testing
    df_train = X_train_scaled.copy()
    df_train["target"] = y_train.values

    df_test = X_test_scaled.copy()
    df_test["target"] = y_test.values

    df_result = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

    print(f"\n[INFO] Dataset preprocessing akhir: {df_result.shape}")
    return df_result


def save_data(df: pd.DataFrame, filepath: str = PREPROCESSED_DATA_PATH) -> None:
    """
    Menyimpan dataset hasil preprocessing ke file CSV.

    Parameters:
        df (pd.DataFrame): DataFrame hasil preprocessing.
        filepath (str): Path output file CSV.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"\n[OK] Dataset preprocessing berhasil disimpan ke: {filepath}")
        print(f"     Total: {df.shape[0]} baris, {df.shape[1]} kolom")
        print(f"     Kolom: {list(df.columns)}")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan data: {e}")
        sys.exit(1)


def main():
    """
    Fungsi utama: orchestrate alur preprocessing.
    """
    print("=" * 50)
    print("  AUTOMATE PREPROCESSING - Diabetes Dataset")
    print("=" * 50)

    # 1. Load data
    df = load_data()

    # 2. EDA singkat
    print("\n" + "=" * 50)
    print("RINGKASAN DATA")
    print("=" * 50)
    print(f"  Jumlah baris : {df.shape[0]}")
    print(f"  Jumlah kolom : {df.shape[1]}")
    print(f"  Kolom        : {list(df.columns)}")
    print(f"  Tipe data    :\n{df.dtypes.to_string()}")

    # 3. Preprocess
    df_preprocessed = preprocess(df)

    # 4. Save data
    save_data(df_preprocessed)

    print("\n" + "=" * 50)
    print("  PREPROCESSING SELESAI - Data siap untuk training!")
    print("=" * 50)


if __name__ == "__main__":
    main()
