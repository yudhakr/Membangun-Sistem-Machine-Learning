# Wine Classification - Machine Learning Pipeline

## 1. Deskripsi Project

Project ini bertujuan untuk melakukan klasifikasi wine menggunakan dataset **Wine** dari Scikit-Learn. Pipeline mencakup preprocessing data, training model **Random Forest Classifier**, hyperparameter tuning dengan **GridSearchCV**, serta tracking eksperimen menggunakan **MLflow** dan **DagsHub**.

Project ini memenuhi 3 level submission MSML Dicoding:
- **BASIC** : Training model dengan `mlflow.autolog()` (`modelling.py`)
- **SKILLED** : Hyperparameter tuning + manual MLflow logging (`modelling_tuning.py`)
- **ADVANCE** : Integrasi DagsHub + artifact tambahan

## 2. Dataset

**Wine Dataset** (Scikit-Learn bawaan):
- **Total sampel**: 178
- **Fitur**: 13 fitur numerik (alcohol, malic_acid, ash, ...)
- **Kelas**: 3 kelas wine (class_0, class_1, class_2)
- **Target**: `target`

Dataset telah melalui preprocessing:
- StandardScaler normalization
- Tidak ada missing values

## 3. Struktur Folder

```
Membangun_model/
├── modelling.py                 # BASIC: training + mlflow.autolog()
├── modelling_tuning.py          # SKILLED + ADVANCE: tuning + manual logging
├── dataset_preprocessing.csv    # Dataset siap training
├── screenshoot_dashboard.jpg    # Screenshot MLflow Dashboard
├── screenshoot_artifak.jpg      # Screenshot MLflow Artifacts
├── requirements.txt             # Dependency Python
├── DagsHub.txt                  # Informasi DagsHub
└── README.md                    # Dokumentasi
```

## 4. Cara Menjalankan MLflow

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Jalankan MLflow Tracking Server

Buka terminal dan jalankan:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Tracking UI akan tersedia di: **http://127.0.0.1:5000**

Biarkan server tetap berjalan di terminal tersebut.

## 5. Cara Menjalankan modelling.py (BASIC)

```bash
python modelling.py
```

File ini menggunakan `mlflow.autolog()` untuk secara otomatis mencatat:
- Parameters model
- Metrics (accuracy)
- Model artifact

## 6. Cara Menjalankan modelling_tuning.py (SKILLED)

### Setup DagsHub (ADVANCE - Opsional)

Sebelum menjalankan, edit `modelling_tuning.py`:

```python
DAGSHUB_USERNAME = "<USERNAME_DAGSHUB_ANDA>"
DAGSHUB_REPO = "<REPO_NAME_ANDA>"
```

### Jalankan Tuning

```bash
python modelling_tuning.py
```

**Tanpa DagsHub**: Script akan berjalan normal dengan tracking ke MLflow lokal.
**Dengan DagsHub**: Script akan men-track experiment ke DagsHub secara otomatis.

## 7. Cara Membuka MLflow UI

1. Jalankan MLflow server:
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```
2. Buka browser: **http://127.0.0.1:5000**
3. Pilih experiment **Wine_Classification_Basic** atau **Wine_Classification_Tuning**
4. Lihat runs, parameters, metrics, dan artifacts

### Cara Generate Screenshot

1. **screenshoot_dashboard.jpg**:
   - Buka MLflow UI di http://127.0.0.1:5000
   - Pilih experiment **Wine_Classification_Tuning**
   - Screenshoot tampilan dashboard (daftar runs + metrics)
   - Simpan sebagai `screenshoot_dashboard.jpg`

2. **screenshoot_artifak.jpg**:
   - Klik salah satu run
   - Scroll ke bagian **Artifacts**
   - Screenshoot tampilan artifacts (confusion_matrix.png, dll)
   - Simpan sebagai `screenshoot_artifak.jpg`

## 8. Cara Integrasi DagsHub

### Langkah-langkah:

1. **Buat akun** di [dagshub.com](https://dagshub.com)
2. **Buat repository** baru di DagsHub
3. **Install dagshub**:
   ```bash
   pip install dagshub
   ```
4. **Edit `modelling_tuning.py`**:
   ```python
   DAGSHUB_USERNAME = "username_anda"
   DAGSHUB_REPO = "nama_repo_anda"
   ```
5. **Jalankan script**:
   ```bash
   python modelling_tuning.py
   ```
6. **Cek DagsHub**: Login ke DagsHub dan lihat experiment di repository Anda.
7. **Simpan URL** ke `DagsHub.txt`.

### Artifact tambahan yang di-track ke DagsHub:
- `confusion_matrix.png`
- `classification_report.txt`
- `feature_importance.png`
- `model_summary.txt`
- `training_log.txt`
- `roc_curve.png`
- Model RandomForest (format MLflow)

## 9. Penjelasan Hyperparameter Tuning

Hyperparameter tuning menggunakan **GridSearchCV** dengan parameter grid:

| Parameter | Nilai yang Diuji |
|-----------|-----------------|
| n_estimators | 50, 100, 200 |
| max_depth | None, 10, 20, 30 |
| min_samples_split | 2, 5, 10 |
| min_samples_leaf | 1, 2, 4 |
| criterion | gini, entropy |

**GridSearchCV** menggunakan:
- **Cross-validation**: 5-fold
- **Scoring**: accuracy
- **Total kombinasi**: 3 x 4 x 3 x 3 x 2 = 216 kombinasi

Setelah tuning, model terbaik dipilih berdasarkan rata-rata accuracy tertinggi dari cross-validation.

## 10. Penjelasan Logging MLflow

### BASIC (`modelling.py` - autolog)
| Yang di-log | Method |
|-------------|--------|
| Parameters | `mlflow.autolog()` otomatis |
| Metrics | `mlflow.autolog()` otomatis |
| Model | `mlflow.autolog()` otomatis |

### SKILLED (`modelling_tuning.py` - manual)
| Yang di-log | Method |
|-------------|--------|
| Dataset info | `mlflow.log_param()` |
| Best parameters | `mlflow.log_params()` |
| Accuracy, Precision, Recall, F1 | `mlflow.log_metrics()` |
| Confusion Matrix image | `mlflow.log_artifact()` |
| Classification Report | `mlflow.log_artifact()` |
| Feature Importance image | `mlflow.log_artifact()` |
| Model Summary | `mlflow.log_artifact()` |
| Training Log | `mlflow.log_artifact()` |
| ROC Curve | `mlflow.log_artifact()` |
| Model | `mlflow.sklearn.log_model()` |

## 11. Requirements

```
pandas
numpy
scikit-learn
mlflow==2.19.0
matplotlib
seaborn
joblib
dagshub
```
