# Wine Classification — MLflow Modelling (Kriteria 2)

## 1. Deskripsi

Project ini melatih model **RandomForestClassifier** pada dataset **Wine** dari Scikit-Learn dengan **MLflow autolog** dan **manual artifact logging** untuk memenuhi Kriteria 2 submission Dicoding.

## 2. Cara Menjalankan

### 2.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 2.2 Jalankan MLflow Tracking Server

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Biarkan server tetap berjalan. Tracking UI: **http://127.0.0.1:5000**

### 2.3 Jalankan Training

```bash
python modelling.py
```

Script akan:
1. Memuat dataset Wine dari Scikit-Learn
2. Split data train/test (80:20)
3. Mengaktifkan `mlflow.sklearn.autolog()` untuk logging otomatis
4. Melatih RandomForestClassifier
5. Mencatat accuracy dan classification report
6. **Manual artifact logging:**
   - `confusion_matrix.png`
   - `classification_report.txt`
   - `feature_importance.png`
7. Mencatat model ke `artifacts/model/`

## 3. Struktur Artifact MLflow

Setelah training selesai, di MLflow UI akan muncul:

```
<run_id>/
├── metrics/
│   ├── accuracy                    (autolog)
│   └── training_score              (autolog)
├── params/
│   ├── max_depth                   (autolog)
│   ├── n_estimators                (autolog)
│   ├── random_state                (autolog)
│   └── ...
├── tags/
├── artifacts/
│   ├── model/                      (log_model manual)
│   │   ├── MLmodel
│   │   ├── conda.yaml
│   │   ├── model.pkl
│   │   ├── python_env.yaml
│   │   └── requirements.txt
│   ├── model_analysis/             (manual artifact)
│   │   ├── confusion_matrix.png
│   │   ├── classification_report.txt
│   │   └── feature_importance.png
│   └── estimator.html              (autolog)
└── ...
```

## 4. Cara Screenshot untuk Submission

### 4.1 Screenshot Dashboard (`screenshoot_dashboard.jpg`)

1. Buka **http://127.0.0.1:5000**
2. Klik experiment **Wine_Classification_Basic**
3. Pilih **run terbaru** (paling atas)
4. Screenshot seluruh halaman yang memperlihatkan:
   - **Run ID** (di bagian atas)
   - **Metrics** (accuracy, training_score)
   - **Parameters** (n_estimators, max_depth, random_state, dll)
5. Simpan sebagai `screenshoot_dashboard.jpg`

### 4.2 Screenshot Artifact (`screenshoot_artifak.jpg`)

1. Di halaman run yang sama, scroll ke **Artifacts** (sebelah kanan)
2. Klik folder **model/**
3. Screenshot yang memperlihatkan:
   - `MLmodel`
   - `conda.yaml`
   - `model.pkl`
   - `python_env.yaml`
   - `requirements.txt`
4. Buka folder **model_analysis/**
5. Screenshot yang memperlihatkan:
   - `confusion_matrix.png`
   - `classification_report.txt`
   - `feature_importance.png`
6. Simpan sebagai `screenshoot_artifak.jpg`

### Tips Screenshot

- **Full window screenshot** — jangan crop terlalu sempit
- **Pastikan folder model/** terbuka — perlihatkan kelima file di dalamnya
- **Sertakan estimator.html** jika terlihat di artifacts (hasil autolog)

## 5. Checklist Artifact

| Artifact | Sumber | Wajib untuk Kriteria 2 |
|----------|--------|------------------------|
| `model/MLmodel` | `mlflow.sklearn.log_model()` | ✅ |
| `model/conda.yaml` | `mlflow.sklearn.log_model()` | ✅ |
| `model/model.pkl` | `mlflow.sklearn.log_model()` | ✅ |
| `model/python_env.yaml` | `mlflow.sklearn.log_model()` | ✅ |
| `model/requirements.txt` | `mlflow.sklearn.log_model()` | ✅ |
| `estimator.html` | `mlflow.sklearn.autolog()` | ✅ (auto) |
| `confusion_matrix.png` | manual `log_artifact()` | ✅ (skilled) |
| `classification_report.txt` | manual `log_artifact()` | ✅ (skilled) |
| `feature_importance.png` | manual `log_artifact()` | ✅ (advance) |

## 6. Requirements

```
pandas
numpy
scikit-learn
mlflow
matplotlib
```
