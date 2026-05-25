

## 3. Struktur Folder

```
Membangun_model/
├── modelling.py                 # Training + MLflow autolog + manual log_model
├── dataset_preprocessing.csv    # Dataset (opsional, tidak digunakan)
├── screenshoot_dashboard.jpg    # Screenshot MLflow Dashboard
├── screenshoot_artifak.jpg      # Screenshot MLflow Artifacts
├── requirements.txt             # Dependency Python
└── README.md                    # Dokumentasi
```

## 4. Cara Menjalankan Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Jalankan Training

```bash
python modelling.py
```



### Jalankan MLflow UI

```bash
mlflow ui
```

Buka browser: **http://127.0.0.1:5000**

## 5. Detail Logging MLflow

| Yang di-log | Method |
|-------------|--------|
| Parameters model | `mlflow.sklearn.autolog()` otomatis |
| Metrics (accuracy) | `mlflow.sklearn.autolog()` otomatis |
| Model artifact | `mlflow.sklearn.autolog()` otomatis |
| Model (manual) | `mlflow.sklearn.log_model(sk_model=model, artifact_path="model")` |

### Struktur Artifact Model

Setelah training selesai, pada MLflow UI akan muncul folder **artifacts/model/** yang berisi:

```
artifacts/model/
├── MLmodel
├── conda.yaml
├── model.pkl
├── python_env.yaml
└── requirements.txt
```

## 6. Cara Screenshot untuk Submission

### Screenshot Dashboard

1. Jalankan `python modelling.py`
2. Jalankan `mlflow ui`
3. Buka **http://127.0.0.1:5000**
4. Pilih experiment **Wine_Classification_Basic**
5. Screenshot tampilan daftar runs
6. Simpan sebagai `screenshoot_dashboard.jpg`

### Screenshot Artifact Model (Kriteria 2)

1. Di MLflow UI, klik run terbaru (paling atas)
2. Scroll ke bagian **Artifacts**
3. Klik folder **model/**
4. Screenshot tampilan yang menunjukkan 5 file berikut:
   - `MLmodel`
   - `conda.yaml`
   - `model.pkl`
   - `python_env.yaml`
   - `requirements.txt`
5. Simpan sebagai `screenshoot_artifak.jpg`

## 7. Requirements

```
pandas
numpy
scikit-learn
mlflow
```
