Kriteria 1

link github :https://github.com/yudhakr/Membangun-Sistem-Machine-Learning.git
# Eksperimen Preprocessing - Wine Classification

## Deskripsi

Project preprocessing untuk dataset **Wine** dari Scikit-Learn. Pipeline mencakup data loading, exploratory data analysis (EDA), data cleaning, feature selection, train-test split, dan scaling.

## Dataset

- **Sumber**: Scikit-Learn (`sklearn.datasets.load_wine`)
- **Sampel**: 178
- **Fitur**: 13 numerik
- **Kelas**: 3 (class_0, class_1, class_2)
- **Target**: `target`

## Struktur Folder

```
Eksperimen_SML_Ayudha/
├── dataset_raw/
│   └── wine_raw.csv
├── preprocessing/
│   ├── Eksperimen_Ayudha.ipynb
│   ├── automate_Ayudha.py
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── .github/workflows/
│   └── preprocessing.yml
├── requirements.txt
└── README.md
```

## Cara Menjalankan

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan notebook

Buka `preprocessing/Eksperimen_Ayudha.ipynb` di Jupyter dan jalankan semua cell.

### 3. Jalankan automate script

```bash
python preprocessing/automate_Ayudha.py
```

Output akan disimpan di `preprocessing/`:
- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

## GitHub Actions

Workflow `.github/workflows/preprocessing.yml` akan otomatis:
- Trigger saat push ke branch `main`
- Install dependencies
- Menjalankan `automate_Ayudha.py`
- Mengupload artifact hasil preprocessing


```bash
# 1. Aktifkan virtual environment
venv\Scripts\activate

# 2. Install dependency
pip install -r requirements.txt

# 3. Jalankan MLflow UI
mlflow ui
```

Buka browser:

```bash
http://127.0.0.1:5000
```

Terminal baru:

```bash
# Basic
python modelling.py

# Skilled / Advance
python modelling_tuning.py
```

Hasil training akan muncul di MLflow UI berupa:

* Metrics
* Parameters
* Artifacts
* Model




## 1. Struktur Folder Keiteria 2

```
Membangun_model/
├── modelling.py                 # Training + MLflow autolog + manual log_model
├── dataset_preprocessing.csv    # Dataset (opsional, tidak digunakan)
├── screenshoot_dashboard.jpg    # Screenshot MLflow Dashboard
├── screenshoot_artifak.jpg      # Screenshot MLflow Artifacts
├── requirements.txt             # Dependency Python
└── README.md                    # Dokumentasi
```

## 2. Cara Menjalankan Project

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

## 3. Detail Logging MLflow

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

## 4. Requirements

```
pandas
numpy
scikit-learn
mlflow
```


