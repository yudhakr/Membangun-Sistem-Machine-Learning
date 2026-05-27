# Wine Classification — MLflow Modelling (Kriteria 2)

## 1. Deskripsi

Project ini memenuhi **Kriteria 2** submission Dicoding dengan dua level:

| Level | File | Metode |
|-------|------|--------|
| **Basic** | `modelling.py` | `mlflow.sklearn.autolog()` — tanpa manual logging |
| **Skilled / Advance** | `modelling_tuning.py` | GridSearchCV + manual logging + artifact |

Dataset: **Wine Dataset** (Scikit-Learn bawaan), 178 sampel, 13 fitur, 3 kelas.

Algoritma: **RandomForestClassifier**.

---

## 2. Cara Menjalankan

### 2.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 2.2 Jalankan MLflow Tracking Server

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Buka **http://127.0.0.1:5000** untuk UI.

### 2.3 Jalankan Basic (autolog only)

```bash
python modelling.py
```

Script akan:
- Load dataset Wine dari Scikit-Learn
- Split train/test (80:20, stratified)
- `mlflow.sklearn.autolog()` aktif → otomatis catat **params**, **metrics**, **model**, **estimator.html**
- Latih RandomForestClassifier (n_estimators=100, max_depth=10)
- Cetak accuracy dan classification report

**Tidak ada manual logging.** Semua dicatat otomatis oleh autolog.

### 2.4 Jalankan Skilled / Advance

```bash
python modelling_tuning.py
```

Script akan:
- Load wine dataset (via CSV)
- GridSearchCV (5-fold) untuk hyperparameter tuning
- Manual logging params, metrics, artifacts via MLflow
- Log confusion_matrix.png, classification_report.txt, feature_importance.png, model_summary.txt, training_log.txt, roc_curve.png
- Log model terbaik ke MLflow

---

## 3. Struktur Artifact MLflow

### Basic (`modelling.py` — autolog only)

Setelah `python modelling.py`, MLflow akan menghasilkan:

```
Run <ID>
├── metrics: accuracy, training_score
├── params: max_depth, n_estimators, random_state, ...
├── artifacts/
│   ├── model/                          ← autolog
│   │   ├── MLmodel
│   │   ├── conda.yaml
│   │   ├── model.pkl
│   │   ├── python_env.yaml
│   │   └── requirements.txt
│   └── estimator.html                  ← autolog
```

### Skilled (`modelling_tuning.py` — manual)

Setelah `python modelling_tuning.py`:

```
Run <ID>
├── metrics: accuracy, precision_macro, recall_macro, f1_macro, ...
├── params: best_*, dataset_shape, cv_folds, ...
├── artifacts/
│   ├── random_forest_model/            ← manual log_model
│   │   ├── MLmodel
│   │   ├── conda.yaml
│   │   ├── model.pkl
│   │   ├── python_env.yaml
│   │   └── requirements.txt
│   ├── evaluation_plots/
│   │   ├── confusion_matrix.png
│   │   └── roc_curve.png
│   ├── evaluation_reports/
│   │   └── classification_report.txt
│   └── model_analysis/
│       ├── feature_importance.png
│       ├── model_summary.txt
│       └── training_log.txt
```

---

## 4. Cara Screenshot untuk Submission

### 4.1 Screenshot Dashboard (`screenshoot_dashboard.jpg`)

Langkah:
1. Buka **http://127.0.0.1:5000**
2. Di sidebar kiri, klik **Wine_Classification_Basic** (jangan klik run-nya)
3. Halaman yang muncul adalah **Experiment View** — berisi **daftar semua run**
4. Screenshot seluruh halaman yang memperlihatkan:
   - **Kolom Run ID / Date** (daftar run)
   - **Kolom Metrics** (accuracy, training_score)
   - **Kolom Parameters** (n_estimators, max_depth, dll)
   - **Source** (modelling.py)
5. **Jangan** klik ke dalam detail run — screenshot dari halaman daftar experiment

> ⚠️ **Perhatian:** Reviewer meminta screenshot **halaman daftar experiment**, bukan halaman detail/overview satu run. Pastikan yang di-screenshot adalah tabel yang menampilkan semua run beserta metrics dan parameters-nya.

### 4.2 Screenshot Artifact (`screenshoot_artifak.jpg`)

Langkah:
1. Klik salah satu **Run ID** pada tabel experiment
2. Scroll ke bagian **Artifacts** (panel kanan)
3. Klik folder **model/** untuk membukanya
4. Screenshot yang memperlihatkan:
   - `MLmodel`
   - `conda.yaml`
   - `model.pkl`
   - `python_env.yaml`
   - `requirements.txt`
   - `estimator.html` (jika terlihat)
5. Simpan sebagai `screenshoot_artifak.jpg`

---

## 5. Checklist Reviewer

| Item | Ada di Basic? | Ada di Skilled? |
|------|:---:|:---:|
| `mlflow.sklearn.autolog()` | ✅ | ❌ (manual) |
| metrics: accuracy | ✅ (autolog) | ✅ (manual) |
| params: n_estimators, max_depth | ✅ (autolog) | ✅ (manual) |
| `model/MLmodel` | ✅ (autolog) | ✅ (manual) |
| `model/model.pkl` | ✅ (autolog) | ✅ (manual) |
| `model/conda.yaml` | ✅ (autolog) | ✅ (manual) |
| `model/python_env.yaml` | ✅ (autolog) | ✅ (manual) |
| `model/requirements.txt` | ✅ (autolog) | ✅ (manual) |
| `estimator.html` | ✅ (autolog) | ❌ |
| confusion_matrix.png | ❌ | ✅ |
| classification_report.txt | ❌ | ✅ |
| feature_importance.png | ❌ | ✅ |
| Screenshot dashboard | ✅ | ✅ |
| Screenshot artifact | ✅ | ✅ |

---

## 6. Requirements

```
pandas
numpy
scikit-learn
mlflow
matplotlib
seaborn
```
