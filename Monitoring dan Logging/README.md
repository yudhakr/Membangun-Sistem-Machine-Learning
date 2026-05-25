# Monitoring dan Logging — MLflow Model Serving

---

## 1. Deskripsi Project

Project ini mengimplementasikan **Model Serving** dan **Monitoring Real-Time** untuk model klasifikasi Wine menggunakan:

- **MLflow Model Serving** — Menyajikan model RandomForestClassifier sebagai REST API
- **Prometheus** — Koleksi metrics real dari system dan model serving
- **Grafana** — Visualisasi dashboard monitoring dan alerting
- **Python (prometheus_client)** — Custom exporter dengan metrics REAL (bukan dummy/random)

Dataset yang digunakan: **Wine Dataset** dari Scikit-Learn (13 fitur, 3 kelas).

---

## 2. Struktur Folder

```
Monitoring dan Logging/
├── 1.bukti_serving/                 # Screenshot model serving aktif
├── 2.prometheus.yml                  # Konfigurasi Prometheus
├── 3.prometheus_exporter.py          # Exporter metrics real (tanpa random)
├── 4.bukti monitoring Prometheus/    # Screenshot Prometheus UI
├── 5.bukti monitoring Grafana/       # Screenshot Grafana dashboard
├── 6.bukti alerting Grafana/         # Screenshot alerting rules
├── 7.inference.py                    # Testing endpoint model
├── requirements.txt                  # Python dependencies
└── README.md                         # Dokumentasi
```

---

## BAGIAN 1 — MODEL SERVING

### 1.1 Mendapatkan RUN_ID

Jalankan training terlebih dahulu:

```bash
cd Membangun_model
python modelling.py
```

Setelah selesai, catat **Run ID** dari output terminal, contoh:

```
MLflow Run ID: abc123def456
```

Atau lihat di MLflow UI: http://127.0.0.1:5000

### 1.2 Menjalankan Model Serving

Gunakan perintah berikut:

```bash
mlflow models serve -m runs:/<RUN_ID>/model -p 5001 --no-conda
```

Contoh dengan Run ID:

```bash
mlflow models serve -m runs:/abc123def456/model -p 5001 --no-conda
```

**Penjelasan:**
- `-m runs:/<RUN_ID>/model` — URI model dari MLflow
- `-p 5001` — Port server (hindari bentrok dengan MLflow UI di 5000)
- `--no-conda` — Gunakan environment Python aktif, bukan create conda env baru

Setelah berjalan, akan muncul:
```
Listening at: http://127.0.0.1:5001
```

### 1.3 Endpoint yang Tersedia

| Endpoint | Method | Fungsi |
|----------|--------|--------|
| `http://127.0.0.1:5001/ping` | GET | Health check |
| `http://127.0.0.1:5001/invocations` | POST | Inference prediction |

### 1.4 Testing Endpoint dengan curl

#### Test Health
```bash
curl http://127.0.0.1:5001/ping
```
Response:
```
OK
```

#### Test Prediction
```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["alcohol","malic_acid","ash","alcalinity_of_ash","magnesium","total_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins","color_intensity","hue","od280/od315_of_diluted_wines","proline"],
      "data": [[14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]]
    }
  }'
```

**Response yang diharapkan:**
```json
{"predictions": [0]}
```
Nilai `0` adalah kelas `class_0`.

### 1.5 Testing dengan inference.py

```bash
cd "Monitoring dan Logging"
python 7.inference.py
```

### 1.6 Screenshot Model Serving yang Benar

Untuk submission, screenshot harus menunjukkan:

1. **Terminal MLflow Serve aktif** — menampilkan `Listening at: http://127.0.0.1:5001`
2. **Terminal curl/inference** — menampilkan response prediction berhasil (status 200)
3. **Response model muncul** — contoh: `{"predictions": [0]}`

Simpan ke folder `1.bukti_serving/` dengan nama `serving_terminal.jpg`.

---

## BAGIAN 2 — PROMETHEUS EXPORTER

### 2.1 Penjelasan

`3.prometheus_exporter.py` adalah custom exporter yang mengumpulkan 2 jenis metrics:

1. **System Metrics** — CPU, memory, disk usage via `psutil` (real)
2. **Model Metrics** — request count, latency, prediction class via inference REAL ke MLflow serving

**Tidak ada data dummy/random.** Semua metrics berasal dari sumber nyata.

### 2.2 Contoh Kode Exporter Lengkap

```python
"""
prometheus_exporter.py
Ekspor metrics real dari MLflow Model Serving untuk Prometheus.
Berjalan di http://localhost:8000/metrics
"""

import json
import sys
import time
import logging
import threading

import psutil
import requests
from prometheus_client import start_http_server, Counter, Histogram, Gauge

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# KONFIGURASI
# ============================================================
MLFLOW_SERVING_URL = "http://127.0.0.1:5001/invocations"
CLASS_NAMES = ["class_0", "class_1", "class_2"]

# Sample data Wine real (bukan random)
SAMPLE_DATA = [
    {
        "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43,
        "alcalinity_of_ash": 15.6, "magnesium": 127.0,
        "total_phenols": 2.8, "flavanoids": 3.06,
        "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29,
        "color_intensity": 5.64, "hue": 1.04,
        "od280/od315_of_diluted_wines": 3.92, "proline": 1065.0
    },
    {
        "alcohol": 12.37, "malic_acid": 0.94, "ash": 1.36,
        "alcalinity_of_ash": 10.6, "magnesium": 88.0,
        "total_phenols": 1.98, "flavanoids": 0.57,
        "nonflavanoid_phenols": 0.28, "proanthocyanins": 0.42,
        "color_intensity": 1.95, "hue": 1.05,
        "od280/od315_of_diluted_wines": 1.82, "proline": 520.0
    },
    {
        "alcohol": 13.71, "malic_acid": 5.65, "ash": 2.45,
        "alcalinity_of_ash": 20.5, "magnesium": 95.0,
        "total_phenols": 1.68, "flavanoids": 0.61,
        "nonflavanoid_phenols": 0.52, "proanthocyanins": 1.06,
        "color_intensity": 7.70, "hue": 0.64,
        "od280/od315_of_diluted_wines": 1.74, "proline": 740.0
    }
]

# --- DEFINE METRICS ---
# BASIC
request_count = Counter("mlflow_request_count", "Total request ke model MLflow", ["endpoint"])
request_latency = Histogram("mlflow_request_latency_seconds", "Latency request model MLflow", ["endpoint"],
                            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))
failed_requests = Counter("mlflow_failed_requests_total", "Total request gagal", ["endpoint", "error_type"])

# SKILLED
cpu_usage = Gauge("system_cpu_usage_percent", "CPU usage dalam persen")
memory_usage = Gauge("system_memory_usage_bytes", "Memory usage dalam bytes", ["type"])

# ADVANCE
disk_usage = Gauge("system_disk_usage_bytes", "Disk usage dalam bytes", ["mount_point", "type"])
prediction_count = Counter("mlflow_prediction_count", "Total prediksi per kelas wine", ["predicted_class"])
response_time = Histogram("mlflow_response_time_seconds", "Response time model MLflow", ["endpoint"],
                          buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0))


def collect_system_metrics():
    """Koleksi metrics sistem dari psutil (REAL, bukan simulasi)."""
    while True:
        try:
            cpu_usage.set(psutil.cpu_percent(interval=1))

            mem = psutil.virtual_memory()
            memory_usage.labels(type="total").set(mem.total)
            memory_usage.labels(type="used").set(mem.used)
            memory_usage.labels(type="available").set(mem.available)
            memory_usage.labels(type="percent").set(mem.percent)

            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage.labels(mount_point=partition.mountpoint, type="total").set(usage.total)
                    disk_usage.labels(mount_point=partition.mountpoint, type="used").set(usage.used)
                    disk_usage.labels(mount_point=partition.mountpoint, type="free").set(usage.free)
                    disk_usage.labels(mount_point=partition.mountpoint, type="percent").set(usage.percent)
                except PermissionError:
                    continue
        except Exception as e:
            logger.warning(f"System metrics error: {e}")

        time.sleep(5)


def collect_model_metrics():
    """Kirim request REAL ke MLflow serving, catat metrics asli."""
    sample_idx = 0

    while True:
        try:
            sample = SAMPLE_DATA[sample_idx]
            sample_idx = (sample_idx + 1) % len(SAMPLE_DATA)

            payload = {
                "dataframe_split": {
                    "columns": list(sample.keys()),
                    "data": [[sample[col] for col in sample]]
                }
            }
            headers = {"Content-Type": "application/json"}

            start = time.time()
            resp = requests.post(MLFLOW_SERVING_URL, data=json.dumps(payload),
                                 headers=headers, timeout=10)
            latency = time.time() - start

            endpoint = "/invocations"
            request_count.labels(endpoint=endpoint).inc()
            request_latency.labels(endpoint=endpoint).observe(latency)
            response_time.labels(endpoint=endpoint).observe(latency)

            if resp.status_code == 200:
                result = resp.json()
                pred = result.get("predictions", [None])
                if isinstance(pred, list) and len(pred) > 0:
                    pred_val = pred[0]
                    class_name = CLASS_NAMES[int(pred_val)]
                    prediction_count.labels(predicted_class=class_name).inc()
            else:
                failed_requests.labels(endpoint=endpoint, error_type="http_error").inc()

        except requests.exceptions.ConnectionError:
            failed_requests.labels(endpoint="/invocations", error_type="connection_error").inc()
            logger.warning("MLflow serving unreachable, retrying...")
        except Exception as e:
            failed_requests.labels(endpoint="/invocations", error_type=str(e)[:50]).inc()
            logger.warning(f"Inference error: {e}")

        time.sleep(5)


def main():
    PORT = 8000

    try:
        start_http_server(PORT)
        logger.info(f"Prometheus exporter running on http://localhost:{PORT}/metrics")

        sys_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        sys_thread.start()

        model_thread = threading.Thread(target=collect_model_metrics, daemon=True)
        model_thread.start()

        logger.info("=" * 50)
        logger.info("  Prometheus Exporter is Running")
        logger.info(f"  Metrics: http://localhost:{PORT}/metrics")
        logger.info("  All metrics are REAL (no random/uniform/choice)")
        logger.info("=" * 50)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Exporter stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Exporter failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### 2.3 Cara Menjalankan Exporter

```bash
cd "Monitoring dan Logging"
pip install -r requirements.txt
python 3.prometheus_exporter.py
```

Exporter akan berjalan di: **http://localhost:8000/metrics**

### 2.4 Cara Mengecek Metrics

Buka browser atau gunakan curl:

```bash
curl http://localhost:8000/metrics
```

Akan menampilkan output seperti:

```
# HELP mlflow_request_count Total request ke model MLflow
# TYPE mlflow_request_count counter
mlflow_request_count{endpoint="/invocations"} 12.0
# HELP system_cpu_usage_percent CPU usage dalam persen
# TYPE system_cpu_usage_percent gauge
system_cpu_usage_percent 23.5
```

### 2.5 Daftar Metrics yang Tersedia

| Metric | Type | Sumber |
|--------|------|--------|
| `mlflow_request_count_total` | Counter | Request real ke MLflow serving |
| `mlflow_request_latency_seconds` | Histogram | Latency real dari response MLflow |
| `mlflow_failed_requests_total` | Counter | Error real (connection / HTTP) |
| `system_cpu_usage_percent` | Gauge | psutil real-time |
| `system_memory_usage_bytes` | Gauge | psutil real-time |
| `system_disk_usage_bytes` | Gauge | psutil real-time |
| `mlflow_prediction_count_total` | Counter | Kelas prediksi real dari MLflow |
| `mlflow_response_time_seconds` | Histogram | Response time real dari MLflow |

---

## BAGIAN 3 — PROMETHEUS

### 3.1 Konfigurasi `prometheus.yml`

File `2.prometheus.yml`:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "wine_monitoring"
    static_configs:
      - targets: ["localhost:8000"]
```

**Penjelasan:**
- `scrape_interval: 5s` — Prometheus mengambil data setiap 5 detik
- `job_name: "wine_monitoring"` — Nama job untuk exporter kita
- `targets: ["localhost:8000"]` — Alamat exporter (Port 8000, bukan 9090)

### 3.2 Cara Menjalankan Prometheus

Download Prometheus dari https://prometheus.io/download/, ekstrak, lalu:

```bash
prometheus.exe --config.file="FULL_PATH/Monitoring dan Logging/2.prometheus.yml"
```

Ganti `FULL_PATH` dengan path absolut ke folder project, contoh:

```bash
prometheus.exe --config.file="E:/Dicoding/Membangun Sistem Machine Learning/Membangun Sistem Machine Learning/Monitoring dan Logging/2.prometheus.yml"
```

Atau via Docker:

```bash
docker run -d --name prometheus -p 9090:9090 -v "FULL_PATH/Monitoring dan Logging/2.prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus
```

### 3.3 Membuka Prometheus UI

Buka browser: **http://localhost:9090**

### 3.4 Memastikan Target UP

1. Klik menu **Status** → **Targets**
2. Pastikan job `wine_monitoring` berstatus **UP** (hijau)
3. Jika tidak UP, periksa apakah exporter sudah berjalan di port 8000

### 3.5 Query Prometheus

Gunakan tab **Graph** untuk menjalankan query berikut:

| Query | Fungsi |
|-------|--------|
| `mlflow_request_count_total` | Total request ke model |
| `system_cpu_usage_percent` | CPU usage saat ini |
| `system_memory_usage_bytes{type="used"}` | Memory terpakai |
| `histogram_quantile(0.95, rate(mlflow_request_latency_seconds_bucket[5m]))` | P95 latency |

### 3.6 Menghasilkan Traffic Inference

Agar grafik Prometheus muncul, exporter perlu mengirim request ke MLflow serving.

**Pastikan 3 komponen berjalan secara bersamaan:**

```
Terminal 1: mlflow models serve -m runs:/<RUN_ID>/model -p 5001 --no-conda
Terminal 2: python 3.prometheus_exporter.py
Terminal 3: prometheus.exe --config.file="...2.prometheus.yml"
```

Exporter otomatis mengirim request ke MLflow serving setiap 5 detik, sehingga grafik Prometheus akan terisi.

Untuk traffic tambahan, jalankan juga:

```bash
python 7.inference.py
```

Atau loop curl:

```bash
for ($i=0; $i -lt 20; $i++) {
  curl -X POST http://127.0.0.1:5001/invocations -H "Content-Type: application/json" -d '{...}'
  Start-Sleep -Seconds 1
}
```

### 3.7 Screenshot Prometheus yang Benar

Untuk submission, screenshot harus menunjukkan:

1. **Status → Targets** — job `wine_monitoring` berstatus **UP**
2. **Query metrics berhasil** — contoh query `mlflow_request_count_total` menghasilkan data
3. **Grafik metrics muncul** — grafik waktu (time series) untuk query yang dijalankan

Simpan ke folder `4.bukti monitoring Prometheus/` dengan nama `prometheus_targets.jpg` dan `prometheus_query.jpg`.

---

## BAGIAN 4 — GRAFANA

### 4.1 Menjalankan Grafana

Via Docker:

```bash
docker run -d --name grafana -p 3000:3000 grafana/grafana
```

Atau install dari: https://grafana.com/grafana/download

### 4.2 Login Grafana

1. Buka **http://localhost:3000**
2. Login default:
   - Username: `admin`
   - Password: `admin`
3. Ubah password jika diminta (lewati jika tidak ingin)

### 4.3 Menambahkan Prometheus sebagai Data Source

1. Klik **Connections** → **Add new connection** → **Prometheus**
2. Pada field **URL**, isi: `http://localhost:9090`
3. Jika Grafana via Docker, gunakan: `http://host.docker.internal:9090`
4. Scroll ke bawah, klik **Save & Test**
5. Pastikan muncul notifikasi hijau: **"Data source is working"**

### 4.4 Membuat Dashboard Monitoring

1. Klik **Dashboards** → **New Dashboard** → **Add visualization**
2. Pilih data source **Prometheus**
3. Buat panel-panel berikut:

#### Panel 1: Request Count
- **Query**: `rate(mlflow_request_count_total[1m])`
- **Legend**: `{{endpoint}}`
- **Visualization**: Time series
- **Panel title**: Request Rate

#### Panel 2: Request Latency
- **Query**: `mlflow_request_latency_seconds_sum / mlflow_request_latency_seconds_count`
- **Legend**: `{{endpoint}}`
- **Visualization**: Time series
- **Panel title**: Average Latency (s)

#### Panel 3: CPU Usage
- **Query**: `system_cpu_usage_percent`
- **Visualization**: Gauge
- **Panel title**: CPU Usage (%)
- **Thresholds**: 80 (yellow), 90 (red)

#### Panel 4: Memory Usage
- **Query A**: `system_memory_usage_bytes{type="used"}`
- **Query B**: `system_memory_usage_bytes{type="total"}`
- **Visualization**: Time series
- **Panel title**: Memory Usage

#### Panel 5 (Opsional): Prediction Distribution
- **Query**: `mlflow_prediction_count_total`
- **Visualization**: Bar gauge
- **Panel title**: Prediction per Class

#### Panel 6 (Opsional): Failed Requests
- **Query**: `rate(mlflow_failed_requests_total[5m])`
- **Visualization**: Stat
- **Panel title**: Failed Request Rate

### 4.5 Query Grafana yang Digunakan

| Panel | Query |
|-------|-------|
| Request Count | `rate(mlflow_request_count_total[1m])` |
| Latency | `mlflow_request_latency_seconds_sum / mlflow_request_latency_seconds_count` |
| CPU Usage | `system_cpu_usage_percent` |
| Memory Used | `system_memory_usage_bytes{type="used"}` |
| Prediction Count | `mlflow_prediction_count_total` |
| Failed Requests | `rate(mlflow_failed_requests_total[5m])` |

### 4.6 Screenshot Grafana yang Benar

Untuk submission, screenshot harus menunjukkan:

1. **Dashboard aktif** — seluruh panel menampilkan data (bukan "No data")
2. **Grafik request count** — ada grafik time series yang naik
3. **Grafik latency** — ada data latency dari request real
4. **Grafik CPU/memory** — menampilkan usage sistem real-time

Simpan ke folder `5.bukti monitoring Grafana/` dengan nama `grafana_dashboard.jpg`.

---

## 5. Alur Menjalankan Keseluruhan

```
Terminal 1: mlflow models serve -m runs:/<RUN_ID>/model -p 5001 --no-conda
Terminal 2: python Monitoring\ dan\ Logging/3.prometheus_exporter.py
Terminal 3: prometheus.exe --config.file="FULL_PATH/Monitoring dan Logging/2.prometheus.yml"
Terminal 4: python Monitoring\ dan\ Logging/7.inference.py          (opsional, untuk traffic tambahan)

Browser 1: http://localhost:9090   (Prometheus UI)
Browser 2: http://localhost:3000   (Grafana UI)
```

---

## 6. Requirements

```
prometheus_client
psutil
requests
numpy
```
