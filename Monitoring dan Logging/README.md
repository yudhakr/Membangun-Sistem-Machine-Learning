# Monitoring dan Logging - MLflow Model Serving

**Dashboard Grafana Username:** yudha2112

---

## 1. Deskripsi Project

Project ini mengimplementasikan **Monitoring dan Logging** untuk model Machine Learning menggunakan:

- **MLflow Model Serving** — Serving model Wine Classification via REST API
- **Prometheus** — Monitoring dan koleksi metrics
- **Grafana** — Visualisasi dashboard dan alerting
- **Python (prometheus_client)** — Custom metrics exporter

---

## 2. Struktur Project

```
Monitoring dan Logging/
├── 1.bukti_serving/              # Screenshot model serving
├── 2.prometheus.yml               # Konfigurasi Prometheus
├── 3.prometheus_exporter.py        # Custom metrics exporter
├── 4.bukti monitoring Prometheus/ # Screenshot Prometheus
├── 5.bukti monitoring Grafana/    # Screenshot Grafana dashboard
├── 6.bukti alerting Grafana/      # Screenshot alerting
├── 7.inference.py                 # Testing endpoint model
├── requirements.txt               # Python dependencies
└── README.md                      # Dokumentasi
```

---

## 3. Cara Menjalankan MLflow Model Serving

### 3.1 Dapatkan Model URI

Cari Run ID dari MLflow (hasil Kriteria 3):

```bash
cd Workflow-CI/MLProject
mlflow ui
```

Buka `http://127.0.0.1:5000`, copy Run ID dari experiment terbaru.

Atau langsung dari direktori mlruns:

```bash
ls MLProject/mlruns/0/
```

### 3.2 Jalankan MLflow Serve

```bash
mlflow models serve \
  --model-uri runs:/<RUN_ID>/model \
  --port 1234 \
  --env-manager local
```

Contoh dengan Run ID:

```bash
mlflow models serve \
  --model-uri runs:/b94ad844050a4e21ad93f2dba8d9c24a/model \
  --port 1234 \
  --env-manager local
```

Model akan berjalan di: **http://127.0.0.1:1234**

### 3.3 Test Endpoint

```bash
# Test health
curl http://127.0.0.1:1234/ping

# Test prediction
python 7.inference.py
```

### 3.4 Contoh Curl Prediction

```bash
curl -X POST http://127.0.0.1:1234/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"],
      "data": [[14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]]
    }
  }'
```

---

## 4. Cara Menjalankan Prometheus

### 4.1 Install Prometheus

Download dari: https://prometheus.io/download/

Atau via Docker:

```bash
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v "$(pwd)/2.prometheus.yml:/etc/prometheus/prometheus.yml" \
  prom/prometheus
```

### 4.2 Jalankan Prometheus Exporter

```bash
cd "Monitoring dan Logging"
pip install -r requirements.txt
python 3.prometheus_exporter.py
```

Exporter berjalan di: **http://localhost:8000/metrics**

### 4.3 Buka Prometheus UI

**http://localhost:9090**

Query metrics:

| Metric | Query |
|--------|-------|
| Request count | `mlflow_request_count_total` |
| Latency | `mlflow_request_latency_seconds` |
| Failed requests | `mlflow_failed_requests_total` |
| CPU | `system_cpu_usage_percent` |
| Memory | `system_memory_usage_bytes` |
| Disk | `system_disk_usage_bytes` |
| Prediction count | `mlflow_prediction_count_total` |
| Active users | `application_active_users` |
| Response time | `mlflow_response_time_seconds` |
| Model accuracy | `mlflow_model_accuracy` |

---

## 5. Cara Menjalankan Grafana

### 5.1 Install Grafana

Via Docker:

```bash
docker run -d \
  --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

Atau download dari: https://grafana.com/grafana/download

### 5.2 Login Grafana

1. Buka **http://localhost:3000**
2. Login: `admin` / `admin` (ubah password jika diminta)

### 5.3 Setup Dashboard (yudha2112)

1. Klik **Connections** → **Add data source**
2. Pilih **Prometheus**
3. Set URL: `http://host.docker.internal:9090` (Docker) atau `http://localhost:9090` (local)
4. Klik **Save & Test**

### 5.4 Create Dashboard

1. Klik **Dashboards** → **New Dashboard**
2. Set dashboard name: **yudha2112**
3. Tambahkan panel berikut:

#### Panel 1: Request Count
- **Query**: `rate(mlflow_request_count_total[1m])`
- **Visualization**: Time series / Stat
- **Title**: Request Count

#### Panel 2: Request Latency
- **Query**: `mlflow_request_latency_seconds_sum / mlflow_request_latency_seconds_count`
- **Visualization**: Time series / Stat
- **Title**: Average Request Latency

#### Panel 3: CPU Usage
- **Query**: `system_cpu_usage_percent`
- **Visualization**: Gauge / Stat
- **Title**: CPU Usage (%)

#### Panel 4: Memory Usage
- **Query**: `system_memory_usage_bytes{type="used"}`
- **Visualization**: Time series
- **Title**: Memory Usage (bytes)

#### Panel 5: Disk Usage
- **Query**: `system_disk_usage_bytes{type="percent"}`
- **Visualization**: Gauge
- **Title**: Disk Usage (%)

#### Panel 6: Failed Requests
- **Query**: `rate(mlflow_failed_requests_total[5m])`
- **Visualization**: Stat / Time series
- **Title**: Failed Requests Rate

#### Panel 7: Prediction Count
- **Query**: `mlflow_prediction_count_total`
- **Visualization**: Bar gauge / Pie chart
- **Title**: Prediction Distribution

#### Panel 8: Active Users
- **Query**: `application_active_users`
- **Visualization**: Stat / Time series
- **Title**: Active Users

#### Panel 9: Response Time
- **Query**: `mlflow_response_time_seconds_sum / mlflow_response_time_seconds_count`
- **Visualization**: Time series
- **Title**: Response Time (avg)

#### Panel 10: Model Accuracy
- **Query**: `mlflow_model_accuracy`
- **Visualization**: Stat / Gauge
- **Title**: Model Accuracy

---

## 6. Cara Membuat Alerting di Grafana

### Alert 1: CPU Usage > 80% (SKILLED)
1. Buka panel **CPU Usage**
2. Klik **Alert** → **Create alert rule**
3. Condition: `WHEN max() OF query (system_cpu_usage_percent) IS ABOVE 80`
4. Evaluation: `for 1m`
5. Set folder dan notification

### Alert 2: Failed Requests > 5 (ADVANCE)
1. Buka panel **Failed Requests**
2. Condition: `WHEN max() OF query (rate(mlflow_failed_requests_total[5m])) IS ABOVE 5`
3. Evaluation: `for 1m`

### Alert 3: Response Time > 2s (ADVANCE)
1. Buka panel **Response Time**
2. Condition: `WHEN max() OF query (mlflow_response_time_seconds_sum / mlflow_response_time_seconds_count) IS ABOVE 2`
3. Evaluation: `for 1m`

### Screenshot Alerting
Simpan screenshot ke folder:
- `6.bukti alerting Grafana/rules_alert.jpg` — Rules alert
- `6.bukti alerting Grafana/notification_alert.jpg` — Notification alert

---

## 7. Daftar Metrics Prometheus

| Metric | Type | Level | Deskripsi |
|--------|------|-------|-----------|
| `mlflow_request_count` | Counter | BASIC | Total request ke model |
| `mlflow_request_latency_seconds` | Histogram | BASIC | Latency request model |
| `mlflow_failed_requests_total` | Counter | BASIC | Request gagal |
| `system_cpu_usage_percent` | Gauge | SKILLED | CPU usage |
| `system_memory_usage_bytes` | Gauge | SKILLED | Memory usage |
| `system_disk_usage_bytes` | Gauge | ADVANCE | Disk usage |
| `mlflow_prediction_count` | Counter | ADVANCE | Prediksi per kelas |
| `application_active_users` | Gauge | ADVANCE | User aktif |
| `mlflow_response_time_seconds` | Histogram | ADVANCE | Response time |
| `mlflow_model_accuracy` | Gauge | ADVANCE | Akurasi model |

---

## 8. Cara Generate Screenshots

### Serving (folder 1.bukti_serving/)
```bash
# Terminal 1: MLflow serve
mlflow models serve --model-uri runs:/<RUN_ID>/model --port 1234 --env-manager local

# Terminal 2: Run inference
python 7.inference.py

# Terminal 3: Curl test
curl -X POST http://127.0.0.1:1234/invocations -H "Content-Type: application/json" -d '{"dataframe_split":{"columns":["alcohol"],"data":[[14.23]]}}'
```
Screenshot terminal output.

### Prometheus (folder 4.bukti monitoring Prometheus/)
1. Buka http://localhost:9090
2. Query beberapa metrics
3. Screenshot tampilan

### Grafana (folder 5.bukti monitoring Grafana/)
1. Buka http://localhost:3000
2. Buka dashboard **yudha2112**
3. Screenshot seluruh dashboard

### Alerting (folder 6.bukti alerting Grafana/)
1. Buka **Alerting** → **Alert rules**
2. Screenshot rules
3. Trigger alert untuk screenshot notification

---

## 9. Requirements

```
prometheus_client==0.21.1
psutil==6.1.1
requests==2.32.3
numpy==2.2.0
```
