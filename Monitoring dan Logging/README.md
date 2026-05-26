# Monitoring dan Logging — MLflow Model Serving

---

## BAGIAN 1 — KONSEP KRITERIA 4

### 1.1 Apa Itu Kriteria 4?

Kriteria 4 pada submission Dicoding "Membangun Sistem Machine Learning" adalah **Model Serving dan Monitoring**. Kriteria ini menguji kemampuan Anda untuk:

- **Menyajikan** (*serve*) model ML yang sudah dilatih sebagai REST API yang bisa diakses oleh aplikasi lain
- **Memantau** (*monitor*) performa model dan sistem secara real-time
- **Menampilkan** metrics dalam dashboard visual

Bukan sekadar menjalankan script Python untuk testing, melainkan membangun sistem observability yang lengkap.

### 1.2 Tujuan Observability dan Monitoring

Observability pada machine learning system bertujuan untuk:

1. **Mengetahui apakah model berjalan** — Serving aktif dan bisa menerima request
2. **Mengetahui performa model** — Seberapa cepat response (latency), berapa banyak request
3. **Mengetahui kesehatan sistem** — CPU, memory, disk usage
4. **Mendeteksi masalah** — Ketika latency naik, request gagal, atau resource habis
5. **Membuktikan ke reviewer** — Bahwa model benar-benar di-deploy dan dimonitor

### 1.3 Mengapa Reviewer Menolak?

| Alasan Penolakan | Penjelasan |
|-----------------|------------|
| Hanya menjalankan `inference.py` | Itu hanya testing client, bukan serving. Reviewer ingin melihat server yang melayani request, bukan script yang mengirim request. |
| Metrics dummy/random | `random.uniform()`, `random.choice()` bukan data real. Metrics harus berasal dari inference asli ke model serving. |
| Tidak ada screenshot monitoring | Tanpa bukti visual (screenshot Prometheus/Grafana), reviewer tidak bisa memverifikasi bahwa monitoring benar-benar berjalan. |

### 1.4 Perbedaan Istilah

| Istilah | Definisi | Contoh |
|---------|----------|--------|
| **Inference biasa** | Memuat model di memory lalu predict langsung | `model.predict(X)` di Python |
| **Model Serving** | Model berjalan sebagai server REST API yang bisa diakses via HTTP | `mlflow models serve` → endpoint `/invocations` |
| **Monitoring** | Mengumpulkan metrics dari system dan model secara periodik | Prometheus scraping metrics tiap 5 detik |
| **Observability** | Kemampuan untuk memahami keadaan sistem dari data yang dikumpulkan | Dashboard Grafana yang menampilkan semua metrics |

### 1.5 Arsitektur Sistem yang Benar

```
                      +------------------+
                      |   MLflow Serving  |
                      |  localhost:5001   |
                      +--------+---------+
                               |
                     Real inference via HTTP
                               |
                      +--------+---------+
                      | Prometheus        |
                      | Exporter (Python) |
                      | localhost:8000    |
                      +--------+---------+
                               |
                     Scrape metrics tiap 5s
                               |
                      +--------+---------+
                      |    Prometheus     |
                      |  localhost:9090   |
                      +--------+---------+
                               |
                     Query metrics via URL
                               |
                      +--------+---------+
                      |     Grafana      |
                      |  localhost:3000   |
                      +------------------+
```

**Alur data:**
1. **Exporter** mengirim request real ke **MLflow Serving** setiap 5 detik
2. **Exporter** mencatat latency, status code, prediction class dari response asli
3. **Exporter** juga mencatat CPU, memory, disk via **psutil**
4. **Prometheus** mengambil (*scrape*) metrics dari **Exporter** setiap 5 detik
5. **Grafana** menampilkan metrics dari **Prometheus** dalam bentuk dashboard

**Tidak ada data random/dummy.** Semua metrics berasal dari sumber nyata.

---

## BAGIAN 2 — MODEL SERVING (WAJIB)

### 2.1 Apa Itu MLflow Model Serving?

MLflow Model Serving adalah fitur MLflow yang memungkinkan model Machine Learning berjalan sebagai **server REST API**. Server ini menunggu request HTTP masuk, melakukan prediksi, dan mengembalikan hasilnya dalam format JSON.

Model yang sudah dilatih dengan `mlflow.sklearn.log_model()` bisa langsung di-serve tanpa menulis kode server tambahan.

### 2.2 Mengapa `python inference.py` BUKAN Serving?

| | `python inference.py` | `mlflow models serve` |
|---|---|---|
| **Peran** | Client / pengirim request | Server / penerima request |
| **Cara kerja** | Script aktif, kirim request, selesai | Server terus berjalan menunggu request |
| **Endpoint** | Tidak ada | `/ping`, `/invocations` |
| **Bukti serving** | Tidak valid | Valid — terminal menunjukkan "Listening at:..." |

`inference.py` hanya **mengetes** apakah serving berjalan. Bukan serving itu sendiri.

### 2.3 Cara Mendapatkan RUN_ID

**Langkah 1:** Jalankan training terlebih dahulu:

```bash
cd Membangun_model
python modelling.py
```

**Langkah 2:** Catat Run ID dari output terminal:

```
2025/01/01 10:00:00 INFO mlflow.utils.autologging: MLflow Run ID: abc123def456
```

Angka `abc123def456` itulah **RUN_ID**.

**Alternatif:** Buka MLflow UI di http://127.0.0.1:5000, klik run terbaru, copy Run ID dari URL:

```
http://127.0.0.1:5000/#/experiments/1/runs/abc123def456
```

### 2.4 Cara Menjalankan Serving

```bash
mlflow models serve -m runs:/<RUN_ID>/model -p 5001 --no-conda
```

**Penjelasan parameter:**

| Parameter | Fungsi |
|-----------|--------|
| `-m runs:/<RUN_ID>/model` | URI model di MLflow. Format: `runs:/<RUN_ID>/model` |
| `-p 5001` | Port server. Gunakan 5001 agar tidak bentrok dengan MLflow UI (port 5000) |
| `--no-conda` | Gunakan environment Python yang sedang aktif. Tanpa flag ini, MLflow akan membuat environment conda baru yang memakan waktu lama |

**Contoh dengan RUN_ID:**

```bash
mlflow models serve -m runs:/abc123def456/model -p 5001 --no-conda
```

**Tunggu hingga muncul:**
```
Listening at: http://127.0.0.1:5001
```

Ini tanda bahwa serving **sudah aktif** dan siap menerima request.

### 2.5 Endpoint yang Tersedia

| Endpoint | Method | Fungsi | Contoh Response |
|----------|--------|--------|-----------------|
| `http://127.0.0.1:5001/ping` | GET | Cek kesehatan server | `OK` |
| `http://127.0.0.1:5001/invocations` | POST | Kirim data untuk diprediksi | `{"predictions": [0]}` |

**Penjelasan:**
- **`/ping`** — Mengembalikan `OK` jika server hidup. Gunakan untuk cek apakah serving jalan.
- **`/invocations`** — Menerima input fitur dalam format JSON dan mengembalikan hasil prediksi.

### 2.6 Cara Test Endpoint

#### A. Via Browser
Buka: **http://127.0.0.1:5001/ping** — akan tampil `OK`.

#### B. Via curl (Command Line)

**Test health:**
```bash
curl http://127.0.0.1:5001/ping
```
Response: `OK`

**Test prediction:**
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

#### C. Via Python (inference.py)
```bash
cd "Monitoring dan Logging"
python 7.inference.py
```

### 2.7 Contoh Request dan Response yang Benar

**Request yang dikirim:**
```json
{
  "dataframe_split": {
    "columns": ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"],
    "data": [[14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]]
  }
}
```

**Response yang benar:**
```json
{
  "predictions": [0]
}
```

- `0` = kelas `class_0`
- `1` = kelas `class_1`
- `2` = kelas `class_2`

### 2.8 Cara Memastikan Serving Aktif

1. Buka terminal, jalankan serving → harus muncul `Listening at: http://127.0.0.1:5001`
2. Buka terminal lain, jalankan `curl http://127.0.0.1:5001/ping` → response `OK`
3. Kirim prediction → response `{"predictions": [...]}`

Jika semua 3 langkah berhasil, **serving sudah benar-benar aktif**.

### 2.9 Screenshot Serving yang Benar

Untuk submission, screenshot harus menunjukkan **3 hal berikut dalam satu gambar atau beberapa gambar:**

1. **Terminal serving aktif** — Menampilkan perintah `mlflow models serve -m runs:/.../model -p 5001 --no-conda` dan output `Listening at: http://127.0.0.1:5001`
2. **Endpoint prediction berhasil** — Terminal curl atau inference.py menunjukkan response status 200
3. **Response prediction muncul** — Output `{"predictions": [0]}` atau hasil prediksi kelas lainnya

**Langkah screenshot:**
1. Buka terminal, jalankan MLflow serve
2. Tunggu hingga muncul `Listening at: http://127.0.0.1:5001`
3. Buka terminal kedua, jalankan curl
4. Screenshot kedua terminal secara bersamaan (atau screenshot terpisah)

Simpan di folder: `1.bukti_serving/serving_terminal.jpg`

---

## BAGIAN 3 — PROMETHEUS EXPORTER

### 3.1 Fungsi Prometheus Exporter

Prometheus Exporter adalah **jembatan** antara aplikasi (MLflow serving) dan Prometheus. Tugasnya:

1. Mengumpulkan metrics dari berbagai sumber (system, aplikasi)
2. Menyajikan metrics dalam format yang bisa dibaca Prometheus
3. Menyediakan endpoint `/metrics` di HTTP port tertentu

### 3.2 Mengapa Metrics Dummy/Random Tidak Diperbolehkan?

Reviewer menolak karena:

- `random.uniform(0.01, 0.3)` → Latency dibuat-buat, bukan dari request asli
- `random.choice(["class_0", "class_1", "class_2"])` → Prediksi dibuat-buat, bukan dari model asli
- `random.randint(1, 50)` → User aktif dibuat-buat

**Ini dianggap curang** karena metrics harus mencerminkan keadaan nyata sistem.

**Solusi:** Ganti semua nilai random dengan data dari sumber nyata:
- Latency → dari response time asli `requests.post()` ke MLflow serving
- Prediction class → dari response JSON asli MLflow
- CPU → dari `psutil.cpu_percent()`
- Memory → dari `psutil.virtual_memory()`

### 3.3 Cara Membuat Metrics REAL

#### a. Inference Asli (bukan random)

```python
import requests
import json

start = time.time()
resp = requests.post(
    "http://127.0.0.1:5001/invocations",
    data=json.dumps(payload),
    headers={"Content-Type": "application/json"},
    timeout=10
)
latency = time.time() - start  # LATENCY ASLI, bukan random
```

#### b. Request Count Asli

```python
request_count.labels(endpoint="/invocations").inc()
```

Setiap kali selesai request, counter bertambah 1.

#### c. Latency Asli

```python
request_latency.labels(endpoint="/invocations").observe(latency)
```

`latency` adalah selisih waktu sebelum dan sesudah request (waktu asli).

#### d. Prediction Class Asli

```python
result = resp.json()
pred = result["predictions"][0]  # 0, 1, atau 2 (ASLI DARI MODEL)
class_name = ["class_0", "class_1", "class_2"][int(pred)]
prediction_count.labels(predicted_class=class_name).inc()
```

#### e. CPU Usage Asli

```python
import psutil
cpu_usage.set(psutil.cpu_percent(interval=1))  # CPU ASLI
```

#### f. Memory Usage Asli

```python
mem = psutil.virtual_memory()
memory_usage.labels(type="used").set(mem.used)  # MEMORY ASLI
memory_usage.labels(type="available").set(mem.available)
memory_usage.labels(type="percent").set(mem.percent)
```

### 3.4 Cara Menggunakan Counter, Histogram, Gauge

| Type | Fungsi | Method | Contoh |
|------|--------|--------|--------|
| **Counter** | Hitungan yang hanya naik (monotonic) | `.inc()` | Jumlah request, jumlah error |
| **Histogram** | Distribusi nilai (latency, response time) | `.observe(value)` | Latency request dalam detik |
| **Gauge** | Nilai yang bisa naik/turun (snapshot) | `.set(value)` | CPU %, memory bytes |

**Kapan menggunakan masing-masing:**
- `Counter` → request count, failed requests, prediction count
- `Histogram` → request latency, response time (karena kita ingin tahu distribusi, misalnya P95)
- `Gauge` → CPU usage, memory usage, disk usage

### 3.5 Cara Membuat Endpoint `/metrics`

```python
from prometheus_client import start_http_server

PORT = 8000
start_http_server(PORT)
```

Setelah ini, Prometheus exporter otomatis menyediakan endpoint `http://localhost:8000/metrics`.

### 3.6 Cara Menjalankan Exporter

```bash
cd "Monitoring dan Logging"
pip install -r requirements.txt
python 3.prometheus_exporter.py
```

Output yang diharapkan:
```
INFO:root:Prometheus exporter running on http://localhost:8000/metrics
INFO:root:System metrics collector started (real psutil data)
INFO:root:Model metrics collector started (real inference to MLflow serving)
```

### 3.7 Cara Memastikan Metrics Muncul

Buka browser: **http://localhost:8000/metrics**

Atau via curl:
```bash
curl http://localhost:8000/metrics
```

Output:
```
# HELP mlflow_request_count Total request ke model MLflow
# TYPE mlflow_request_count counter
mlflow_request_count{endpoint="/invocations"} 12.0
# HELP system_cpu_usage_percent CPU usage dalam persen
# TYPE system_cpu_usage_percent gauge
system_cpu_usage_percent 23.5
# HELP system_memory_usage_bytes Memory usage dalam bytes
# TYPE system_memory_usage_bytes gauge
system_memory_usage_bytes{type="used"} 8.123e9
```

Jika muncul seperti di atas, **exporter berjalan dengan benar**.

### 3.8 Conflict Port dan Solusinya

**Masalah:** Jika port 8000 sudah dipakai aplikasi lain, exporter akan error:
```
OSError: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000): ...
```

**Solusi:**
1. Cek port yang dipakai: `netstat -ano | findstr :8000`
2. Hentikan aplikasi yang memakai port 8000
3. Atau ganti PORT di `3.prometheus_exporter.py` ke port lain (misal 8001)
4. Jangan lupa update `2.prometheus.yml` juga jika port diganti

### 3.9 Kenapa Exporter Port 8000, Bukan 9090?

**Port 9090** sudah dipakai oleh **Prometheus UI**. Jika exporter juga menggunakan 9090, akan terjadi bentrok.

Pembagian port:
| Komponen | Port |
|----------|------|
| MLflow UI / Tracking Server | 5000 |
| MLflow Model Serving | 5001 |
| Prometheus Exporter | **8000** |
| Prometheus UI | 9090 |
| Grafana UI | 3000 |

### 3.10 Contoh Kode Exporter Lengkap dan Benar

File `Monitoring dan Logging/3.prometheus_exporter.py`:

**Struktur kode:**
1. **Import library** — `requests`, `psutil`, `prometheus_client`
2. **Definisi metrics** — Counter, Histogram, Gauge
3. **Fungsi `collect_system_metrics()`** — Kumpulkan CPU, memory, disk via psutil (real)
4. **Fungsi `collect_model_metrics()`** — Kirim request real ke MLflow serving, catat latency & prediction
5. **Fungsi `main()`** — Start HTTP server di port 8000, jalankan thread

**Tidak ada:**
- ❌ `import random`
- ❌ `random.uniform()`
- ❌ `random.choice()`
- ❌ `random.randint()`

**Yang ada:**
- ✅ `requests.post(MLFLOW_SERVING_URL, ...)` — inference real
- ✅ `time.time()` — latency real
- ✅ `psutil.cpu_percent()` — CPU real
- ✅ `psutil.virtual_memory()` — memory real

---

## BAGIAN 4 — PROMETHEUS

### 4.1 Apa Fungsi Prometheus?

Prometheus adalah sistem monitoring dan alerting open-source. Fungsinya:

1. **Scraping** — Mengambil data metrics dari target secara periodik
2. **Storage** — Menyimpan data metrics dalam time-series database
3. **Query** — Menyediakan bahasa query (PromQL) untuk menganalisis data
4. **Alerting** — Memberi notifikasi ketika kondisi tertentu terpenuhi

### 4.2 Cara Kerja Scraping Metrics

```
Prometheus ──── GET http://localhost:8000/metrics ────► Exporter
     │                                                      │
     │  <─── mlflow_request_count 12                         │
     │  <─── system_cpu_usage_percent 23.5                   │
     │  <─── system_memory_usage_bytes{type="used"} 8.1e9    │
     │                                                      │
     └─────► Simpan ke Time-Series DB                       │
```

Prometheus secara berkala (tiap `scrape_interval` detik) mengakses URL `/metrics` dari target, membaca data, dan menyimpannya.

### 4.3 Struktur `prometheus.yml`

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "wine_monitoring"
    static_configs:
      - targets: ["localhost:8000"]
```

### 4.4 Penjelasan Setiap Bagian

| Bagian | Nilai | Arti |
|--------|-------|------|
| `scrape_interval: 5s` | 5 detik | Prometheus mengambil data setiap 5 detik |
| `job_name: "wine_monitoring"` | Nama job | Label untuk mengidentifikasi sumber data |
| `targets: ["localhost:8000"]` | alamat:port | Alamat exporter yang akan di-scrape |

**Mengapa 5 detik?** Karena exporter juga mengirim request ke MLflow setiap 5 detik. Interval scraping harus sama atau lebih cepat agar tidak ada data yang terlewat.

### 4.5 Cara Menjalankan Prometheus di Windows

**Manual (download dari https://prometheus.io/download/):**

1. Download `prometheus-2.x.x.windows-amd64.zip`
2. Ekstrak ke folder, misal `C:\prometheus`
3. Copy file `2.prometheus.yml` atau jalankan dengan:

```bash
prometheus.exe --config.file="E:\Dicoding\Membangun Sistem Machine Learning\Membangun Sistem Machine Learning\Monitoring dan Logging\2.prometheus.yml"
```

**Via Docker (lebih mudah):**

```bash
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v "E:\Dicoding\Membangun Sistem Machine Learning\Membangun Sistem Machine Learning\Monitoring dan Logging\2.prometheus.yml:/etc/prometheus/prometheus.yml" \
  prom/prometheus
```

Atau via docker-compose (direkomendasikan):

```bash
cd "Monitoring dan Logging"
docker compose up -d
```

### 4.6 Cara Mengatasi Error Umum

#### Error: `query.active` / Port already in use

**Masalah:** Port 9090 sudah dipakai.

**Cek:**
```bash
netstat -ano | findstr :9090
```

**Solusi:**
- Hentikan proses yang memakai port 9090
- Atau jalankan Prometheus di port lain: `--web.listen-address=0.0.0.0:9091`

#### Error: Targets DOWN

**Penyebab:** Prometheus tidak bisa menghubungi exporter di `localhost:8000`.

**Cek:**
1. Apakah exporter sudah jalan? `python 3.prometheus_exporter.py`
2. Apakah port benar? Cek di `http://localhost:8000/metrics`
3. Apakah firewall memblokir?

### 4.7 Cara Memastikan Targets UP

1. Buka **http://localhost:9090/targets**
2. Cari baris `wine_monitoring`
3. Status harus **UP** (hijau)

```
Status:  UP
Labels:  job="wine_monitoring"
Scrape URL:  http://localhost:8000/metrics
Last Scrape:  2.3s ago
```

Jika masih DOWN, periksa:
- Exporter berjalan? (`python 3.prometheus_exporter.py`)
- Port exporter sesuai dengan `prometheus.yml`?

### 4.8 Cara Menggunakan Query Prometheus

1. Buka **http://localhost:9090**
2. Klik tab **Graph**
3. Ketik query di kotak input
4. Klik **Execute**
5. Lihat hasil di tab **Table** (tabel) atau **Graph** (grafik)

### 4.9 Penjelasan Query Prometheus

#### `mlflow_request_count_total`

Menampilkan total request yang diterima model.

```
mlflow_request_count{endpoint="/invocations"}  42
```

Arti: Sudah ada 42 request ke endpoint `/invocations`.

#### `system_cpu_usage_percent`

Menampilkan penggunaan CPU saat ini.

```
system_cpu_usage_percent  35.2
```

Arti: CPU terpakai 35.2%.

#### `system_memory_usage_bytes`

Menampilkan penggunaan memory.

```
system_memory_usage_bytes{type="used"}  8.123456e9
system_memory_usage_bytes{type="total"}  16.0e9
```

Arti: Memory terpakai 8.1 GB dari total 16 GB.

#### `histogram_quantile(0.95, rate(mlflow_request_latency_seconds_bucket[5m]))`

Query paling penting untuk monitoring latency.

**Penjelasan:**
- `rate(...[5m])` — Hitung rata-rata request per detik dalam 5 menit terakhir
- `mlflow_request_latency_seconds_bucket` — Bucket latency (0.005, 0.01, 0.025, ...)
- `histogram_quantile(0.95, ...)` — Hitung percentile ke-95

**Artinya:** "95% request selesai dalam waktu X detik". Jika hasilnya 0.5, berarti 95% request selesai dalam 0.5 detik.

### 4.10 Cara Menghasilkan Traffic Inference

Agar grafik Prometheus muncul, harus ada data yang masuk. Ada 2 sumber traffic:

**Sumber 1: Exporter (otomatis)**
Exporter sudah mengirim request ke MLflow setiap 5 detik. Ini cukup untuk mengisi grafik.

**Sumber 2: Manual (traffic tambahan)**
Jalankan `7.inference.py` atau loop curl untuk traffic lebih padat:

```bash
# PowerShell
for ($i=0; $i -lt 30; $i++) {
  curl.exe -X POST http://127.0.0.1:5001/invocations `
    -H "Content-Type: application/json" `
    -d '{"dataframe_split":{"columns":["alcohol","malic_acid","ash","alcalinity_of_ash","magnesium","total_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins","color_intensity","hue","od280/od315_of_diluted_wines","proline"],"data":[[14.23,1.71,2.43,15.6,127.0,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065.0]]}}'
  Start-Sleep -Seconds 1
}
```

### 4.11 Screenshot Prometheus yang Benar

Untuk submission, Anda perlu 2-3 screenshot:

**1. Status → Targets UP**
- Buka http://localhost:9090/targets
- Screenshoot baris `wine_monitoring` dengan status **UP** (hijau)
- Simpan sebagai `4.bukti monitoring Prometheus/prometheus_targets_up.jpg`

**2. Query Berhasil (Execute)**
- Masuk ke tab **Graph**
- Query: `mlflow_request_count_total`
- Klik **Execute**
- Screenshot hasil query (tabel atau grafik)
- Simpan sebagai `4.bukti monitoring Prometheus/prometheus_query_request_count.jpg`

**3. Grafik Metrics Muncul**
- Query: `rate(mlflow_request_count_total[1m])`
- Switch ke **Graph**
- Screenshot grafik yang menunjukkan garis naik
- Simpan sebagai `4.bukti monitoring Prometheus/prometheus_graph.jpg`

---

## BAGIAN 5 — GRAFANA

### 5.1 Apa Fungsi Grafana?

Grafana adalah platform visualisasi data open-source. Fungsinya:

1. **Dashboard** — Menampilkan metrics dalam bentuk grafik, gauge, tabel
2. **Multi-source** — Bisa connect ke berbagai data source (Prometheus, InfluxDB, dll)
3. **Alerting** — Memberi notifikasi ketika metrics melebihi threshold
4. **Visualization** — Berbagai jenis grafik (time series, bar, gauge, stat)

### 5.2 Cara Install Grafana

**Via Docker (direkomendasikan):**
```bash
docker run -d --name grafana -p 3000:3000 grafana/grafana
```

**Manual:** Download dari https://grafana.com/grafana/download

### 5.3 Cara Menjalankan Grafana

```bash
docker run -d --name grafana -p 3000:3000 grafana/grafana
```

Atau jika sudah ada di docker-compose:
```bash
cd "Monitoring dan Logging"
docker compose up -d grafana
```

Tunggu beberapa detik, lalu buka http://localhost:3000.

### 5.4 Login Default Grafana

| Field | Value |
|-------|-------|
| URL | http://localhost:3000 |
| Username | `admin` |
| Password | `admin` |

Setelah login pertama, Grafana akan meminta ubah password. Klik **Skip** jika tidak ingin.

### 5.5 Cara Menambahkan Prometheus Data Source

1. Buka http://localhost:3000
2. Kiri: **Connections** → **Add new connection**
3. Cari **Prometheus**, klik
4. Klik **Add new data source**
5. Isi **URL**: `http://localhost:9090`
   - Jika Grafana via Docker: `http://host.docker.internal:9090`
   - Jika Grafana lokal: `http://localhost:9090`
6. Scroll ke bawah, klik **Save & Test**
7. Harus muncul **"Data source is working"** (hijau)

### 5.6 Cara Menghubungkan Grafana ke Prometheus

Jika Grafana berjalan di **Docker** dan Prometheus juga di **Docker**, gunakan network internal:

```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
    depends_on: [prometheus]
```

URL datasource di Grafana: `http://prometheus:9090` (gunakan nama service, bukan localhost).

Jika Grafana di **Docker** dan Prometheus di **Windows (host)**, URL datasource: `http://host.docker.internal:9090`.

### 5.7 Cara Membuat Dashboard Monitoring

1. Klik **Dashboards** → **New Dashboard** → **Add visualization**
2. Pilih data source **Prometheus**
3. Masukkan query PromQL
4. Pilih visualization type
5. Atur judul panel
6. Klik **Apply**
7. Ulangi untuk panel berikutnya
8. Klik **Save dashboard** (ikon floppy disk di atas)

### 5.8 Cara Membuat Panel-Panel Monitoring

#### Panel 1: Request Count

| Setting | Value |
|---------|-------|
| **Query** | `rate(mlflow_request_count_total[1m])` |
| **Legend** | `{{endpoint}}` |
| **Visualization** | Time series |
| **Title** | Request Rate |
| **Unit** | ops (operations per second) |

Cara membuat:
1. Di tab **Query**, masukkan `rate(mlflow_request_count_total[1m])`
2. Klik **Run queries** — pastikan muncul data
3. Di tab **Settings**, ganti judul ke "Request Rate"
4. Klik **Apply**

#### Panel 2: Latency

| Setting | Value |
|---------|-------|
| **Query** | `mlflow_request_latency_seconds_sum / mlflow_request_latency_seconds_count` |
| **Legend** | `{{endpoint}}` |
| **Visualization** | Time series |
| **Title** | Average Latency (s) |
| **Unit** | seconds |

#### Panel 3: CPU Usage

| Setting | Value |
|---------|-------|
| **Query** | `system_cpu_usage_percent` |
| **Visualization** | Gauge |
| **Title** | CPU Usage (%) |
| **Unit** | percent |
| **Thresholds** | Base: green (0), yellow (80), red (90) |

#### Panel 4: Memory Usage

| Setting | Value |
|---------|-------|
| **Query A** | `system_memory_usage_bytes{type="used"}` |
| **Query B** | `system_memory_usage_bytes{type="total"}` |
| **Visualization** | Time series |
| **Title** | Memory Usage |
| **Unit** | bytes |

### 5.9 Visualization Graph yang Tersedia

| Jenis | Fungsi | Cocok Untuk |
|-------|--------|-------------|
| **Time series** | Grafik garis seiring waktu | Request count, latency, memory |
| **Gauge** | Indikator seperti speedometer | CPU usage, disk usage |
| **Stat** | Angka besar dengan icon | Request count total, failed count |
| **Bar gauge** | Batang horizontal/vertikal | Prediction per class |
| **Pie chart** | Diagram lingkaran | Distribusi prediksi |

### 5.10 Contoh Dashboard Monitoring Profesional

Dashboard yang baik memiliki:

1. **Header row** — Judul dashboard dan waktu
2. **Row 1: Overview** — Stat penting (total request, avg latency, error rate)
3. **Row 2: Performance** — Time series request count + latency
4. **Row 3: System** — CPU + memory gauge
5. **Row 4: Advanced** — Prediction distribution, disk usage

### 5.11 Screenshot Grafana yang Benar

Untuk submission, screenshot harus menunjukkan:

1. **Dashboard aktif** — Seluruh panel menampilkan data (tidak ada tulisan "No data")
2. **Datasource connected** — Bisa screenshot halaman datasource yang menunjukkan "Data source is working"
3. **Grafik request count** — Time series dengan data naik
4. **Grafik latency** — Ada data latency
5. **Grafik CPU/memory** — Menampilkan usage real-time

Simpan di folder `5.bukti monitoring Grafana/grafana_dashboard.jpg`.

---

## BAGIAN 6 — DOCKER (OPSIONAL TAPI DIREKOMENDASIKAN)

### 6.1 Keuntungan Menggunakan Docker

1. **Tidak perlu install manual** — Prometheus dan Grafala langsung jalan
2. **Portable** — Bisa dijalankan di laptop mana pun
3. **Terpisah** — Tidak mengotori system dengan installasi
4. **Tampilan profesional** — Screenshot Docker Desktop menarik untuk reviewer

### 6.2 Cara Install Docker Desktop

1. Download dari https://www.docker.com/products/docker-desktop/
2. Install dan restart komputer
3. Buka Docker Desktop, pastikan status **Running**

### 6.3 docker-compose.yml

File `Monitoring dan Logging/docker-compose.yml`:

```yaml
version: "3.8"

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./2.prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    restart: unless-stopped
    depends_on:
      - prometheus
```

**Penjelasan:**
- `image` — Nama image Docker yang akan di-download
- `container_name` — Nama container (mudah diidentifikasi)
- `ports` — Mapping port host:container
- `volumes` — File konfigurasi prometheus.yml di-mount ke container
- `depends_on` — Grafana akan start setelah Prometheus
- `restart: unless-stopped` — Auto-restart jika container mati

### 6.4 Cara Menjalankan

```bash
cd "Monitoring dan Logging"
docker compose up -d
```

Flag `-d` artinya **detached mode** (berjalan di background).

Output:
```
[+] Running 2/2
 ✔ Container prometheus  Started
 ✔ Container grafana     Started
```

### 6.5 Cara Memastikan Container Running

```bash
docker ps
```

Output:
```
CONTAINER ID   IMAGE                  STATUS         PORTS
abc123         prom/prometheus        Up 2 minutes   0.0.0.0:9090->9090/tcp
def456         grafana/grafana        Up 2 minutes   0.0.0.0:3000->3000/tcp
```

Atau buka **Docker Desktop** → lihat container **Running**.

### 6.6 Cara Mendapatkan Tampilan Docker Desktop

Untuk screenshot submission:
1. Buka Docker Desktop
2. Klik **Containers** di kiri
3. Pastikan container `prometheus` dan `grafana` status **Running**
4. Screenshot seluruh tampilan
5. Simpan sebagai `docker_desktop.jpg`

---

## BAGIAN 7 — STRUKTUR FOLDER SUBMISSION

### 7.1 Struktur Final

```
Monitoring dan Logging/
├── 1.bukti_serving/
│   └── serving_terminal.jpg          # Screenshot MLflow serve aktif + response
│
├── 2.prometheus.yml                   # Konfigurasi Prometheus
│
├── 3.prometheus_exporter.py           # Exporter metrics real (tanpa random)
│
├── 4.bukti monitoring Prometheus/
│   ├── prometheus_targets_up.jpg      # Status → Targets → UP
│   └── prometheus_query_request_count.jpg  # Query berhasil + grafik
│
├── 5.bukti monitoring Grafana/
│   └── grafana_dashboard.jpg          # Dashboard aktif dengan semua panel
│
├── 6.bukti alerting Grafana/
│   └── alert_rules.jpg               # (Opsional) Rules alerting
│
├── 7.inference.py                     # Testing endpoint model
│
├── docker-compose.yml                 # (Opsional) Docker Compose
│
├── requirements.txt                   # Python dependencies
│
└── README.md                          # Dokumentasi ini
```

### 7.2 Isi Screenshot per Folder

| Folder | Screenshot Wajib | Isi Screenshot |
|--------|-------------------|----------------|
| `1.bukti_serving/` | ✅ | Terminal MLflow serve (Listening at: 5001) + curl response prediction |
| `4.bukti monitoring Prometheus/` | ✅ | http://localhost:9090/targets status UP + query grafik |
| `5.bukti monitoring Grafana/` | ✅ | Dashboard dengan panel request, latency, CPU, memory (ada data) |
| `6.bukti alerting Grafana/` | ❌ (Opsional) | Alert rules yang sudah dibuat |

### 7.3 Cara Membuat Screenshot yang Baik

1. **Full window screenshot** — Jangan crop terlalu kecil
2. **Pastikan data muncul** — Jangan screenshot saat "No data" atau "Waiting for data"
3. **Beri waktu** — Tunggu 1-2 menit setelah semua services jalan agar data terisi
4. **Cahaya cukup** — Screenshot jelas, tidak blur
5. **Nama file jelas** — Gunakan nama yang deskriptif

---

## BAGIAN 8 — VALIDASI FINAL

### Checklist Sebelum Upload

Gunakan checklist berikut untuk memastikan submission valid:

#### A. Model Serving

| No | Item | Status |
|----|------|--------|
| 1 | `mlflow models serve -m runs:/<RUN_ID>/model -p 5001 --no-conda` berjalan | ☐ |
| 2 | Terminal menunjukkan `Listening at: http://127.0.0.1:5001` | ☐ |
| 3 | `curl http://127.0.0.1:5001/ping` → `OK` | ☐ |
| 4 | `curl -X POST http://127.0.0.1:5001/invocations` → response prediction | ☐ |
| 5 | Screenshot serving ada di `1.bukti_serving/` | ☐ |

#### B. Prometheus Exporter

| No | Item | Status |
|----|------|--------|
| 6 | `python 3.prometheus_exporter.py` berjalan tanpa error | ☐ |
| 7 | `http://localhost:8000/metrics` menampilkan semua metrics | ☐ |
| 8 | Tidak ada `import random` di `3.prometheus_exporter.py` | ☐ |
| 9 | Tidak ada `random.uniform()`, `random.choice()`, `random.randint()` | ☐ |
| 10 | Metrics berasal dari `requests.post()` ke MLflow (bukan random) | ☐ |
| 11 | CPU/memory dari `psutil` (bukan random) | ☐ |
| 12 | Port exporter = 8000 (bukan 9090) | ☐ |

#### C. Prometheus

| No | Item | Status |
|----|------|--------|
| 13 | `prometheus.exe --config.file=...` atau `docker compose up -d` berjalan | ☐ |
| 14 | `http://localhost:9090/targets` → status **UP** | ☐ |
| 15 | Query `mlflow_request_count_total` menghasilkan data | ☐ |
| 16 | Query `system_cpu_usage_percent` menghasilkan data | ☐ |
| 17 | Query `system_memory_usage_bytes` menghasilkan data | ☐ |
| 18 | Query `histogram_quantile(0.95, rate(...))` menghasilkan data | ☐ |
| 19 | Grafik Prometheus muncul (time series) | ☐ |
| 20 | Screenshot Prometheus ada di `4.bukti monitoring Prometheus/` | ☐ |

#### D. Grafana

| No | Item | Status |
|----|------|--------|
| 21 | Grafana berjalan di `http://localhost:3000` | ☐ |
| 22 | Login admin/admin berhasil | ☐ |
| 23 | Prometheus datasource terhubung ("Data source is working") | ☐ |
| 24 | Dashboard dengan panel Request Count ada data | ☐ |
| 25 | Dashboard dengan panel Latency ada data | ☐ |
| 26 | Dashboard dengan panel CPU Usage ada data | ☐ |
| 27 | Dashboard dengan panel Memory Usage ada data | ☐ |
| 28 | Screenshot Grafana ada di `5.bukti monitoring Grafana/` | ☐ |

#### E. Final Check

| No | Item | Status |
|----|------|--------|
| 29 | Semua screenshot jelas dan terbaca | ☐ |
| 30 | Tidak ada metrics dummy/random di kode | ☐ |
| 31 | Folder `1.bukti_serving/` terisi | ☐ |
| 32 | Folder `4.bukti monitoring Prometheus/` terisi | ☐ |
| 33 | Folder `5.bukti monitoring Grafana/` terisi | ☐ |
| 34 | README.md terdokumentasi dengan baik | ☐ |
| 35 | `requirements.txt` berisi dependensi yang diperlukan | ☐ |

### Catatan Penting

1. **Jalankan semua services secara bersamaan** — MLflow serve, exporter, Prometheus, dan Grafana harus berjalan semua agar data terisi
2. **Tunggu data terisi sebelum screenshot** — Minimal 1-2 menit setelah semua services jalan
3. **Gunakan Docker** jika memungkinkan — Lebih mudah dan tampilan lebih profesional
4. **Jangan gunakan random** — Reviewer akan mengecek kode, pastikan tidak ada `import random`
5. **Screenshot harus jelas** — Jangan screenshot yang blur atau terlalu kecil

---

## Lampiran: Alur Menjalankan (Quick Start)

### Jalankan Semua Services

```
Terminal 1:  mlflow models serve -m runs:/<RUN_ID>/model -p 5001 --no-conda
Terminal 2:  python Monitoring\ dan\ Logging/3.prometheus_exporter.py
Terminal 3:  cd Monitoring\ dan\ Logging && docker compose up -d
             (Prometheus di :9090, Grafana di :3000)
Terminal 4:  python Monitoring\ dan\ Logging/7.inference.py
             (Traffic tambahan untuk grafik)
```

### Akses Semua UI

| Service | URL |
|---------|-----|
| MLflow Model Serving | http://127.0.0.1:5001 |
| Prometheus Exporter | http://localhost:8000/metrics |
| Prometheus UI | http://localhost:9090 |
| Grafana UI | http://localhost:3000 |

### Screenshot yang Wajib

1. `1.bukti_serving/serving_terminal.jpg` — Terminal serve + curl response
2. `4.bukti monitoring Prometheus/prometheus_targets_up.jpg` — Targets UP
3. `4.bukti monitoring Prometheus/prometheus_query_request_count.jpg` — Query + grafik
4. `5.bukti monitoring Grafana/grafana_dashboard.jpg` — Dashboard dengan data

---

Semoga dengan panduan ini submission Kriteria 4 Anda lolos review. Kunci utamanya: **semua metrics harus REAL, bukan random; serving harus benar-benar aktif sebagai server; screenshot harus menunjukkan data yang nyata.**
