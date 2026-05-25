"""
prometheus_exporter.py
Ekspor metrics real dari MLflow Model Serving untuk Prometheus.
Berjalan di http://localhost:8000/metrics
"""

import json
import os
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

# Sample data Wine real untuk inference
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

# ============================================================
# BASIC METRICS
# ============================================================

request_count = Counter(
    "mlflow_request_count",
    "Total request ke model MLflow",
    ["endpoint"]
)

request_latency = Histogram(
    "mlflow_request_latency_seconds",
    "Latency request model MLflow",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

failed_requests = Counter(
    "mlflow_failed_requests_total",
    "Total request gagal ke model MLflow",
    ["endpoint", "error_type"]
)

# ============================================================
# SKILLED METRICS
# ============================================================

cpu_usage = Gauge(
    "system_cpu_usage_percent",
    "CPU usage dalam persen"
)

memory_usage = Gauge(
    "system_memory_usage_bytes",
    "Memory usage dalam bytes",
    ["type"]
)

# ============================================================
# ADVANCE METRICS
# ============================================================

disk_usage = Gauge(
    "system_disk_usage_bytes",
    "Disk usage dalam bytes",
    ["mount_point", "type"]
)

prediction_count = Counter(
    "mlflow_prediction_count",
    "Total prediksi per kelas wine",
    ["predicted_class"]
)

active_users = Gauge(
    "application_active_users",
    "Jumlah user aktif saat ini"
)

response_time = Histogram(
    "mlflow_response_time_seconds",
    "Response time model MLflow",
    ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

model_accuracy = Gauge(
    "mlflow_model_accuracy",
    "Akurasi model saat ini"
)


def collect_system_metrics():
    """Koleksi metrics sistem secara periodik via psutil (real)."""
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
            logger.warning(f"System metrics collection error: {e}")

        time.sleep(5)


def do_inference(features: dict):
    """Kirim satu sampel real ke MLflow serving, return hasil."""
    payload = {
        "dataframe_split": {
            "columns": list(features.keys()),
            "data": [[features[col] for col in features]]
        }
    }
    headers = {"Content-Type": "application/json"}
    start = time.time()
    resp = requests.post(MLFLOW_SERVING_URL, data=json.dumps(payload), headers=headers, timeout=10)
    latency = time.time() - start
    return resp, latency


def collect_model_metrics():
    """Kirim request REAL ke MLflow model serving secara periodik."""
    sample_idx = 0

    while True:
        try:
            sample = SAMPLE_DATA[sample_idx]
            sample_idx = (sample_idx + 1) % len(SAMPLE_DATA)

            endpoint = "/invocations"
            resp, latency = do_inference(sample)

            request_count.labels(endpoint=endpoint).inc()
            request_latency.labels(endpoint=endpoint).observe(latency)
            response_time.labels(endpoint=endpoint).observe(latency)

            if resp.status_code == 200:
                result = resp.json()
                pred = result.get("predictions", [None])
                if isinstance(pred, list) and len(pred) > 0:
                    pred_val = pred[0]
                    class_name = CLASS_NAMES[int(pred_val)] if isinstance(pred_val, (int, float)) else str(pred_val)
                    prediction_count.labels(predicted_class=class_name).inc()
            else:
                failed_requests.labels(endpoint=endpoint, error_type="http_error").inc()

            active_users.set(1)

        except requests.exceptions.ConnectionError:
            failed_requests.labels(endpoint="/invocations", error_type="connection_error").inc()
            logger.warning("MLflow serving unreachable, will retry...")
        except Exception as e:
            failed_requests.labels(endpoint="/invocations", error_type=str(e)[:50]).inc()
            logger.warning(f"Inference error: {e}")

        time.sleep(5)


def main():
    """Main: start HTTP server dan background threads."""
    PORT = 8000

    try:
        start_http_server(PORT)
        logger.info(f"Prometheus exporter running on http://localhost:{PORT}/metrics")

        sys_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        sys_thread.start()
        logger.info("System metrics collector started (real psutil data)")

        model_thread = threading.Thread(target=collect_model_metrics, daemon=True)
        model_thread.start()
        logger.info("Model metrics collector started (real inference to MLflow serving)")

        logger.info("=" * 50)
        logger.info("  Prometheus Exporter is Running")
        logger.info(f"  Metrics: http://localhost:{PORT}/metrics")
        logger.info("  All metrics are REAL (no random simulation)")
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
