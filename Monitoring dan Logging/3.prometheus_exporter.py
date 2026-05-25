"""
prometheus_exporter.py
Ekspor metrics sistem dan model MLflow untuk Prometheus.
Berjalan di http://localhost:8000/metrics
"""

import os
import sys
import time
import random
import logging
import threading

import psutil
import requests
from prometheus_client import start_http_server, Counter, Histogram, Gauge

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# BASIC METRICS
# ============================================================

# Total request yang diterima
request_count = Counter(
    "mlflow_request_count",
    "Total request ke model MLflow",
    ["endpoint"]
)

# Latency histogram (detik)
request_latency = Histogram(
    "mlflow_request_latency_seconds",
    "Latency request model MLflow",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

# Jumlah request gagal
failed_requests = Counter(
    "mlflow_failed_requests_total",
    "Total request gagal ke model MLflow",
    ["endpoint", "error_type"]
)

# ============================================================
# SKILLED METRICS
# ============================================================

# CPU usage (persen)
cpu_usage = Gauge(
    "system_cpu_usage_percent",
    "CPU usage dalam persen"
)

# Memory usage (bytes)
memory_usage = Gauge(
    "system_memory_usage_bytes",
    "Memory usage dalam bytes",
    ["type"]
)

# ============================================================
# ADVANCE METRICS
# ============================================================

# Disk usage (bytes)
disk_usage = Gauge(
    "system_disk_usage_bytes",
    "Disk usage dalam bytes",
    ["mount_point", "type"]
)

# Total prediksi per kelas
prediction_count = Counter(
    "mlflow_prediction_count",
    "Total prediksi per kelas wine",
    ["predicted_class"]
)

# Active users (simulasi)
active_users = Gauge(
    "application_active_users",
    "Jumlah user aktif saat ini"
)

# Response time (detik, sama dengan latency)
response_time = Histogram(
    "mlflow_response_time_seconds",
    "Response time model MLflow",
    ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Model accuracy (simulasi)
model_accuracy = Gauge(
    "mlflow_model_accuracy",
    "Akurasi model saat ini"
)


def collect_system_metrics():
    """Koleksi metrics sistem secara periodik."""
    while True:
        try:
            # CPU
            cpu_usage.set(psutil.cpu_percent(interval=1))

            # Memory
            mem = psutil.virtual_memory()
            memory_usage.labels(type="total").set(mem.total)
            memory_usage.labels(type="used").set(mem.used)
            memory_usage.labels(type="available").set(mem.available)
            memory_usage.labels(type="percent").set(mem.percent)

            # Disk
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


def simulate_activity():
    """Simulasi aktivitas user dan request untuk demonstrasi metrics."""
    while True:
        try:
            # Simulasi active users (random 1-50)
            active_users.set(random.randint(1, 50))

            # Simulasi model accuracy (fluktuasi kecil antara 0.92-1.0)
            accuracy = round(random.uniform(0.92, 1.0), 4)
            model_accuracy.set(accuracy)

            # Simulasi request
            endpoint = "/invocations"
            with request_latency.labels(endpoint=endpoint).time():
                time.sleep(random.uniform(0.01, 0.3))

            response_time.labels(endpoint=endpoint).observe(random.uniform(0.01, 0.3))
            request_count.labels(endpoint=endpoint).inc()

            # Simulasi prediksi per kelas
            kelas = random.choice(["class_0", "class_1", "class_2"])
            prediction_count.labels(predicted_class=kelas).inc()

            # Simulasi failed requests (5% chance)
            if random.random() < 0.05:
                failed_requests.labels(
                    endpoint=endpoint,
                    error_type="prediction_error"
                ).inc()

        except Exception as e:
            logger.warning(f"Activity simulation error: {e}")

        time.sleep(2)


def main():
    """Main: start HTTP server dan background threads."""
    PORT = 8000

    try:
        # Start Prometheus HTTP server
        start_http_server(PORT)
        logger.info(f"Prometheus exporter running on http://localhost:{PORT}/metrics")

        # Start system metrics collector thread
        sys_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        sys_thread.start()
        logger.info("System metrics collector started")

        # Start activity simulator thread
        act_thread = threading.Thread(target=simulate_activity, daemon=True)
        act_thread.start()
        logger.info("Activity simulator started")

        logger.info("=" * 50)
        logger.info("  Prometheus Exporter is Running")
        logger.info(f"  Metrics: http://localhost:{PORT}/metrics")
        logger.info("  Basic   : request_count, request_latency, failed_requests")
        logger.info("  Skilled : cpu_usage, memory_usage")
        logger.info("  Advance : disk_usage, prediction_count, active_users")
        logger.info("            response_time, model_accuracy")
        logger.info("=" * 50)

        # Keep main thread alive
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
