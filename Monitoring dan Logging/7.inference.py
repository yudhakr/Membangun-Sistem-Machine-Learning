"""
inference.py
Testing endpoint MLflow Model Serving.
Mengirim sample data Wine dan menampilkan response prediksi.
"""

import json
import time
import logging

import requests
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Endpoint MLflow Model Serving (port 5001 dengan --no-conda)
MODEL_ENDPOINT = "http://127.0.0.1:5001/invocations"

# Sample data Wine (3 sampel, satu untuk setiap kelas)
SAMPLE_DATA = [
    {
        "alcohol": 14.23,
        "malic_acid": 1.71,
        "ash": 2.43,
        "alcalinity_of_ash": 15.6,
        "magnesium": 127.0,
        "total_phenols": 2.8,
        "flavanoids": 3.06,
        "nonflavanoid_phenols": 0.28,
        "proanthocyanins": 2.29,
        "color_intensity": 5.64,
        "hue": 1.04,
        "od280/od315_of_diluted_wines": 3.92,
        "proline": 1065.0
    },
    {
        "alcohol": 12.37,
        "malic_acid": 0.94,
        "ash": 1.36,
        "alcalinity_of_ash": 10.6,
        "magnesium": 88.0,
        "total_phenols": 1.98,
        "flavanoids": 0.57,
        "nonflavanoid_phenols": 0.28,
        "proanthocyanins": 0.42,
        "color_intensity": 1.95,
        "hue": 1.05,
        "od280/od315_of_diluted_wines": 1.82,
        "proline": 520.0
    },
    {
        "alcohol": 13.71,
        "malic_acid": 5.65,
        "ash": 2.45,
        "alcalinity_of_ash": 20.5,
        "magnesium": 95.0,
        "total_phenols": 1.68,
        "flavanoids": 0.61,
        "nonflavanoid_phenols": 0.52,
        "proanthocyanins": 1.06,
        "color_intensity": 7.70,
        "hue": 0.64,
        "od280/od315_of_diluted_wines": 1.74,
        "proline": 740.0
    }
]


def predict_single(features: dict) -> dict:
    """
    Kirim satu sampel ke MLflow model serving.

    Parameters:
        features (dict): Dictionary fitur wine.

    Returns:
        dict: Response dari model.
    """
    # MLflow serving expects dataframe_split format
    payload = {
        "dataframe_split": {
            "columns": list(features.keys()),
            "data": [[features[col] for col in features]]
        }
    }

    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    response = requests.post(
        MODEL_ENDPOINT,
        data=json.dumps(payload),
        headers=headers,
        timeout=10
    )
    latency = time.time() - start_time

    return {
        "status_code": response.status_code,
        "latency_seconds": round(latency, 4),
        "response": response.json() if response.status_code == 200 else response.text
    }


def predict_batch(samples: list) -> list:
    """
    Kirim batch sampel ke MLflow model serving.

    Parameters:
        samples (list): List dictionary fitur wine.

    Returns:
        list: List response dari model.
    """
    results = []
    for i, sample in enumerate(samples):
        logger.info(f"Predicting sample {i + 1}/{len(samples)}...")
        result = predict_single(sample)
        results.append(result)
        time.sleep(0.5)
    return results


def main():
    """
    Fungsi utama: testing MLflow model serving.
    """
    CLASS_NAMES = ["class_0", "class_1", "class_2"]

    logger.info("=" * 60)
    logger.info("  MLflow Model Serving - Inference Test")
    logger.info("=" * 60)
    logger.info(f"  Endpoint: {MODEL_ENDPOINT}")
    logger.info(f"  Samples : {len(SAMPLE_DATA)}")
    logger.info("Starting inference...")

    # Test koneksi
    try:
        health = requests.get("http://127.0.0.1:5001/ping", timeout=5)
        logger.info(f"  Server status: {'ONLINE' if health.status_code == 200 else 'UNKNOWN'}")
    except requests.exceptions.ConnectionError:
        logger.error("  Server is OFFLINE! Jalankan 'mlflow models serve' terlebih dahulu.")
        logger.error("  Command: mlflow models serve -m runs:/<RUN_ID>/model -p 5001 --no-conda")
        return

    logger.info("Sending predictions...")

    # Kirim prediksi
    results = predict_batch(SAMPLE_DATA)

    # Tampilkan hasil
    print()
    print("=" * 60)
    print("  HASIL PREDIKSI")
    print("=" * 60)

    for i, result in enumerate(results):
        print(f"\n  Sample {i + 1}:")
        print(f"  {'Status':<20}: {result['status_code']}")
        print(f"  {'Latency':<20}: {result['latency_seconds']}s")

        if result["status_code"] == 200:
            resp = result["response"]
            # MLflow returns predictions in various formats
            if isinstance(resp, list):
                pred = resp[0]
            elif isinstance(resp, dict):
                pred = resp.get("predictions", resp)
                if isinstance(pred, list):
                    pred = pred[0]
            else:
                pred = resp

            class_name = CLASS_NAMES[int(pred)] if isinstance(pred, (int, float)) else str(pred)
            print(f"  {'Predicted Class':<20}: {class_name}")
        else:
            print(f"  {'Error':<20}: {result['response']}")

    print()
    print("=" * 60)
    print("  Inference test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
