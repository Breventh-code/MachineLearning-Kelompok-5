"""Memuat pipeline hasil training dan menjalankan prediksi.

PENTING: joblib/pickle hanya boleh digunakan untuk berkas tepercaya.
Simpan pipeline (preprocessing + estimator) dari kedua ML Engineer
ke folder models/ sesuai nama berkas yang ditentukan.
"""
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from backend.schema import CLASS_LABELS, FEATURE_ORDER

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "supervised": ROOT / "models" / "random_forest_pipeline.pkl",
    "unsupervised": ROOT / "models" / "kmeans_pipeline.pkl",
}


class ModelUnavailable(Exception):
    pass


class ModelPredictionError(Exception):
    pass


@lru_cache(maxsize=8)
def _load_cached(path_string, modified_time_ns):
    return joblib.load(path_string)


def load_model(kind):
    path = MODEL_PATHS[kind]
    if not path.is_file():
        raise ModelUnavailable(
            f"Model {kind} belum tersedia. Tambahkan '{path.name}' ke folder models/."
        )
    try:
        return _load_cached(str(path), path.stat().st_mtime_ns)
    except Exception as exc:
        raise ModelUnavailable(
            f"Model '{path.name}' gagal dimuat. Periksa format dan versi library."
        ) from exc


def get_model_status():
    response = {}
    for kind, path in MODEL_PATHS.items():
        try:
            load_model(kind)
            response[kind] = {"ready": True, "filename": path.name, "message": "Model siap digunakan."}
        except ModelUnavailable as exc:
            response[kind] = {"ready": False, "filename": path.name, "message": str(exc)}
    return response


def _make_dataframe(features):
    # Urutan/nama kolom harus identik dengan saat training.
    return pd.DataFrame([[features[name] for name in FEATURE_ORDER]], columns=FEATURE_ORDER)


def _python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def predict_supervised(features):
    model = load_model("supervised")
    X = _make_dataframe(features)
    try:
        value = _python_scalar(model.predict(X)[0])
        probabilities = []
        if hasattr(model, "predict_proba"):
            row = model.predict_proba(X)[0]
            classes = getattr(model, "classes_", list(range(len(row))))
            for klass, probability in zip(classes, row):
                klass = _python_scalar(klass)
                probabilities.append({
                    "class_id": str(klass),
                    "label": CLASS_LABELS.get(str(klass), str(klass)),
                    "probability": round(float(probability), 6),
                })
        return {
            "prediction": str(value),
            "label": CLASS_LABELS.get(str(value), str(value)),
            "probabilities": probabilities,
            "note": "Hasil model eksperimental untuk pembelajaran, bukan diagnosis industri tersertifikasi.",
        }
    except Exception as exc:
        raise ModelPredictionError("Klasifikasi gagal. Pastikan skema fitur sama dengan saat training.") from exc


def predict_unsupervised(features):
    model = load_model("unsupervised")
    X = _make_dataframe(features)
    try:
        cluster_id = int(model.predict(X)[0])
        distances = []
        if hasattr(model, "transform"):
            try:
                row = np.ravel(model.transform(X))
                distances = [
                    {"cluster": int(i), "distance": round(float(d), 6)}
                    for i, d in enumerate(row)
                ]
            except (AttributeError, TypeError, ValueError):
                pass  # Distances bersifat opsional.
        return {
            "cluster": cluster_id,
            "label": f"Cluster {cluster_id}",
            "distances": distances,
            "note": "Nomor cluster bukan label jenis kerusakan. Interpretasi memerlukan analisis lanjutan.",
        }
    except Exception as exc:
        raise ModelPredictionError("Clustering gagal. Pastikan skema fitur sama dengan saat training.") from exc
