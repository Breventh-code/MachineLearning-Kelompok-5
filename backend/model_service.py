"""Inference dua model SmartBearing yang menggunakan 13 fitur CWRU TERSKALA.

Model dari ML Engineer dilatih pada train_scaled.csv (X tanpa kolom label).
Input web adalah 9 fitur mentah; transform_input menghitung 4 fitur
baru dan menerapkan scaler yang dibuat Data Engineer, tepat satu kali.

joblib.load() hanya aman digunakan pada artefak yang dipercaya.
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import FEATURES, transform_input

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_PATHS = {
    "supervised": MODELS_DIR / "random_forest_pipeline.pkl",
    "unsupervised": MODELS_DIR / "kmeans_pipeline.pkl",
}
REQUIRED_PREPROCESSING = ("scaler.pkl", "label_encoder.pkl", "feature_list.json")


class ModelUnavailable(Exception):
    pass


class ModelPredictionError(Exception):
    pass


@lru_cache(maxsize=16)
def _load_cached(path_string, modified_time_ns):
    return joblib.load(path_string)


def _load_artifact(path):
    return _load_cached(str(path), path.stat().st_mtime_ns)


def _verify_artifacts():
    missing = [name for name in REQUIRED_PREPROCESSING if not (MODELS_DIR / name).is_file()]
    if missing:
        raise ModelUnavailable(
            "Jalankan preprocessing CWRU terlebih dahulu; artefak belum ada: "
            + ", ".join(missing)
        )
    try:
        details = json.loads((MODELS_DIR / "feature_list.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelUnavailable("feature_list.json tidak dapat dibaca.") from exc
    if details.get("features") != FEATURES or details.get("target") != "fault_type":
        raise ModelUnavailable("feature_list.json bukan konfigurasi final 4 kelas / 13 fitur.")


def load_model(kind):
    if kind not in MODEL_PATHS:
        raise ValueError("Jenis model tidak tersedia")
    _verify_artifacts()
    path = MODEL_PATHS[kind]
    if not path.is_file():
        raise ModelUnavailable(f"Model {kind} belum tersedia: {path.name}.")
    try:
        model = _load_artifact(path)
    except Exception as exc:
        raise ModelUnavailable(f"Model {path.name} gagal dibaca. Periksa model dan versi library.") from exc
    if not hasattr(model, "predict"):
        raise ModelUnavailable(f"{path.name} tidak memiliki method predict().")
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and n_features != len(FEATURES):
        raise ModelUnavailable(f"{path.name} dilatih untuk {n_features} fitur; seharusnya 13.")
    return model


def get_model_status():
    result = {}
    for kind, path in MODEL_PATHS.items():
        try:
            load_model(kind)
            result[kind] = {"ready": True, "filename": path.name, "message": "Model siap digunakan."}
        except ModelUnavailable as exc:
            result[kind] = {"ready": False, "filename": path.name, "message": str(exc)}
    return result


def _prepare_features(raw_values):
    """Gunakan schema/scaler yang sama dengan pipeline data engineer."""
    try:
        scaled = transform_input(raw_values, model_dir=MODELS_DIR)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise ModelPredictionError("Preprocessing input gagal: " + str(exc)) from exc
    if scaled.shape != (1, len(FEATURES)):
        raise ModelPredictionError("Bentuk fitur tidak sesuai kontrak model 13 fitur.")
    return pd.DataFrame(scaled, columns=FEATURES)


def _encoder():
    try:
        return _load_artifact(MODELS_DIR / "label_encoder.pkl")
    except Exception as exc:
        raise ModelPredictionError("Label encoder gagal dimuat.") from exc


def _class_label(value, encoder):
    """class indices are from sklearn LabelEncoder, never guess their order."""
    try:
        return str(encoder.inverse_transform([int(value)])[0])
    except (ValueError, TypeError, IndexError) as exc:
        raise ModelPredictionError("Model mengeluarkan class id yang tidak ada di label encoder.") from exc


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def predict_supervised(features):
    model = load_model("supervised")
    X = _prepare_features(features)
    encoder = _encoder()
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
                    "label": _class_label(klass, encoder),
                    "probability": round(float(probability), 6),
                })
        return {
            "prediction": str(value),
            "label": _class_label(value, encoder),
            "probabilities": probabilities,
            "note": "Hasil model eksperimental, bukan diagnosis industri tersertifikasi.",
        }
    except ModelPredictionError:
        raise
    except Exception as exc:
        raise ModelPredictionError("Klasifikasi gagal. Cek kecocokan model dengan 13 fitur scaled.") from exc


def predict_unsupervised(features):
    model = load_model("unsupervised")
    X = _prepare_features(features)
    try:
        cluster_id = int(model.predict(X)[0])
        distances = []
        if hasattr(model, "transform"):
            try:
                distance_row = np.ravel(model.transform(X))
                distances = [{"cluster": i, "distance": round(float(d), 6)}
                             for i, d in enumerate(distance_row)]
            except (AttributeError, TypeError, ValueError):
                pass
        return {
            "cluster": cluster_id,
            "label": f"Cluster {cluster_id}",
            "distances": distances,
            "note": "Nomor cluster bukan label jenis kerusakan tanpa validasi interpretasi.",
        }
    except Exception as exc:
        raise ModelPredictionError("Clustering gagal. Cek kecocokan KMeans dengan 13 fitur scaled.") from exc
