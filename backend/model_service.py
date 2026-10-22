"""Inference dua model SmartBearing yang menggunakan 13 fitur CWRU TERSKALA.

Model dari ML Engineer dilatih pada train_scaled.csv (X tanpa kolom label).
Input web adalah 9 fitur mentah; transform_input menghitung 4 fitur
baru dan menerapkan scaler yang dibuat Data Engineer, tepat satu kali.

joblib.load() hanya aman digunakan pada artefak yang dipercaya.
"""
import json
import hashlib
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import FEATURES, transform_input
from sklearn.cluster import Birch

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_PATHS = {
    "supervised": MODELS_DIR / "random_forest_pipeline.pkl",
    "unsupervised": MODELS_DIR / "best_unsupervised_model.pkl",
}
REQUIRED_PREPROCESSING = ("scaler.pkl", "label_encoder.pkl", "feature_list.json")
BIRCH_PROFILE = MODELS_DIR / "birch_cluster_profiles.json"


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
    if kind == "unsupervised" and not isinstance(model, Birch):
        raise ModelUnavailable("File unsupervised harus berisi model BIRCH hasil training.")
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
            result[kind] = {"ready": True, "filename": path.name, "algorithm": "BIRCH" if kind == "unsupervised" else "Random Forest", "message": "Model siap digunakan."}
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


def _read_birch_profile(cluster_id):
    """Read empirical TEST composition only for exactly the bundled BIRCH model.

    Do not use the composition if a different model was deployed or retrained:
    cluster IDs are arbitrary and may change.
    """
    try:
        raw = BIRCH_PROFILE.read_text(encoding="utf-8")
        info = json.loads(raw)
        model_path = MODEL_PATHS["unsupervised"]
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if info.get("model_sha256") != digest:
            return None
        record = info.get("cluster_profiles", {}).get(str(cluster_id))
        if not isinstance(record, dict):
            return None
        counts = record["counts"]
        total = sum(int(v) for v in counts.values())
        if total < 1:
            return None
        composition = [
            {"label": label, "count": int(counts.get(label, 0)),
             "share": round(int(counts.get(label, 0)) / total, 6)}
            for label in ("Ball", "IR", "Normal", "OR")
        ]
        return {
            "title": str(record["title"]),
            "description": str(record["description"]),
            "sample_count": total,
            "total_test_samples": int(info["n_test_samples"]),
            "composition": composition,
            "source": str(info["source"]),
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def predict_unsupervised(features):
    """Predict a BIRCH cluster for 13 scaled features, NOT a fault class."""
    model = load_model("unsupervised")
    X = _prepare_features(features)
    try:
        # BIRCH was fitted on ndarray. Keep feature order from preprocessing.py;
        # passing ndarray avoids feature-name warning without changing scaling.
        cluster_id = int(model.predict(X.to_numpy(dtype=float))[0])
        profile = _read_birch_profile(cluster_id)
        return {
            "algorithm": "BIRCH",
            "cluster": cluster_id,
            "label": f"Cluster {cluster_id}",
            "cluster_profile": profile,
            "distances": [],  # BIRCH has no KMeans-like transform-to-K-centroids API.
            "note": "Kelompok pola getaran, bukan klasifikasi kondisi bearing. Persentase profil berasal dari DATA TEST, bukan probabilitas input.",
        }
    except Exception as exc:
        raise ModelPredictionError("Prediksi BIRCH gagal. Periksa kesesuaian model dengan 13 fitur terstandardisasi.") from exc
