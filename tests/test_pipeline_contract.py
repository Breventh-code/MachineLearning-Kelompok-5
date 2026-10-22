"""Tes integrasi preprocessing dan inferensi dengan DATA SINTETIS, tanpa dataset CWRU.

Model yang digunakan hanya untuk memverifikasi kontrak API: bukan model proyek!
"""
import pytest

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.ensemble import RandomForestClassifier

from preprocessing import BASE_FEATURES, FEATURES, run_pipeline, transform_input


def _test_data():
    rows = []
    rng = np.random.default_rng(42)
    for i, fault in enumerate(["Normal", "Ball_007", "IR_007", "OR_007"]):
        for j in range(18):
            rms = float(0.25 + i * 0.30 + abs(rng.normal(scale=0.035)))
            rows.append({
                "max": 3 * rms, "min": -2.4 * rms,
                "mean": float(rng.normal(0.05 * i, 0.004)),
                "sd": rms * 0.97, "rms": rms,
                "skewness": float(rng.normal(scale=0.1)),
                "kurtosis": float(2.5 + 0.2 * i + abs(rng.normal(scale=0.03))),
                "crest": 3.0, "form": 1.15,
                "fault": fault,
            })
    return pd.DataFrame(rows)


def test_preprocessing_and_both_model_contracts(tmp_path, monkeypatch):
    import backend.model_service as svc
    data_root = tmp_path / "data"
    data_root.mkdir()
    (_ := data_root / "raw").mkdir()
    _test_data().to_csv(data_root / "raw" / "feature_time_48k_2048_load_1.csv", index=False)
    model_dir = tmp_path / "models"
    out = run_pipeline(data_dir=data_root, out_dir=data_root / "processed", model_dir=model_dir)
    assert out["X_train"].shape[1] == 13
    assert len(out["label_encoder"].classes_) == 4
    assert out["features"] == FEATURES

    raw_input = {name: float(out["df_clean"].iloc[0][name]) for name in BASE_FEATURES}
    expected = out["scaler"].transform(out["df_clean"].iloc[[0]][FEATURES])
    actual = transform_input(raw_input, model_dir=model_dir)
    np.testing.assert_allclose(actual, expected)

    rf = RandomForestClassifier(n_estimators=8, random_state=42).fit(out["X_train"], out["y_train"])
    km = Birch(n_clusters=3, threshold=0.8).fit(out["X_train"].to_numpy())
    joblib.dump(rf, model_dir / "random_forest_pipeline.pkl")
    joblib.dump(km, model_dir / "best_unsupervised_model.pkl")
    monkeypatch.setattr(svc, "MODELS_DIR", model_dir)
    monkeypatch.setattr(svc, "MODEL_PATHS", {
        "supervised": model_dir / "random_forest_pipeline.pkl",
        "unsupervised": model_dir / "best_unsupervised_model.pkl",
    })
    sup = svc.predict_supervised(raw_input)
    unsup = svc.predict_unsupervised(raw_input)
    assert sup["label"] in {"Ball", "IR", "Normal", "OR"}
    assert sum(p["probability"] for p in sup["probabilities"]) == pytest.approx(1, abs=1e-5)
    assert isinstance(unsup["cluster"], int)
    assert unsup["algorithm"] == "BIRCH"
    assert unsup["distances"] == []
    assert unsup["cluster_profile"] is None
