"""Test API dan halaman tanpa memerlukan file model hasil training."""
import pytest
from backend import create_app


@pytest.fixture()
def client():
    app = create_app({"TESTING": True})
    return app.test_client()


VALID = {
    "rms": 0.1, "std_dev": 0.11, "peak_to_peak": 0.6,
    "kurtosis": 3.0, "skewness": 0.2, "crest_factor": 3.4,
    "signal_energy": 20.0, "dominant_frequency": 123.0,
}


@pytest.mark.parametrize("path", ["/", "/supervised", "/unsupervised", "/about"])
def test_pages_accessible(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert b"SmartBearing" in response.data


def test_unknown_page(client):
    assert client.get("/does-not-exist").status_code == 404


def test_health_and_status(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.get_json()
    assert "supervised" in data["models"]
    assert len(data["feature_order"]) == 8
    assert client.get("/api/health").get_json()["ok"] is True


@pytest.mark.parametrize("kind", ["supervised", "unsupervised"])
def test_missing_feature_rejected(client, kind):
    response = client.post(f"/api/predict/{kind}", json={"features": {"rms": 0.1}})
    assert response.status_code == 400
    assert "std_dev" in response.get_json()["details"]


def test_nan_and_negative_rejected(client):
    bad = dict(VALID, rms=-1, std_dev="NaN")
    response = client.post("/api/predict/supervised", json={"features": bad})
    assert response.status_code == 400
    assert "rms" in response.get_json()["details"]
    assert "std_dev" in response.get_json()["details"]


def test_unexpected_feature_rejected(client):
    response = client.post("/api/predict/supervised", json={"features": dict(VALID, extra=1)})
    assert response.status_code == 400
    assert "unknown" in response.get_json()["details"]


def test_json_required(client):
    response = client.post("/api/predict/supervised", data="x", content_type="text/plain")
    assert response.status_code == 400


def test_supervised_response_contract(client, monkeypatch):
    import backend.routes as routes
    monkeypatch.setattr(routes, "predict_supervised", lambda features: {
        "prediction": "0", "label": "Normal", "probabilities": [
            {"class_id": "0", "label": "Normal", "probability": 1.0}
        ], "note": "mock test",
    })
    response = client.post("/api/predict/supervised", json={"features": VALID})
    assert response.status_code == 200
    assert response.get_json()["result"]["label"] == "Normal"


def test_unsupervised_response_contract(client, monkeypatch):
    import backend.routes as routes
    monkeypatch.setattr(routes, "predict_unsupervised", lambda features: {
        "cluster": 1, "label": "Cluster 1", "distances": [{"cluster": 1, "distance": 0.2}], "note": "mock test",
    })
    response = client.post("/api/predict/unsupervised", json={"features": VALID})
    assert response.status_code == 200
    assert response.get_json()["result"]["cluster"] == 1


def test_model_unavailable_returns_503(client, monkeypatch):
    import backend.routes as routes
    from backend.model_service import ModelUnavailable
    def missing(_features):
        raise ModelUnavailable("Model supervised belum tersedia.")
    monkeypatch.setattr(routes, "predict_supervised", missing)
    response = client.post("/api/predict/supervised", json={"features": VALID})
    assert response.status_code == 503
    assert response.get_json()["ok"] is False
