"""Pengujian endpoint Flask tanpa model terlatih dari dataset asli."""
import pytest
from backend import create_app


@pytest.fixture()
def client():
    return create_app({"TESTING": True}).test_client()


VALID = {
    "max": 0.8, "min": -0.7, "mean": 0.02, "sd": 0.3, "rms": 0.31,
    "skewness": 0.08, "kurtosis": 3.2, "crest": 2.58, "form": 1.15,
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
    assert set(data["models"]) == {"supervised", "unsupervised"}
    assert len(data["feature_order"]) == 9
    assert client.get("/api/health").get_json()["ok"] is True


@pytest.mark.parametrize("kind", ["supervised", "unsupervised"])
def test_missing_feature_rejected(client, kind):
    response = client.post(f"/api/predict/{kind}", json={"features": {"max": 0.1}})
    assert response.status_code == 400
    assert "rms" in response.get_json()["details"]


def test_nan_and_negative_rejected(client):
    bad = dict(VALID, rms=-1, sd="NaN")
    response = client.post("/api/predict/supervised", json={"features": bad})
    assert response.status_code == 400
    assert "rms" in response.get_json()["details"]
    assert "sd" in response.get_json()["details"]


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
        "prediction": "2", "label": "Normal", "probabilities": [
            {"class_id": "2", "label": "Normal", "probability": 1.0}
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

    def unavailable(_features):
        raise ModelUnavailable("Model supervised belum tersedia.")

    monkeypatch.setattr(routes, "predict_supervised", unavailable)
    response = client.post("/api/predict/supervised", json={"features": VALID})
    assert response.status_code == 503
    assert response.get_json()["ok"] is False
