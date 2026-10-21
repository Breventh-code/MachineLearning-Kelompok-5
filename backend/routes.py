"""Halaman HTML dan endpoint REST API SmartBearing."""
from flask import Blueprint, current_app, jsonify, render_template, request
from backend.model_service import (
    ModelPredictionError, ModelUnavailable,
    get_model_status, predict_supervised, predict_unsupervised,
)
from backend.schema import FEATURE_FIELDS, FEATURE_ORDER
from backend.validators import InputValidationError, validate_features

web = Blueprint("web", __name__)


@web.get("/")
def home():
    return render_template("index.html", page="home")


@web.get("/supervised")
def supervised():
    return render_template("supervised.html", page="supervised", feature_fields=FEATURE_FIELDS)


@web.get("/unsupervised")
def unsupervised():
    return render_template("unsupervised.html", page="unsupervised", feature_fields=FEATURE_FIELDS)


@web.get("/about")
def about():
    return render_template("about.html", page="about")


@web.get("/api/health")
def health():
    return jsonify({"ok": True, "service": "SmartBearing CWRU", "models": get_model_status()})


@web.get("/api/status")
def status():
    return jsonify({"ok": True, "models": get_model_status(), "feature_order": FEATURE_ORDER})


def _run_inference(kind):
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "Kirim JSON valid dengan field 'features'."}), 400
    try:
        features = validate_features(body.get("features"))
        prediction = (
            predict_supervised(features)
            if kind == "supervised" else predict_unsupervised(features)
        )
        return jsonify({"ok": True, "result": prediction})
    except InputValidationError as exc:
        return jsonify({"ok": False, "error": str(exc), "details": exc.details}), 400
    except ModelUnavailable as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    except ModelPredictionError as exc:
        current_app.logger.exception("Prediction failed: %s", kind)
        return jsonify({"ok": False, "error": str(exc)}), 500


@web.post("/api/predict/supervised")
def supervised_api():
    return _run_inference("supervised")


@web.post("/api/predict/unsupervised")
def unsupervised_api():
    return _run_inference("unsupervised")


@web.app_errorhandler(404)
def page_not_found(_error):
    return render_template("404.html", page=""), 404
