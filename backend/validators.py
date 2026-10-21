"""Validasi fitur numerik dari request JSON."""
import math
from backend.schema import FEATURE_FIELDS, FEATURE_ORDER


class InputValidationError(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


def validate_features(raw):
    if not isinstance(raw, dict):
        raise InputValidationError("Field 'features' harus berupa objek JSON.")

    details = {}
    results = {}
    extra = sorted(set(raw) - set(FEATURE_ORDER))
    if extra:
        details["unknown"] = f"Fitur tidak dikenal: {', '.join(extra)}"

    for spec in FEATURE_FIELDS:
        name = spec["name"]
        value = raw.get(name)
        if value is None or value == "":
            details[name] = "Wajib diisi."
            continue
        if isinstance(value, bool):
            details[name] = "Harus berupa angka, bukan boolean."
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            details[name] = "Harus berupa angka yang valid."
            continue
        if not math.isfinite(numeric):
            details[name] = "Tidak boleh NaN atau tak hingga."
            continue
        minimum = spec.get("min")
        if minimum is not None and numeric < minimum:
            details[name] = f"Nilai minimal {minimum}."
            continue
        results[name] = numeric

    if details:
        raise InputValidationError("Input tidak valid. Periksa setiap fitur.", details)
    return results
