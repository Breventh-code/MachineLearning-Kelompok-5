"""Sumber tunggal nama/urutan fitur untuk form dan inferensi model."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_FILE = ROOT / "models" / "feature_schema.json"
with SCHEMA_FILE.open(encoding="utf-8") as handle:
    SCHEMA = json.load(handle)

FEATURE_FIELDS = SCHEMA["features"]
FEATURE_ORDER = [field["name"] for field in FEATURE_FIELDS]
CLASS_LABELS = SCHEMA["class_labels"]
