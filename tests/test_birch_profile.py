"""Test cluster composition metadata is tied to the exact BIRCH model artifact."""
import json
import hashlib
from pathlib import Path


def test_birch_profile_matches_model():
    root = Path(__file__).resolve().parents[1] / "models"
    profile = json.loads((root / "birch_cluster_profiles.json").read_text(encoding="utf-8"))
    model = (root / "best_unsupervised_model.pkl").read_bytes()
    assert hashlib.sha256(model).hexdigest() == profile["model_sha256"]
    assert profile["algorithm"] == "BIRCH"
    groups = profile["cluster_profiles"]
    assert set(groups) == {"0", "1", "2"}
    assert sum(sum(g["counts"].values()) for g in groups.values()) == 460
    assert {k: sum(v["counts"].values()) for k, v in groups.items()} == {"0": 379, "1": 32, "2": 49}
