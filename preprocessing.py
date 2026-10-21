"""Final CWRU feature-based preprocessing (4-class baseline).

Source format: Kaggle brjapon/cwru-bearing-datasets, CSV feature_time_*.csv.
Input columns (9): max, min, mean, sd, rms, skewness, kurtosis, crest, form.
Target: fault -> fault_type (Ball, IR, Normal, OR).
Output (13): 9 source features + 4 deterministic derived features.

CLI from repository root:
    python preprocessing.py --data-dir data

For web/API inference:
    from preprocessing import transform_input
    X_scaled = transform_input(values, model_dir="models")

IMPORTANT: models trained on train_scaled.csv already take SCALED data; do not
apply a second StandardScaler inside the RandomForest / KMeans estimator.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_STATE = 42
DEFAULT_PATTERN = "feature_time_48k_2048_load_1.csv"
BASE_FEATURES = ["max", "min", "mean", "sd", "rms", "skewness", "kurtosis", "crest", "form"]
ENGINEERED_FEATURES = ["peak_to_peak", "abs_peak", "energy", "sd_rms_ratio"]
FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
TARGET_CANDIDATES = ("fault", "label", "class", "target")
EXPECTED_CLASSES = {"Ball", "IR", "Normal", "OR"}

# 9 fields are entered by users; 4 derived fields are calculated in Python.
FEATURE_DESCRIPTIONS = {
    "max": ("Maximum", "Nilai maksimum amplitudo getaran"),
    "min": ("Minimum", "Nilai minimum amplitudo getaran"),
    "mean": ("Mean", "Rata-rata amplitudo getaran"),
    "sd": ("Standard Deviation", "Simpangan baku sinyal getaran"),
    "rms": ("RMS", "Root Mean Square sinyal getaran"),
    "skewness": ("Skewness", "Kemiringan distribusi getaran"),
    "kurtosis": ("Kurtosis", "Kurtosis sesuai CSV CWRU yang digunakan"),
    "crest": ("Crest", "Faktor crest dari dataset CWRU"),
    "form": ("Form", "Faktor bentuk dari dataset CWRU"),
}


def _input_files(data_dir: str | Path, file_pattern: str) -> list[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {root.resolve()}")
    candidates = sorted(
        p for p in root.rglob(file_pattern)
        if p.is_file() and p.suffix.lower() == ".csv"
        and not any(part.lower() in {"processed", "models", ".ipynb_checkpoints"} for part in p.parts)
    )
    if not candidates:
        available = [p.name for p in root.rglob("*.csv") if "processed" not in p.parts]
        raise FileNotFoundError(
            f"Tidak ditemukan CSV '{file_pattern}' di {root.resolve()}. "
            f"Letakkan CSV dataset Kaggle di data/raw/. CSV yang terdeteksi: {available[:8]}"
        )
    return candidates


def find_target(df: pd.DataFrame) -> str:
    """Detect actual source class column (must not use fault_type as raw target)."""
    for column in TARGET_CANDIDATES:
        if column in df.columns:
            return column
    raise KeyError(f"Kolom target (fault/label/class/target) tidak ada. Kolom: {list(df.columns)}")


def load_raw(data_dir: str | Path = "data", file_pattern: str = DEFAULT_PATTERN) -> pd.DataFrame:
    """Read feature CSVs recursively; do not accidentally read our outputs."""
    root = Path(data_dir)
    frames = []
    for path in _input_files(root, file_pattern):
        frame = pd.read_csv(path)
        frame.columns = [str(col).strip().lower() for col in frame.columns]
        frame = frame.loc[:, ~frame.columns.str.startswith("unnamed")].copy()
        missing = [name for name in BASE_FEATURES if name not in frame.columns]
        if missing:
            raise ValueError(f"CSV '{path.name}' tidak punya fitur wajib: {missing}")
        source_target = find_target(frame)
        if source_target != "fault":
            frame = frame.rename(columns={source_target: "fault"})
        # Avoid accidental inclusion of previous intermediate columns as model features.
        name = path.name
        rate = re.search(r"(\d+)k", name.lower())
        load = re.search(r"load[_-]?(\d+)", name.lower())
        frame["sampling_khz"] = int(rate.group(1)) if rate else np.nan
        frame["load_hp"] = int(load.group(1)) if load else np.nan
        frame["source_file"] = str(path.relative_to(root))
        frame["source_row"] = np.arange(len(frame))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame, target: str = "fault") -> pd.DataFrame:
    """Normalize numeric columns & labels and remove invalid/duplicate rows."""
    if target not in df.columns:
        raise KeyError(f"Kolom target '{target}' tidak ditemukan")
    missing = [name for name in BASE_FEATURES if name not in df.columns]
    if missing:
        raise ValueError(f"Kolom fitur dasar tidak lengkap: {missing}")
    result = df.copy()
    for col in BASE_FEATURES:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result[BASE_FEATURES] = result[BASE_FEATURES].replace([np.inf, -np.inf], np.nan)
    result[target] = result[target].astype("string").str.strip()
    result[target] = result[target].replace("", pd.NA)
    result = result.dropna(subset=BASE_FEATURES + [target])
    result = result.drop_duplicates(subset=BASE_FEATURES + [target]).reset_index(drop=True)
    if result.empty:
        raise ValueError("Data kosong setelah cleaning. Periksa CSV sumber.")
    return result


def add_fault_type(df: pd.DataFrame, target: str = "fault") -> pd.DataFrame:
    """Map detailed source labels explicitly. Never silently treat unknown labels as Normal."""
    result = df.copy()
    raw = result[target].astype(str).str.strip()
    prefixes = raw.str.extract(r"^(Ball|IR|OR|Normal)(?:_|$)", flags=re.IGNORECASE, expand=False)
    normalized = prefixes.str.lower().map({
        "ball": "Ball", "ir": "IR", "or": "OR", "normal": "Normal"
    })
    if normalized.isna().any():
        unknown = sorted(raw[normalized.isna()].unique().tolist())
        raise ValueError(f"Label fault tidak dikenali: {unknown[:12]}. Periksa mapping label.")
    result["fault_type"] = normalized
    return result


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features replicate the submitted Data Engineer notebook.

    energy = RMS**2 denotes MEAN SQUARE amplitude (not window total energy).
    """
    result = df.copy()
    result["peak_to_peak"] = result["max"] - result["min"]
    result["abs_peak"] = result[["max", "min"]].abs().max(axis=1)
    result["energy"] = result["rms"] ** 2
    result["sd_rms_ratio"] = result["sd"] / (result["rms"] + 1e-12)
    return result


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Fitur tidak lengkap: {missing}")
    return FEATURES.copy()


def run_pipeline(
    data_dir: str | Path = "data",
    out_dir: str | Path = "data/processed",
    model_dir: str | Path = "models",
    target: str = "fault_type",
    test_size: float = 0.2,
    file_pattern: str = DEFAULT_PATTERN,
    save: bool = True,
) -> dict:
    """End-to-end preparation and artifact export; deliberately DOES NOT train ML models.

    Split is stratified by WINDOW/ROW. On one file containing multiple windows
    from the same recordings, this is not a recording-independent evaluation.
    """
    if target not in {"fault", "fault_type"}:
        raise ValueError("target hanya mendukung 'fault_type' (4 kelas) atau 'fault' (detail)")
    if not 0 < test_size < 1:
        raise ValueError("test_size harus di antara 0 dan 1")
    raw = load_raw(data_dir, file_pattern=file_pattern)
    df = add_features(add_fault_type(clean(raw, target="fault")))
    features = get_feature_cols(df)
    if target == "fault_type":
        found = set(df[target].unique())
        if found != EXPECTED_CLASSES:
            raise ValueError(f"Untuk klasifikasi 4 kelas harus ada {sorted(EXPECTED_CLASSES)}; ditemukan {sorted(found)}")

    labels = LabelEncoder().fit(df[target].astype(str))
    y = labels.transform(df[target].astype(str))
    row_ids = np.arange(len(df))
    train_ids, test_ids = train_test_split(
        row_ids, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    X_train_raw = df.iloc[train_ids][features]
    X_test_raw = df.iloc[test_ids][features]
    y_train, y_test = y[train_ids], y[test_ids]

    scaler = StandardScaler().fit(X_train_raw)
    X_train = pd.DataFrame(scaler.transform(X_train_raw), columns=features)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=features)

    result = {
        "df_raw": raw, "df_clean": df, "features": features, "target": target,
        "label_encoder": labels, "scaler": scaler,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "train_ids": train_ids, "test_ids": test_ids,
        "split_warning": (
            "Stratified random ROW/WINDOW split; beberapa window bisa berasal dari rekaman "
            "asli yang sama. Hasil test bukan pengukuran generalisasi antar-rekaman. "
            "Jika tersedia recording_id sebenarnya, gunakan group-aware split."
        ),
    }
    if save:
        out_path, model_path = Path(out_dir), Path(model_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model_path.mkdir(parents=True, exist_ok=True)

        df.to_csv(out_path / "cwru_clean.csv", index=False)
        X_train.assign(label=y_train).to_csv(out_path / "train_scaled.csv", index=False)
        X_test.assign(label=y_test).to_csv(out_path / "test_scaled.csv", index=False)
        # Split manifest stores metadata; MUST NOT be used as model input.
        manifest = df[["source_file", "source_row", "fault", "fault_type"]].copy()
        manifest["split"] = "test"
        manifest.loc[train_ids, "split"] = "train"
        manifest.to_csv(out_path / "split_manifest.csv", index=False)

        joblib.dump(scaler, model_path / "scaler.pkl")
        joblib.dump(labels, model_path / "label_encoder.pkl")
        feature_list = {
            "features": features, "base_features": BASE_FEATURES,
            "engineered_features": ENGINEERED_FEATURES, "target": target,
            "classes": list(labels.classes_), "model_input": "13_scaled_features",
            "input_form": "9_unscaled_base_features", "random_state": RANDOM_STATE,
            "source_pattern": file_pattern,
        }
        (model_path / "feature_list.json").write_text(
            json.dumps(feature_list, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        web_schema = {
            "description": "9 fitur CWRU asli diisi pengguna; 4 fitur turunan dihitung otomatis di backend.",
            "features": [
                {
                    "name": name, "label": FEATURE_DESCRIPTIONS[name][0],
                    "hint": FEATURE_DESCRIPTIONS[name][1],
                    **({"min": 0} if name in {"sd", "rms", "crest", "form"} else {}),
                }
                for name in BASE_FEATURES
            ],
            "class_labels": {str(i): str(name) for i, name in enumerate(labels.classes_)},
        }
        (model_path / "feature_schema.json").write_text(
            json.dumps(web_schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = {
            "source_csv": sorted(df["source_file"].unique().tolist()),
            "rows_raw": int(len(raw)), "rows_clean": int(len(df)),
            "rows_train": int(len(train_ids)), "rows_test": int(len(test_ids)),
            "feature_count": len(features), "target": target,
            "class_distribution": {str(k): int(v) for k, v in df[target].value_counts().items()},
            "warning": result["split_warning"],
        }
        (out_path / "preprocessing_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def transform_input(values: Mapping[str, object], model_dir: str | Path = "models") -> np.ndarray:
    """Validate 9 raw user features and return 1x13 scaled model-ready ndarray."""
    if not isinstance(values, Mapping):
        raise TypeError("Input harus berupa dictionary nama fitur -> nilai")
    missing = [key for key in BASE_FEATURES if key not in values]
    if missing:
        raise ValueError(f"Input kurang fitur: {missing}")
    parsed: dict[str, float] = {}
    for key in BASE_FEATURES:
        try:
            parsed[key] = float(values[key])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Fitur '{key}' harus angka") from exc
        if not np.isfinite(parsed[key]):
            raise ValueError(f"Fitur '{key}' harus bernilai finite")
    if any(parsed[key] < 0 for key in ("sd", "rms", "crest", "form")):
        raise ValueError("sd, rms, crest, dan form tidak boleh negatif")

    model_path = Path(model_dir)
    details = json.loads((model_path / "feature_list.json").read_text(encoding="utf-8"))
    feature_order = details["features"]
    if feature_order != FEATURES:
        raise ValueError("Urutan fitur di feature_list.json tidak konsisten dengan preprocessing.py")
    raw_row = add_features(pd.DataFrame([parsed]))[feature_order]
    scaler = joblib.load(model_path / "scaler.pkl")
    scaled = scaler.transform(raw_row)
    if not np.isfinite(scaled).all():
        raise ValueError("Input menghasilkan fitur non-finite setelah preprocessing")
    return scaled


def main() -> None:
    parser = argparse.ArgumentParser(description="CWRU 4-class data-prep for SmartBearing")
    parser.add_argument("--data-dir", default="data", help="Folder CSV sumber (rekursif)")
    parser.add_argument("--file-pattern", default=DEFAULT_PATTERN, help="Nama/pola CSV CWRU")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--target", default="fault_type", choices=["fault_type", "fault"])
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    output = run_pipeline(
        data_dir=args.data_dir, file_pattern=args.file_pattern,
        out_dir=args.out_dir, model_dir=args.model_dir,
        target=args.target, test_size=args.test_size,
    )
    print("DATASET      :", output["df_clean"].shape)
    print("FITUR MODEL  :", output["features"])
    print("KELAS        :", list(output["label_encoder"].classes_))
    print("TRAIN / TEST :", output["X_train"].shape, output["X_test"].shape)
    print("OUTPUT       :", args.out_dir, "dan", args.model_dir)
    print("CATATAN      :", output["split_warning"])


if __name__ == "__main__":
    main()
