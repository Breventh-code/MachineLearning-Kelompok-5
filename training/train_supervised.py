
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

print("Memulai training Random Forest...", flush=True)
print("Root proyek:", ROOT, flush=True)

train_path = DATA / "train_scaled.csv"
test_path = DATA / "test_scaled.csv"
feature_path = MODELS / "feature_list.json"

for path in [train_path, test_path, feature_path]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

with open(feature_path, encoding="utf-8") as f:
    features = json.load(f)["features"]

if "label" not in train.columns or "label" not in test.columns:
    raise ValueError("Kolom 'label' tidak ditemukan dalam CSV.")

X_train = train[features]
X_test = test[features]
y_train = train["label"].astype(int)
y_test = test["label"].astype(int)

print("Jumlah fitur:", len(features), flush=True)
print("Data training:", X_train.shape, flush=True)
print("Mulai fit model...", flush=True)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred), flush=True)
print("F1 macro:", f1_score(y_test, pred, average="macro"), flush=True)
print(classification_report(y_test, pred, zero_division=0), flush=True)

MODELS.mkdir(parents=True, exist_ok=True)
output = MODELS / "random_forest_pipeline.pkl"
joblib.dump(model, output)

print("MODEL BERHASIL DISIMPAN:", output, flush=True)
