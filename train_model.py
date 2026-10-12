
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "crop_raw.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]
TARGET = "label"

# Use the real dataset if it exists; otherwise create demo data.
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=500,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=42
    )
    df = pd.DataFrame(X, columns=FEATURES)
    df[TARGET] = [f"crop_{value}" for value in y]

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, ARTIFACTS_DIR / "crop_model.joblib")

metrics = {
    "accuracy": round(float(accuracy), 4),
    "model": "RandomForestClassifier"
}

with open(BASE_DIR / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model training completed.")
print("Accuracy:", metrics["accuracy"])
print("Model saved to artifacts/crop_model.joblib")
print("Metrics saved to metrics.json")
