
import json
from pathlib import Path

METRICS_PATH = Path(__file__).parent / "metrics.json"
MINIMUM_ACCURACY = 0.85

with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print(f"Model accuracy: {accuracy:.2%}")
print(f"Required accuracy: {MINIMUM_ACCURACY:.2%}")

if accuracy < MINIMUM_ACCURACY:
    print("QUALITY GATE FAILED")
    raise SystemExit(1)

print("QUALITY GATE PASSED")
