
import json
import unittest
from pathlib import Path

import joblib


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "artifacts" / "crop_model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"


class CropPipelineTest(unittest.TestCase):

    def test_model_file_exists(self):
        self.assertTrue(
            MODEL_PATH.exists(),
            "Trained model file was not created"
        )

    def test_metrics_file_exists(self):
        self.assertTrue(
            METRICS_PATH.exists(),
            "Metrics file was not created"
        )

    def test_accuracy_is_valid(self):
        with open(METRICS_PATH, "r") as file:
            metrics = json.load(file)

        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_model_can_be_loaded(self):
        model = joblib.load(MODEL_PATH)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
