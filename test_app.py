
import unittest
from unittest.mock import patch

from app import app


class CropAPITest(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_prediction(self):
        with patch("app.MODEL_PATH") as model_path, \
             patch("app.joblib.load") as load_model:
            model_path.exists.return_value = True
            load_model.return_value.predict.return_value = ["rice"]

            response = self.client.post(
                "/predict",
                json={
                    "N": 80,
                    "P": 40,
                    "K": 40,
                    "temperature": 25,
                    "humidity": 85,
                    "ph": 6.2,
                    "rainfall": 200
                }
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.get_json()["predicted_crop"], "banana"
            )

    def test_missing_feature(self):
        response = self.client.post(
            "/predict",
            json={"N": 80}
        )
        self.assertEqual(response.status_code, 400)

    def test_non_numeric_feature(self):
        response = self.client.post(
            "/predict",
            json={
                "N": "abc",
                "P": 40,
                "K": 40,
                "temperature": 25,
                "humidity": 85,
                "ph": 6.2,
                "rainfall": 200
            }
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
