import copy
import unittest

from fastapi.testclient import TestClient
from app import app


class TestApp(unittest.TestCase):
    def setUp(self) -> None:
        self.data = {
            "data": [[69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0],
                     [69, 0, 0, 140, 239, 0, 0, 151, 0, 1.8, 0, 2, 0],
                     [66, 0, 0, 150, 226, 0, 0, 114, 0, 2.6, 2, 0, 0]],
            "col_names": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
                          "slope", "ca", "thal"]
        }

    def test_root_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), "Welcome to Heart Disease Predictor! It is running and ready")

    def test_health_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, b'{"detail":"Model is ready. Everything is OK"}')

    def test_predict_endpoint_errors(self):
        with TestClient(app) as client:
            response = client.get("/predict")
            self.assertEqual(response.status_code, 405)

            response = client.post("/predict", json={})
            self.assertEqual(response.status_code, 400)

            bad_json = copy.copy(self.data)
            bad_json["data"] = [i[:-1] for i in bad_json["data"]]
            assert bad_json != self.data
            response = client.post("predict", json=bad_json)
            self.assertEqual(response.status_code, 400)

            bad_json = copy.copy(self.data)
            bad_json["data"] = []
            response = client.post("predict", json=bad_json)
            self.assertEqual(response.status_code, 500)  # correct headers but no data = error on prediction stage

            bad_json = copy.copy(self.data)
            bad_json["data"] = list(map(str, bad_json["data"]))

            response = client.post("predict", json=bad_json)
            self.assertEqual(response.status_code, 400)

            bad_json = copy.copy(self.data)
            bad_json["col_names"] = ["bad", "names"]
            response = client.post("predict", json=bad_json)
            self.assertEqual(response.status_code, 400)

    def test_predict_endpoint_ok(self):
        with TestClient(app) as client:
            response = client.post("/predict", json=self.data)
            self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
