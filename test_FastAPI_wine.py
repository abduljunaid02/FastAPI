from FastaAPI_Practice import wineapp
from fastapi.testclient import TestClient

client = TestClient(wineapp)

def test_predict():
    payload = {
        "alcohol": 13.64,
  "malic_acid": 3.10,
  "ash": 2.56,
  "alcalinity_of_ash": 15.2,
  "magnesium": 116.0,
  "total_phenols": 2.70,
  "flavanoids": 3.03,
  "nonflavanoid_phenols": 0.17,
  "proanthocyanins": 1.66,
  "color_intensity": 5.10,
  "hue": 0.96,
  "od280/od315_of_diluted_wines": 3.36,
  "proline": 845.0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "Class_type" in response.json()
