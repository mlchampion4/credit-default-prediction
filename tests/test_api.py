import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict(client):
    payload = {
    "SEX": 2,
    "EDUCATION": 1,
    "MARRIAGE": 3,
    "AGE": 29,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 85000.0,
    "BILL_AMT2": 78000.0,
    "BILL_AMT3": 72000.0,
    "BILL_AMT4": 65000.0,
    "BILL_AMT5": 58000.0,
    "BILL_AMT6": 52000.0,
    "PAY_AMT1": 0.0,
    "PAY_AMT2": 0.0,
    "PAY_AMT3": 1500.0,
    "PAY_AMT4": 2000.0,
    "PAY_AMT5": 2500.0,
    "PAY_AMT6": 3000.0,
    "LIMIT_BAL": 100000.0
    }
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["risk"] in ("low", "medium", "high")
    assert len(body["top_factors"]) == 3

def test_predict_batch(client):
    payload = {
        "clients": [
            {
                "SEX": 1, "EDUCATION": 3, "MARRIAGE": 1, "AGE": 38,
                "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 22000.0, "BILL_AMT2": 21000.0, "BILL_AMT3": 20000.0,
                "BILL_AMT4": 18000.0, "BILL_AMT5": 17000.0, "BILL_AMT6": 16000.0,
                "PAY_AMT1": 8000.0, "PAY_AMT2": 7000.0, "PAY_AMT3": 6500.0,
                "PAY_AMT4": 6000.0, "PAY_AMT5": 5500.0, "PAY_AMT6": 5000.0,
                "LIMIT_BAL": 70000.0,
            },
            {
                "SEX": 2, "EDUCATION": 1, "MARRIAGE": 3, "AGE": 29,
                "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 85000.0, "BILL_AMT2": 78000.0, "BILL_AMT3": 72000.0,
                "BILL_AMT4": 65000.0, "BILL_AMT5": 58000.0, "BILL_AMT6": 52000.0,
                "PAY_AMT1": 0.0, "PAY_AMT2": 0.0, "PAY_AMT3": 1500.0,
                "PAY_AMT4": 2000.0, "PAY_AMT5": 2500.0, "PAY_AMT6": 3000.0,
                "LIMIT_BAL": 100000.0,
            }
        ]
    }
    r = client.post('/predict_batch', json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2