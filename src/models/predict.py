from pathlib import Path
import joblib
import json
import pandas as pd
from src.features.build import transform_features

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "default_pipeline.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

def _dummy_client() -> dict:
    return {
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 2,
        "AGE": 46,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 5000.0,
        "BILL_AMT2": 4500.0,
        "BILL_AMT3": 4200.0,
        "BILL_AMT4": 4000.0,
        "BILL_AMT5": 3800.0,
        "BILL_AMT6": 3776.0,
        "PAY_AMT1": 26000.0,
        "PAY_AMT2": 0.0,
        "PAY_AMT3": 3450.0,
        "PAY_AMT4": 2300.0,
        "PAY_AMT5": 1750.0,
        "PAY_AMT6": 2850.0,
        "LIMIT_BAL": 50000.0,
    }

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}. Запусти python -m src.models.train")
    return joblib.load(MODEL_PATH)

def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {"optimal_threshold": 0.5}
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)

def _prepare(client: dict) -> pd.DataFrame:
    return transform_features(pd.DataFrame([client]))

def _prepare_batch(clients: list[dict]) -> pd.DataFrame:
    return transform_features(pd.DataFrame(clients))

def _risk_level(proba: float, threshold: float) -> str:
    if proba >= threshold:
        return 'high'
    elif proba >= threshold * 0.5:
        return 'medium'
    else:
        return 'low'

def predict_one(client: dict) -> dict:
    model = load_model()
    metadata = load_metadata()
    threshold = metadata.get('optimal_threshold', 0.5)

    X = _prepare(client)
    proba = float(model.predict_proba(X)[:, 1][0])

    risk = _risk_level(proba, threshold)

    return {
        'default_probability': round(proba, 4),
        'risk': risk,
        'threshold': threshold,
        'recommendation': _recommend(risk)
    }

def _recommend(risk: str) -> str:
    if risk == 'high':
        return 'Предложить реструктуризацию и взять на особый контроль'
    if risk == 'medium':
        return 'Напомнить о платеже, проверить вовлечённость'
    return 'Клиент лоялен, стандартный мониторинг'

def predict_batch(clients: list[dict]) -> list[dict]:
    model = load_model()
    metadata = load_metadata()
    threshold = metadata.get('optimal_threshold', 0.5)

    df = predict_batch(clients)

    probas = model.predict_proba(df)[:, 1]

    results = []
    for proba in probas:
        risk = (
            'high' if proba >= threshold
            else 'medium' if proba >= threshold * 0.5
            else 'low'
        )
        results.append({
            'default_probability': round(proba, 4),
            'risk': risk,
            'threshold': threshold,
            'recommendation': _recommend(risk)
        })
    return results

if __name__ == '__main__':
    sample_client = _dummy_client()

    print(json.dumps(predict_one(sample_client), indent=2, ensure_ascii=False))