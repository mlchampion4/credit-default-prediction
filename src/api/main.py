from contextlib import asynccontextmanager
import json
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from src.models.predict import _recommend, _prepare, _prepare_batch, _dummy_client, _risk_level, METADATA_PATH, MODEL_PATH
from src.api.entities import Client, PredictionResponse, BatchRequest, BatchPredictionResponse

_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    _state['model'] = joblib.load(MODEL_PATH)
    _state['threshold'] = 0.43

    dummy_X = _prepare(_dummy_client())

    _state['explainer'] = shap.TreeExplainer(_state['model'], dummy_X)
    _state['feature_names'] = _state['model'].feature_name_

    yield
    _state.clear()

app = FastAPI(
    title='Default prediction API',
    version='1.0',
    description='API для предстказания кредитного дефолта',
    lifespan=lifespan
)

def _explain(X: pd.DataFrame, top_n: int = 3) -> list[dict]:
    shap_vals = _state['explainer'].shap_values(X)
    feature_names = _state['feature_names']
    sv = shap_vals[0]
    print(X.columns.to_list())
    top_idx = np.argsort(np.abs(sv))[::-1][:top_n]

    factors = []
    for idx in top_idx:
        factors.append({
            'feature': str(feature_names[idx]),
            'shap_value': round(float(sv[idx]), 4),
            'direction': 'to_default' if sv[idx] > 0 else 'from_default'
        })
    return factors

@app.get('/health')
def health() -> dict:
    return {
        'status': 'ok',
        'model': type(_state['model']).__name__,
        'threshold': _state['threshold']
    }

@app.get('/metadata')
def metadata():
    if not METADATA_PATH.exists():
        raise HTTPException(404, 'metadata.json is not found')
    with open(METADATA_PATH, encoding='utf-8') as f:
        return json.load(f)

@app.post('/predict', response_model=PredictionResponse)
def predict(client: Client) -> dict:
    X = _prepare(client.model_dump())
    proba = float(_state['model'].predict_proba(X)[:, 1][0])
    risk = _risk_level(proba, _state['threshold'])
    top_factors = _explain(X, 3)

    return PredictionResponse(
        default_probability=proba,
        risk=risk,
        threshold=_state['threshold'],
        recommendation=_recommend(risk),
        top_factors=top_factors
    )

@app.post('/predict_batch', response_model=BatchPredictionResponse)
def predict_batch(request: BatchRequest):
    X = _prepare_batch([c.model_dump() for c in request.clients])
    probas = _state['model'].predict_proba(X)[:, 1]

    results: list[PredictionResponse] = []

    for idx, proba in enumerate(probas):
        row = X.iloc[[idx]]
        risk = _risk_level(proba, _state['threshold'])
        results.append(PredictionResponse(
            default_probability=proba,
            risk=risk,
            threshold=_state['threshold'],
            recommendation=_recommend(risk),
            top_factors=_explain(row, 3)
        ))

    return BatchPredictionResponse(
        predictions=results,
        count=len(results)
    )