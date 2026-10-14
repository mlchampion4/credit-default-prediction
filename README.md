# 🎯 Credit Card Default Prediction — End-to-End ML System

Полный ML-пайплайн: EDA → feature engineering → сравнение моделей с Optuna → SHAP → FastAPI → Docker → тесты.

## 📊 Метрики финальной модели

**Модель:** LightGBM (Optuna) · **Порог:** 0.43 (по бизнес-метрике)

| Метрика | Значение |
|---|---|
| ROC-AUC (test) | **0.782** |
| PR-AUC | **0.49** |
| Gap (train-test) | **0.040** |
| Optimal threshold | **0.43** |

## 🎯 Бизнес-задача

Датасет — существующий портфель кредитных карт (UCI, 30 000 клиентов).
Модель прогнозирует, кто уйдёт в дефолт в следующем месяце.

**Цель:** проактивная работа с рискованными клиентами — напоминания,
реструктуризация, особый контроль. Мы не отказываем клиенту —
он уже в портфеле. Мы минимизируем потери.

## 🎯 Бизнес-порог

Порог подобран по формуле прибыли с учётом LGD:

- `LGD = 0.6` → порог **0.53**.
- `LGD = 0.8` → порог **0.38**.

Sensitivity analysis — в `notebooks/05_ab_test.ipynb`.

## 🏗️ Архитектура

CSV → features → LightGBM → FastAPI → SHAP → CRM

## 🚀 Быстрый старт

git clone https://github.com/USERNAME/default-project.git
cd default-project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.models.train
pytest tests/ -v
uvicorn src.api.main:app --reload

API: http://localhost:8000/docs

## 🐳 Docker

docker build -t default-api -f docker/Dockerfile .
docker run -p 8000:8000 -v $(pwd)/models:/app/models default-api

## 🔌 Пример запроса

POST /predict — тело запроса:

{
  "SEX": 2, "EDUCATION": 1, "MARRIAGE": 3, "AGE": 29,
  "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": 0, "PAY_6": 0,
  "BILL_AMT1": 85000.0, "BILL_AMT2": 78000.0, "BILL_AMT3": 72000.0,
  "BILL_AMT4": 65000.0, "BILL_AMT5": 58000.0, "BILL_AMT6": 52000.0,
  "PAY_AMT1": 0.0, "PAY_AMT2": 0.0, "PAY_AMT3": 1500.0,
  "PAY_AMT4": 2000.0, "PAY_AMT5": 2500.0, "PAY_AMT6": 3000.0,
  "LIMIT_BAL": 100000.0
}

Ответ:

{
  "default_probability": 0.740,
  "risk": "high",
  "threshold": 0.43,
  "recommendation": "Предложить реструктуризацию и взять на особый контроль",
  "top_factors": [
    {"feature": "PAY_max", "shap_value": 1.1578, "direction": "to_default"},
    {"feature": "PAY_AMT1", "shap_value": 0.3938, "direction": "to_default"},
    {"feature": "UTILIZATION", "shap_value": 0.3537, "direction": "to_default"}
  ]
}

## 📁 Структура

default-project/
├── src/
│   ├── data/load.py
│   ├── features/build.py
│   ├── models/{train,predict}.py
│   └── api/main.py
├── tests/
├── docker/
├── notebooks/
├── models/
├── requirements.txt
└── README.md

## 🛠️ Стек

pandas · scikit-learn · LightGBM · CatBoost · Optuna · SHAP · FastAPI · Docker · pytest

## 👤 Автор

Драбудько Никита · GitHub: https://github.com/mlchampion4