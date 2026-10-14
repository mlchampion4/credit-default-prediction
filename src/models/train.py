from pathlib import Path
from datetime import datetime
import json
import joblib

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.load import load_data
from src.features.build import transform_features

RANDOM_STATE = 42
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
OPTIMAL_THRESHOLD = 0.43

PARAMS = {
        "boosting_type": "gbdt",
        "class_weight": "balanced",
        "colsample_bytree": 0.7423037751345191,
        "importance_type": "split",
        "learning_rate": 0.008573284834129728,
        "max_depth": 6,
        "min_child_samples": 19,
        "min_child_weight": 0.001,
        "min_split_gain": 0.0,
        "n_estimators": 450,
        "num_leaves": 29,
        "random_state": 42,
        "reg_alpha": 0.0018604271097014996,
        "reg_lambda": 0.7526859917125899,
        "subsample": 0.8397151514472229,
        "subsample_for_bin": 200000,
        "subsample_freq": 0,
        "verbose": -1
    }

def train():
    df = transform_features(load_data())
    X = df.drop(columns=['default.payment.next.month'])
    y = df['default.payment.next.month']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    model = LGBMClassifier(**PARAMS)

    model.fit(X_train, y_train)

    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    train_roc = roc_auc_score(y_train, train_proba)
    test_roc = roc_auc_score(y_test, test_proba)
    test_pr = average_precision_score(y_test, test_proba)
    gap = train_roc - test_roc

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1
    )

    print(f"Train ROC-AUC: {train_roc:.4f}")
    print(f"Test  ROC-AUC: {test_roc:.4f}")
    print(f"Gap:           {gap:.4f}")
    print(f"Test  PR-AUC:  {test_pr:.4f}")
    print(f"CV    ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "default_pipeline.pkl")

    metadata = {
        "model": type(model).__name__,
        "version": "1.0",
        "trained_at": datetime.now().isoformat(),
        "reason": "Лучший test ROC-AUC + почти минимальный gap + статистически не отличается от CatBoost и RandomForest",
        "metrics": {
            "train_roc_auc": float(train_roc),
            "test_roc_auc": float(test_roc),
            "gap": float(gap),
            "test_pr_auc": float(test_pr),
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
        },
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "hyperparams": model.get_params()
    }

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved: {MODELS_DIR / 'default_pipeline.pkl'}")
    print(f"Saved: {MODELS_DIR / 'metadata.json'}")

    return metadata

if __name__ == "__main__":
    train()