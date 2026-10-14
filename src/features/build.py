import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.data.load import load_data

bill_features = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
pay_features = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
payamt_features = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

numeric_features = [
    'AGE',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
    'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'BILL_mean', 'BILL_std', 'BILL_grow', 'BILL_trend',
    'never_paid', 'UTILIZATION', 'LIMIT_BAL_log',
    'PAY_max', 'has_delay'
]

categorical_features = [
    'SEX',
    'EDUCATION',
    'MARRIAGE',
]

def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['BILL_mean'] = df[bill_features].mean(axis=1)
    df['BILL_std'] = df[bill_features].std(axis=1)
    df['BILL_grow'] = (df['BILL_AMT6'] > df['BILL_AMT1']).astype(int)
    df['BILL_trend'] = df['BILL_AMT6'] - df['BILL_AMT1']
    df = df.drop(columns=bill_features)

    df['PAY_max'] = np.max(df[pay_features], axis=1)
    df['has_delay'] = (df['PAY_max'] > 0).astype(int)
    df = df.drop(columns=pay_features)

    df['never_paid'] = (df[payamt_features].sum(axis=1) == 0).astype(int)
    df[payamt_features] = np.log1p(df[payamt_features])

    df['UTILIZATION'] = df['BILL_mean'] / df['LIMIT_BAL']
    df['LIMIT_BAL_log'] = np.log1p(df['LIMIT_BAL'])
    df = df.drop(columns=['LIMIT_BAL'])

    return df

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

if __name__ == '__main__':
    df = transform_features(load_data())
    print(f"Shape after FE: {df.shape}")
    print(f"New columns: {[c for c in df.columns if c not in load_data().columns]}")

    preprocessor = build_preprocessor()
    X = df.drop(columns=["default.payment.next.month"])
    X_transformed = preprocessor.fit_transform(X)
    print(f"After preprocessor: {X_transformed.shape}")

    models_dir = Path(__file__).resolve().parents[2] / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(preprocessor, models_dir / "preprocessor.pkl")
    print(f"Saved: {models_dir / 'preprocessor.pkl'}")
