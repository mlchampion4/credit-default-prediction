import pandas as pd
import pytest
from src.features.build import transform_features

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "SEX": [1],
        "EDUCATION": [2],
        "MARRIAGE": [2],
        "AGE": [46],
        "PAY_0": [0],
        "PAY_2": [2],
        "PAY_3": [0],
        "PAY_4": [0],
        "PAY_5": [0],
        "PAY_6": [0],
        "BILL_AMT1": [5000.0],
        "BILL_AMT2": [4500.0],
        "BILL_AMT3": [4200.0],
        "BILL_AMT4": [4000.0],
        "BILL_AMT5": [3800.0],
        "BILL_AMT6": [3776.0],
        "PAY_AMT1": [26000.0],
        "PAY_AMT2": [0.0],
        "PAY_AMT3": [3450.0],
        "PAY_AMT4": [2300.0],
        "PAY_AMT5": [1750.0],
        "PAY_AMT6": [2850.0],
        "LIMIT_BAL": [50000.0],
    })

def test_transform_features_adds_colums(sample_df):
    out = transform_features(sample_df)
    expected = {'BILL_mean', 'BILL_std', 'BILL_grow', 'BILL_trend', 'PAY_max', 'has_delay', 'never_paid', 'UTILIZATION', 'LIMIT_BAL_log'}
    assert expected.issubset(out.columns)

def test_bill_mean(sample_df):
    out = transform_features(sample_df)
    assert int(out['BILL_mean'].iloc[0]) == 4212

def test_pay_max(sample_df):
    out = transform_features(sample_df)
    assert out['PAY_max'].iloc[0] == 2