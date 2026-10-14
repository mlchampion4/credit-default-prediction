from pydantic import BaseModel, Field

class Client(BaseModel):
    SEX: int = Field(..., ge=1, le=2, examples=[1])
    EDUCATION: int = Field(..., ge=1, le=6, examples=[2])
    MARRIAGE: int = Field(..., ge=1, le=3, examples=[2])
    AGE: int = Field(..., ge=18, le=100, examples=[46])

    PAY_0: int = Field(..., ge=-2, le=9, examples=[0])
    PAY_2: int = Field(..., ge=-2, le=9, examples=[0])
    PAY_3: int = Field(..., ge=-2, le=9, examples=[0])
    PAY_4: int = Field(..., ge=-2, le=9, examples=[0])
    PAY_5: int = Field(..., ge=-2, le=9, examples=[0])
    PAY_6: int = Field(..., ge=-2, le=9, examples=[0])

    BILL_AMT1: float = Field(..., ge=0, examples=[5000.0])
    BILL_AMT2: float = Field(..., ge=0, examples=[4500.0])
    BILL_AMT3: float = Field(..., ge=0, examples=[4200.0])
    BILL_AMT4: float = Field(..., ge=0, examples=[4000.0])
    BILL_AMT5: float = Field(..., ge=0, examples=[3800.0])
    BILL_AMT6: float = Field(..., ge=0, examples=[3776.0])

    PAY_AMT1: float = Field(..., ge=0, examples=[26000.0])
    PAY_AMT2: float = Field(..., ge=0, examples=[0.0])
    PAY_AMT3: float = Field(..., ge=0, examples=[3450.0])
    PAY_AMT4: float = Field(..., ge=0, examples=[2300.0])
    PAY_AMT5: float = Field(..., ge=0, examples=[1750.0])
    PAY_AMT6: float = Field(..., ge=0, examples=[2850.0])

    LIMIT_BAL: float = Field(..., gt=0, examples=[50000.0])

class PredictionResponse(BaseModel):
    default_probability: float
    risk: str
    threshold: float
    recommendation: str
    top_factors: list[dict]

class BatchRequest(BaseModel):
    clients: list[Client]

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int