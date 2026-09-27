from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from src.prediction import predict

app = FastAPI(
    title="Credit Risk Modelling API",
    description=(
        "Predicts loan default probability and converts it into a "
        "CIBIL-style 300-900 credit score with a rating band."
    ),
    version="1.0.0",
)

class CreditRiskInput(BaseModel):
    age: int
    income: float
    loan_amount: float
    loan_tenure_months: int
    avg_dpd_per_delinquency: float
    delinquency_ratio: float
    credit_utilization_ratio: float
    num_open_accounts: int
    residence_type: Literal["Owned", "Rented", "Mortgage"]
    loan_purpose: Literal["Education", "Home", "Auto", "Personal"]
    loan_type: Literal["Unsecured", "Secured"]

class CreditRiskOutput(BaseModel):
    probability: float
    credit_score: int
    rating: str

@app.get("/ping")
def hello():
    return "Hello"

@app.post("/predict_credit_risk", response_model=CreditRiskOutput)
def predict_credit_risk(input_data: CreditRiskInput):
    print("Request received")
    try:
        probability, credit_score, rating = predict(input_data.age, input_data.income, input_data.loan_amount,
                                                    input_data.loan_tenure_months, input_data.avg_dpd_per_delinquency,
                                                    input_data.delinquency_ratio, input_data.credit_utilization_ratio,
                                                    input_data.num_open_accounts, input_data.residence_type,
                                                    input_data.loan_purpose, input_data.loan_type)
        return CreditRiskOutput(probability=probability, credit_score=credit_score, rating=rating)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))