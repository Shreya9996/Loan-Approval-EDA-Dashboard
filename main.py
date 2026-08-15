import joblib
import pandas as pd
import requests

from fastapi import FastAPI,HTTPException
from pydantic import BaseModel

from fastapi import Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")


app = FastAPI()

model = joblib.load("Loan_model.joblib")
features = joblib.load("Loan_features.joblib")
ss = joblib.load("standard_scaler.joblib")
mms = joblib.load("minmax_scaler.joblib")


class LoanFeatures(BaseModel):
    name:str
    age:int
    email: str
    phone:str
    person_income: float
    person_home_ownership: str
    loan_int_rate: float
    loan_percent_income: float
    previous_loan_defaults_on_file: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.post("/predict")
def predict(data: LoanFeatures):

    try:
        home_ownership_mapping = {
            "RENT": 0,
            "OWN": 1,
            "MORTGAGE": 2,
            "OTHER": 3
        }

        previous_loan_defaults_mapping = {
            "NO": 0,
            "YES": 1
        }

        input_df = pd.DataFrame([{
            "person_income": data.person_income,
            "person_home_ownership": home_ownership_mapping[data.person_home_ownership],
            "loan_int_rate": data.loan_int_rate,
            "loan_percent_income": data.loan_percent_income,
            "previous_loan_defaults_on_file": previous_loan_defaults_mapping[data.previous_loan_defaults_on_file]
        }])

        input_df[['person_income', 'loan_percent_income']] = ss.transform(
            input_df[['person_income', 'loan_percent_income']]
        )

        input_df[['loan_int_rate']] = mms.transform(
            input_df[['loan_int_rate']]
        )

        input_df = input_df[features]

        prediction = model.predict(input_df)[0]

        result = "Approved" if prediction == 1 else "Rejected"

        payload = {
        "name": data.name,
        "age": data.age,
        "income": data.person_income,
        "home_ownership": data.person_home_ownership,
        "interest_rate": data.loan_int_rate,
        "loan_percent_income": data.loan_percent_income,
        "previous_default": data.previous_loan_defaults_on_file,
        "prediction": result,
        "email": data.email,
        "phone": data.phone
    }

        try:
            requests.post(
                "http://localhost:5678/webhook-test/Loan_Application",
                json=payload,
                timeout=5
            )
        except Exception as webhook_error:
            print("Webhook Error:", webhook_error)

        return {
            "prediction": result
        }

    except Exception as e:
        print("ERROR OCCURRED:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed: {str(e)}"
        )
    