import joblib
import pandas as pd

model = joblib.load("app/model.pkl")

def predict_price(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return float(round(prediction, 2))