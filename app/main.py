from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict_price

app = FastAPI(title="Airbnb Pricing API")

class PriceInput(BaseModel):
    neighbourhood_group: str
    room_type: str
    minimum_nights: int
    number_of_reviews: int
    reviews_per_month: float
    availability_365: int
    month: int  # month of year

@app.get("/")
def root():
    return {"message": "Airbnb Pricing API is running"}

@app.post("/predict")
def get_price(input_data: PriceInput):
    try:
        price = predict_price(input_data.dict())
        return {
            "predicted_price": float(price),
            "currency": "USD",
            "unit": "per night"
        }
    except Exception as e:
        print("💥 Prediction error:", e)
        return {"error": str(e)}