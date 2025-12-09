from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI(title="Intent Classifier")

class Intent(BaseModel):
    text: str

@app.post("/predict")
def predict_intent(data: Intent):
    """
    Calls the predict.py module for inference
    """
    prediction = predict(data.text)
    return {"prediction": str(prediction)}
