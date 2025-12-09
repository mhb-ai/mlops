# src/predict.py
import joblib
from .utils import clean_text

# Load trained model & vectorizer
vectorizer, model = joblib.load("models/model.pkl")

def predict(text: str):
    """
    Returns model prediction for a single text input
    """
    cleaned_text = clean_text(text)
    X = vectorizer.transform([cleaned_text])
    return model.predict(X)[0]

# Optional test
if __name__ == "__main__":
    sample_text = "I am going to north for vaccations this winter"
    print("Prediction:", predict(sample_text))
