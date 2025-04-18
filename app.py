import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Get absolute path of the model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# Check if model, vectorizer, and label encoder exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Ensure the model is trained and saved.")

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}. Ensure it was saved during training.")

if not os.path.exists(LABEL_ENCODER_PATH):
    raise FileNotFoundError(f"Label encoder file not found: {LABEL_ENCODER_PATH}. Ensure it was saved during training.")

# Load trained model, vectorizer, and label encoder
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Initialize FastAPI app
app = FastAPI(title="Fake News Detection API (Naïve Bayes)", description="Detects whether a given news statement is Fake or Real.")

# Pydantic model for request validation
class NewsInput(BaseModel):
    statement: str

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Fake News Detection API. Use /predict to make predictions."}

# Prediction endpoint using only statement
@app.post("/predict")
def predict(news: NewsInput):
    """
    Apply the same text preprocessing before making predictions.
    """
    try:
        # Convert text into numerical representation using the trained vectorizer
        input_vectorized = vectorizer.transform([news.statement])

        # Make prediction using BernoulliNB model
        prediction_encoded = model.predict(input_vectorized)[0]

        # Decode label back to original class name
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        return {"prediction": prediction_label}
    
    except Exception as e:
        return {"error": f"Prediction Error: {str(e)}"}
