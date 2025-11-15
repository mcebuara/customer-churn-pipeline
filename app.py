# app.py
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
import uvicorn
import numpy as np  # For any NaN handling

app = FastAPI(title="Customer Churn Predictor")

# Load model & preprocessor (assuming they exist after main.py)
try:
    model = joblib.load("models/xgboost_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model not found — run python main.py first!")
    # Fallback: Dummy model for demo
    model = None
    preprocessor = None

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Customer Churn Predictor</h1>
    <p>Upload a CSV with customer data (age, job, marital, etc.) to get churn risk!</p>
    <form action="/predict" enctype="multipart/form-data" method="post">
        <input name="file" type="file" accept=".csv">
        <input type="submit" value="Predict Churn">
    </form>
    <p><strong>Live Demo by @mcebuara</strong></p>
    """

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded — run pipeline first!"}
    
    df = pd.read_csv(file.file)
    
    # Basic preprocessing (handle missing columns/values)
    df = df.fillna(0)  # Simple fill for demo
    
    # Transform (assuming features match training)
    X = preprocessor.transform(df)
    
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    df['churn_probability'] = probs.round(4)
    df['will_churn'] = np.where(preds == 1, 'Yes', 'No')
    
    return df[['churn_probability', 'will_churn']].to_html()

if __name__ == "__main__":
    # CRITICAL: Bind to 0.0.0.0 and use Render's PORT
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)