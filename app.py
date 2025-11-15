# app.py
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
import uvicorn
import numpy as np
from pathlib import Path

app = FastAPI(title="Customer Churn Predictor")

MODEL_PATH = "models/xgboost_model.pkl"
PREPROC_PATH = "models/preprocessor.pkl"

# === AUTO-RUN PIPELINE IF MODEL MISSING ===
if not Path(MODEL_PATH).exists() or not Path(PREPROC_PATH).exists():
    print("Model not found — running pipeline...")
    import subprocess
    result = subprocess.run(["python", "main.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("PIPELINE FAILED:", result.stderr)
    else:
        print("PIPELINE COMPLETE! Model ready.")

# Load model
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROC_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None
    preprocessor = None

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Customer Churn Predictor</h1>
    <p>Upload a CSV to predict churn risk!</p>
    <form action="/predict" enctype="multipart/form-data" method="post">
        <input name="file" type="file" accept=".csv" required>
        <input type="submit" value="Predict Churn">
    </form>
    <p><strong>By @mcebuara</strong> | <a href="https://github.com/mcebuara/customer-churn-pipeline">GitHub</a></p>
    """

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not available. Try again later."}
    
    df = pd.read_csv(file.file)
    df = df.fillna(0)
    
    try:
        X = preprocessor.transform(df)
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        result_df = pd.DataFrame({
            'churn_probability': probs.round(4),
            'will_churn': np.where(preds == 1, 'Yes', 'No')
        })
        return result_df.to_html(index=False)
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)