# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
import uvicorn

app = FastAPI(title="Customer Churn Predictor")

# Load model & preprocessor
model = joblib.load("models/xgboost_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Customer Churn Predictor</h1>
    <form action="/predict" enctype="multipart/form-data" method="post">
        <input name="file" type="file" accept=".csv">
        <input type="submit" value="Predict Churn">
    </form>
    """

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    X = preprocessor.transform(df)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    df['churn_probability'] = probs
    df['will_churn'] = preds.map({0: 'No', 1: 'Yes'})
    
    return df[['churn_probability', 'will_churn']].to_html()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)