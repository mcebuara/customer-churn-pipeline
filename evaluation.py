# src/evaluation.py
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
from pathlib import Path

def load_model_and_data():
    # Load test data saved by models.py
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    model = joblib.load("models/xgboost_model.pkl")
    return model, X_test, y_test

def evaluate_model():
    model, X_test, y_test = load_model_and_data()
    
    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Metrics
    ap = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    # Business metric: Save $500 per retained customer, cost $50 to contact
    retention_cost = 50
    acquisition_cost = 500
    saved = ((y_test == 1) & (y_pred == 1)).sum() * (acquisition_cost - retention_cost)
    
    print(f"\nEVALUATION RESULTS:")
    print(f"   Average Precision: {ap:.4f}")
    print(f"   Customers saved: {((y_test == 1) & (y_pred == 1)).sum()}")
    print(f"   Estimated $ saved: ${saved:,.0f}")
    
    # Create reports folder
    Path("reports").mkdir(exist_ok=True)
    
    # Plot Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/pr_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("reports/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_metric("average_precision", ap)
        mlflow.log_metric("customers_saved", ((y_test == 1) & (y_pred == 1)).sum())
        mlflow.log_metric("dollars_saved", saved)
        mlflow.log_artifact("reports/pr_curve.png")
        mlflow.log_artifact("reports/confusion_matrix.png")
    
    print("Evaluation complete! Check reports/ folder.")

if __name__ == "__main__":
    evaluate_model()