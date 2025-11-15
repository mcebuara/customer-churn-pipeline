# main.py
import os
from data_ingestion import download_bank_data, download_telecom_data
from processing import merge_and_save
from features import create_features, save_features, load_cleaned_data
from models import train_model
from evaluation import evaluate_model

def run_pipeline():
    print("Starting Customer Churn Pipeline...\n")
    
    # Step 1: Download
    print("1. Downloading real data...")
    download_bank_data()
    download_telecom_data()
    
    # Step 2: Clean
    print("\n2. Cleaning data...")
    merge_and_save()
    
    # Step 3: Features
    print("\n3. Creating smart features...")
    df = load_cleaned_data()
    df = create_features(df)
    save_features(df)
    
    # Step 4: Train
    print("\n4. Training XGBoost with Optuna...")
    train_model()
    
    # Step 5: Evaluate
    print("\n5. Evaluating model...")
    evaluate_model()
    
    print("\nPIPELINE COMPLETE! Check reports/ and models/")

if __name__ == "__main__":
    run_pipeline()