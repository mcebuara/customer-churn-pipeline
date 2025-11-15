import pandas as pd
import os
import urllib.request
import zipfile
import yaml

# Load config from same-level configs/ folder
def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

def download_bank_data():
    """Download real bank marketing data from UCI"""
    os.makedirs("data/raw", exist_ok=True)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    zip_path = "data/raw/bank.zip"
    
    print("Downloading bank data...")
    urllib.request.urlretrieve(url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/raw")
    
    # Read the main CSV
    df = pd.read_csv("data/raw/bank-additional/bank-additional-full.csv", sep=";")
    df.to_csv("data/raw/bank_full.csv", index=False)
    print(f"Bank data saved! {len(df)} customers.")

def download_telecom_data():
    """Download real telecom churn data"""
    url = "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df.to_csv("data/raw/telecom.csv", index=False)
    print(f"Telecom data saved! {len(df)} customers.")

if __name__ == "__main__":
    download_bank_data()
    download_telecom_data()