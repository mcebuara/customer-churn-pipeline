import pandas as pd
import yaml
from pathlib import Path
import numpy as np

# Load config
def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

def load_data():
    # Read bank_full.csv (saved with commas)
    bank = pd.read_csv("data/raw/bank_full.csv")  # NO sep=";"
    telecom = pd.read_csv("data/raw/telecom.csv")
    return bank, telecom

def clean_bank_data(df):
    # Replace 'unknown' with NaN
    df = df.replace("unknown", pd.NA)
    
    # Fix column types
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['pdays'] = pd.to_numeric(df['pdays'], errors='coerce')
    
    # Create churn label: 1 = yes, 0 = no
    df['churn'] = (df['y'] == 'yes').astype(int)
    
    return df

def clean_telecom_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(bool)
    df['churn'] = (df['Churn'] == 'Yes').astype(int)
    return df

def merge_and_save():
    bank, telecom = load_data()
    
    bank_clean = clean_bank_data(bank)
    telecom_clean = clean_telecom_data(telecom)
    
    # Use bank data (larger)
    final_df = bank_clean.copy()
    
    # Add fake support calls
    np.random.seed(42)
    final_df['support_calls'] = np.random.randint(0, 10, size=len(final_df))
    
    # Save
    Path("data/processed").mkdir(exist_ok=True)
    final_df.to_csv("data/processed/cleaned_data.csv", index=False)
    print(f"Cleaned data saved! {len(final_df)} rows.")

if __name__ == "__main__":
    merge_and_save()