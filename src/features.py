import pandas as pd
import numpy as np
from pathlib import Path

def load_cleaned_data():
    return pd.read_csv("data/processed/cleaned_data.csv")

def create_features(df):
    """
    Create smart features that help predict churn
    """
    # 1. Age groups
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 30, 40, 55, 100], 
                            labels=['young', 'adult', 'middle', 'senior'])

    # 2. Total contacts (campaign + previous)
    df['total_contacts'] = df['campaign'] + df['previous']

    # 3. Was contacted before? (pdays != 999)
    df['previously_contacted'] = (df['pdays'] != 999).astype(int)

    # 4. Campaign efficiency (duration per contact)
    df['duration_per_contact'] = df['duration'] / (df['campaign'] + 1)

    # 5. Month encoding (seasonal effect)
    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    df['month_num'] = df['month'].map(month_map)

    # 6. Job risk score (simple mapping)
    high_risk_jobs = ['student', 'unemployed', 'housemaid']
    df['high_risk_job'] = df['job'].isin(high_risk_jobs).astype(int)

    # 7. Support calls impact (fake but useful)
    df['high_support'] = (df['support_calls'] >= 5).astype(int)

    print(f"Created {len(df.columns) - 21} new features!")
    return df

def save_features(df):
    Path("data/processed").mkdir(exist_ok=True)
    df.to_csv("data/processed/features_data.csv", index=False)
    print(f"Features saved! {len(df)} rows, {len(df.columns)} columns.")

if __name__ == "__main__":
    df = load_cleaned_data()
    df = create_features(df)
    save  = save_features(df)