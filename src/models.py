# src/models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import optuna
import joblib
import mlflow
import mlflow.xgboost
from pathlib import Path


# Load features
def load_features():
    return pd.read_csv("data/processed/features_data.csv")


# Preprocessing + SMOTE
def prepare_data(df):
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    smote = SMOTE(random_state=42)
    X_train_res = preprocessor.fit_transform(X_train)
    X_train_res, y_train_res = smote.fit_resample(X_train_res, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_res, X_test_processed, y_train_res, y_test, preprocessor


# Optuna objective
def objective(trial, X_train, y_train):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model.score(X_train, y_train)


def train_model():
    df = load_features()
    X_train_res, X_test_processed, y_train_res, y_test, preprocessor = prepare_data(df)
    
    with mlflow.start_run():
        # === OPTUNA TUNING ===
        def run_optimization(X_train, y_train):
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: objective(trial, X_train, y_train),
                n_trials=30
            )
            return study.best_params
        
        best_params = run_optimization(X_train_res, y_train_res)
        mlflow.log_params(best_params)
        
        # === FINAL MODEL ===
        final_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
        final_model.fit(X_train_res, y_train_res)
        
        # === SAVE ===
        Path("models").mkdir(exist_ok=True)
        joblib.dump(preprocessor, "models/preprocessor.pkl")
        joblib.dump(final_model, "models/xgboost_model.pkl")
        
        mlflow.xgboost.log_model(final_model, "xgboost_model")
        mlflow.log_artifact("models/preprocessor.pkl")
        
        print(f"Model trained! Best params: {best_params}")
        print(f"Train accuracy: {final_model.score(X_train_res, y_train_res):.4f}")
        
        # === SAVE TEST DATA ===
        Path("data/processed").mkdir(exist_ok=True)
        pd.DataFrame(X_test_processed).to_csv("data/processed/X_test.csv", index=False)
        pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)
        print("Test data saved for evaluation.")
        
        return final_model, X_test_processed, y_test, preprocessor


if __name__ == "__main__":
    train_model()