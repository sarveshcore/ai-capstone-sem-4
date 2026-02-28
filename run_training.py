# ABOUTME: Orchestrates the full training pipeline: load → preprocess → train → evaluate → save.
# ABOUTME: Run this script to retrain the model: `python run_training.py`

import joblib
import os

from src.data import load_data
from src.preprocess import split_data
from src.train import evaluate_model, train_model

DATA_PATH = os.path.join('data', 'german_credit_data.csv')
MODEL_OUTPUT_PATH = os.path.join('models', 'credit_risk_model_v2.pkl')


def main():
    print("=== Credit Risk Model Training Pipeline ===\n")

    print("[1/4] Loading and cleaning data...")
    df = load_data(DATA_PATH)
    print(f"  Dataset shape: {df.shape}")

    print("\n[2/4] Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    print("\n[3/4] Training model with SMOTE + GridSearchCV...")
    model = train_model(X_train, y_train)

    print("\n[4/4] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"\n{metrics['report']}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Model saved → {MODEL_OUTPUT_PATH}")


if __name__ == '__main__':
    main()
