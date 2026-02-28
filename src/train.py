# ABOUTME: Trains the credit risk model using SMOTE oversampling and GridSearchCV.
# ABOUTME: Provides train_model() which returns the best fitted pipeline and its evaluation metrics.

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.preprocess import build_preprocessor


PARAM_GRID = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__class_weight': ['balanced', 'balanced_subsample'],
}


def train_model(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5):
    """Fit a SMOTE + RandomForest pipeline via GridSearchCV. Returns the best estimator."""
    pipeline = ImbPipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42)),
    ])

    print("Running GridSearchCV (this may take a few minutes)...")
    grid_search = GridSearchCV(pipeline, PARAM_GRID, cv=cv, scoring='recall', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best params: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a fitted model and return a dict of metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'report': classification_report(y_test, y_pred),
    }
    return metrics
