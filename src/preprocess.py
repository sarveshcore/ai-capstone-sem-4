# ABOUTME: Builds the sklearn ColumnTransformer preprocessor and splits data into train/test sets.
# ABOUTME: Provides build_preprocessor() and split_data() used by the training pipeline.

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ['Age', 'Credit amount', 'Duration']
CATEGORICAL_FEATURES = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that scales numerics and encodes categoricals."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_FEATURES),
        ]
    )


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split a cleaned DataFrame into X_train, X_test, y_train, y_test."""
    y = df['Risk']
    X = df.drop('Risk', axis=1)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
