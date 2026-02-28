# ABOUTME: Unit tests for the credit risk training pipeline (data → preprocess → predict).
# ABOUTME: Run with: pytest tests/

import os
import sys
import pandas as pd
import pytest

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import load_data
from src.preprocess import build_preprocessor, split_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.predict import make_prediction


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'german_credit_data.csv')


class TestDataLoading:
    def test_loads_csv_returns_dataframe(self):
        df = load_data(DATA_PATH)
        assert isinstance(df, pd.DataFrame)

    def test_drops_unnamed_column(self):
        df = load_data(DATA_PATH)
        assert 'Unnamed: 0' not in df.columns

    def test_target_is_binary(self):
        df = load_data(DATA_PATH)
        assert set(df['Risk'].unique()).issubset({0, 1})

    def test_no_na_string_in_account_columns(self):
        df = load_data(DATA_PATH)
        assert 'NA' not in df['Saving accounts'].values
        assert 'NA' not in df['Checking account'].values


class TestPreprocessing:
    def setup_method(self):
        self.df = load_data(DATA_PATH)

    def test_split_data_shapes(self):
        X_train, X_test, y_train, y_test = split_data(self.df)
        total = len(self.df)
        assert len(X_train) + len(X_test) == total
        assert len(y_train) + len(y_test) == total

    def test_preprocessor_transforms_training_data(self):
        X_train, _, y_train, _ = split_data(self.df)
        preprocessor = build_preprocessor()
        transformed = preprocessor.fit_transform(X_train)
        assert transformed.shape[0] == len(X_train)

    def test_feature_columns_present(self):
        df = load_data(DATA_PATH)
        X = df.drop('Risk', axis=1)
        for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            assert col in X.columns, f"Missing expected column: {col}"


class TestPrediction:
    def test_make_prediction_structure(self):
        """make_prediction returns correct keys with a dummy sklearn pipeline."""
        from unittest.mock import MagicMock
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.8, 0.2]]

        input_df = pd.DataFrame({
            'Age': [30], 'Sex': ['male'], 'Job': [2], 'Housing': ['own'],
            'Saving accounts': ['little'], 'Checking account': ['moderate'],
            'Credit amount': [2000], 'Duration': [12], 'Purpose': ['car'],
        })

        result = make_prediction(mock_model, input_df)
        assert 'prediction' in result
        assert 'risk_label' in result
        assert 'confidence' in result
        assert result['prediction'] == 0
        assert result['risk_label'] == 'Low Risk (Good)'

    def test_make_prediction_bad_risk(self):
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.2, 0.8]]

        input_df = pd.DataFrame({'dummy': [1]})
        result = make_prediction(mock_model, input_df)
        assert result['prediction'] == 1
        assert result['risk_label'] == 'High Risk (Bad)'
        assert result['confidence'] == 80.0
