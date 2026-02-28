# ABOUTME: Handles loading and initial cleaning of the German credit dataset.
# ABOUTME: Provides load_data() which returns a clean DataFrame ready for preprocessing.

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the German credit CSV and apply basic cleaning."""
    df = pd.read_csv(path)

    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # Fill missing/NA-string values with 'unknown' for account columns
    df['Saving accounts'] = df['Saving accounts'].replace('NA', 'unknown').fillna('unknown')
    df['Checking account'] = df['Checking account'].replace('NA', 'unknown').fillna('unknown')

    # Encode target: bad=1, good=0
    df['Risk'] = df['Risk'].map({'bad': 1, 'good': 0})

    return df
