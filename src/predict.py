# ABOUTME: Inference helper that wraps model.predict and returns structured prediction results.
# ABOUTME: Provides make_prediction() consumed by the Streamlit app.

import pandas as pd


def make_prediction(model, input_data: pd.DataFrame) -> dict:
    """
    Run inference on a single-row DataFrame.

    Returns a dict with:
        - prediction: int (0=good, 1=bad)
        - risk_label: str
        - confidence: float (0–100)
    """
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    if prediction == 1:
        risk_label = "High Risk (Bad)"
        confidence = probabilities[1] * 100
    else:
        risk_label = "Low Risk (Good)"
        confidence = probabilities[0] * 100

    return {
        'prediction': int(prediction),
        'risk_label': risk_label,
        'confidence': round(confidence, 1),
        'default_probability': round(float(probabilities[1]) * 100, 1),
    }
