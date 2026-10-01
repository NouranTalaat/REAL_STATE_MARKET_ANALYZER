from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ml.features import build_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def load_model(
    market_type: str,
):
    if market_type not in {
        "sale",
        "rent",
    }:
        raise ValueError(
            "market_type must be 'sale' or 'rent'."
        )

    model_path = (
        MODELS_DIR
        / f"{market_type}_model.pkl"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(
        model_path
    )


def predict_price(
    data: pd.DataFrame,
    market_type: str,
) -> np.ndarray:

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    model = load_model(
        market_type
    )

    features = build_features(
        data
    )

    predictions_log = model.predict(
        features
    )

    predictions = np.expm1(
        predictions_log
    )

    return np.maximum(
        predictions,
        0,
    )


def predict_single(
    property_data: dict,
    market_type: str,
) -> float:

    dataframe = pd.DataFrame(
        [property_data]
    )

    prediction = predict_price(
        dataframe,
        market_type,
    )

    return float(
        prediction[0]
    )