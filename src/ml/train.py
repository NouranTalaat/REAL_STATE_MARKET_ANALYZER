
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.ml.evaluate import evaluate_predictions
from src.ml.features import build_features, get_feature_lists


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


MARKET_TYPE_COLUMN = "offering_type"

MARKET_TYPE_MAPPING = {
    "sale": "Residential for Sale",
    "rent": "Residential for Rent",
}


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def build_model(market_type: str):
    if market_type == "sale":
        return ExtraTreesRegressor(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    if market_type == "rent":
        return RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unsupported market type: {market_type}"
    )


def train_model(
    df: pd.DataFrame,
    market_type: str,
) -> dict:
    # ---------------------------------------------------------
    # Validate market type
    # ---------------------------------------------------------

    if market_type not in MARKET_TYPE_MAPPING:
        raise ValueError(
            f"Unsupported market type: {market_type}. "
            f"Expected one of: {list(MARKET_TYPE_MAPPING)}"
        )

    # ---------------------------------------------------------
    # Validate market column
    # ---------------------------------------------------------

    if MARKET_TYPE_COLUMN not in df.columns:
        raise ValueError(
            f"Required market column '{MARKET_TYPE_COLUMN}' "
            "was not found in the training data."
        )

    expected_market_value = MARKET_TYPE_MAPPING[
        market_type
    ]

    # ---------------------------------------------------------
    # Filter Sale / Rent
    # ---------------------------------------------------------

    market_df = df[
        df[MARKET_TYPE_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        == expected_market_value.lower()
    ].copy()

    if market_df.empty:
        available_values = (
            df[MARKET_TYPE_COLUMN]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        raise ValueError(
            f"No records found for market type: {market_type}. "
            f"Expected '{expected_market_value}' in "
            f"'{MARKET_TYPE_COLUMN}'. "
            f"Available values: {available_values}"
        )

    # ---------------------------------------------------------
    # Validate target
    # ---------------------------------------------------------

    market_df["price_egp"] = pd.to_numeric(
        market_df["price_egp"],
        errors="coerce",
    )

    market_df = market_df[
        market_df["price_egp"].notna()
        & (market_df["price_egp"] > 0)
    ].copy()

    if market_df.empty:
        raise ValueError(
            f"No valid price records found for market type: "
            f"{market_type}"
        )

    # ---------------------------------------------------------
    # Prepare ML dataset
    # ---------------------------------------------------------

    X, y = _prepare_training_data(
        market_df
    )

    # ---------------------------------------------------------
    # Train / Test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # ---------------------------------------------------------
    # Feature definitions
    # ---------------------------------------------------------

    numeric_features, categorical_features = (
        get_feature_lists()
    )

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = build_model(
        market_type
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    print(
        f"\nTraining {market_type.upper()} model..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )

    metrics = evaluate_predictions(
        y_true_log=y_test,
        y_pred_log=predictions,
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODELS_DIR
        / f"{market_type}_model.pkl"
    )

    joblib.dump(
        pipeline,
        model_path,
        compress=3,
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = {
        "market_type": market_type,
        "market_column": MARKET_TYPE_COLUMN,
        "market_value": expected_market_value,
        "target": "price_egp",
        "prediction_scale": "log1p",
        "trained_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "training_rows": int(
            len(X_train)
        ),
        "test_rows": int(
            len(X_test)
        ),
        "feature_count": int(
            len(numeric_features)
            + len(categorical_features)
        ),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_type": type(model).__name__,
        "model_parameters": model.get_params(),
        "metrics": metrics,
        "leakage_protection": [
            "price_egp",
            "price_per_sqm",
            "target",
            "log_price",
            "log_target",
        ],
        "random_state": 42,
    }

    metadata_path = (
        MODELS_DIR
        / f"{market_type}_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "market_type": market_type,
        "market_value": expected_market_value,
        "model_path": str(
            model_path
        ),
        "metadata_path": str(
            metadata_path
        ),
        "training_rows": len(
            X_train
        ),
        "test_rows": len(
            X_test
        ),
        "metrics": metrics,
    }


def _prepare_training_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for model training.

    Important:
    The target is extracted BEFORE build_features()
    because build_features() intentionally removes target
    and target-derived columns to prevent leakage.
    """

    working_df = df.copy()

    # ---------------------------------------------------------
    # Validate target
    # ---------------------------------------------------------

    if "price_egp" not in working_df.columns:
        raise ValueError(
            "Training dataset must contain 'price_egp'."
        )

    # ---------------------------------------------------------
    # Extract target BEFORE feature engineering
    # ---------------------------------------------------------

    target = pd.to_numeric(
        working_df["price_egp"],
        errors="coerce",
    )

    valid_target_mask = (
        target.notna()
        & (target > 0)
    )

    working_df = working_df.loc[
        valid_target_mask
    ].copy()

    target = target.loc[
        valid_target_mask
    ].copy()

    if working_df.empty:
        raise ValueError(
            "No valid positive price records found."
        )

    # ---------------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------------

    features = build_features(
        working_df
    )

    # ---------------------------------------------------------
    # Feature definitions
    # ---------------------------------------------------------

    numeric_features, categorical_features = (
        get_feature_lists()
    )

    feature_columns = (
        numeric_features
        + categorical_features
    )

    # ---------------------------------------------------------
    # Validate features
    # ---------------------------------------------------------

    missing_features = [
        column
        for column in feature_columns
        if column not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing engineered features: "
            + ", ".join(
                missing_features
            )
        )

    # ---------------------------------------------------------
    # Build X
    # ---------------------------------------------------------

    X = features[
        feature_columns
    ].copy()

    # ---------------------------------------------------------
    # Build y
    #
    # Log transformation reduces the impact of extreme
    # real-estate prices and makes the regression target
    # more stable.
    # ---------------------------------------------------------

    y = pd.Series(
        np.log1p(target),
        index=target.index,
        name="log_price",
    )

    return X, y


def train_all(
    df: pd.DataFrame,
) -> list[dict]:
    results = []

    for market_type in (
        "sale",
        "rent",
    ):
        result = train_model(
            df=df,
            market_type=market_type,
        )

        results.append(
            result
        )

    return results
