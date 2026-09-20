"""
TRAIN SPECIALIZED REAL ESTATE VALUATION ENGINE

Architecture:

Market
  -> Property Type
      -> CatBoost
      -> ExtraTrees
      -> RandomForest
      -> Validation-weighted Ensemble

A market-level fallback model is always trained.

Specialized models are only trained when a property type
has enough observations.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import engine
from src.ml.features import (
    build_features,
    get_feature_lists,
)


MODELS_DIR = (
    PROJECT_ROOT
    / "models"
    / "valuation_engine"
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_STATE = 42

MIN_SPECIALIZED_ROWS = 500

MARKET_VALUES = {
    "sale": "Residential for Sale",
    "rent": "Residential for Rent",
}


# =====================================================================
# DATA
# =====================================================================

def load_data() -> pd.DataFrame:

    query = """
        SELECT *
        FROM analytics.property_listings
        WHERE price_egp IS NOT NULL
          AND price_egp > 0
          AND offering_type IS NOT NULL;
    """

    return pd.read_sql(
        query,
        engine,
    )


# =====================================================================
# METRICS
# =====================================================================

def evaluate(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> dict[str, float]:

    y_true = np.expm1(
        y_true_log
    )

    y_pred = np.maximum(
        0,
        np.expm1(
            y_pred_log
        ),
    )

    return {
        "mae_egp": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse_egp": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "median_absolute_error_egp": float(
            median_absolute_error(
                y_true,
                y_pred,
            )
        ),
    }


# =====================================================================
# FEATURE PREPARATION
# =====================================================================

def prepare_features(
    df: pd.DataFrame,
):

    numeric_features, categorical_features = (
        get_feature_lists()
    )

    X = build_features(
        df.drop(
            columns=["price_egp"],
            errors="ignore",
        ).copy()
    )

    selected_features = (
        numeric_features
        + categorical_features
    )

    selected_features = [
        feature
        for feature in selected_features
        if feature in X.columns
    ]

    X = X[
        selected_features
    ].copy()

    for column in categorical_features:

        if column in X.columns:

            X[column] = (
                X[column]
                .fillna("__MISSING__")
                .astype(str)
            )

    categorical_features = [
        feature
        for feature in categorical_features
        if feature in X.columns
    ]

    numeric_features = [
        feature
        for feature in numeric_features
        if feature in X.columns
    ]

    return (
        X,
        numeric_features,
        categorical_features,
    )


# =====================================================================
# CATBOOST
# =====================================================================

def train_catboost(
    X_train,
    y_train,
    X_validation,
    y_validation,
    categorical_features,
):

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=3000,
        learning_rate=0.04,
        depth=10,
        l2_leaf_reg=7,
        random_seed=RANDOM_STATE,
        random_strength=1,
        border_count=128,
        od_type="Iter",
        od_wait=150,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
        eval_set=(
            X_validation,
            y_validation,
        ),
        use_best_model=True,
    )

    return model


# =====================================================================
# SKLEARN MODELS
# =====================================================================

def build_sklearn_models(
    numeric_features,
    categorical_features,
):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        )
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
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
                ),
                categorical_features,
            ),
        ]
    )

    models = {}

    models["extra_trees"] = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=300,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models["random_forest"] = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return models


# =====================================================================
# TRAIN ONE MODEL FAMILY
# =====================================================================

def train_model_family(
    df: pd.DataFrame,
    market_type: str,
    segment_name: str,
):

    X, numeric_features, categorical_features = (
        prepare_features(df)
    )

    y = np.log1p(
        df["price_egp"]
        .astype(float)
        .to_numpy()
    )

    (
        X_train,
        X_temp,
        y_train,
        y_temp,
    ) = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    (
        X_validation,
        X_test,
        y_validation,
        y_test,
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    trained_models = {}

    # ---------------------------------------------------------------
    # CATBOOST
    # ---------------------------------------------------------------

    print(
        f"\n[{segment_name}] CatBoost"
    )

    start = time.time()

    catboost = train_catboost(
        X_train,
        y_train,
        X_validation,
        y_validation,
        categorical_features,
    )

    catboost_time = time.time() - start

    catboost_prediction = catboost.predict(
        X_validation,
        cat_features=categorical_features,
    )

    catboost_metrics = evaluate(
        y_validation,
        catboost_prediction,
    )

    trained_models["catboost"] = {
        "model": catboost,
        "metrics": catboost_metrics,
        "training_seconds": catboost_time,
    }

    # ---------------------------------------------------------------
    # EXTRA TREES / RANDOM FOREST
    # ---------------------------------------------------------------

    sklearn_models = build_sklearn_models(
        numeric_features,
        categorical_features,
    )

    for model_name, model in sklearn_models.items():

        print(
            f"[{segment_name}] "
            f"{model_name}"
        )

        start = time.time()

        model.fit(
            X_train,
            y_train,
        )

        elapsed = time.time() - start

        prediction = model.predict(
            X_validation
        )

        metrics = evaluate(
            y_validation,
            prediction,
        )

        trained_models[model_name] = {
            "model": model,
            "metrics": metrics,
            "training_seconds": elapsed,
        }

    # ---------------------------------------------------------------
    # ENSEMBLE WEIGHTS
    # ---------------------------------------------------------------

    # Lower validation RMSE gets higher weight.
    inverse_errors = {}

    for name, result in trained_models.items():

        rmse = max(
            result["metrics"]["rmse_egp"],
            1.0,
        )

        inverse_errors[name] = 1.0 / rmse

    total_weight = sum(
        inverse_errors.values()
    )

    for name, result in trained_models.items():

        result["ensemble_weight"] = (
            inverse_errors[name]
            / total_weight
        )

    # ---------------------------------------------------------------
    # TEST ENSEMBLE
    # ---------------------------------------------------------------

    test_predictions = []

    test_weights = []

    for name, result in trained_models.items():

        model = result["model"]

        if name == "catboost":

            prediction = model.predict(
                X_test,
                cat_features=categorical_features,
            )

        else:

            prediction = model.predict(
                X_test
            )

        test_predictions.append(
            prediction
        )

        test_weights.append(
            result["ensemble_weight"]
        )

    ensemble_prediction = np.average(
        np.vstack(
            test_predictions
        ),
        axis=0,
        weights=test_weights,
    )

    ensemble_metrics = evaluate(
        y_test,
        ensemble_prediction,
    )

    print(
        f"\n[{segment_name}] ENSEMBLE TEST"
    )

    print(
        f"MAE: "
        f"{ensemble_metrics['mae_egp']:,.2f}"
    )

    print(
        f"RMSE: "
        f"{ensemble_metrics['rmse_egp']:,.2f}"
    )

    print(
        f"R²: "
        f"{ensemble_metrics['r2']:.4f}"
    )

    # ---------------------------------------------------------------
    # RETRAIN MODELS ON ALL SEGMENT DATA
    # ---------------------------------------------------------------

    print(
        f"\n[{segment_name}] "
        f"Retraining ensemble on all data..."
    )

    final_models = {}

    # CatBoost
    final_catboost = CatBoostRegressor(
        loss_function="RMSE",
        iterations=3000,
        learning_rate=0.04,
        depth=10,
        l2_leaf_reg=7,
        random_seed=RANDOM_STATE,
        random_strength=1,
        border_count=128,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    final_catboost.fit(
        X,
        y,
        cat_features=categorical_features,
    )

    final_models["catboost"] = final_catboost

    # sklearn models
    final_sklearn_models = build_sklearn_models(
        numeric_features,
        categorical_features,
    )

    for name, model in final_sklearn_models.items():

        model.fit(
            X,
            y,
        )

        final_models[name] = model

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    model_entries = []

    for name, model in final_models.items():

        filename = (
            f"{market_type}_"
            f"{segment_name}_"
            f"{name}.pkl"
            .replace(" ", "_")
            .lower()
        )

        path = (
            MODELS_DIR
            / filename
        )

        joblib.dump(
            model,
            path,
            compress=3,
        )

        model_entries.append(
            {
                "model_name": name,
                "model_type": (
                    "catboost"
                    if name == "catboost"
                    else "sklearn"
                ),
                "model_path": filename,
                "ensemble_weight": float(
                    trained_models[name][
                        "ensemble_weight"
                    ]
                ),
            }
        )

    return {
        "segment": segment_name,
        "rows": len(df),
        "feature_columns": list(
            X.columns
        ),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "validation_models": {
            name: {
                "metrics": result["metrics"],
                "ensemble_weight": result[
                    "ensemble_weight"
                ],
            }
            for name, result
            in trained_models.items()
        },
        "ensemble_test_metrics": ensemble_metrics,
        "models": model_entries,
    }


# =====================================================================
# MARKET TRAINING
# =====================================================================

def train_market(
    df: pd.DataFrame,
    market_type: str,
    market_value: str,
):

    market_df = df[
        df["offering_type"]
        == market_value
    ].copy()

    print("\n" + "#" * 80)
    print(
        f"MARKET: {market_type.upper()}"
    )
    print(
        f"Rows: {len(market_df):,}"
    )
    print("#" * 80)

    # ---------------------------------------------------------------
    # MARKET MODEL
    # ---------------------------------------------------------------

    market_result = train_model_family(
        market_df,
        market_type,
        "market",
    )

    specialized_models = {}

    # ---------------------------------------------------------------
    # PROPERTY TYPE SEGMENTS
    # ---------------------------------------------------------------

    property_counts = (
        market_df[
            "property_type"
        ]
        .fillna("__MISSING__")
        .astype(str)
        .value_counts()
    )

    print("\nProperty type distribution:")

    print(
        property_counts.to_string()
    )

    for property_type, count in (
        property_counts.items()
    ):

        if count < MIN_SPECIALIZED_ROWS:

            print(
                f"\nSkipping specialized model "
                f"for '{property_type}' "
                f"({count} rows < "
                f"{MIN_SPECIALIZED_ROWS})"
            )

            continue

        segment_df = market_df[
            market_df[
                "property_type"
            ]
            .fillna("__MISSING__")
            .astype(str)
            == property_type
        ].copy()

        segment_key = (
            property_type
            .strip()
            .lower()
        )

        result = train_model_family(
            segment_df,
            market_type,
            segment_key,
        )

        specialized_models[
            segment_key
        ] = result

    return {
        "market_models": market_result[
            "models"
        ],
        "market_validation_models": market_result[
            "validation_models"
        ],
        "market_test_metrics": market_result[
            "ensemble_test_metrics"
        ],
        "feature_columns": market_result[
            "feature_columns"
        ],
        "numeric_features": market_result[
            "numeric_features"
        ],
        "categorical_features": market_result[
            "categorical_features"
        ],
        "validation_r2": market_result[
            "ensemble_test_metrics"
        ]["r2"],
        "specialized_models": {
            key: {
                "rows": value["rows"],
                "models": value["models"],
                "validation_models": value[
                    "validation_models"
                ],
                "test_metrics": value[
                    "ensemble_test_metrics"
                ],
            }
            for key, value
            in specialized_models.items()
        },
    }


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 80)
    print(
        "REAL ESTATE MARKET INTELLIGENCE PLATFORM"
    )
    print(
        "SPECIALIZED REAL ESTATE VALUATION ENGINE"
    )
    print("=" * 80)

    print(
        "\nLoading data from SQL Server..."
    )

    df = load_data()

    print(
        f"Loaded {len(df):,} records."
    )

    engine_metadata = {
        "engine": (
            "Hierarchical Real Estate "
            "Valuation Engine"
        ),
        "version": "1.0",
        "trained_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "random_state": RANDOM_STATE,
        "minimum_specialized_rows": (
            MIN_SPECIALIZED_ROWS
        ),
        "target": "price_egp",
        "prediction_scale": "log1p",
        "markets": {},
    }

    for market_type, market_value in (
        MARKET_VALUES.items()
    ):

        engine_metadata[
            "markets"
        ][market_type] = train_market(
            df,
            market_type,
            market_value,
        )

    metadata_path = (
        MODELS_DIR
        / "engine_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            engine_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print(
        "VALUATION ENGINE TRAINING COMPLETE"
    )
    print("=" * 80)

    print(
        f"\nMetadata:"
    )

    print(
        metadata_path
    )


if __name__ == "__main__":
    main()  
      