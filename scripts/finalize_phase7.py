"""
REAL ESTATE MARKET INTELLIGENCE PLATFORM
PHASE 7 - FINAL ML MODEL SELECTION & TRAINING

Workflow:
1. Load validated data from SQL Server
2. Build leakage-safe features
3. Split into train / validation / test
4. Tune CatBoost using validation data
5. Evaluate final configuration on untouched test data
6. Retrain selected configuration on ALL valid market data
7. Save production model + metadata
8. Run a real prediction sanity check
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import engine
from src.ml.features import build_features, get_feature_lists


MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

MARKET_TYPE_COLUMN = "offering_type"

MARKET_TYPE_MAPPING = {
    "sale": "Residential for Sale",
    "rent": "Residential for Rent",
}


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def load_training_data() -> pd.DataFrame:
    query = """
        SELECT *
        FROM analytics.property_listings
        WHERE price_egp IS NOT NULL
          AND price_egp > 0
          AND offering_type IS NOT NULL;
    """

    return pd.read_sql(query, engine)


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def evaluate_predictions(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> dict[str, float]:

    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    y_pred = np.maximum(y_pred, 0)

    return {
        "mae_egp": float(mean_absolute_error(y_true, y_pred)),
        "rmse_egp": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "r2": float(r2_score(y_true, y_pred)),
        "median_absolute_error_egp": float(
            median_absolute_error(y_true, y_pred)
        ),
    }


# ---------------------------------------------------------------------
# CATBOOST PREPARATION
# ---------------------------------------------------------------------

def prepare_catboost_features(
    X: pd.DataFrame,
    categorical_features: list[str],
) -> pd.DataFrame:

    X = X.copy()

    for column in categorical_features:
        if column in X.columns:
            X[column] = X[column].fillna("__MISSING__").astype(str)

    return X


# ---------------------------------------------------------------------
# SPLIT
# ---------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:

    # 70% train
    # 15% validation
    # 15% untouched test

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ---------------------------------------------------------------------
# CATBOOST TUNING
# ---------------------------------------------------------------------

def tune_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    categorical_features: list[str],
) -> tuple[dict, CatBoostRegressor]:

    print("\n" + "=" * 80)
    print("CATBOOST TUNING")
    print("=" * 80)

    configurations = [
        {
            "name": "catboost_balanced",
            "iterations": 3500,
            "learning_rate": 0.035,
            "depth": 8,
            "l2_leaf_reg": 5,
            "random_strength": 1,
        },
        {
            "name": "catboost_deeper",
            "iterations": 3000,
            "learning_rate": 0.04,
            "depth": 10,
            "l2_leaf_reg": 7,
            "random_strength": 1,
        },
        {
            "name": "catboost_regularized",
            "iterations": 3500,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 10,
            "random_strength": 0.5,
        },
    ]

    best_model = None
    best_config = None
    best_rmse = float("inf")

    for index, config in enumerate(configurations, start=1):

        print(
            f"\n[{index}/{len(configurations)}] "
            f"{config['name']}"
        )

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=config["iterations"],
            learning_rate=config["learning_rate"],
            depth=config["depth"],
            l2_leaf_reg=config["l2_leaf_reg"],
            random_seed=RANDOM_STATE,
            random_strength=config["random_strength"],
            border_count=128,
            od_type="Iter",
            od_wait=150,
            verbose=500,
            allow_writing_files=False,
            thread_count=-1,
        )

        start = time.time()

        model.fit(
            X_train,
            y_train,
            cat_features=categorical_features,
            eval_set=(X_validation, y_validation),
            use_best_model=True,
        )

        elapsed = time.time() - start

        validation_prediction = model.predict(X_validation)

        metrics = evaluate_predictions(
            y_validation.to_numpy(),
            validation_prediction,
        )

        print(
            f"Validation MAE: "
            f"{metrics['mae_egp']:,.2f} EGP"
        )

        print(
            f"Validation RMSE: "
            f"{metrics['rmse_egp']:,.2f} EGP"
        )

        print(
            f"Validation R²: "
            f"{metrics['r2']:.4f}"
        )

        print(
            f"Median Absolute Error: "
            f"{metrics['median_absolute_error_egp']:,.2f} EGP"
        )

        print(
            f"Training time: "
            f"{elapsed:.2f}s"
        )

        # RMSE is the primary selection metric.
        if metrics["rmse_egp"] < best_rmse:

            best_rmse = metrics["rmse_egp"]
            best_model = model
            best_config = {
                **config,
                "validation_metrics": metrics,
                "training_seconds": elapsed,
                "best_iteration": int(model.get_best_iteration()),
            }

    print("\n" + "-" * 80)
    print("BEST CONFIGURATION")
    print("-" * 80)

    print(json.dumps(best_config, indent=2))

    return best_config, best_model


# ---------------------------------------------------------------------
# FINAL TEST
# ---------------------------------------------------------------------

def evaluate_on_untouched_test(
    model: CatBoostRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:

    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 80)

    predictions = model.predict(X_test)

    metrics = evaluate_predictions(
        y_test.to_numpy(),
        predictions,
    )

    print(
        f"\nMAE: "
        f"{metrics['mae_egp']:,.2f} EGP"
    )

    print(
        f"RMSE: "
        f"{metrics['rmse_egp']:,.2f} EGP"
    )

    print(
        f"R²: "
        f"{metrics['r2']:.4f}"
    )

    print(
        f"Median Absolute Error: "
        f"{metrics['median_absolute_error_egp']:,.2f} EGP"
    )

    return metrics


# ---------------------------------------------------------------------
# FINAL PRODUCTION TRAINING
# ---------------------------------------------------------------------

def train_final_model(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    categorical_features: list[str],
    best_config: dict,
) -> CatBoostRegressor:

    print("\n" + "=" * 80)
    print("FINAL PRODUCTION MODEL TRAINING")
    print("=" * 80)

    best_iteration = best_config["best_iteration"]

    # Give the final model the complete dataset.
    # We use the selected iteration count from validation tuning.
    final_iterations = max(best_iteration + 1, 500)

    print(f"Final training rows: {len(X_all):,}")
    print(f"Final iterations: {final_iterations:,}")

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=final_iterations,
        learning_rate=best_config["learning_rate"],
        depth=best_config["depth"],
        l2_leaf_reg=best_config["l2_leaf_reg"],
        random_seed=RANDOM_STATE,
        random_strength=best_config["random_strength"],
        border_count=128,
        verbose=500,
        allow_writing_files=False,
        thread_count=-1,
    )

    start = time.time()

    model.fit(
        X_all,
        y_all,
        cat_features=categorical_features,
    )

    elapsed = time.time() - start

    print(
        f"\nFinal model training completed in "
        f"{elapsed:.2f} seconds."
    )

    return model


# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

def save_model_and_metadata(
    market_type: str,
    market_value: str,
    model: CatBoostRegressor,
    best_config: dict,
    test_metrics: dict,
    feature_count: int,
    numeric_features: list[str],
    categorical_features: list[str],
    total_rows: int,
) -> None:

    model_path = MODELS_DIR / f"{market_type}_model.pkl"
    metadata_path = MODELS_DIR / f"{market_type}_metadata.json"

    joblib.dump(
        model,
        model_path,
        compress=3,
    )

    metadata = {
        "market_type": market_type,
        "market_column": MARKET_TYPE_COLUMN,
        "market_value": market_value,
        "target": "price_egp",
        "prediction_scale": "log1p",

        "trained_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "training_rows_final": total_rows,

        "evaluation_method": {
            "split": "70% train / 15% validation / 15% untouched test",
            "random_state": RANDOM_STATE,
            "test_used_for_model_selection": False,
        },

        "feature_count": feature_count,
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),

        "numeric_features": numeric_features,
        "categorical_features": categorical_features,

        "model_type": "CatBoostRegressor",

        "hyperparameters": {
            "iterations": int(model.get_params()["iterations"]),
            "learning_rate": model.get_params()["learning_rate"],
            "depth": model.get_params()["depth"],
            "l2_leaf_reg": model.get_params()["l2_leaf_reg"],
            "random_strength": model.get_params()["random_strength"],
            "border_count": model.get_params()["border_count"],
            "random_seed": RANDOM_STATE,
        },

        "selection_validation_metrics": best_config[
            "validation_metrics"
        ],

        "untouched_test_metrics": test_metrics,

        "leakage_protection": [
            "price_egp",
            "price_per_sqm",
            "target",
            "log_price",
            "log_target",
        ],
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nModel saved:")
    print(model_path)

    print("\nMetadata saved:")
    print(metadata_path)


# ---------------------------------------------------------------------
# PREDICTION SANITY CHECK
# ---------------------------------------------------------------------

def prediction_sanity_check(
    model: CatBoostRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:

    print("\n" + "=" * 80)
    print("PREDICTION SANITY CHECK")
    print("=" * 80)

    sample = X_test.iloc[[0]]

    prediction_log = model.predict(sample)[0]

    prediction_egp = max(
        0,
        float(np.expm1(prediction_log)),
    )

    actual_egp = float(
        np.expm1(y_test.iloc[0])
    )

    print(
        f"\nActual price:     "
        f"{actual_egp:,.2f} EGP"
    )

    print(
        f"Predicted price:  "
        f"{prediction_egp:,.2f} EGP"
    )

    print(
        f"Absolute error:   "
        f"{abs(actual_egp - prediction_egp):,.2f} EGP"
    )


# ---------------------------------------------------------------------
# MARKET PIPELINE
# ---------------------------------------------------------------------

def process_market(
    df: pd.DataFrame,
    market_type: str,
    market_value: str,
) -> None:

    print("\n\n" + "#" * 80)
    print(f"FINALIZING MARKET: {market_type.upper()}")
    print(f"VALUE: {market_value}")
    print("#" * 80)

    market_df = df[
        df[MARKET_TYPE_COLUMN] == market_value
    ].copy()

    market_df = market_df[
        market_df["price_egp"].notna()
        & (market_df["price_egp"] > 0)
    ].copy()

    print(
        f"\nTotal valid rows: "
        f"{len(market_df):,}"
    )

    # -----------------------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------------------

    X_raw = market_df.drop(
        columns=["price_egp"],
        errors="ignore",
    )

    y = np.log1p(
        market_df["price_egp"].astype(float)
    )

    X = build_features(X_raw)

    numeric_features, categorical_features = (
        get_feature_lists()
    )

    selected_features = (
        numeric_features +
        categorical_features
    )

    selected_features = [
        feature
        for feature in selected_features
        if feature in X.columns
    ]

    X = X[selected_features].copy()

    X = prepare_catboost_features(
        X,
        categorical_features,
    )

    categorical_features = [
        feature
        for feature in categorical_features
        if feature in X.columns
    ]

    # -----------------------------------------------------------------
    # Train / validation / test
    # -----------------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(X, pd.Series(y, index=X.index))

    print(
        f"Train rows:       {len(X_train):,}"
    )

    print(
        f"Validation rows:  {len(X_validation):,}"
    )

    print(
        f"Test rows:        {len(X_test):,}"
    )

    # -----------------------------------------------------------------
    # Tuning
    # -----------------------------------------------------------------

    best_config, selected_model = tune_catboost(
        X_train,
        y_train,
        X_validation,
        y_validation,
        categorical_features,
    )

    # -----------------------------------------------------------------
    # Untouched test evaluation
    # -----------------------------------------------------------------

    test_metrics = evaluate_on_untouched_test(
        selected_model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------------------
    # Final production model on ALL rows
    # -----------------------------------------------------------------

    final_model = train_final_model(
        X,
        pd.Series(y, index=X.index),
        categorical_features,
        best_config,
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    save_model_and_metadata(
        market_type=market_type,
        market_value=market_value,
        model=final_model,
        best_config=best_config,
        test_metrics=test_metrics,
        feature_count=len(selected_features),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        total_rows=len(X),
    )

    # -----------------------------------------------------------------
    # Sanity check
    # -----------------------------------------------------------------

    prediction_sanity_check(
        final_model,
        X_test,
        y_test,
    )

    print(
        f"\nFINAL {market_type.upper()} MODEL READY."
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 80)
    print("REAL ESTATE MARKET INTELLIGENCE PLATFORM")
    print("PHASE 7 - FINAL ML MODELIZATION")
    print("=" * 80)

    print(
        "\n[1/4] Loading validated data from SQL Server..."
    )

    df = load_training_data()

    print(
        f"Loaded {len(df):,} records."
    )

    print("\nMarket distribution:")
    print(
        df[MARKET_TYPE_COLUMN]
        .value_counts()
        .to_string()
    )

    print(
        "\n[2/4] Finalizing SALE CatBoost model..."
    )

    process_market(
        df,
        "sale",
        MARKET_TYPE_MAPPING["sale"],
    )

    print(
        "\n[3/4] Finalizing RENT CatBoost model..."
    )

    process_market(
        df,
        "rent",
        MARKET_TYPE_MAPPING["rent"],
    )

    print(
        "\n[4/4] Phase 7 finalization complete."
    )

    print("\n" + "=" * 80)
    print("PHASE 7 - ML COMPLETE")
    print("=" * 80)

    print(
        "\nProduction models:"
    )

    print(
        f"  {MODELS_DIR / 'sale_model.pkl'}"
    )

    print(
        f"  {MODELS_DIR / 'rent_model.pkl'}"
    )

    print(
        "\nNext phase:"
    )

    print(
        "  PHASE 8 - MODEL SERVING"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()