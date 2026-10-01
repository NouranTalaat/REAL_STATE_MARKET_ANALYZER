from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import engine
from src.ml.features import (
    build_features,
    get_feature_lists,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data() -> pd.DataFrame:
    query = """
        SELECT *
        FROM analytics.property_listings
        WHERE price_egp IS NOT NULL
          AND price_egp > 0
          AND offering_type IS NOT NULL;
    """

    return pd.read_sql(query, engine)


def prepare_market_data(
    df: pd.DataFrame,
    market_value: str,
) -> tuple[pd.DataFrame, pd.Series]:

    market_df = df[
        df["offering_type"] == market_value
    ].copy()

    target = pd.to_numeric(
        market_df["price_egp"],
        errors="coerce",
    )

    valid = target.notna() & (target > 0)

    market_df = market_df.loc[valid].copy()
    target = target.loc[valid]

    features = build_features(market_df)

    numeric_features, categorical_features = (
        get_feature_lists()
    )

    feature_columns = (
        numeric_features
        + categorical_features
    )

    X = features[feature_columns].copy()

    y = np.log1p(
        target.astype(float)
    )

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def prepare_catboost_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: list[str],
):
    X_train = X_train.copy()
    X_test = X_test.copy()

    for column in categorical_features:
        X_train[column] = (
            X_train[column]
            .fillna("Unknown")
            .astype(str)
        )

        X_test[column] = (
            X_test[column]
            .fillna("Unknown")
            .astype(str)
        )

    return X_train, X_test


def prepare_histgradient_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: list[str],
):
    X_train = X_train.copy()
    X_test = X_test.copy()

    for column in categorical_features:
        combined = pd.concat(
            [
                X_train[column],
                X_test[column],
            ],
            axis=0,
        ).fillna("Unknown").astype(str)

        categories = pd.Index(
            combined.unique()
        )

        X_train[column] = pd.Categorical(
            X_train[column]
            .fillna("Unknown")
            .astype(str),
            categories=categories,
        )

        X_test[column] = pd.Categorical(
            X_test[column]
            .fillna("Unknown")
            .astype(str),
            categories=categories,
        )

    return X_train, X_test


def evaluate(
    y_true_log: pd.Series,
    y_pred_log: np.ndarray,
) -> dict:

    y_true = np.maximum(
        np.expm1(y_true_log),
        0,
    )

    y_pred = np.maximum(
        np.expm1(y_pred_log),
        0,
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


def run_catboost(
    X_train,
    X_test,
    y_train,
    y_test,
    categorical_features,
):
    X_train, X_test = prepare_catboost_data(
        X_train,
        X_test,
        categorical_features,
    )

    cat_indices = [
        X_train.columns.get_loc(column)
        for column in categorical_features
    ]

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2500,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=5,
        random_seed=RANDOM_STATE,
        random_strength=1,
        border_count=128,
        od_type="Iter",
        od_wait=100,
        verbose=200,
        allow_writing_files=False,
        thread_count=-1,
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        cat_features=cat_indices,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    elapsed = time.perf_counter() - start

    predictions = model.predict(X_test)

    return model, evaluate(
        y_test,
        predictions,
    ), elapsed


def run_histgradient(
    X_train,
    X_test,
    y_train,
    y_test,
    categorical_features,
):
    X_train, X_test = prepare_histgradient_data(
        X_train,
        X_test,
        categorical_features,
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=1000,
        max_leaf_nodes=63,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=75,
        random_state=RANDOM_STATE,
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    elapsed = time.perf_counter() - start

    predictions = model.predict(X_test)

    return model, evaluate(
        y_test,
        predictions,
    ), elapsed


def run_extratrees(
    X_train,
    X_test,
    y_train,
    y_test,
    numeric_features,
    categorical_features,
):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
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
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "encoder",
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

    model = ExtraTreesRegressor(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    from sklearn.pipeline import make_pipeline

    pipeline = make_pipeline(
        preprocessor,
        model,
    )

    start = time.perf_counter()

    pipeline.fit(
        X_train,
        y_train,
    )

    elapsed = time.perf_counter() - start

    predictions = pipeline.predict(
        X_test
    )

    return pipeline, evaluate(
        y_test,
        predictions,
    ), elapsed


def run_randomforest(
    X_train,
    X_test,
    y_train,
    y_test,
    numeric_features,
    categorical_features,
):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
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
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "encoder",
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

    model = RandomForestRegressor(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    from sklearn.pipeline import make_pipeline

    pipeline = make_pipeline(
        preprocessor,
        model,
    )

    start = time.perf_counter()

    pipeline.fit(
        X_train,
        y_train,
    )

    elapsed = time.perf_counter() - start

    predictions = pipeline.predict(
        X_test
    )

    return pipeline, evaluate(
        y_test,
        predictions,
    ), elapsed


def benchmark_market(
    df: pd.DataFrame,
    market_type: str,
    market_value: str,
):
    print("\n" + "=" * 80)
    print(f"MARKET: {market_type.upper()}")
    print(f"VALUE: {market_value}")
    print("=" * 80)

    X, y = prepare_market_data(
        df,
        market_value,
    )

    numeric_features, categorical_features = (
        get_feature_lists()
    )

    X_train, X_test, y_train, y_test = (
        split_data(X, y)
    )

    print(
        f"Total rows: {len(X):,}"
    )

    print(
        f"Train rows: {len(X_train):,}"
    )

    print(
        f"Test rows: {len(X_test):,}"
    )

    results = []

    # -------------------------------------------------------------
    # CatBoost
    # -------------------------------------------------------------

    print("\n[1/4] CatBoost")

    cat_model, cat_metrics, cat_time = (
        run_catboost(
            X_train,
            X_test,
            y_train,
            y_test,
            categorical_features,
        )
    )

    results.append(
        {
            "model": "CatBoost",
            **cat_metrics,
            "training_seconds": cat_time,
        }
    )

    # -------------------------------------------------------------
    # HistGradientBoosting
    # -------------------------------------------------------------

    print("\n[2/4] HistGradientBoosting - SKIPPED")
    print(
        "Skipped because scikit-learn categorical cardinality "
        "limit is exceeded by the location features."
    )

    # -------------------------------------------------------------
    # ExtraTrees
    # -------------------------------------------------------------

    print("\n[3/4] ExtraTrees")

    extra_model, extra_metrics, extra_time = (
        run_extratrees(
            X_train,
            X_test,
            y_train,
            y_test,
            numeric_features,
            categorical_features,
        )
    )

    results.append(
        {
            "model": "ExtraTrees",
            **extra_metrics,
            "training_seconds": extra_time,
        }
    )

    # -------------------------------------------------------------
    # RandomForest
    # -------------------------------------------------------------

    print("\n[4/4] RandomForest")

    rf_model, rf_metrics, rf_time = (
        run_randomforest(
            X_train,
            X_test,
            y_train,
            y_test,
            numeric_features,
            categorical_features,
        )
    )

    results.append(
        {
            "model": "RandomForest",
            **rf_metrics,
            "training_seconds": rf_time,
        }
    )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=[
            "r2",
            "mae_egp",
        ],
        ascending=[
            False,
            True,
        ],
    )

    print("\n" + "-" * 80)
    print("MODEL BENCHMARK RESULTS")
    print("-" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:,.4f}"
            ),
        )
    )

    best_model_name = (
        results_df.iloc[0]["model"]
    )

    print(
        f"\nBest model for {market_type}: "
        f"{best_model_name}"
    )

    return {
        "market_type": market_type,
        "market_value": market_value,
        "results": results_df.to_dict(
            orient="records"
        ),
        "best_model": best_model_name,
    }


def main():
    print("=" * 80)
    print(
        "REAL ESTATE MARKET INTELLIGENCE PLATFORM"
    )
    print(
        "PHASE 7 - ADVANCED ML BENCHMARK"
    )
    print("=" * 80)

    print(
        "\nLoading data from SQL Server..."
    )

    df = load_data()

    print(
        f"Loaded {len(df):,} records."
    )

    print("\nStarting benchmark...")

    sale_results = benchmark_market(
        df,
        "sale",
        "Residential for Sale",
    )

    rent_results = benchmark_market(
        df,
        "rent",
        "Residential for Rent",
    )

    output = {
        "sale": sale_results,
        "rent": rent_results,
    }

    output_path = (
        PROJECT_ROOT
        / "models"
        / "benchmark_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        "\nBenchmark results saved to:"
    )

    print(output_path)

    print(
        "\n" + "=" * 80
    )

    print(
        "ADVANCED ML BENCHMARK FINISHED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()