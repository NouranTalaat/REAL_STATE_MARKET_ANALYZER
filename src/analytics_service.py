import math

import pandas as pd

from src.sql_analytics import (
    load_market_summary,
    load_robust_metrics,
    load_location_intelligence,
    load_city_market_profile,
    load_opportunity_signals,
    load_data_quality_monitor,
)


def clean_for_json(value):
    """
    Convert non-JSON-compliant numeric values to None.

    NaN and infinite values are converted to None so that
    FastAPI can safely serialize the response as JSON.
    """

    if pd.isna(value):
        return None

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    """
    Convert a DataFrame into JSON-safe records.
    """

    records = df.to_dict(orient="records")

    return [
        {
            key: clean_for_json(value)
            for key, value in record.items()
        }
        for record in records
    ]


def get_market_summary() -> pd.DataFrame:
    return load_market_summary()


def get_robust_metrics() -> pd.DataFrame:
    return load_robust_metrics()


def get_location_intelligence() -> pd.DataFrame:
    return load_location_intelligence()


def get_city_market_profile() -> pd.DataFrame:
    return load_city_market_profile()


def get_opportunity_signals() -> pd.DataFrame:
    return load_opportunity_signals()


def get_data_quality_status() -> pd.DataFrame:
    return load_data_quality_monitor()


def get_market_overview() -> dict:
    """
    Build a unified JSON-safe market intelligence response.
    """

    market_summary = get_market_summary()
    robust_metrics = get_robust_metrics()
    location_intelligence = get_location_intelligence()
    city_market_profile = get_city_market_profile()
    opportunity_signals = get_opportunity_signals()
    data_quality = get_data_quality_status()

    return {
        "market_summary": dataframe_to_records(market_summary),
        "robust_metrics": dataframe_to_records(robust_metrics),
        "location_intelligence": dataframe_to_records(
            location_intelligence
        ),
        "city_market_profile": dataframe_to_records(
            city_market_profile
        ),
        "opportunity_signals": dataframe_to_records(
            opportunity_signals
        ),
        "data_quality": dataframe_to_records(data_quality),
    }