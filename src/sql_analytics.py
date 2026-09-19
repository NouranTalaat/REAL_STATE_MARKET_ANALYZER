import pandas as pd

from src.database import engine


def load_view(view_name: str) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM {view_name};
    """

    return pd.read_sql(query, engine)


def load_market_overview() -> pd.DataFrame:
    return load_view("analytics.vw_market_overview")


def load_market_summary() -> pd.DataFrame:
    return load_view("analytics.vw_market_summary")


def load_robust_metrics() -> pd.DataFrame:
    return load_view("analytics.vw_market_robust_metrics")


def load_location_intelligence() -> pd.DataFrame:
    return load_view("analytics.vw_location_intelligence")


def load_city_market_profile() -> pd.DataFrame:
    return load_view("analytics.vw_city_market_profile")


def load_opportunity_signals() -> pd.DataFrame:
    return load_view("analytics.vw_market_opportunity_signals")


def load_data_quality_monitor() -> pd.DataFrame:
    return load_view("analytics.vw_data_quality_monitor")