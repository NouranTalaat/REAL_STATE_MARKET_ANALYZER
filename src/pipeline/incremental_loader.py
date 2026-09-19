from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from src.database import engine
from src.utils.logging import get_logger


logger = get_logger(__name__)


TARGET_SCHEMA = "analytics"
TARGET_TABLE = "property_listings"


def get_existing_listing_ids() -> set:
    query = text(
        f"""
        SELECT listing_id
        FROM {TARGET_SCHEMA}.{TARGET_TABLE};
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).fetchall()

    return {
        row[0]
        for row in rows
        if row[0] is not None
    }


def prepare_dataframe_for_sql(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    integer_columns = [
        "listing_id",
        "bedrooms",
        "bathrooms",
        "images_count",
        "rera",
        "agent_id",
        "broker_id",
        "listing_age_days",
    ]

    numeric_columns = [
        "price_egp",
        "lat",
        "lon",
        "area_value",
        "price_per_sqm",
    ]

    boolean_columns = [
        "is_premium",
        "is_verified",
        "is_featured",
        "is_new_construction",
        "is_direct_from_dev",
        "is_exclusive",
        "has_view_360",
        "agent_is_super",
        "has_images",
        "has_video",
    ]

    datetime_columns = [
        "listed_date",
        "scraped_at",
    ]

    for column in integer_columns:
        if column in dataframe.columns:
            dataframe[column] = (
                pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                ).astype("Int64")
            )

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    for column in boolean_columns:
        if column in dataframe.columns:
            dataframe[column] = (
                dataframe[column]
                .astype("boolean")
            )

    for column in datetime_columns:
        if column in dataframe.columns:
            dataframe[column] = (
                pd.to_datetime(
                    dataframe[column],
                    errors="coerce",
                    utc=True,
                )
                .dt
                .tz_localize(None)
            )

    dataframe = (
        dataframe.astype(object)
        .where(pd.notna(dataframe), None)
    )

    return dataframe


def load_new_listings(
    dataframe: pd.DataFrame,
) -> dict:

    if dataframe.empty:
        return {
            "source_rows": 0,
            "existing_rows": 0,
            "new_rows": 0,
            "inserted_rows": 0,
        }

    if "listing_id" not in dataframe.columns:
        raise ValueError(
            "listing_id is required for incremental loading."
        )

    dataframe = dataframe.copy()

    null_listing_ids = int(
        dataframe["listing_id"].isna().sum()
    )

    if null_listing_ids > 0:
        raise ValueError(
            "Incremental loading received "
            f"{null_listing_ids} records without "
            "listing_id. Clean them before loading."
        )

    existing_ids = get_existing_listing_ids()

    source_ids = set(
        dataframe["listing_id"].tolist()
    )

    new_ids = source_ids - existing_ids

    new_dataframe = dataframe[
        dataframe["listing_id"].isin(new_ids)
    ].copy()

    source_count = len(dataframe)

    existing_count = len(
        source_ids & existing_ids
    )

    new_count = len(new_dataframe)

    logger.info(
        "Incremental load detected %s existing listings "
        "and %s new listings.",
        existing_count,
        new_count,
    )

    if new_dataframe.empty:
        return {
            "source_rows": source_count,
            "existing_rows": existing_count,
            "new_rows": 0,
            "inserted_rows": 0,
        }

    new_dataframe = prepare_dataframe_for_sql(
        new_dataframe
    )

    loaded_at = (
        datetime.now(timezone.utc)
        .replace(tzinfo=None)
    )

    new_dataframe["loaded_at"] = loaded_at

    columns = list(
        new_dataframe.columns
    )

    with engine.begin() as connection:

        new_dataframe.to_sql(
            name=TARGET_TABLE,
            con=connection,
            schema=TARGET_SCHEMA,
            if_exists="append",
            index=False,
            chunksize=5000,
            method=None,
        )

    logger.info(
        "Incremental load completed: "
        "%s new listings inserted.",
        new_count,
    )

    return {
        "source_rows": source_count,
        "existing_rows": existing_count,
        "new_rows": new_count,
        "inserted_rows": new_count,
        "inserted_columns": columns,
    }


def get_target_row_count() -> int:

    query = text(
        f"""
        SELECT COUNT(*) AS total_rows
        FROM {TARGET_SCHEMA}.{TARGET_TABLE};
        """
    )

    with engine.connect() as connection:
        return int(
            connection.execute(query).scalar()
        )


def get_latest_loaded_at():

    query = text(
        f"""
        SELECT MAX(loaded_at) AS latest_loaded_at
        FROM {TARGET_SCHEMA}.{TARGET_TABLE};
        """
    )

    with engine.connect() as connection:
        return connection.execute(query).scalar()