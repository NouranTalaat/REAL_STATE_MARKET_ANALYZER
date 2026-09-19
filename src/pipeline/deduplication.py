from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger


logger = get_logger(__name__)


def deduplicate_listings(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    if dataframe.empty:
        return dataframe.copy()

    dataframe = dataframe.copy()

    initial_count = len(dataframe)

    if "listing_id" not in dataframe.columns:
        raise ValueError(
            "listing_id column is required for "
            "production deduplication."
        )

    # ---------------------------------------------------------
    # Reject records without a valid business key
    # ---------------------------------------------------------

    null_listing_ids = int(
        dataframe["listing_id"].isna().sum()
    )

    if null_listing_ids > 0:
        logger.warning(
            "%s records rejected because listing_id "
            "is missing.",
            null_listing_ids,
        )

        dataframe = dataframe[
            dataframe["listing_id"].notna()
        ].copy()

    # ---------------------------------------------------------
    # Deduplicate using listing_id
    # ---------------------------------------------------------

    dataframe = dataframe.drop_duplicates(
        subset=["listing_id"],
        keep="last",
    )

    removed_count = initial_count - len(dataframe)

    logger.info(
        "Deduplication completed: %s records removed/rejected.",
        removed_count,
    )

    return dataframe


def get_duplicate_statistics(
    dataframe: pd.DataFrame,
) -> dict:

    if dataframe.empty:
        return {
            "total_rows": 0,
            "null_listing_ids": 0,
            "duplicate_rows": 0,
            "duplicate_listing_ids": 0,
            "unique_listing_ids": 0,
        }

    null_listing_ids = 0
    duplicate_listing_ids = 0
    unique_listing_ids = 0

    if "listing_id" in dataframe.columns:

        null_listing_ids = int(
            dataframe["listing_id"].isna().sum()
        )

        non_null_ids = dataframe[
            "listing_id"
        ].dropna()

        duplicate_listing_ids = int(
            non_null_ids.duplicated(
                keep=False
            ).sum()
        )

        unique_listing_ids = int(
            non_null_ids.nunique()
        )

    duplicate_rows = int(
        dataframe.duplicated().sum()
    )

    return {
        "total_rows": int(len(dataframe)),
        "null_listing_ids": null_listing_ids,
        "duplicate_rows": duplicate_rows,
        "duplicate_listing_ids":
            duplicate_listing_ids,
        "unique_listing_ids":
            unique_listing_ids,
    }