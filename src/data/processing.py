from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.validation import (
    BOOLEAN_COLUMNS,
    SENSITIVE_COLUMNS,
    build_quality_report,
    validate_numeric_ranges,
)
from src.utils.config import PROCESSED_DATA_DIR, REPORTS_DIR
from src.utils.logging import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Column handling
# ---------------------------------------------------------------------------

def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case."""

    dataframe = dataframe.copy()

    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    return dataframe


# ---------------------------------------------------------------------------
# General cleaning
# ---------------------------------------------------------------------------

def clean_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean whitespace and convert empty strings to missing values."""

    dataframe = dataframe.copy()

    text_columns = dataframe.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        dataframe[column] = dataframe[column].astype("string").str.strip()

        dataframe[column] = dataframe[column].replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "null": pd.NA,
                "NULL": pd.NA,
            }
        )

    return dataframe


def normalize_categories(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize important categorical columns."""

    dataframe = dataframe.copy()

    columns_to_normalize = [
        "category",
        "listing_type",
        "property_type",
        "offering_type",
        "completion_status",
        "price_period",
        "price_currency",
        "city",
        "town",
        "district",
        "subdistrict",
        "area_unit",
        "furnished",
        "listing_level",
        "payment_method",
    ]

    for column in columns_to_normalize:
        if column not in dataframe.columns:
            continue

        dataframe[column] = (
            dataframe[column]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    return dataframe


# ---------------------------------------------------------------------------
# Numeric cleaning
# ---------------------------------------------------------------------------

def clean_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric fields safely and remove impossible values."""

    dataframe = dataframe.copy()

    numeric_columns = [
        "internal_id",
        "price_egp",
        "lat",
        "lon",
        "area_value",
        "images_count",
        "rera",
        "agent_id",
        "broker_id",
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    # Impossible prices
    if "price_egp" in dataframe.columns:
        dataframe.loc[
            dataframe["price_egp"] <= 0,
            "price_egp",
        ] = np.nan

    # Impossible areas
    if "area_value" in dataframe.columns:
        dataframe.loc[
            dataframe["area_value"] <= 0,
            "area_value",
        ] = np.nan

    # Coordinates
    if "lat" in dataframe.columns:
        dataframe.loc[
            ~dataframe["lat"].between(-90, 90),
            "lat",
        ] = np.nan

    if "lon" in dataframe.columns:
        dataframe.loc[
            ~dataframe["lon"].between(-180, 180),
            "lon",
        ] = np.nan

    # Negative image counts
    if "images_count" in dataframe.columns:
        dataframe.loc[
            dataframe["images_count"] < 0,
            "images_count",
        ] = np.nan

    return dataframe


# ---------------------------------------------------------------------------
# Boolean cleaning
# ---------------------------------------------------------------------------

def normalize_boolean_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize boolean columns to pandas nullable boolean."""

    dataframe = dataframe.copy()

    true_values = {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }

    false_values = {
        "false",
        "0",
        "no",
        "n",
        "f",
    }

    for column in BOOLEAN_COLUMNS:
        if column not in dataframe.columns:
            continue

        def convert_boolean(value):
            if pd.isna(value):
                return pd.NA

            if isinstance(value, bool):
                return value

            normalized = str(value).strip().lower()

            if normalized in true_values:
                return True

            if normalized in false_values:
                return False

            return pd.NA

        dataframe[column] = dataframe[column].map(convert_boolean).astype(
            "boolean"
        )

    return dataframe


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def parse_dates(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert date columns to datetime."""

    dataframe = dataframe.copy()

    date_columns = [
        "listed_date",
        "scraped_at",
    ]

    for column in date_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(
                dataframe[column],
                errors="coerce",
                utc=True,
            )

    return dataframe


# ---------------------------------------------------------------------------
# Property-specific normalization
# ---------------------------------------------------------------------------

def normalize_bedrooms_and_bathrooms(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Extract numeric bedroom and bathroom counts."""

    dataframe = dataframe.copy()

    for column in ["bedrooms", "bathrooms"]:
        if column not in dataframe.columns:
            continue

        dataframe[column] = (
            dataframe[column]
            .astype("string")
            .str.extract(r"(\d+(?:\.\d+)?)", expand=False)
        )

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    return dataframe


def normalize_area_unit(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardize area units."""

    dataframe = dataframe.copy()

    if "area_unit" not in dataframe.columns:
        return dataframe

    dataframe["area_unit"] = (
        dataframe["area_unit"]
        .astype("string")
        .str.lower()
        .str.strip()
    )

    dataframe["area_unit"] = dataframe["area_unit"].replace(
        {
            "sqm": "sqm",
            "m2": "sqm",
            "m²": "sqm",
            "square meter": "sqm",
            "square meters": "sqm",
        }
    )

    return dataframe


def normalize_furnished(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize furnished status."""

    dataframe = dataframe.copy()

    if "furnished" not in dataframe.columns:
        return dataframe

    dataframe["furnished"] = (
        dataframe["furnished"]
        .astype("string")
        .str.lower()
        .str.strip()
    )

    dataframe["furnished"] = dataframe["furnished"].replace(
        {
            "yes": "furnished",
            "true": "furnished",
            "1": "furnished",
            "no": "not_furnished",
            "false": "not_furnished",
            "0": "not_furnished",
            "partly": "partly_furnished",
            "partial": "partly_furnished",
        }
    )

    return dataframe


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def create_market_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create reusable market intelligence features."""

    dataframe = dataframe.copy()

    # Price per square meter
    if {"price_egp", "area_value"}.issubset(dataframe.columns):
        dataframe["price_per_sqm"] = np.where(
            dataframe["area_value"] > 0,
            dataframe["price_egp"] / dataframe["area_value"],
            np.nan,
        )

    # Listing age
    if "listed_date" in dataframe.columns:
        current_reference_date = pd.Timestamp.now(tz="UTC")

        dataframe["listing_age_days"] = (
            current_reference_date - dataframe["listed_date"]
        ).dt.days

        dataframe.loc[
            dataframe["listing_age_days"] < 0,
            "listing_age_days",
        ] = np.nan

    # Location hierarchy
    location_columns = [
        column
        for column in [
            "city",
            "town",
            "district",
            "subdistrict",
        ]
        if column in dataframe.columns
    ]

    if location_columns:
        dataframe["location_key"] = (
            dataframe[location_columns]
            .fillna("")
            .astype("string")
            .agg(" | ".join, axis=1)
            .str.strip(" |")
        )

        dataframe["location_key"] = dataframe["location_key"].replace(
            "",
            pd.NA,
        )

    # Has media
    if "images_count" in dataframe.columns:
        dataframe["has_images"] = (
            dataframe["images_count"].fillna(0) > 0
        )

    if "video_url" in dataframe.columns:
        dataframe["has_video"] = (
            dataframe["video_url"].notna()
            & dataframe["video_url"].astype("string").str.strip().ne("")
        )

    return dataframe


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

def remove_sensitive_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove personal contact information from processed data."""

    dataframe = dataframe.copy()

    columns_to_remove = [
        column
        for column in SENSITIVE_COLUMNS
        if column in dataframe.columns
    ]

    if columns_to_remove:
        logger.info(
            "Removing sensitive columns: %s",
            ", ".join(columns_to_remove),
        )

        dataframe = dataframe.drop(columns=columns_to_remove)

    return dataframe


# ---------------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------------

def remove_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records using listing_id where possible."""

    dataframe = dataframe.copy()

    initial_rows = len(dataframe)

    if "listing_id" in dataframe.columns:
        dataframe = dataframe.drop_duplicates(
            subset=["listing_id"],
            keep="last",
        )
    else:
        dataframe = dataframe.drop_duplicates()

    removed = initial_rows - len(dataframe)

    logger.info("Duplicate records removed: %s", removed)

    return dataframe


# ---------------------------------------------------------------------------
# Final validation
# ---------------------------------------------------------------------------

def apply_final_quality_rules(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply final sanity checks without aggressively deleting data."""

    dataframe = dataframe.copy()

    if "price_per_sqm" in dataframe.columns:
        dataframe.loc[
            dataframe["price_per_sqm"] <= 0,
            "price_per_sqm",
        ] = np.nan

    if "listing_age_days" in dataframe.columns:
        dataframe.loc[
            dataframe["listing_age_days"] > 3650,
            "listing_age_days",
        ] = np.nan

    return dataframe


# ---------------------------------------------------------------------------
# Main ETL
# ---------------------------------------------------------------------------

def transform_property_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Run the complete transformation pipeline."""

    logger.info("Starting data transformation.")

    dataframe = standardize_column_names(dataframe)
    dataframe = clean_text_columns(dataframe)
    dataframe = normalize_categories(dataframe)
    dataframe = clean_numeric_columns(dataframe)
    dataframe = normalize_boolean_columns(dataframe)
    dataframe = parse_dates(dataframe)
    dataframe = normalize_bedrooms_and_bathrooms(dataframe)
    dataframe = normalize_area_unit(dataframe)
    dataframe = normalize_furnished(dataframe)
    dataframe = remove_duplicates(dataframe)
    dataframe = create_market_features(dataframe)
    dataframe = apply_final_quality_rules(dataframe)
    dataframe = remove_sensitive_columns(dataframe)

    logger.info(
        "Transformation completed: %s rows, %s columns.",
        dataframe.shape[0],
        dataframe.shape[1],
    )

    return dataframe


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_processed_data(
    dataframe: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Save processed dataset."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "propertyfinder_clean.csv"

    dataframe.to_csv(
        output_path,
        index=False,
    )

    logger.info("Processed dataset saved to: %s", output_path)

    return output_path


def save_quality_report(
    report: dict,
    output_path: Path | None = None,
) -> Path:
    """Save data-quality report as JSON."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = REPORTS_DIR / "data_quality_report.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    logger.info("Data quality report saved to: %s", output_path)

    return output_path