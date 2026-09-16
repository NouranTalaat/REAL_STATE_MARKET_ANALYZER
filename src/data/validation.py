from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "listing_id",
    "category",
    "listing_type",
    "property_type",
    "price_egp",
    "city",
    "area_value",
    "listed_date",
}

NUMERIC_COLUMNS = [
    "price_egp",
    "lat",
    "lon",
    "area_value",
    "images_count",
    "rera",
]

BOOLEAN_COLUMNS = [
    "is_premium",
    "is_verified",
    "is_featured",
    "is_new_construction",
    "is_direct_from_dev",
    "is_exclusive",
    "has_view_360",
    "agent_is_super",
]

SENSITIVE_COLUMNS = [
    "agent_name",
    "agent_email",
    "broker_name",
    "broker_email",
    "broker_phone",
    "contact_phone",
    "contact_whatsapp",
    "contact_email",
]


def validate_required_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return required columns that are missing."""
    return sorted(REQUIRED_COLUMNS - set(dataframe.columns))


def validate_numeric_ranges(dataframe: pd.DataFrame) -> dict[str, int]:
    """Count invalid values in important numeric fields."""

    checks = {}

    if "price_egp" in dataframe.columns:
        checks["invalid_price"] = int(
            (dataframe["price_egp"].notna() & (dataframe["price_egp"] <= 0)).sum()
        )

    if "area_value" in dataframe.columns:
        checks["invalid_area"] = int(
            (dataframe["area_value"].notna() & (dataframe["area_value"] <= 0)).sum()
        )

    if "lat" in dataframe.columns:
        checks["invalid_latitude"] = int(
            (
                dataframe["lat"].notna()
                & ~dataframe["lat"].between(-90, 90)
            ).sum()
        )

    if "lon" in dataframe.columns:
        checks["invalid_longitude"] = int(
            (
                dataframe["lon"].notna()
                & ~dataframe["lon"].between(-180, 180)
            ).sum()
        )

    if "images_count" in dataframe.columns:
        checks["invalid_images_count"] = int(
            (
                dataframe["images_count"].notna()
                & (dataframe["images_count"] < 0)
            ).sum()
        )

    return checks


def validate_duplicates(dataframe: pd.DataFrame) -> dict[str, int]:
    """Return duplicate statistics."""

    result = {
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }

    if "listing_id" in dataframe.columns:
        result["duplicate_listing_ids"] = int(
            dataframe["listing_id"].duplicated(keep=False).sum()
        )

    return result


def validate_missing_values(dataframe: pd.DataFrame) -> dict[str, int]:
    """Return missing-value counts for all columns."""

    missing = dataframe.isna().sum()

    return {
        column: int(count)
        for column, count in missing.items()
        if count > 0
    }


def validate_sensitive_columns(dataframe: pd.DataFrame) -> list[str]:
    """Identify sensitive columns that should not enter the processed dataset."""

    return [
        column
        for column in SENSITIVE_COLUMNS
        if column in dataframe.columns
    ]


def build_quality_report(
    raw_dataframe: pd.DataFrame,
    processed_dataframe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a structured data-quality report."""

    required_columns_missing = validate_required_columns(raw_dataframe)

    report: dict[str, Any] = {
        "dataset": {
            "raw_rows": int(len(raw_dataframe)),
            "raw_columns": int(len(raw_dataframe.columns)),
        },
        "required_columns": {
            "missing": required_columns_missing,
            "status": "PASS" if not required_columns_missing else "FAIL",
        },
        "duplicates": validate_duplicates(raw_dataframe),
        "missing_values": validate_missing_values(raw_dataframe),
        "numeric_validation": validate_numeric_ranges(raw_dataframe),
        "sensitive_columns_detected": validate_sensitive_columns(raw_dataframe),
    }

    if processed_dataframe is not None:
        report["processed_dataset"] = {
            "rows": int(len(processed_dataframe)),
            "columns": int(len(processed_dataframe.columns)),
        }

        report["processed_duplicates"] = {
            "duplicate_rows": int(processed_dataframe.duplicated().sum()),
        }

    return report