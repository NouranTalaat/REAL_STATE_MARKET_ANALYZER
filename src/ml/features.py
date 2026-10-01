from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Leakage protection
# ---------------------------------------------------------------------

LEAKAGE_COLUMNS = {
    "price_egp",
    "price_per_sqm",
    "target",
    "log_price",
    "log_target",
}


# ---------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------

NUMERIC_FEATURES = [
    "lat",
    "lon",
    "bedrooms",
    "bathrooms",
    "area_value",
    "is_premium",
    "is_featured",
    "images_count",
    "has_view_360",
    "listing_year",
    "listing_month",
    "listing_day",
    "listing_dayofweek",
    "log_area",
    "total_rooms",
    "bed_bath_ratio",
    "area_per_bedroom",
    "area_per_bathroom",
    "bathrooms_per_bedroom",
    "has_video",
    "has_360_view",
    "has_amenities",
    "has_district",
    "has_subdistrict",
    "location_completeness",
    "listing_quality_score",
    "agent_language_count",
    "bedrooms_per_100sqm",
    "is_large_property",
    "is_high_bedroom",
    "is_new_construction",
    "is_direct_from_dev",
    "is_exclusive",
    "direct_developer",
    "super_agent",
]


CATEGORICAL_FEATURES = [
    "property_type",
    "city",
    "town",
    "district",
    "subdistrict",
    "furnished",
    "listing_level",
    "payment_method",
    "area_size_category",
    "bedroom_category",
    "bathroom_category",
    "city_town",
    "town_district",
    "district_subdistrict",
    "property_location",
    "area_category",
    "city_property_type",
    "district_property_type",
    "furnished_property_type",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    result = numerator / denominator.replace(0, np.nan)

    return result.replace([np.inf, -np.inf], np.nan)


def _to_binary(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }
    ).astype(int)


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML-safe engineered features.

    Important:
    - Target columns are never used to create model features.
    - price_egp and price_per_sqm are explicitly removed.
    """

    data = df.copy()

    # -------------------------------------------------------------
    # Date features
    # -------------------------------------------------------------

    if "listed_date" in data.columns:
        listed_date = pd.to_datetime(
            data["listed_date"],
            errors="coerce",
        )

    else:
        listed_date = pd.Series(
            pd.NaT,
            index=data.index,
            dtype="datetime64[ns]",
        )

    # If listed_date is unavailable, reconstruct it from
    # listing_age_days when possible.
    if "listing_age_days" in data.columns:
        listing_age_days = pd.to_numeric(
            data["listing_age_days"],
            errors="coerce",
        )

        fallback_date = (
            pd.Timestamp.today().normalize()
            - pd.to_timedelta(
                listing_age_days,
                unit="D",
            )
        )

        listed_date = listed_date.fillna(
            fallback_date
        )

    # Final fallback: use the current date rather than
    # allowing numeric date features to become NaN.
    listed_date = listed_date.fillna(
        pd.Timestamp.today().normalize()
    )

    data["listing_year"] = listed_date.dt.year
    data["listing_month"] = listed_date.dt.month
    data["listing_day"] = listed_date.dt.day
    data["listing_dayofweek"] = listed_date.dt.dayofweek

    # -------------------------------------------------------------
    # Area features
    # -------------------------------------------------------------

    if "area_value" in data.columns:
        area = pd.to_numeric(
            data["area_value"],
            errors="coerce",
        )

        data["log_area"] = np.log1p(
            area.clip(lower=0)
        )

    else:
        data["log_area"] = np.nan

    # -------------------------------------------------------------
    # Room features
    # -------------------------------------------------------------

    bedrooms = pd.to_numeric(
        data.get("bedrooms", pd.Series(index=data.index)),
        errors="coerce",
    )

    bathrooms = pd.to_numeric(
        data.get("bathrooms", pd.Series(index=data.index)),
        errors="coerce",
    )

    area = pd.to_numeric(
        data.get("area_value", pd.Series(index=data.index)),
        errors="coerce",
    )

    data["total_rooms"] = bedrooms.fillna(0) + bathrooms.fillna(0)

    data["bed_bath_ratio"] = _safe_divide(
        bedrooms,
        bathrooms,
    )

    data["area_per_bedroom"] = _safe_divide(
        area,
        bedrooms,
    )

    data["area_per_bathroom"] = _safe_divide(
        area,
        bathrooms,
    )

    data["bathrooms_per_bedroom"] = _safe_divide(
        bathrooms,
        bedrooms,
    )

    data["bedrooms_per_100sqm"] = (
        _safe_divide(
            bedrooms * 100,
            area,
        )
    )

    # -------------------------------------------------------------
    # Binary / quality features
    # -------------------------------------------------------------

    if "video_url" in data.columns:
        data["has_video"] = (
            data["video_url"]
            .notna()
            .astype(int)
        )
    else:
        data["has_video"] = 0

    if "has_view_360" in data.columns:
        data["has_360_view"] = _to_binary(
            data["has_view_360"]
        )
    else:
        data["has_360_view"] = 0

    if "amenities" in data.columns:
        data["has_amenities"] = (
            data["amenities"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .astype(int)
        )
    else:
        data["has_amenities"] = 0

    if "district" in data.columns:
        data["has_district"] = (
            data["district"]
            .notna()
            .astype(int)
        )
    else:
        data["has_district"] = 0

    if "subdistrict" in data.columns:
        data["has_subdistrict"] = (
            data["subdistrict"]
            .notna()
            .astype(int)
        )
    else:
        data["has_subdistrict"] = 0

    location_fields = [
        column
        for column in [
            "city",
            "town",
            "district",
            "subdistrict",
        ]
        if column in data.columns
    ]

    if location_fields:
        data["location_completeness"] = (
            data[location_fields]
            .notna()
            .sum(axis=1)
        )
    else:
        data["location_completeness"] = 0

    quality_components = [
        data["has_images"]
        if "has_images" in data.columns
        else pd.Series(0, index=data.index),

        data["has_video"],

        data["has_360_view"],

        data["has_amenities"],

        data["has_district"],

        data["has_subdistrict"],

        data["is_verified"]
        if "is_verified" in data.columns
        else pd.Series(0, index=data.index),

        data["is_featured"]
        if "is_featured" in data.columns
        else pd.Series(0, index=data.index),
    ]

    data["listing_quality_score"] = sum(
        component.fillna(0).astype(int)
        for component in quality_components
    )

    # -------------------------------------------------------------
    # Agent features
    # -------------------------------------------------------------

    if "agent_languages" in data.columns:
        data["agent_language_count"] = (
            data["agent_languages"]
            .fillna("")
            .astype(str)
            .apply(
                lambda value: (
                    len(
                        [
                            item
                            for item in value.split(",")
                            if item.strip()
                        ]
                    )
                    if value.strip()
                    else 0
                )
            )
        )
    else:
        data["agent_language_count"] = 0

    if "agent_is_super" in data.columns:
        data["super_agent"] = _to_binary(
            data["agent_is_super"]
        )
    else:
        data["super_agent"] = 0

    if "is_direct_from_dev" in data.columns:
        data["direct_developer"] = _to_binary(
            data["is_direct_from_dev"]
        )
    else:
        data["direct_developer"] = 0

    # -------------------------------------------------------------
    # Property characteristics
    # -------------------------------------------------------------

    if "area_value" in data.columns:
        data["is_large_property"] = (
            pd.to_numeric(
                data["area_value"],
                errors="coerce",
            )
            >= 200
        ).astype(int)
    else:
        data["is_large_property"] = 0

    if "bedrooms" in data.columns:
        data["is_high_bedroom"] = (
            pd.to_numeric(
                data["bedrooms"],
                errors="coerce",
            )
            >= 4
        ).astype(int)
    else:
        data["is_high_bedroom"] = 0

    # -------------------------------------------------------------
    # Category features
    # -------------------------------------------------------------

    if "area_value" in data.columns:
        area = pd.to_numeric(
            data["area_value"],
            errors="coerce",
        )

        data["area_size_category"] = pd.cut(
            area,
            bins=[
                -np.inf,
                60,
                100,
                150,
                250,
                np.inf,
            ],
            labels=[
                "small",
                "medium",
                "large",
                "very_large",
                "luxury_size",
            ],
        ).astype("object")

        data["area_category"] = pd.cut(
            area,
            bins=[
                -np.inf,
                80,
                120,
                180,
                np.inf,
            ],
            labels=[
                "compact",
                "standard",
                "spacious",
                "very_spacious",
            ],
        ).astype("object")
    else:
        data["area_size_category"] = "unknown"
        data["area_category"] = "unknown"

    if "bedrooms" in data.columns:
        bedrooms = pd.to_numeric(
            data["bedrooms"],
            errors="coerce",
        )

        data["bedroom_category"] = pd.cut(
            bedrooms,
            bins=[
                -np.inf,
                0,
                1,
                2,
                3,
                np.inf,
            ],
            labels=[
                "studio",
                "one_bedroom",
                "two_bedrooms",
                "three_bedrooms",
                "four_plus_bedrooms",
            ],
        ).astype("object")
    else:
        data["bedroom_category"] = "unknown"

    if "bathrooms" in data.columns:
        bathrooms = pd.to_numeric(
            data["bathrooms"],
            errors="coerce",
        )

        data["bathroom_category"] = pd.cut(
            bathrooms,
            bins=[
                -np.inf,
                1,
                2,
                3,
                np.inf,
            ],
            labels=[
                "one_or_less",
                "two",
                "three",
                "four_plus",
            ],
        ).astype("object")
    else:
        data["bathroom_category"] = "unknown"

    # -------------------------------------------------------------
    # Location interaction features
    # -------------------------------------------------------------

    def combine_columns(
        first: str,
        second: str,
        output: str,
    ) -> None:
        if first in data.columns and second in data.columns:
            data[output] = (
                data[first]
                .fillna("Unknown")
                .astype(str)
                + " | "
                + data[second]
                .fillna("Unknown")
                .astype(str)
            )
        else:
            data[output] = "Unknown"

    combine_columns(
        "city",
        "town",
        "city_town",
    )

    combine_columns(
        "town",
        "district",
        "town_district",
    )

    combine_columns(
        "district",
        "subdistrict",
        "district_subdistrict",
    )

    combine_columns(
        "city",
        "property_type",
        "city_property_type",
    )

    combine_columns(
        "district",
        "property_type",
        "district_property_type",
    )

    combine_columns(
        "furnished",
        "property_type",
        "furnished_property_type",
    )

    if "city" in data.columns:
        city = data["city"].fillna("Unknown").astype(str)
    else:
        city = pd.Series(
            "Unknown",
            index=data.index,
        )

    if "town" in data.columns:
        town = data["town"].fillna("Unknown").astype(str)
    else:
        town = pd.Series(
            "Unknown",
            index=data.index,
        )

    if "district" in data.columns:
        district = (
            data["district"]
            .fillna("Unknown")
            .astype(str)
        )
    else:
        district = pd.Series(
            "Unknown",
            index=data.index,
        )

    if "subdistrict" in data.columns:
        subdistrict = (
            data["subdistrict"]
            .fillna("Unknown")
            .astype(str)
        )
    else:
        subdistrict = pd.Series(
            "Unknown",
            index=data.index,
        )

    data["property_location"] = (
        city
        + " | "
        + town
        + " | "
        + district
        + " | "
        + subdistrict
    )

    # -------------------------------------------------------------
    # Remove target / leakage columns
    # -------------------------------------------------------------

    leakage_columns_present = [
        column
        for column in LEAKAGE_COLUMNS
        if column in data.columns
    ]

    if leakage_columns_present:
        data = data.drop(
            columns=leakage_columns_present
        )

    return data


# ---------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------

def get_feature_lists() -> Tuple[list[str], list[str]]:
    return (
        NUMERIC_FEATURES.copy(),
        CATEGORICAL_FEATURES.copy(),
    )


# ---------------------------------------------------------------------
# ML dataset preparation
# ---------------------------------------------------------------------

def prepare_ml_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and log-transformed target.

    Returns:
        X: engineered ML features
        y: log1p(price_egp)
    """

    if "price_egp" not in df.columns:
        raise ValueError(
            "Input dataframe must contain 'price_egp'."
        )

    target = pd.to_numeric(
        df["price_egp"],
        errors="coerce",
    )

    valid_target = (
        target.notna()
        & (target > 0)
    )

    clean_df = df.loc[valid_target].copy()
    target = target.loc[valid_target]

    features = build_features(clean_df)

    numeric_features, categorical_features = (
        get_feature_lists()
    )

    expected_features = (
        numeric_features
        + categorical_features
    )

    missing_features = [
        column
        for column in expected_features
        if column not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing engineered features: "
            + ", ".join(missing_features)
        )

    X = features[expected_features].copy()

    y = np.log1p(
        target.astype(float)
    )

    return X, y