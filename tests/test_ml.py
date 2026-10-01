from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.features import (
    build_features,
    prepare_ml_dataset,
)
from src.ml.valuation_engine import (
    RealEstateValuationEngine,
)


def sample_property() -> dict:
    return {
        "listing_type": "sale",
        "property_type": "Apartment",
        "completion_status": "Ready",
        "city": "Cairo",
        "town": "New Cairo",
        "district": "First Settlement",
        "subdistrict": "North",
        "furnished": "No",
        "listing_level": "Ground",
        "payment_method": "Cash",
        "lat": 30.0,
        "lon": 31.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "area_value": 150,
        "is_premium": 0,
        "is_featured": 0,
        "images_count": 10,
        "has_view_360": 0,
        "has_video": 1,
        "has_360_view": 0,
        "has_amenities": 1,
        "has_district": 1,
        "has_subdistrict": 1,
        "is_new_construction": 1,
        "is_direct_from_dev": 0,
        "is_exclusive": 0,
        "is_verified": 1,
        "agent_is_super": 1,
        "agent_languages": "English,Arabic",
        "listed_date": "2026-01-15",
    }


def test_feature_engineering_does_not_keep_target():
    df = pd.DataFrame(
        [sample_property()]
    )

    result = build_features(df)

    assert "price_egp" not in result.columns
    assert "price_per_sqm" not in result.columns


def test_feature_engineering_creates_expected_features():
    df = pd.DataFrame(
        [sample_property()]
    )

    result = build_features(df)

    expected = [
        "listing_year",
        "listing_month",
        "listing_day",
        "listing_dayofweek",
        "log_area",
        "total_rooms",
        "bed_bath_ratio",
        "area_per_bedroom",
        "bathrooms_per_bedroom",
        "city_town",
        "town_district",
        "district_subdistrict",
        "city_property_type",
    ]

    for column in expected:
        assert column in result.columns


def test_prepare_ml_dataset():
    df = pd.DataFrame(
        [
            {
                **sample_property(),
                "price_egp": 2_500_000,
            },
            {
                **sample_property(),
                "price_egp": 3_000_000,
            },
        ]
    )

    X, y = prepare_ml_dataset(df)

    assert len(X) == 2
    assert len(y) == 2
    assert "price_egp" not in X.columns
    assert np.isfinite(y).all()


def test_valuation_engine_predicts_sale_property():
    engine = RealEstateValuationEngine()

    result = engine.predict(
        property_data=sample_property(),
        market_type="sale",
    )

    assert isinstance(result, dict)

    assert result["market_type"] == "sale"
    assert result["market_value"] == "Residential for Sale"

    assert isinstance(
        result["estimated_value_egp"],
        float,
    )

    assert result["estimated_value_egp"] >= 0

    assert result["property_type"] == "Apartment"

    assert result["model_resolution"] in {
        "market",
        "specialized",
    }

    assert "confidence" in result
    assert "model_predictions" in result

    assert len(result["model_predictions"]) == 3