import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "propertyfinder.csv"
)

SALE_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "final_sale_model.pkl"
)

RENT_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "final_rent_model.pkl"
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Real Estate Market Analyzer API",
    description="API for analyzing the Egyptian real estate market",
    version="1.0.0",
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    """Load the PropertyFinder dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    return pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )


# =========================================================
# LOAD MODELS
# =========================================================

def load_sale_model():

    if not SALE_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Sale model not found: {SALE_MODEL_FILE}"
        )

    return joblib.load(SALE_MODEL_FILE)


def load_rent_model():

    if not RENT_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Rent model not found: {RENT_MODEL_FILE}"
        )

    return joblib.load(RENT_MODEL_FILE)


# =========================================================
# PREPARE SALE INPUT
# =========================================================

def prepare_sale_input(data: dict) -> pd.DataFrame:

    sale_features = [
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
        "is_luxury_property",
        "is_large_property",
        "is_high_bedroom",
        "is_new_construction",
        "is_direct_from_dev",
        "is_exclusive",
        "direct_developer",
        "super_agent",
        "property_type",
        "completion_status",
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

    input_df = pd.DataFrame([data])

    for column in sale_features:
        if column not in input_df.columns:
            input_df[column] = None

    return input_df[sale_features]


# =========================================================
# PREPARE RENT INPUT
# =========================================================

def prepare_rent_input(data: dict) -> pd.DataFrame:

    rent_features = [
        "area_value",
        "bedrooms",
        "bathrooms",
        "log_area",
        "bedrooms_per_100sqm",
        "bathrooms_per_bedroom",
        "is_luxury_property",
        "is_large_property",
        "is_high_bedroom",
        "is_new_construction",
        "is_direct_from_dev",
        "property_type",
        "city",
        "town",
        "district",
        "subdistrict",
        "furnished",
        "listing_level",
        "city_town",
        "town_district",
        "district_subdistrict",
        "area_category",
        "bedroom_category",
        "city_property_type",
        "district_property_type",
        "furnished_property_type",
    ]

    input_df = pd.DataFrame([data])

    for column in rent_features:
        if column not in input_df.columns:
            input_df[column] = None

    return input_df[rent_features]


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Real Estate Market Analyzer API is running"
    }


# =========================================================
# GET PROPERTIES
# =========================================================

@app.get("/properties")
def get_properties(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
):

    df = load_data()

    properties = df.iloc[
        skip:skip + limit
    ]

    properties_json = properties.to_json(
        orient="records",
        date_format="iso",
    )

    return {
        "total": len(df),
        "skip": skip,
        "limit": limit,
        "properties": json.loads(properties_json),
    }


# =========================================================
# MARKET OVERVIEW
# =========================================================

@app.get("/market/overview")
def market_overview():

    df = load_data()

    sale_df = df[
        df["category"] == "buy"
    ]

    rent_df = df[
        df["category"] == "rent"
    ]

    return {
        "total_listings": len(df),
        "sale_listings": len(sale_df),
        "rent_listings": len(rent_df),
        "median_sale_price": sale_df["price_egp"].median(),
        "median_rent": rent_df["price_egp"].median(),
    }


# =========================================================
# PROPERTY TYPES
# =========================================================

@app.get("/market/property-types")
def property_types():

    df = load_data()

    result = (
        df["property_type"]
        .value_counts()
        .reset_index()
    )

    result.columns = [
        "property_type",
        "listings",
    ]

    return {
        "property_types": result.to_dict(
            orient="records"
        )
    }


# =========================================================
# LOCATIONS
# =========================================================

@app.get("/market/locations")
def market_locations(
    city: str | None = None,
):

    df = load_data()

    if city:

        df = df[
            df["city"]
            .astype(str)
            .str.lower()
            == city.lower()
        ]

        if df.empty:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"No listings found "
                    f"for city: {city}"
                ),
            )

    result = (
        df["city"]
        .value_counts()
        .reset_index()
    )

    result.columns = [
        "city",
        "listings",
    ]

    return {
        "city_filter": city,
        "locations": result.to_dict(
            orient="records"
        ),
    }


# =========================================================
# PRICE ANALYSIS
# =========================================================

@app.get("/market/prices")
def price_analysis():

    df = load_data()

    sale_df = df[
        df["category"] == "buy"
    ]

    rent_df = df[
        df["category"] == "rent"
    ]

    return {
        "sale": {
            "min": sale_df["price_egp"].min(),
            "max": sale_df["price_egp"].max(),
            "mean": sale_df["price_egp"].mean(),
            "median": sale_df["price_egp"].median(),
        },
        "rent": {
            "min": rent_df["price_egp"].min(),
            "max": rent_df["price_egp"].max(),
            "mean": rent_df["price_egp"].mean(),
            "median": rent_df["price_egp"].median(),
        },
    }


# =========================================================
# SALE PRICE PREDICTION
# =========================================================

@app.post("/predict/sale")
def predict_sale(data: dict):

    try:

        model = load_sale_model()

        X_input = prepare_sale_input(data)

        prediction_log = model.predict(X_input)[0]

        prediction = max(
            0,
            float(np.expm1(prediction_log)),
        )

        return {
            "prediction_type": "sale_price",
            "predicted_price_egp": round(
                prediction,
                2,
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# =========================================================
# RENT PRICE PREDICTION
# =========================================================

@app.post("/predict/rent")
def predict_rent(data: dict):

    try:

        model = load_rent_model()

        X_input = prepare_rent_input(data)

        prediction_log = model.predict(X_input)[0]

        prediction = max(
            0,
            float(np.expm1(prediction_log)),
        )

        return {
            "prediction_type": "rent_price",
            "predicted_rent_egp": round(
                prediction,
                2,
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )