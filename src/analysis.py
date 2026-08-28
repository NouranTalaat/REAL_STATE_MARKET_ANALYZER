
# ============================================================
# REAL ESTATE MARKET ANALYZER
# ANALYSIS MODULE
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "DATA"
    / "raw"
    / "propertyfinder.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the PropertyFinder dataset.
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    return pd.read_csv(
        DATA_FILE,
        low_memory=False
    )


# ============================================================
# MARKET OVERVIEW
# ============================================================

def market_overview(df):
    """
    Return a general overview of the real estate market.
    """

    total_listings = len(df)

    sale_df = df[
        df["category"].astype(str).str.lower() == "buy"
    ]

    rent_df = df[
        df["category"].astype(str).str.lower() == "rent"
    ]

    return {
        "total_listings": total_listings,
        "sale_listings": len(sale_df),
        "rent_listings": len(rent_df),

        "sale_percentage": round(
            len(sale_df) / total_listings * 100,
            2
        ) if total_listings else 0,

        "rent_percentage": round(
            len(rent_df) / total_listings * 100,
            2
        ) if total_listings else 0,

        "median_sale_price": (
            sale_df["price_egp"].median()
            if not sale_df.empty
            else None
        ),

        "median_rent_price": (
            rent_df["price_egp"].median()
            if not rent_df.empty
            else None
        )
    }


# ============================================================
# PRICE SUMMARY
# ============================================================

def price_summary(df):
    """
    Calculate price statistics for Sale and Rent.
    """

    results = {}

    for category, name in [
        ("buy", "sale"),
        ("rent", "rent")
    ]:

        subset = df[
            df["category"].astype(str).str.lower()
            == category
        ]

        if subset.empty:
            results[name] = {}
            continue

        prices = subset["price_egp"].dropna()

        results[name] = {
            "min": float(prices.min()),
            "max": float(prices.max()),
            "mean": float(prices.mean()),
            "median": float(prices.median()),
            "std": float(prices.std())
        }

    return results


# ============================================================
# PROPERTY TYPE ANALYSIS
# ============================================================

def property_type_analysis(df):
    """
    Analyze listings by property type.
    """

    result = (
        df["property_type"]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    result.columns = [
        "property_type",
        "listings"
    ]

    return result


# ============================================================
# PROPERTY TYPE PRICE ANALYSIS
# ============================================================

def property_type_prices(df, category):
    """
    Calculate price statistics by property type.

    category:
        buy
        rent
    """

    subset = df[
        df["category"].astype(str).str.lower()
        == category.lower()
    ].copy()

    if subset.empty:
        return pd.DataFrame()
    result = (
        subset
        .groupby("property_type", observed=True)
        ["price_egp"]
        .agg(
            listings="count",
            mean_price="mean",
            median_price="median",
            min_price="min",
            max_price="max"
        )
        .reset_index()
        .sort_values(
            "median_price",
            ascending=False
        )
    )

    return result


# ============================================================
# LOCATION ANALYSIS
# ============================================================

def location_analysis(df):
    """
    Analyze listings by city.
    """

    result = (
        df["city"]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    result.columns = [
        "city",
        "listings"
    ]

    return result


# ============================================================
# CITY PRICE ANALYSIS
# ============================================================

def city_price_analysis(df, category):
    """
    Calculate price statistics by city.
    """

    subset = df[
        df["category"].astype(str).str.lower()
        == category.lower()
    ].copy()

    if subset.empty:
        return pd.DataFrame()

    result = (
        subset
        .groupby("city", observed=True)
        ["price_egp"]
        .agg(
            listings="count",
            mean_price="mean",
            median_price="median"
        )
        .reset_index()
        .sort_values(
            "median_price",
            ascending=False
        )
    )

    return result


# ============================================================
# TOWN ANALYSIS
# ============================================================

def town_analysis(df):
    """
    Analyze listings by town.
    """

    result = (
        df["town"]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    result.columns = [
        "town",
        "listings"
    ]

    return result


# ============================================================
# DISTRICT ANALYSIS
# ============================================================

def district_analysis(df):
    """
    Analyze listings by district.
    """

    result = (
        df["district"]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    result.columns = [
        "district",
        "listings"
    ]

    return result


# ============================================================
# PROPERTY CHARACTERISTICS
# ============================================================

def property_characteristics(df):
    """
    Analyze property characteristics.
    """

    numeric_columns = [
        "area_value",
        "bedrooms",
        "bathrooms"
    ]

    available_columns = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    if not available_columns:
        return pd.DataFrame()

    return (
        df[available_columns]
        .describe()
        .T
        .reset_index()
        .rename(
            columns={
                "index": "feature"
            }
        )
    )


# ============================================================
# BEDROOM PRICE ANALYSIS
# ============================================================

def bedroom_price_analysis(df, category):
    """
    Analyze prices according to number of bedrooms.
    """

    subset = df[
        df["category"].astype(str).str.lower()
        == category.lower()
    ].copy()

    subset = subset.dropna(
        subset=["bedrooms", "price_egp"]
    )

    if subset.empty:
        return pd.DataFrame()

    result = (
        subset
        .groupby("bedrooms")
        ["price_egp"]
        .agg(
            listings="count",
            mean_price="mean",
            median_price="median"
        )
        .reset_index()
        .sort_values("bedrooms")
    )

    return result


# ============================================================
# AREA PRICE ANALYSIS
# ============================================================
def area_price_analysis(df, category):
    """
    Analyze relationship between area and price.
    """

    subset = df[
        df["category"].astype(str).str.lower()
        == category.lower()
    ].copy()

    subset = subset[
        [
            "area_value",
            "price_egp"
        ]
    ].dropna()

    if subset.empty:
        return pd.DataFrame()

    return subset


# ============================================================
# FURNISHED ANALYSIS
# ============================================================

def furnished_analysis(df, category):
    """
    Compare furnished and unfurnished properties.
    """

    subset = df[
        df["category"].astype(str).str.lower()
        == category.lower()
    ].copy()

    if "furnished" not in subset.columns:
        return pd.DataFrame()

    result = (
        subset
        .groupby("furnished", observed=True)
        ["price_egp"]
        .agg(
            listings="count",
            mean_price="mean",
            median_price="median"
        )
        .reset_index()
        .sort_values(
            "median_price",
            ascending=False
        )
    )

    return result


# ============================================================
# AREA PRICE PER SQM
# ============================================================

def price_per_sqm_analysis(df, category):
    """
    Analyze price per square meter.
    """

    subset = df[
        df["category"].astype(str).str.lower()
        == category.lower()
    ].copy()

    required = [
        "area_value",
        "price_egp"
    ]

    if not all(
        column in subset.columns
        for column in required
    ):
        return pd.DataFrame()

    subset = subset[
        (subset["area_value"] > 0)
        & (subset["price_egp"] > 0)
    ].copy()

    if subset.empty:
        return pd.DataFrame()

    subset["price_per_sqm"] = (
        subset["price_egp"]
        / subset["area_value"]
    )

    return subset[
        [
            "property_type",
            "city",
            "area_value",
            "price_egp",
            "price_per_sqm"
        ]
    ]


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def numeric_correlation(df):
    """
    Calculate correlations between numerical variables.
    """

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.corr()


# ============================================================
# TOP LOCATIONS
# ============================================================

def top_locations(df, n=10):
    """
    Return the most active cities by listing count.
    """

    result = (
        df["city"]
        .astype(str)
        .value_counts()
        .head(n)
        .reset_index()
    )

    result.columns = [
        "city",
        "listings"
    ]

    return result


# ============================================================
# TOP PROPERTY TYPES
# ============================================================

def top_property_types(df, n=10):
    """
    Return the most common property types.
    """

    result = (
        df["property_type"]
        .astype(str)
        .value_counts()
        .head(n)
        .reset_index()
    )

    result.columns = [
        "property_type",
        "listings"
    ]

    return result


# ============================================================
# COMPLETE MARKET ANALYSIS
# ============================================================

def run_market_analysis(df):
    """
    Run the complete market analysis.
    """

    analysis = {

        "market_overview":
            market_overview(df),

        "price_summary":
            price_summary(df),

        "property_types":
            property_type_analysis(df),

        "locations":
            location_analysis(df),

        "towns":
            town_analysis(df),

        "districts":
            district_analysis(df),
        "property_characteristics":
            property_characteristics(df),

        "sale_property_type_prices":
            property_type_prices(
                df,
                "buy"
            ),

        "rent_property_type_prices":
            property_type_prices(
                df,
                "rent"
            ),

        "sale_city_prices":
            city_price_analysis(
                df,
                "buy"
            ),

        "rent_city_prices":
            city_price_analysis(
                df,
                "rent"
            ),

        "sale_bedroom_prices":
            bedroom_price_analysis(
                df,
                "buy"
            ),

        "rent_bedroom_prices":
            bedroom_price_analysis(
                df,
                "rent"
            ),

        "sale_furnished":
            furnished_analysis(
                df,
                "buy"
            ),

        "rent_furnished":
            furnished_analysis(
                df,
                "rent"
            )
    }

    return analysis


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("REAL ESTATE MARKET ANALYSIS")
    print("=" * 70)

    df = load_data()

    print(
        f"\nDataset loaded successfully."
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    # --------------------------------------------------------
    # MARKET OVERVIEW
    # --------------------------------------------------------

    overview = market_overview(df)

    print("\n")
    print("=" * 70)
    print("MARKET OVERVIEW")
    print("=" * 70)

    for key, value in overview.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:,.2f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    # --------------------------------------------------------
    # PRICE SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PRICE SUMMARY")
    print("=" * 70)

    summary = price_summary(df)

    for category, values in summary.items():

        print(
            f"\n{category.upper()}"
        )

        for key, value in values.items():

            print(
                f"{key}: {value:,.2f} EGP"
            )

    # --------------------------------------------------------
    # TOP LOCATIONS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOP 10 CITIES")
    print("=" * 70)

    print(
        top_locations(df, 10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # TOP PROPERTY TYPES
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOP PROPERTY TYPES")
    print("=" * 70)

    print(
        top_property_types(df, 10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # SALE CITY PRICES
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SALE PRICE BY CITY")
    print("=" * 70)

    print(
        city_price_analysis(
            df,
            "buy"
        )
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # RENT CITY PRICES
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RENT PRICE BY CITY")
    print("=" * 70)

    print(
        city_price_analysis(
            df,
            "rent"
        )
        .head(10)
        .to_string(index=False)
    )

    print("\n")
    print("=" * 70)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)