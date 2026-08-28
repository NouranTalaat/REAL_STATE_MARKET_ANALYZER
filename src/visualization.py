# ============================================================
# REAL ESTATE MARKET ANALYZER
# DATA VISUALIZATION
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_ROOT
    / "DATA"
    / "raw"
    / "propertyfinder.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    df = pd.read_csv(
        RAW_FILE,
        low_memory=False
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = df.copy()

    # Make sure price is numeric
    df["price_egp"] = pd.to_numeric(
        df["price_egp"],
        errors="coerce"
    )

    # Remove invalid prices
    df = df[
        df["price_egp"] > 0
    ]

    return df


# ============================================================
# 1. SALE VS RENT DISTRIBUTION
# ============================================================

def plot_sale_vs_rent(df):

    counts = (
        df["category"]
        .value_counts()
        .reindex(["buy", "rent"])
    )

    labels = [
        "Sale",
        "Rent"
    ]

    plt.figure(figsize=(8, 6))

    plt.bar(
        labels,
        counts.values
    )

    plt.title(
        "Sale vs Rent Listings"
    )

    plt.xlabel(
        "Listing Type"
    )

    plt.ylabel(
        "Number of Listings"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 2. SALE VS RENT PERCENTAGE
# ============================================================

def plot_market_share(df):

    counts = (
        df["category"]
        .value_counts()
        .reindex(["buy", "rent"])
    )

    labels = [
        "Sale",
        "Rent"
    ]

    plt.figure(figsize=(7, 7))

    plt.pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title(
        "Sale vs Rent Market Share"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 3. MEDIAN SALE VS RENT PRICE
# ============================================================

def plot_median_prices(df):

    sale_median = df.loc[
        df["category"] == "buy",
        "price_egp"
    ].median()

    rent_median = df.loc[
        df["category"] == "rent",
        "price_egp"
    ].median()

    values = [
        sale_median,
        rent_median
    ]

    labels = [
        "Sale",
        "Rent"
    ]

    plt.figure(figsize=(8, 6))

    plt.bar(
        labels,
        values
    )

    plt.title(
        "Median Property Price"
    )

    plt.xlabel(
        "Listing Type"
    )

    plt.ylabel(
        "Median Price (EGP)"
    )

    plt.ticklabel_format(
        style="plain",
        axis="y"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 4. TOP 10 CITIES
# ============================================================

def plot_top_cities(df):

    top_cities = (
        df["city"]
        .value_counts()
        .head(10)
        .sort_values()
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_cities.index.astype(str),
        top_cities.values
    )

    plt.title(
        "Top 10 Cities by Number of Listings"
    )

    plt.xlabel(
        "Number of Listings"
    )

    plt.ylabel(
        "City"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
#  5. TOP PROPERTY TYPES
# ============================================================

def plot_top_property_types(df):

    property_counts = (
        df["property_type"]
        .astype("string")
        .value_counts()
        .head(10)
        .sort_values()
    )

    plt.figure(figsize=(12, 7))

    plt.barh(
        property_counts.index.astype(str),
        property_counts.values
    )

    plt.title("Top 10 Property Types")
    plt.xlabel("Number of Listings")
    plt.ylabel("Property Type")

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. SALE MEDIAN PRICE BY CITY
# ============================================================

def plot_sale_price_by_city(df):

    sale_df = df[
        df["category"] == "buy"
    ]

    city_prices = (
        sale_df
        .groupby("city")["price_egp"]
        .median()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        city_prices.index.astype(str),
        city_prices.values
    )

    plt.title(
        "Top 10 Cities by Median Sale Price"
    )

    plt.xlabel(
        "Median Sale Price (EGP)"
    )

    plt.ylabel(
        "City"
    )

    plt.ticklabel_format(
        style="plain",
        axis="x"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 7. RENT MEDIAN PRICE BY CITY
# ============================================================

def plot_rent_price_by_city(df):

    rent_df = df[
        df["category"] == "rent"
    ]

    city_prices = (
        rent_df
        .groupby("city")["price_egp"]
        .median()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        city_prices.index.astype(str),
        city_prices.values
    )

    plt.title(
        "Top 10 Cities by Median Rent Price"
    )

    plt.xlabel(
        "Median Rent Price (EGP)"
    )

    plt.ylabel(
        "City"
    )

    plt.ticklabel_format(
        style="plain",
        axis="x"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 8. SALE PRICE DISTRIBUTION
# ============================================================

def plot_sale_price_distribution(df):

    sale_prices = df.loc[
        df["category"] == "buy",
        "price_egp"
    ].dropna()

    # Use log scale because real-estate prices are highly skewed
    sale_prices = sale_prices[
        sale_prices > 0
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        np.log1p(sale_prices),
        bins=50
    )

    plt.title(
        "Sale Price Distribution (Log Scale)"
    )

    plt.xlabel(
        "Log(1 + Sale Price)"
    )

    plt.ylabel(
        "Number of Listings"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 9. RENT PRICE DISTRIBUTION
# ============================================================

def plot_rent_price_distribution(df):

    rent_prices = df.loc[
        df["category"] == "rent",
        "price_egp"
    ].dropna()

    rent_prices = rent_prices[
        rent_prices > 0
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        np.log1p(rent_prices),
        bins=50
    )

    plt.title(
        "Rent Price Distribution (Log Scale)"
    )

    plt.xlabel(
        "Log(1 + Rent Price)"
    )

    plt.ylabel(
        "Number of Listings"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 10. AREA VS SALE PRICE
# ============================================================

def plot_area_vs_sale_price(df):

    sale_df = df[
        df["category"] == "buy"
    ].copy()

    sale_df["area_value"] = pd.to_numeric(
        sale_df["area_value"],
        errors="coerce"
    )
    sale_df = sale_df[
        (sale_df["area_value"] > 0)
        &
        (sale_df["price_egp"] > 0)
    ]

    # Remove extreme outliers only for visualization
    sale_df = sale_df[
        sale_df["area_value"]
        <= sale_df["area_value"].quantile(0.99)
    ]

    sale_df = sale_df[
        sale_df["price_egp"]
        <= sale_df["price_egp"].quantile(0.99)
    ]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        sale_df["area_value"],
        sale_df["price_egp"],
        alpha=0.3
    )

    plt.title(
        "Area vs Sale Price"
    )

    plt.xlabel(
        "Area (sqm)"
    )

    plt.ylabel(
        "Sale Price (EGP)"
    )

    plt.ticklabel_format(
        style="plain",
        axis="y"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 11. AREA VS RENT PRICE
# ============================================================

def plot_area_vs_rent_price(df):

    rent_df = df[
        df["category"] == "rent"
    ].copy()

    rent_df["area_value"] = pd.to_numeric(
        rent_df["area_value"],
        errors="coerce"
    )

    rent_df = rent_df[
        (rent_df["area_value"] > 0)
        &
        (rent_df["price_egp"] > 0)
    ]

    rent_df = rent_df[
        rent_df["area_value"]
        <= rent_df["area_value"].quantile(0.99)
    ]

    rent_df = rent_df[
        rent_df["price_egp"]
        <= rent_df["price_egp"].quantile(0.99)
    ]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        rent_df["area_value"],
        rent_df["price_egp"],
        alpha=0.3
    )

    plt.title(
        "Area vs Rent Price"
    )

    plt.xlabel(
        "Area (sqm)"
    )

    plt.ylabel(
        "Rent Price (EGP)"
    )

    plt.ticklabel_format(
        style="plain",
        axis="y"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 12. SALE PRICE BY PROPERTY TYPE
# ============================================================

def plot_sale_price_by_property_type(df):

    sale_df = df[
        df["category"] == "buy"
    ]

    result = (
        sale_df
        .groupby("property_type")["price_egp"]
        .median()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        result.index.astype(str),
        result.values
    )

    plt.title(
        "Median Sale Price by Property Type"
    )

    plt.xlabel(
        "Median Sale Price (EGP)"
    )

    plt.ylabel(
        "Property Type"
    )

    plt.ticklabel_format(
        style="plain",
        axis="x"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 13. RENT PRICE BY PROPERTY TYPE
# ============================================================

def plot_rent_price_by_property_type(df):

    rent_df = df[
        df["category"] == "rent"
    ]

    result = (
        rent_df
        .groupby("property_type")["price_egp"]
        .median()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        result.index.astype(str),
        result.values
    )

    plt.title(
        "Median Rent Price by Property Type"
    )

    plt.xlabel(
        "Median Rent Price (EGP)"
    )

    plt.ylabel(
        "Property Type"
    )

    plt.ticklabel_format(
        style="plain",
        axis="x"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 14. BEDROOMS VS SALE PRICE
# ============================================================

def plot_bedrooms_vs_sale_price(df):

    sale_df = df[
        df["category"] == "buy"
    ].copy()

    sale_df["bedrooms"] = pd.to_numeric(
        sale_df["bedrooms"],
        errors="coerce"
    )

    result = (
        sale_df
        .dropna(subset=["bedrooms"])
        .groupby("bedrooms")["price_egp"]
        .median()
    )

    result = result[
        result.index <= 8
    ]
    plt.figure(figsize=(10, 6))

    plt.plot(
        result.index,
        result.values,
        marker="o"
    )

    plt.title(
        "Median Sale Price by Number of Bedrooms"
    )

    plt.xlabel(
        "Bedrooms"
    )

    plt.ylabel(
        "Median Sale Price (EGP)"
    )

    plt.ticklabel_format(
        style="plain",
        axis="y"
    )

    plt.xticks(
        result.index
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 15. BEDROOMS VS RENT PRICE
# ============================================================

def plot_bedrooms_vs_rent_price(df):

    rent_df = df[
        df["category"] == "rent"
    ].copy()

    rent_df["bedrooms"] = pd.to_numeric(
        rent_df["bedrooms"],
        errors="coerce"
    )

    result = (
        rent_df
        .dropna(subset=["bedrooms"])
        .groupby("bedrooms")["price_egp"]
        .median()
    )

    result = result[
        result.index <= 8
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        result.index,
        result.values,
        marker="o"
    )

    plt.title(
        "Median Rent Price by Number of Bedrooms"
    )

    plt.xlabel(
        "Bedrooms"
    )

    plt.ylabel(
        "Median Rent Price (EGP)"
    )

    plt.ticklabel_format(
        style="plain",
        axis="y"
    )

    plt.xticks(
        result.index
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 16. MAIN VISUALIZATION FUNCTION
# ============================================================

def run_all_visualizations():

    print("=" * 70)
    print("REAL ESTATE MARKET VISUALIZATION")
    print("=" * 70)

    df = load_data()

    df = prepare_data(df)

    print(
        f"Dataset loaded successfully."
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    print("\nGenerating visualizations...")

    plot_sale_vs_rent(df)

    plot_market_share(df)

    plot_median_prices(df)

    plot_top_cities(df)

    plot_top_property_types(df)

    plot_sale_price_by_city(df)

    plot_rent_price_by_city(df)

    plot_sale_price_distribution(df)

    plot_rent_price_distribution(df)

    plot_area_vs_sale_price(df)

    plot_area_vs_rent_price(df)

    plot_sale_price_by_property_type(df)

    plot_rent_price_by_property_type(df)

    plot_bedrooms_vs_sale_price(df)

    plot_bedrooms_vs_rent_price(df)

    print("\n" + "=" * 70)
    print("DATA VISUALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 70)