"""
Real Estate Market Intelligence Platform
Phase 3 - SQL Data Layer
Step 3 - CSV -> SQL Server Staging Loader
"""

from pathlib import Path
import urllib.parse

import pandas as pd
from sqlalchemy import create_engine, text


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "propertyfinder_clean.csv"
)

SQL_SERVER = "localhost"
DATABASE = "REAL_ESTATE_MARKET_INTELLIGENCE"

SCHEMA = "staging"
TABLE = "property_listings_raw"

CHUNK_SIZE = 5_000


# ============================================================
# DATABASE CONNECTION
# ============================================================

def create_database_engine():
    """Create a SQLAlchemy engine for SQL Server."""

    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={DATABASE};"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )

    connection_url = urllib.parse.quote_plus(connection_string)

    engine = create_engine(
        f"mssql+pyodbc:///?odbc_connect={connection_url}",
        fast_executemany=True,
    )

    return engine


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare CSV data for SQL Server."""

    # Integer columns
    integer_columns = [
        "internal_id",
        "bedrooms",
        "bathrooms",
        "images_count",
        "rera",
        "agent_id",
        "broker_id",
        "listing_age_days",
    ]

    for column in integer_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).astype("Int64")

    # Numeric / decimal columns
    numeric_columns = [
        "price_egp",
        "area_value",
        "lat",
        "lon",
        "price_per_sqm",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Boolean columns
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

    for column in boolean_columns:
        if column in df.columns:
            df[column] = df[column].astype("boolean")

    # Datetime columns
    datetime_columns = [
        "listed_date",
        "scraped_at",
    ]

    for column in datetime_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    # Convert pandas NA/NaN to Python None
    df = df.astype(object).where(pd.notna(df), None)

    return df


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load processed CSV into SQL Server staging table."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{CSV_PATH}"
        )

    print("=" * 70)
    print("REAL ESTATE MARKET INTELLIGENCE PLATFORM")
    print("SQL SERVER STAGING LOADER")
    print("=" * 70)

    print("\nSource file:")
    print(CSV_PATH)

    print("\nTarget:")
    print(
        f"{DATABASE}.{SCHEMA}.{TABLE}"
    )

    print(
        f"\nChunk size: {CHUNK_SIZE:,}"
    )

    engine = create_database_engine()

    total_rows = 0
    chunk_number = 0

    try:
        # ----------------------------------------------------
        # Test database connection
        # ----------------------------------------------------

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        print("\n✓ SQL Server connection successful")

        # ----------------------------------------------------
        # Read CSV in chunks
        # ----------------------------------------------------

        chunks = pd.read_csv(
            CSV_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        for chunk in chunks:

            chunk_number += 1

            print(
                f"\nProcessing chunk #{chunk_number}..."
            )

            # Prepare chunk
            chunk = prepare_chunk(chunk)

            # Load into SQL Server
            chunk.to_sql(
                name=TABLE,
                con=engine,
                schema=SCHEMA,
                if_exists="append",
                index=False,
                chunksize=CHUNK_SIZE,
                method=None,
            )

            rows_loaded = len(chunk)

            total_rows += rows_loaded

            print(
                f"✓ Loaded {rows_loaded:,} rows"
            )

            print(
                f"✓ Total loaded: {total_rows:,}"
            )

        # ----------------------------------------------------
        # Final verification
        # ----------------------------------------------------

        with engine.connect() as connection:

            result = connection.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS row_count
                    FROM {SCHEMA}.{TABLE}
                    """
                )
            )

            sql_row_count = result.scalar()

        print("\n" + "=" * 70)
        print("LOAD COMPLETED")
        print("=" * 70)

        print(
            f"Rows processed by Python : {total_rows:,}"
        )

        print(
            f"Rows stored in SQL Server: {sql_row_count:,}"
        )

        if total_rows == sql_row_count:
            print(
                "\n✓ Row count validation PASSED"
            )
        else:
            print(
                "\n⚠ Row count mismatch detected"
            )

    finally:

        engine.dispose()

        print(
            "\nDatabase connection closed."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    load_data()