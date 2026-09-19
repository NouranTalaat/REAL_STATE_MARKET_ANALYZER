"""
Real Estate Market Intelligence Platform
SQL Server Staging Loader
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.database import engine


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "propertyfinder_clean.csv"
)

SCHEMA = "staging"
TABLE = "property_listings_raw"

CHUNK_SIZE = 5_000


def prepare_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare CSV data for SQL Server.
    """

    df = df.copy()

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

    numeric_columns = [
        "price_egp",
        "area_value",
        "lat",
        "lon",
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
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).astype("Int64")

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    for column in boolean_columns:
        if column in df.columns:
            df[column] = df[column].astype("boolean")

    for column in datetime_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
                utc=True,
            ).dt.tz_localize(None)

    return df.astype(object).where(
        pd.notna(df),
        None,
    )


def load_data():
    """
    Load processed CSV into SQL Server staging.
    """

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )

    print("=" * 70)
    print("REAL ESTATE MARKET INTELLIGENCE PLATFORM")
    print("SQL SERVER STAGING LOADER")
    print("=" * 70)

    print(f"\nSource: {CSV_PATH}")
    print(f"Target: {SCHEMA}.{TABLE}")

    total_rows = 0
    chunk_number = 0

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        print("\nSQL Server connection successful.")

        chunks = pd.read_csv(
            CSV_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        with engine.begin() as connection:

            connection.execute(
                text(
                    f"""
                    TRUNCATE TABLE {SCHEMA}.{TABLE};
                    """
                )
            )

            for chunk in chunks:

                chunk_number += 1

                chunk = prepare_chunk(chunk)

                chunk.to_sql(
                    name=TABLE,
                    con=connection,
                    schema=SCHEMA,
                    if_exists="append",
                    index=False,
                    chunksize=CHUNK_SIZE,
                    method=None,
                )

                rows_loaded = len(chunk)
                total_rows += rows_loaded

                print(
                    f"Chunk #{chunk_number}: "
                    f"{rows_loaded:,} rows"
                )

        with engine.connect() as connection:

            result = connection.execute(
                text(
                    f"""
                    SELECT COUNT(*) AS row_count
                    FROM {SCHEMA}.{TABLE};
                    """
                )
            )

            sql_row_count = int(result.scalar())

        print("\n" + "=" * 70)
        print("STAGING LOAD COMPLETED")
        print("=" * 70)

        print(
            f"Rows processed: {total_rows:,}"
        )

        print(
            f"Rows in SQL Server: {sql_row_count:,}"
        )

        if total_rows != sql_row_count:
            raise RuntimeError(
                "Staging row count validation failed."
            )

        print("\nRow count validation: PASS")

        return {
            "processed_rows": total_rows,
            "sql_rows": sql_row_count,
            "status": "SUCCESS",
        }

    finally:
        engine.dispose()


if __name__ == "__main__":
    load_data()