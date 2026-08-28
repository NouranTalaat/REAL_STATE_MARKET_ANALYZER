from pathlib import Path

import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_ROOT
    / "DATA"
    / "raw"
    / "propertyfinder.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "DATA"
    / "processed"
)

PROCESSED_FILE = (
    PROCESSED_DIR
    / "cleaned_propertyfinder.csv"
)


# =========================================================
# LOAD RAW DATA
# =========================================================

def load_data():
    """Load the raw PropertyFinder dataset."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    return pd.read_csv(
        RAW_FILE,
        low_memory=False
    )


# =========================================================
# SAVE PROCESSED DATA
# =========================================================

def save_processed_data(df):
    """Save the processed dataset as CSV."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_FILE,
        index=False
    )

    print("=" * 70)
    print("PROCESSED DATASET SAVED SUCCESSFULLY")
    print("=" * 70)

    print(f"File: {PROCESSED_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")