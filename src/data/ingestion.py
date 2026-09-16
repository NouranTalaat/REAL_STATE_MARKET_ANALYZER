from pathlib import Path

import pandas as pd

from src.utils.config import PROPERTYFINDER_FILE
from src.utils.logging import get_logger


logger = get_logger(__name__)


class DataIngestionError(Exception):
    """
    Raised when data ingestion fails.
    """


def validate_raw_file(file_path: Path) -> None:
    """
    Validate that the raw dataset exists and is a CSV file.
    """

    logger.info("Validating raw data file: %s", file_path)

    if not file_path.exists():
        raise DataIngestionError(
            f"Raw data file does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise DataIngestionError(
            f"Raw data path is not a file: {file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise DataIngestionError(
            f"Expected a CSV file, received: {file_path.suffix}"
        )

    logger.info("Raw data file validation successful.")


def load_propertyfinder_data(
    file_path: Path = PROPERTYFINDER_FILE,
) -> pd.DataFrame:
    """
    Load the PropertyFinder dataset into a pandas DataFrame.
    """

    validate_raw_file(file_path)

    logger.info("Loading PropertyFinder dataset...")

    try:
        dataframe = pd.read_csv(file_path)

    except Exception as exc:
        raise DataIngestionError(
            f"Failed to read dataset: {exc}"
        ) from exc

    if dataframe.empty:
        raise DataIngestionError(
            "Dataset was loaded successfully, but it is empty."
        )

    logger.info(
        "Dataset loaded successfully: %s rows, %s columns.",
        dataframe.shape[0],
        dataframe.shape[1],
    )

    logger.info(
        "Memory usage: %.2f MB",
        dataframe.memory_usage(deep=True).sum() / (1024 ** 2),
    )

    return dataframe


def get_dataset_summary(dataframe: pd.DataFrame) -> dict:
    """
    Return a basic summary of the ingested dataset.
    """

    return {
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "column_names": dataframe.columns.tolist(),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "missing_cells": int(dataframe.isna().sum().sum()),
    }


if __name__ == "__main__":
    from src.utils.logging import setup_logging

    setup_logging()

    df = load_propertyfinder_data()

    summary = get_dataset_summary(df)

    logger.info("Dataset summary:")
    logger.info("Rows: %s", summary["rows"])
    logger.info("Columns: %s", summary["columns"])
    logger.info("Duplicate rows: %s", summary["duplicate_rows"])
    logger.info("Missing cells: %s", summary["missing_cells"])