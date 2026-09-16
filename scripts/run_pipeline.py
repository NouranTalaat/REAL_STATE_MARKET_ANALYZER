import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data.ingestion import (
    get_dataset_summary,
    load_propertyfinder_data,
)

from src.utils.config import ensure_project_directories
from src.utils.logging import get_logger, setup_logging


logger = get_logger(__name__)


def main() -> None:
    """
    Run the data ingestion pipeline.
    """

    setup_logging()

    logger.info("=" * 70)
    logger.info("REAL ESTATE MARKET INTELLIGENCE PLATFORM")
    logger.info("DATA INGESTION PIPELINE")
    logger.info("=" * 70)

    ensure_project_directories()

    dataframe = load_propertyfinder_data()

    summary = get_dataset_summary(dataframe)

    logger.info("-" * 70)
    logger.info("INGESTION SUMMARY")
    logger.info("-" * 70)

    logger.info("Rows: %s", summary["rows"])
    logger.info("Columns: %s", summary["columns"])
    logger.info("Duplicate rows: %s", summary["duplicate_rows"])
    logger.info("Missing cells: %s", summary["missing_cells"])

    logger.info("-" * 70)
    logger.info("First 5 rows:")
    logger.info("\n%s", dataframe.head().to_string())

    logger.info("-" * 70)
    logger.info("Data ingestion completed successfully.")
    logger.info("-" * 70)


if __name__ == "__main__":
    main()