from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.ingestion import load_propertyfinder_data
from src.data.processing import (
    save_processed_data,
    save_quality_report,
    transform_property_data,
)
from src.data.validation import (
    build_quality_report,
    validate_numeric_ranges,
    validate_required_columns,
)
from src.utils.config import ensure_project_directories
from src.utils.logging import get_logger, setup_logging


logger = get_logger(__name__)


def main() -> None:
    setup_logging()

    logger.info("=" * 80)
    logger.info("REAL ESTATE MARKET INTELLIGENCE PLATFORM")
    logger.info("PHASE 2 - ETL + DATA QUALITY")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # 1. Project directories
    # ------------------------------------------------------------------

    ensure_project_directories()

    # ------------------------------------------------------------------
    # 2. Extract
    # ------------------------------------------------------------------

    logger.info("-" * 80)
    logger.info("STEP 1: EXTRACT")
    logger.info("-" * 80)

    raw_dataframe = load_propertyfinder_data()

    logger.info(
        "Raw dataset: %s rows × %s columns",
        raw_dataframe.shape[0],
        raw_dataframe.shape[1],
    )

    # ------------------------------------------------------------------
    # 3. Validate raw data
    # ------------------------------------------------------------------

    logger.info("-" * 80)
    logger.info("STEP 2: VALIDATE RAW DATA")
    logger.info("-" * 80)

    missing_required_columns = validate_required_columns(raw_dataframe)

    if missing_required_columns:
        raise ValueError(
            "Required columns are missing: "
            + ", ".join(missing_required_columns)
        )

    numeric_issues = validate_numeric_ranges(raw_dataframe)

    logger.info(
        "Required-column validation: PASS"
    )

    logger.info(
        "Duplicate rows: %s",
        raw_dataframe.duplicated().sum(),
    )

    logger.info(
        "Missing cells: %s",
        raw_dataframe.isna().sum().sum(),
    )

    logger.info(
        "Numeric quality issues: %s",
        numeric_issues,
    )

    # ------------------------------------------------------------------
    # 4. Transform
    # ------------------------------------------------------------------

    logger.info("-" * 80)
    logger.info("STEP 3: TRANSFORM")
    logger.info("-" * 80)

    processed_dataframe = transform_property_data(raw_dataframe)

    # ------------------------------------------------------------------
    # 5. Final quality report
    # ------------------------------------------------------------------

    logger.info("-" * 80)
    logger.info("STEP 4: FINAL DATA QUALITY CHECK")
    logger.info("-" * 80)

    quality_report = build_quality_report(
        raw_dataframe=raw_dataframe,
        processed_dataframe=processed_dataframe,
    )

    logger.info(
        "Processed rows: %s",
        processed_dataframe.shape[0],
    )

    logger.info(
        "Processed columns: %s",
        processed_dataframe.shape[1],
    )

    logger.info(
        "Processed duplicate rows: %s",
        processed_dataframe.duplicated().sum(),
    )

    # ------------------------------------------------------------------
    # 6. Save processed data
    # ------------------------------------------------------------------

    logger.info("-" * 80)
    logger.info("STEP 5: SAVE PROCESSED DATA")
    logger.info("-" * 80)

    processed_path = save_processed_data(processed_dataframe)

    # ------------------------------------------------------------------
    # 7. Save quality report
    # ------------------------------------------------------------------

    report_path = save_quality_report(quality_report)

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("PHASE 2 COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    logger.info(
        "Raw dataset:       %s rows × %s columns",
        raw_dataframe.shape[0],
        raw_dataframe.shape[1],
    )

    logger.info(
        "Processed dataset: %s rows × %s columns",
        processed_dataframe.shape[0],
        processed_dataframe.shape[1],
    )

    logger.info(
        "Processed file:    %s",
        processed_path,
    )

    logger.info(
        "Quality report:    %s",
        report_path,
    )

    logger.info("=" * 80)


if __name__ == "__main__":
    main()