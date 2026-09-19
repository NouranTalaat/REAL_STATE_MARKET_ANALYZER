from __future__ import annotations

from datetime import datetime, timezone

from src.data.ingestion import load_propertyfinder_data
from src.data.processing import transform_property_data
from src.data.validation import (
    build_quality_report,
)
from src.pipeline.deduplication import (
    deduplicate_listings,
    get_duplicate_statistics,
)
from src.pipeline.incremental_loader import (
    get_latest_loaded_at,
    get_target_row_count,
    load_new_listings,
)
from src.pipeline.monitoring import (
    complete_pipeline_run,
    ensure_monitoring_table,
    start_pipeline_run,
)
from src.utils.logging import (
    get_logger,
    setup_logging,
)


logger = get_logger(__name__)


PIPELINE_NAME = (
    "real_estate_market_pipeline"
)


def run_pipeline() -> dict:

    ensure_monitoring_table()

    run_id = start_pipeline_run(
        PIPELINE_NAME
    )

    started_at = (
        datetime.now(timezone.utc)
        .replace(tzinfo=None)
    )

    source_rows = 0
    processed_rows = 0
    inserted_rows = 0
    rejected_rows = 0

    try:

        logger.info("=" * 70)
        logger.info(
            "PHASE 5 DATA PIPELINE STARTED"
        )
        logger.info("=" * 70)

        # -------------------------------------------------
        # 1. INGESTION
        # -------------------------------------------------

        raw_dataframe = (
            load_propertyfinder_data()
        )

        source_rows = len(
            raw_dataframe
        )

        logger.info(
            "Source dataset loaded: %s rows.",
            source_rows,
        )

        # -------------------------------------------------
        # 2. ETL / TRANSFORMATION
        # -------------------------------------------------

        processed_dataframe = (
            transform_property_data(
                raw_dataframe
            )
        )

        logger.info(
            "Transformation completed: %s rows.",
            len(processed_dataframe),
        )

        # -------------------------------------------------
        # 3. QUALITY VALIDATION
        # -------------------------------------------------

        quality_report = build_quality_report(
            raw_dataframe=raw_dataframe,
            processed_dataframe=processed_dataframe,
        )

        if (
            quality_report["required_columns"]["status"]
            != "PASS"
        ):
            raise ValueError(
                "Required-column validation failed: "
                + str(
                    quality_report[
                        "required_columns"
                    ]["missing"]
                )
            )

        logger.info(
            "Required-column validation: PASS"
        )

        listing_id_quality = quality_report[
            "listing_id_validation"
        ]

        if listing_id_quality["missing_column"]:
            raise ValueError(
                "listing_id column is missing."
            )

        logger.info(
            "Listing ID validation: %s",
            listing_id_quality["status"],
        )

        # -------------------------------------------------
        # 4. DEDUPLICATION / REJECTION
        # -------------------------------------------------

        duplicate_statistics = (
            get_duplicate_statistics(
                processed_dataframe
            )
        )

        rejected_rows = duplicate_statistics[
            "null_listing_ids"
        ]

        logger.info(
            "Duplicate statistics before "
            "pipeline deduplication: %s",
            duplicate_statistics,
        )

        processed_dataframe = (
            deduplicate_listings(
                processed_dataframe
            )
        )

        processed_rows = len(
            processed_dataframe
        )

        logger.info(
            "Valid processed rows after "
            "deduplication: %s",
            processed_rows,
        )

        if rejected_rows > 0:
            logger.warning(
                "Rejected records due to missing "
                "listing_id: %s",
                rejected_rows,
            )

        # -------------------------------------------------
        # 5. INCREMENTAL SQL LOAD
        # -------------------------------------------------

        load_result = load_new_listings(
            processed_dataframe
        )

        inserted_rows = load_result[
            "inserted_rows"
        ]

        logger.info(
            "Incremental load result: %s",
            load_result,
        )

        # -------------------------------------------------
        # 6. FINAL DATABASE STATE
        # -------------------------------------------------

        target_row_count = (
            get_target_row_count()
        )

        latest_loaded_at = (
            get_latest_loaded_at()
        )

        # -------------------------------------------------
        # 7. MONITORING
        # -------------------------------------------------

        complete_pipeline_run(
            run_id=run_id,
            status="SUCCESS",
            started_at=started_at,
            source_rows=source_rows,
            processed_rows=processed_rows,
            inserted_rows=inserted_rows,
        )

        result = {
            "status": "SUCCESS",
            "run_id": run_id,
            "pipeline": PIPELINE_NAME,
            "source_rows": source_rows,
            "processed_rows": processed_rows,
            "rejected_rows": rejected_rows,
            "inserted_rows": inserted_rows,
            "target_row_count": target_row_count,
            "latest_loaded_at": latest_loaded_at,
            "duplicate_statistics":
                duplicate_statistics,
            "quality_status":
                quality_report[
                    "required_columns"
                ]["status"],
            "listing_id_quality":
                listing_id_quality,
        }

        logger.info("=" * 70)
        logger.info(
            "PIPELINE COMPLETED SUCCESSFULLY"
        )
        logger.info(
            "Run ID: %s",
            run_id,
        )
        logger.info(
            "Source rows: %s",
            source_rows,
        )
        logger.info(
            "Processed rows: %s",
            processed_rows,
        )
        logger.info(
            "Rejected rows: %s",
            rejected_rows,
        )
        logger.info(
            "Inserted rows: %s",
            inserted_rows,
        )
        logger.info(
            "SQL rows: %s",
            target_row_count,
        )
        logger.info("=" * 70)

        return result

    except Exception as exc:

        logger.exception(
            "Pipeline execution failed."
        )

        complete_pipeline_run(
            run_id=run_id,
            status="FAILED",
            started_at=started_at,
            source_rows=source_rows,
            processed_rows=processed_rows,
            inserted_rows=inserted_rows,
            error_message=str(exc),
        )

        raise


if __name__ == "__main__":

    setup_logging()

    result = run_pipeline()

    print("\nPIPELINE RESULT")
    print("=" * 70)

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )