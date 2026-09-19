from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import text

from src.database import engine
from src.utils.logging import get_logger


logger = get_logger(__name__)


PIPELINE_RUNS_TABLE = "pipeline_runs"


def ensure_monitoring_table() -> None:
    """
    Create the pipeline monitoring table if it does not exist.
    """

    query = text(
        f"""
        IF OBJECT_ID(
            'analytics.{PIPELINE_RUNS_TABLE}',
            'U'
        ) IS NULL
        BEGIN
            CREATE TABLE analytics.{PIPELINE_RUNS_TABLE} (
                run_id INT IDENTITY(1,1) PRIMARY KEY,
                pipeline_name NVARCHAR(200) NOT NULL,
                status NVARCHAR(50) NOT NULL,
                started_at DATETIME2 NOT NULL,
                completed_at DATETIME2 NULL,
                duration_seconds FLOAT NULL,
                source_rows INT NULL,
                processed_rows INT NULL,
                inserted_rows INT NULL,
                error_message NVARCHAR(MAX) NULL
            );
        END;
        """
    )

    with engine.begin() as connection:
        connection.execute(query)


def start_pipeline_run(pipeline_name: str) -> int:
    """
    Create a new pipeline run record.
    """

    started_at = datetime.now(timezone.utc).replace(
        tzinfo=None
    )

    query = text(
        f"""
        INSERT INTO analytics.{PIPELINE_RUNS_TABLE} (
            pipeline_name,
            status,
            started_at
        )
        OUTPUT INSERTED.run_id
        VALUES (
            :pipeline_name,
            :status,
            :started_at
        );
        """
    )

    with engine.begin() as connection:
        result = connection.execute(
            query,
            {
                "pipeline_name": pipeline_name,
                "status": "RUNNING",
                "started_at": started_at,
            },
        )

        run_id = result.scalar_one()

    logger.info(
        "Pipeline run started: run_id=%s",
        run_id,
    )

    return int(run_id)


def complete_pipeline_run(
    run_id: int,
    status: str,
    started_at: datetime,
    source_rows: int,
    processed_rows: int,
    inserted_rows: int,
    error_message: str | None = None,
) -> None:
    """
    Complete a pipeline run and store execution metrics.
    """

    completed_at = datetime.now(timezone.utc).replace(
        tzinfo=None
    )

    duration_seconds = (
        completed_at - started_at
    ).total_seconds()

    query = text(
        f"""
        UPDATE analytics.{PIPELINE_RUNS_TABLE}
        SET
            status = :status,
            completed_at = :completed_at,
            duration_seconds = :duration_seconds,
            source_rows = :source_rows,
            processed_rows = :processed_rows,
            inserted_rows = :inserted_rows,
            error_message = :error_message
        WHERE run_id = :run_id;
        """
    )

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "run_id": run_id,
                "status": status,
                "completed_at": completed_at,
                "duration_seconds": duration_seconds,
                "source_rows": source_rows,
                "processed_rows": processed_rows,
                "inserted_rows": inserted_rows,
                "error_message": error_message,
            },
        )

    logger.info(
        "Pipeline run completed: run_id=%s status=%s duration=%.2fs",
        run_id,
        status,
        duration_seconds,
    )


def get_latest_pipeline_run() -> dict | None:
    """
    Return the latest pipeline execution.
    """

    query = text(
        f"""
        SELECT TOP 1
            run_id,
            pipeline_name,
            status,
            started_at,
            completed_at,
            duration_seconds,
            source_rows,
            processed_rows,
            inserted_rows,
            error_message
        FROM analytics.{PIPELINE_RUNS_TABLE}
        ORDER BY run_id DESC;
        """
    )

    with engine.connect() as connection:
        result = connection.execute(query).mappings().first()

    return dict(result) if result else None


def get_pipeline_statistics() -> dict:
    """
    Return high-level pipeline execution statistics.
    """

    query = text(
        f"""
        SELECT
            COUNT(*) AS total_runs,
            SUM(
                CASE
                    WHEN status = 'SUCCESS' THEN 1
                    ELSE 0
                END
            ) AS successful_runs,
            SUM(
                CASE
                    WHEN status = 'FAILED' THEN 1
                    ELSE 0
                END
            ) AS failed_runs,
            MAX(completed_at) AS last_completed_at
        FROM analytics.{PIPELINE_RUNS_TABLE};
        """
    )

    with engine.connect() as connection:
        result = connection.execute(query).mappings().first()

    return {
        "total_runs": int(result["total_runs"] or 0),
        "successful_runs": int(result["successful_runs"] or 0),
        "failed_runs": int(result["failed_runs"] or 0),
        "last_completed_at": result["last_completed_at"],
    }
