from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.pipeline.monitoring import (
    get_latest_pipeline_run,
    get_pipeline_statistics,
)


router = APIRouter(
    prefix="/pipeline",
    tags=["Pipeline Monitoring"],
)


def serialize_datetime(value):

    if value is None:
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return value


@router.get("/status")
def pipeline_status():

    try:
        latest_run = get_latest_pipeline_run()

        if latest_run is None:
            return {
                "status": "NO_RUNS",
                "data": None,
            }

        latest_run = dict(latest_run)

        latest_run["started_at"] = serialize_datetime(
            latest_run.get("started_at")
        )

        latest_run["completed_at"] = serialize_datetime(
            latest_run.get("completed_at")
        )

        return {
            "status": "OK",
            "data": latest_run,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to retrieve "
                f"pipeline status: {exc}"
            ),
        )


@router.get("/statistics")
def pipeline_statistics():

    try:

        statistics = get_pipeline_statistics()

        statistics["last_completed_at"] = (
            serialize_datetime(
                statistics.get("last_completed_at")
            )
        )

        return {
            "status": "OK",
            "data": statistics,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to retrieve "
                f"pipeline statistics: {exc}"
            ),
        )