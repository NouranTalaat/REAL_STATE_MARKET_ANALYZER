from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from src.database import engine
from src.prediction_service import validate_model_artifacts


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown lifecycle.

    Startup:
        - Validate critical configuration dependencies.
        - Verify database connectivity.
        - Verify required ML artifacts.

    Shutdown:
        - Dispose SQLAlchemy connection pool.
    """

    logger.info(
        "Starting Real Estate Market Intelligence API..."
    )

    # --------------------------------------------------------
    # Database startup validation
    # --------------------------------------------------------

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        logger.info(
            "Database connectivity check: OK"
        )

    except Exception:
        logger.exception(
            "Database connectivity check failed. "
            "Application will start, but readiness may remain unavailable."
        )

    # --------------------------------------------------------
    # ML artifact validation
    # --------------------------------------------------------

    try:
        validate_model_artifacts()

        logger.info(
            "ML model artifact validation: OK"
        )

    except Exception:
        logger.exception(
            "ML model artifact validation failed. "
            "Application will start, but readiness may remain unavailable."
        )

    logger.info(
        "Real Estate Market Intelligence API startup completed."
    )

    try:
        yield

    finally:
        logger.info(
            "Shutting down Real Estate Market Intelligence API..."
        )

        engine.dispose()

        logger.info(
            "Database connection pool disposed."
        )