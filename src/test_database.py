from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.core.config import get_settings


settings = get_settings()


engine: Engine = create_engine(
    settings.database_url,
    pool_pre_ping=settings.database_pool_pre_ping,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
)


def get_connection():
    """
    Create a database connection.

    The caller is responsible for closing the connection.
    """
    return engine.connect()


def test_connection() -> str:
    """
    Verify database connectivity and return the current database name.
    """
    with engine.connect() as connection:
        result = connection.execute(
            text(
                "SELECT DB_NAME() AS current_database;"
            )
        )

        return result.scalar()