from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not configured."
    )


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


def get_connection():
    return engine.connect()


def test_connection():
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT DB_NAME() AS current_database;")
        )

        return result.scalar()