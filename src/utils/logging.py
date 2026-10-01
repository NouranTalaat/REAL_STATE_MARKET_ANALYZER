from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class StructuredFormatter(logging.Formatter):
    """
    Format application logs as structured JSON.

    JSON logs are easier to consume by:
    - Docker
    - Cloud logging systems
    - ELK / OpenSearch
    - AWS CloudWatch
    - log aggregation platforms
    """

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(
            record.created,
            tz=timezone.utc,
        ).isoformat()

        log_data: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        structured_fields = getattr(
            record,
            "structured_fields",
            None,
        )

        if isinstance(structured_fields, dict):
            log_data.update(structured_fields)

        if record.exc_info:
            log_data["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(
            log_data,
            ensure_ascii=False,
            default=str,
        )


def setup_logging() -> None:
    """
    Configure application-wide structured logging.
    """

    handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(
        StructuredFormatter()
    )

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger.
    """

    return logging.getLogger(name)


def log_structured(
    logger: logging.Logger,
    level: int,
    message: str,
    **fields: Any,
) -> None:
    """
    Write a structured log entry.

    Example:

        log_structured(
            logger,
            logging.INFO,
            "Request completed",
            request_id="abc",
            method="GET",
            status_code=200,
        )
    """

    logger.log(
        level,
        message,
        extra={
            "structured_fields": fields,
        },
    )