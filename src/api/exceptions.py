from __future__ import annotations

from typing import Any


class AppException(Exception):
    """
    Base application exception.

    All expected application-level exceptions should inherit
    from this class so they can be handled centrally by FastAPI.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "APPLICATION_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details


class ValidationException(AppException):
    """Generic application validation error."""

    def __init__(
        self,
        message: str = "Request validation failed.",
        *,
        details: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="VALIDATION_ERROR",
            status_code=422,
            details=details,
        )


class ResourceNotFoundException(AppException):
    """Raised when an expected resource does not exist."""

    def __init__(
        self,
        message: str = "Requested resource was not found.",
        *,
        details: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="RESOURCE_NOT_FOUND",
            status_code=404,
            details=details,
        )


class DatabaseException(AppException):
    """Raised when a database operation cannot be completed."""

    def __init__(
        self,
        message: str = "A database operation failed.",
        *,
        details: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="DATABASE_ERROR",
            status_code=503,
            details=details,
        )


class ModelException(AppException):
    """Raised when required ML model artifacts are unavailable."""

    def __init__(
        self,
        message: str = "The prediction service is unavailable.",
        *,
        details: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="MODEL_ERROR",
            status_code=503,
            details=details,
        )


class PredictionException(AppException):
    """Raised when property valuation cannot be completed."""

    def __init__(
        self,
        message: str = "Prediction could not be completed.",
        *,
        details: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="PREDICTION_ERROR",
            status_code=422,
            details=details,
        )