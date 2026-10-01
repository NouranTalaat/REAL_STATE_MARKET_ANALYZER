from __future__ import annotations

from datetime import datetime, timezone
from typing import Generic, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T")


class ResponseMeta(BaseModel):
    """
    Metadata returned with every standardized API response.
    """

    request_id: str = Field(
        ...,
        description="Unique identifier for tracing the request.",
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the response was generated.",
    )

    version: str = Field(
        default="2.0.0",
        description="API version.",
    )


class ErrorDetail(BaseModel):
    """
    Structured API error information.
    """

    code: str = Field(
        ...,
        description="Machine-readable error code.",
    )

    message: str = Field(
        ...,
        description="Human-readable error message.",
    )

    details: dict | list | None = Field(
        default=None,
        description="Additional structured error information.",
    )


class APIResponse(BaseModel, Generic[T]):
    """
    Standard response envelope for successful and failed API responses.
    """

    success: bool = Field(
        ...,
        description="Whether the request completed successfully.",
    )

    data: T | None = Field(
        default=None,
        description="Response payload.",
    )

    meta: ResponseMeta

    error: ErrorDetail | None = Field(
        default=None,
        description="Structured error information.",
    )


class PaginationMeta(ResponseMeta):
    """
    Metadata for paginated endpoints.
    """

    page: int = Field(
        default=1,
        ge=1,
    )

    page_size: int = Field(
        default=20,
        ge=1,
    )

    total: int | None = Field(
        default=None,
        ge=0,
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Standard response envelope for paginated resources.
    """

    success: bool = True

    data: list[T]

    meta: PaginationMeta

    error: ErrorDetail | None = None