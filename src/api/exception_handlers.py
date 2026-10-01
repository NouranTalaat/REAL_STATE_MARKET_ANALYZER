from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.exceptions import AppException
from src.api.schemas.common import APIResponse
from src.api.schemas.errors import build_error_response
from src.core.config import get_settings
from src.utils.logging import get_logger, log_structured


logger = get_logger(__name__)

settings = get_settings()


def _request_id(request: Request) -> str:
    return getattr(
        request.state,
        "request_id",
        "unknown",
    )


def _json_error_response(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | list[Any] | None = None,
) -> JSONResponse:
    """
    Build the single standardized error response used by the API.
    """

    request_id = _request_id(request)

    response: APIResponse[None] = build_error_response(
        code=code,
        message=message,
        request_id=request_id,
        version=settings.api_version,
        details=details,
    )

    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(mode="json"),
        headers={
            "X-Request-ID": request_id,
        },
    )


async def app_exception_handler(
    request: Request,
    exc: AppException,
) -> JSONResponse:
    """
    Handle all expected application exceptions centrally.
    """

    request_id = _request_id(request)

    log_structured(
        logger,
        logging.WARNING,
        "Application exception",
        service=settings.service_name,
        environment=settings.environment,
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        error_code=exc.code,
        status_code=exc.status_code,
    )

    return _json_error_response(
        request=request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Handle FastAPI/Pydantic request validation errors.
    """

    request_id = _request_id(request)

    details = {
        "validation_errors": exc.errors(),
    }

    log_structured(
        logger,
        logging.WARNING,
        "Request validation failed",
        service=settings.service_name,
        environment=settings.environment,
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        error_code="REQUEST_VALIDATION_ERROR",
        status_code=422,
    )

    return _json_error_response(
        request=request,
        status_code=422,
        code="REQUEST_VALIDATION_ERROR",
        message="Request validation failed.",
        details=details,
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Final safety net for unexpected exceptions.

    Internal exception details are intentionally not exposed
    to API clients.
    """

    request_id = _request_id(request)

    log_structured(
        logger,
        logging.ERROR,
        "Unhandled application exception",
        service=settings.service_name,
        environment=settings.environment,
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        error_code="INTERNAL_SERVER_ERROR",
        status_code=500,
        exception_type=type(exc).__name__,
    )

    logger.exception(
        "Unhandled exception traceback"
    )

    return _json_error_response(
        request=request,
        status_code=500,
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected internal error occurred.",
    )