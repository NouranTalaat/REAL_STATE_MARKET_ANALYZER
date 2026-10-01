from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.api.exception_handlers import (
    app_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from src.api.exceptions import AppException
from src.api.lifespan import lifespan
from src.api.middleware import RequestContextMiddleware
from src.api.routes.market import router as market_router
from src.api.routes.pipeline import router as pipeline_router
from src.api.routes.predictions import (
    router as predictions_router,
)
from src.api.routes.properties import (
    router as properties_router,
)
from src.api.security import (
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from src.core.config import get_settings
from src.database import engine
from src.prediction_service import get_model_status
from src.utils.logging import setup_logging


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

setup_logging()

settings = get_settings()


# ============================================================
# OPENAPI TAG METADATA
# ============================================================

openapi_tags = [
    {
        "name": "System",
        "description": (
            "Application health, readiness, and service "
            "metadata endpoints."
        ),
    },
    {
        "name": "Market",
        "description": (
            "Real estate market analytics and market-level "
            "intelligence endpoints."
        ),
    },
    {
        "name": "Properties",
        "description": (
            "Property search and property-level data access "
            "endpoints."
        ),
    },
    {
        "name": "Predictions",
        "description": (
            "Machine learning valuation and model status "
            "endpoints."
        ),
    },
    {
        "name": "Pipeline",
        "description": (
            "Data pipeline monitoring and operational "
            "endpoints."
        ),
    },
]


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=settings.app_name,
    description=(
        "Production-grade API for Egyptian real estate "
        "market intelligence, analytics, property search, "
        "machine learning valuation, and pipeline monitoring.\n\n"
        "## API Version\n"
        "The current public API is exposed under `/api/v2`.\n\n"
        "## Architecture\n"
        "The API follows a layered architecture with dedicated "
        "routing, service, repository, machine learning, "
        "configuration, and operational components.\n\n"
        "## Security Baseline\n"
        "The service includes trusted-host validation, CORS "
        "configuration, request-size protection, security "
        "response headers, request IDs, and centralized "
        "exception handling."
    ),
    version=settings.api_version,
    debug=settings.debug,
    lifespan=lifespan,
    openapi_tags=openapi_tags,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ============================================================
# SECURITY MIDDLEWARE
# ============================================================

# Trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.trusted_hosts,
)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
    ],
    allow_headers=[
        "Accept",
        "Content-Type",
        "Authorization",
        "X-Request-ID",
    ],
)


# Request size protection
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_request_size_bytes=(
        settings.max_request_size_bytes
    ),
)


# Security headers
app.add_middleware(
    SecurityHeadersMiddleware,
    environment=settings.environment,
)


# ============================================================
# REQUEST CONTEXT MIDDLEWARE
# ============================================================

app.add_middleware(
    RequestContextMiddleware,
)


# ============================================================
# CENTRALIZED EXCEPTION HANDLERS
# ============================================================

app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler,
)

app.add_exception_handler(
    AppException,
    app_exception_handler,
)

app.add_exception_handler(
    Exception,
    unhandled_exception_handler,
)


# ============================================================
# API V2 ROUTES
# ============================================================

app.include_router(
    market_router,
    prefix=settings.api_prefix,
)

app.include_router(
    properties_router,
    prefix=settings.api_prefix,
)

app.include_router(
    predictions_router,
    prefix=settings.api_prefix,
)

app.include_router(
    pipeline_router,
    prefix=settings.api_prefix,
)


# ============================================================
# SYSTEM ENDPOINTS
# ============================================================

@app.get(
    "/",
    tags=["System"],
    summary="Service information",
    description=(
        "Returns basic metadata about the running "
        "Real Estate Market Intelligence API service."
    ),
)
def home():
    return {
        "message": (
            f"{settings.app_name} is running"
        ),
        "service": settings.service_name,
        "version": settings.api_version,
        "api_version": "v2",
        "environment": settings.environment,
        "status": "healthy",
    }


@app.get(
    "/health",
    tags=["System"],
    summary="Liveness health check",
    description=(
        "Lightweight liveness probe used to verify that "
        "the API process is running and able to serve "
        "HTTP requests."
    ),
)
def health_check():
    """
    Lightweight liveness probe.

    This endpoint intentionally does not check external
    dependencies. It only confirms that the application
    process is alive and able to serve requests.
    """

    return {
        "status": "healthy",
        "service": settings.service_name,
        "version": settings.api_version,
        "api_version": "v2",
    }


@app.get(
    "/ready",
    tags=["System"],
    summary="Service readiness check",
    description=(
        "Checks whether critical runtime dependencies "
        "required by the API are available.\n\n"
        "The readiness check verifies:\n"
        "- SQL Server database connectivity\n"
        "- Required machine learning model artifacts\n\n"
        "Returns HTTP 200 when all dependencies are ready "
        "and HTTP 503 otherwise."
    ),
)
def readiness_check():
    """
    Production readiness probe.

    Verifies critical runtime dependencies:
        - SQL Server database
        - Required ML model artifacts
    """

    checks = {
        "database": "unknown",
        "models": "unknown",
    }

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    try:
        with engine.connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

        checks["database"] = "ready"

    except Exception:
        checks["database"] = "unavailable"

    # --------------------------------------------------------
    # ML models
    # --------------------------------------------------------

    try:
        model_status = get_model_status()

        if not isinstance(
            model_status,
            dict,
        ):
            checks["models"] = "unavailable"

        else:
            sale_ready = model_status.get(
                "sale_model",
                {},
            ).get(
                "exists",
                False,
            )

            rent_ready = model_status.get(
                "rent_model",
                {},
            ).get(
                "exists",
                False,
            )

            checks["models"] = (
                "ready"
                if sale_ready and rent_ready
                else "unavailable"
            )

    except Exception:
        checks["models"] = "unavailable"

    # --------------------------------------------------------
    # Overall readiness
    # --------------------------------------------------------

    is_ready = all(
        value == "ready"
        for value in checks.values()
    )

    response = {
        "status": (
            "ready"
            if is_ready
            else "not_ready"
        ),
        "service": settings.service_name,
        "version": settings.api_version,
        "api_version": "v2",
        "environment": settings.environment,
        "checks": checks,
    }

    if not is_ready:
        return JSONResponse(
            status_code=503,
            content=response,
        )

    return response