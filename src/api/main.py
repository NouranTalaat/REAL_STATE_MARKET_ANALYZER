from fastapi import FastAPI

from src.api.routes.market import (
    router as market_router,
)
from src.api.routes.pipeline import (
    router as pipeline_router,
)
from src.api.routes.predictions import (
    router as predictions_router,
)
from src.api.routes.properties import (
    router as properties_router,
)


app = FastAPI(
    title="Real Estate Market Intelligence API",
    description=(
        "Production-oriented API for Egyptian "
        "real estate market intelligence, "
        "analytics, property search, "
        "machine learning model serving, "
        "and production pipeline monitoring."
    ),
    version="2.0.0",
)


app.include_router(
    market_router
)

app.include_router(
    properties_router
)

app.include_router(
    predictions_router
)

app.include_router(
    pipeline_router
)


@app.get(
    "/",
    tags=["System"],
)
def home():

    return {
        "message": (
            "Real Estate Market Intelligence "
            "API is running"
        ),
        "version": "2.0.0",
        "status": "healthy",
    }


@app.get(
    "/health",
    tags=["System"],
)
def health_check():

    return {
        "status": "healthy",
        "service": (
            "real-estate-market-"
            "intelligence-api"
        ),
        "version": "2.0.0",
    }