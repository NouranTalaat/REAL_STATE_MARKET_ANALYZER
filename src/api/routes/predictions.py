from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.schemas.common import APIResponse
from src.api.schemas.prediction import (
    PredictionRequest,
    PredictionResult,
)
from src.api.services.prediction_service import PredictionService
from src.core.config import get_settings
from src.prediction_service import get_model_status


settings = get_settings()


router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
)


prediction_service = PredictionService()


@router.get(
    "/status",
    response_model=APIResponse[dict],
)
def prediction_status(
    request: Request,
):
    result = get_model_status()

    return APIResponse(
        success=True,
        data=result,
        meta={
            "request_id": request.state.request_id,
            "version": settings.api_version,
        },
    )


@router.post(
    "/valuate",
    response_model=APIResponse[PredictionResult],
)
def valuate_property(
    request: PredictionRequest,
    http_request: Request,
):
    result = prediction_service.valuate(
        property_data=request.property_data,
        market_type=request.market_type,
    )

    return APIResponse(
        success=True,
        data=result,
        meta={
            "request_id": http_request.state.request_id,
            "version": settings.api_version,
        },
    )