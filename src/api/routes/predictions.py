from fastapi import APIRouter, HTTPException

from src.prediction_service import get_model_status


router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
)


@router.get("/status")
def prediction_status():
    try:
        return get_model_status()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check prediction models: {str(e)}",
        )