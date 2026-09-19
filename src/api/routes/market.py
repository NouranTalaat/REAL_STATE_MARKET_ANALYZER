from fastapi import APIRouter, HTTPException

from src.analytics_service import (
    get_market_summary,
    get_robust_metrics,
    get_location_intelligence,
    get_city_market_profile,
    get_opportunity_signals,
    get_data_quality_status,
    get_market_overview,
)


router = APIRouter(
    prefix="/analytics",
    tags=["Market Analytics"],
)


@router.get("/overview")
def market_overview():
    """
    Return a unified market intelligence overview.
    """

    try:
        return get_market_overview()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build market overview: {str(e)}",
        )


@router.get("/market-summary")
def market_summary():

    try:
        df = get_market_summary()

        return {
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load market summary: {str(e)}",
        )


@router.get("/robust-metrics")
def robust_metrics():

    try:
        df = get_robust_metrics()

        return {
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load robust metrics: {str(e)}",
        )


@router.get("/location-intelligence")
def location_intelligence():

    try:
        df = get_location_intelligence()

        return {
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load location intelligence: {str(e)}",
        )


@router.get("/city-market-profile")
def city_market_profile():

    try:
        df = get_city_market_profile()

        return {
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load city market profile: {str(e)}",
        )


@router.get("/opportunity-signals")
def opportunity_signals():

    try:
        df = get_opportunity_signals()

        return {
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load opportunity signals: {str(e)}",
        )


@router.get("/data-quality")
def data_quality():

    try:
        df = get_data_quality_status()

        return {
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load data quality status: {str(e)}",
        )