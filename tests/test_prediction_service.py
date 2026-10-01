from __future__ import annotations

import pytest

from src.api.exceptions import ModelException, PredictionException
from src.api.services.prediction_service import PredictionService


class FakeValuationEngine:
    def predict(
        self,
        *,
        property_data,
        market_type,
    ):
        return {
            "market_type": market_type,
            "market_value": "Residential for Sale",
            "property_type": property_data["property_type"],
            "model_resolution": "standard",
            "estimated_value_egp": 1_000_000.0,
            "confidence": {
                "score": 0.8,
                "level": "high",
                "model_agreement": 0.9,
                "model_quality": 0.8,
                "reference_r2": 0.7,
            },
            "model_predictions": [],
        }


class FailingPredictionEngine:
    def predict(
        self,
        *,
        property_data,
        market_type,
    ):
        raise ValueError("area_value cannot be negative.")


class MissingModelEngine:
    def predict(
        self,
        *,
        property_data,
        market_type,
    ):
        raise FileNotFoundError("model missing")


def test_prediction_service_returns_valuation_result():
    service = PredictionService(
        valuation_engine=FakeValuationEngine()
    )

    result = service.valuate(
        property_data={
            "property_type": "Apartment",
            "area_value": 150,
        },
        market_type="sale",
    )

    assert result["market_type"] == "sale"
    assert result["property_type"] == "Apartment"
    assert result["estimated_value_egp"] == 1_000_000.0


def test_prediction_service_translates_validation_error():
    service = PredictionService(
        valuation_engine=FailingPredictionEngine()
    )

    with pytest.raises(PredictionException) as exc_info:
        service.valuate(
            property_data={
                "property_type": "Apartment",
                "area_value": -10,
            },
            market_type="sale",
        )

    assert exc_info.value.code == "PREDICTION_ERROR"
    assert "area_value" in exc_info.value.message


def test_prediction_service_translates_missing_model_error():
    service = PredictionService(
        valuation_engine=MissingModelEngine()
    )

    with pytest.raises(ModelException) as exc_info:
        service.valuate(
            property_data={
                "property_type": "Apartment",
                "area_value": 150,
            },
            market_type="sale",
        )

    assert exc_info.value.code == "MODEL_ERROR"
    assert exc_info.value.status_code == 503