from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    """
    Strong API contract for real-estate valuation requests.

    The property payload remains a dictionary because the existing
    valuation engine expects a dictionary and owns the detailed
    business validation rules.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    market_type: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Real estate market: sale or rent.",
        examples=["sale"],
    )

    property_data: dict[str, Any] = Field(
        ...,
        min_length=1,
        description="Property features used for valuation.",
    )

    @field_validator("market_type")
    @classmethod
    def normalize_market_type(
        cls,
        value: str,
    ) -> str:
        return value.strip().lower()

    @field_validator("property_data")
    @classmethod
    def validate_property_data(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:

        # Keep this layer focused on API-level contract checks.
        # Detailed business validation remains inside the
        # RealEstateValuationEngine.

        if not value:
            raise ValueError(
                "property_data cannot be empty."
            )

        return value


class ConfidenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float
    level: str
    model_agreement: float
    model_quality: float
    reference_r2: float


class ModelPredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prediction_egp: float
    weight: float


class PredictionResult(BaseModel):
    """
    Business-level prediction payload.

    This represents the actual valuation result and is
    intentionally separated from the API response envelope.
    """

    model_config = ConfigDict(extra="forbid")

    market_type: str
    market_value: str
    property_type: str
    model_resolution: str
    estimated_value_egp: float
    confidence: ConfidenceResponse
    model_predictions: list[ModelPredictionResponse]