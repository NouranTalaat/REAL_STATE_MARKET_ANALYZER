from __future__ import annotations

from typing import Any

from src.api.exceptions import (
    ModelException,
    PredictionException,
)
from src.ml.valuation_engine import RealEstateValuationEngine


class PredictionService:
    """
    Application service responsible for orchestrating
    real-estate valuation requests.

    This layer sits between the API route and the
    underlying valuation engine.

    Responsibilities:
    - Validate service-level prerequisites.
    - Orchestrate the valuation operation.
    - Translate low-level failures into application exceptions.
    - Keep HTTP concerns outside the business layer.
    """

    def __init__(
        self,
        valuation_engine: RealEstateValuationEngine | None = None,
    ) -> None:
        self.valuation_engine = (
            valuation_engine
            if valuation_engine is not None
            else RealEstateValuationEngine()
        )

    def valuate(
        self,
        *,
        property_data: dict[str, Any],
        market_type: str,
    ) -> dict[str, Any]:
        """
        Execute a real-estate valuation.

        Parameters
        ----------
        property_data:
            Property features required by the valuation engine.

        market_type:
            Supported market type, such as sale or rent.

        Returns
        -------
        dict[str, Any]
            Business-level valuation result.

        Raises
        ------
        PredictionException
            When the input cannot be processed by the
            valuation engine.

        ModelException
            When a required model artifact is unavailable.
        """

        try:
            return self.valuation_engine.predict(
                property_data=property_data,
                market_type=market_type,
            )

        except FileNotFoundError as exc:
            raise ModelException(
                message="Required prediction model is unavailable."
            ) from exc

        except (ValueError, TypeError) as exc:
            raise PredictionException(
                message=str(exc),
            ) from exc