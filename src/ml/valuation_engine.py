
"""
REAL ESTATE MARKET INTELLIGENCE PLATFORM
SPECIALIZED REAL ESTATE VALUATION ENGINE

Hierarchical valuation architecture:

Market
    -> Qualified Property Type
        -> Ensemble
            -> CatBoost
            -> ExtraTrees
            -> RandomForest

Fallback:
Qualified Specialized Model -> Market Model

Important:
Only specialized segments that demonstrated acceptable
segment-level test performance are eligible for routing.

The inference layer also applies strict input validation
to prevent target leakage and invalid predictions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.ml.features import build_features


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
    / "valuation_engine"
)


# ======================================================================
# MARKET VALUES
# ======================================================================

MARKET_VALUES = {
    "sale": "Residential for Sale",
    "rent": "Residential for Rent",
}


# ======================================================================
# STRICT INPUT VALIDATION
# ======================================================================

# These fields contain the target or information derived directly
# from the target and therefore MUST NOT be supplied during inference.

FORBIDDEN_INPUT_FIELDS = {
    "price_egp",
    "price_per_sqm",
    "target",
    "log_price",
    "log_target",
}


# Fields that cannot logically be negative.

NON_NEGATIVE_FIELDS = {
    "area_value",
    "bedrooms",
    "bathrooms",
    "images_count",
    "listing_age_days",
}


# Fields that must be strictly greater than zero.

POSITIVE_FIELDS = {
    "area_value",
}


# Geographic boundaries.

LATITUDE_MIN = -90.0
LATITUDE_MAX = 90.0

LONGITUDE_MIN = -180.0
LONGITUDE_MAX = 180.0


# ======================================================================
# QUALIFIED SPECIALIZED ROUTING POLICY
# ======================================================================

#
# These segments were selected from the completed Phase 7 benchmark.
#
# Sale:
#   Chalet       R² = 0.6518
#   Townhouse    R² = 0.6800
#   Twin House   R² = 0.5292
#
# Rent:
#   Duplex       R² = 0.5037
#   Penthouse    R² = 0.6625
#
# Other trained specialized models remain available on disk but are
# intentionally NOT used for production routing.
#

QUALIFIED_SPECIALIZED_SEGMENTS = {
    "sale": {
        "chalet",
        "townhouse",
        "twin house",
    },
    "rent": {
        "duplex",
        "penthouse",
    },
}


class RealEstateValuationEngine:
    """
    Production valuation engine.

    Resolution order:

    1. Qualified specialized market + property_type ensemble
    2. Market-level ensemble
    3. Error if no model exists

    The engine never routes to a specialized model simply because
    the model exists. The property type must be explicitly qualified.

    Strict validation is applied before feature generation to prevent
    target leakage and invalid inference inputs.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        models_dir: Path | None = None,
    ):

        self.models_dir = (
            models_dir
            if models_dir is not None
            else MODELS_DIR
        )

        self.metadata = self._load_metadata()

    # ==================================================================
    # METADATA
    # ==================================================================

    def _load_metadata(self) -> dict[str, Any]:

        metadata_path = (
            self.models_dir
            / "engine_metadata.json"
        )

        if not metadata_path.exists():

            raise FileNotFoundError(
                "Valuation engine metadata not found: "
                f"{metadata_path}"
            )

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    # ==================================================================
    # MODEL LOADING
    # ==================================================================

    def _load_model(
        self,
        model_path: str,
    ):

        path = (
            self.models_dir
            / model_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        return joblib.load(path)

    # ==================================================================
    # STRICT INPUT VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_finite_number(
        value: Any,
        field_name: str,
    ) -> None:

        """
        Validate that a numeric value is finite.

        None is treated as a missing optional value and is allowed.
        """

        if value is None:

            return

        try:

            numeric_value = float(value)

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                f"{field_name} must be numeric."
            )

        if not math.isfinite(
            numeric_value
        ):

            raise ValueError(
                f"{field_name} must be finite "
                "(not NaN or infinite)."
            )

    def _validate_property_input(
        self,
        property_data: dict[str, Any],
        market: str,
    ) -> None:

        """
        Strict validation for inference inputs.

        Goals:

        - Prevent target leakage.
        - Reject invalid numeric values.
        - Reject impossible geographic values.
        - Validate required property_type.
        - Validate market consistency.
        """

        if not isinstance(
            property_data,
            dict,
        ):

            raise TypeError(
                "property_data must be a dictionary."
            )

        if not property_data:

            raise ValueError(
                "property_data cannot be empty."
            )

        # --------------------------------------------------------------
        # TARGET / LEAKAGE PROTECTION
        # --------------------------------------------------------------

        forbidden_fields = sorted(
            set(property_data.keys())
            & FORBIDDEN_INPUT_FIELDS
        )

        if forbidden_fields:

            raise ValueError(
                "Prediction input contains forbidden "
                "target/leakage fields: "
                f"{forbidden_fields}"
            )

        # --------------------------------------------------------------
        # REQUIRED PROPERTY TYPE
        # --------------------------------------------------------------

        property_type = property_data.get(
            "property_type"
        )

        if property_type is None:

            raise ValueError(
                "property_type is required."
            )

        if not isinstance(
            property_type,
            str,
        ):

            raise TypeError(
                "property_type must be a string."
            )

        if not property_type.strip():

            raise ValueError(
                "property_type cannot be empty."
            )

        # --------------------------------------------------------------
        # NUMERIC VALIDATION
        # --------------------------------------------------------------

        for field_name in NON_NEGATIVE_FIELDS:

            if field_name not in property_data:

                continue

            value = property_data[field_name]

            # Missing values are allowed for optional fields.
            if value is None:

                continue

            # Pandas / NumPy missing values are treated as missing,
            # not as invalid user input.
            try:

                if pd.isna(value):

                    continue

            except (
                TypeError,
                ValueError,
            ):

                pass

            self._validate_finite_number(
                value=value,
                field_name=field_name,
            )

            numeric_value = float(value)

            if numeric_value < 0:

                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        # --------------------------------------------------------------
        # POSITIVE FIELDS
        # --------------------------------------------------------------

        for field_name in POSITIVE_FIELDS:

            if field_name not in property_data:

                continue

            value = property_data[field_name]

            if value is None:

                continue

            try:

                if pd.isna(value):

                    continue

            except (
                TypeError,
                ValueError,
            ):

                pass

            numeric_value = float(value)

            if not math.isfinite(
                numeric_value
            ):

                raise ValueError(
                    f"{field_name} must be finite."
                )

            if numeric_value <= 0:

                raise ValueError(
                    f"{field_name} must be greater than zero."
                )

        # --------------------------------------------------------------
        # LATITUDE
        # --------------------------------------------------------------

        if "lat" in property_data:

            value = property_data["lat"]

            if value is not None:

                try:

                    if not pd.isna(value):

                        self._validate_finite_number(
                            value=value,
                            field_name="lat",
                        )

                        latitude = float(value)

                        if not (
                            LATITUDE_MIN
                            <= latitude
                            <= LATITUDE_MAX
                        ):

                            raise ValueError(
                                "lat must be between "
                                "-90 and 90."
                            )

                except (
                    TypeError,
                    ValueError,
                ):

                    raise

        # --------------------------------------------------------------
        # LONGITUDE
        # --------------------------------------------------------------

        if "lon" in property_data:

            value = property_data["lon"]

            if value is not None:

                try:

                    if not pd.isna(value):

                        self._validate_finite_number(
                            value=value,
                            field_name="lon",
                        )

                        longitude = float(value)

                        if not (
                            LONGITUDE_MIN
                            <= longitude
                            <= LONGITUDE_MAX
                        ):

                            raise ValueError(
                                "lon must be between "
                                "-180 and 180."
                            )

                except (
                    TypeError,
                    ValueError,
                ):

                    raise

        # --------------------------------------------------------------
        # MARKET CONSISTENCY
        # --------------------------------------------------------------

        supplied_offering_type = (
            property_data.get(
                "offering_type"
            )
        )

        if supplied_offering_type is not None:

            if not isinstance(
                supplied_offering_type,
                str,
            ):

                raise TypeError(
                    "offering_type must be a string."
                )

            normalized_offering = (
                supplied_offering_type
                .strip()
                .lower()
            )

            expected_offering = (
                MARKET_VALUES[market]
                .lower()
            )

            if (
                normalized_offering
                != expected_offering
            ):

                raise ValueError(
                    "offering_type does not match "
                    f"market_type='{market}'. "
                    f"Expected '{MARKET_VALUES[market]}'."
                )

    # ==================================================================
    # FEATURE PREPARATION
    # ==================================================================

    @staticmethod
    def _prepare_features(
        property_data: pd.DataFrame,
        feature_columns: list[str],
        categorical_features: list[str],
    ) -> pd.DataFrame:

        features = build_features(
            property_data.copy()
        )

        missing_columns = [
            column
            for column in feature_columns
            if column not in features.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required features: "
                f"{missing_columns}"
            )

        features = features[
            feature_columns
        ].copy()

        for column in categorical_features:

            if column in features.columns:

                features[column] = (
                    features[column]
                    .fillna("__MISSING__")
                    .astype(str)
                )

        return features

    # ==================================================================
    # SINGLE MODEL PREDICTION
    # ==================================================================

    @staticmethod
    def _predict_model(
        model,
        X: pd.DataFrame,
        model_type: str,
        categorical_features: list[str],
    ) -> np.ndarray:

        """
        Generate model prediction.

        CatBoost already knows its categorical features from training.
        Therefore cat_features must NOT be passed to predict().
        """

        if model_type == "catboost":

            prediction_log = model.predict(X)

        else:

            prediction_log = model.predict(X)

        prediction_log = np.asarray(
            prediction_log,
            dtype=float,
        )

        if prediction_log.size == 0:

            raise ValueError(
                "Model returned an empty prediction."
            )

        if not np.all(
            np.isfinite(prediction_log)
        ):

            raise ValueError(
                "Model returned a non-finite prediction."
            )

        return prediction_log

    # ==================================================================
    # ENSEMBLE PREDICTION
    # ==================================================================

    def _ensemble_prediction(
        self,
        model_entries: list[dict[str, Any]],
        X: pd.DataFrame,
        categorical_features: list[str],
    ) -> tuple[
        np.ndarray,
        list[dict[str, Any]],
    ]:

        if not model_entries:

            raise ValueError(
                "No models available for ensemble prediction."
            )

        predictions = []
        weights = []
        model_outputs = []

        for entry in model_entries:

            if "model_path" not in entry:

                raise ValueError(
                    "Model entry is missing 'model_path'."
                )

            if "ensemble_weight" not in entry:

                raise ValueError(
                    "Model entry is missing "
                    "'ensemble_weight'."
                )

            model = self._load_model(
                entry["model_path"]
            )

            prediction_log = self._predict_model(
                model=model,
                X=X,
                model_type=entry["model_type"],
                categorical_features=categorical_features,
            )

            predictions.append(
                prediction_log
            )

            weight = float(
                entry["ensemble_weight"]
            )

            if not math.isfinite(weight):

                raise ValueError(
                    "Ensemble weight must be finite."
                )

            if weight < 0:

                raise ValueError(
                    "Ensemble weight cannot be negative."
                )

            weights.append(weight)

            prediction_egp = float(
                np.expm1(
                    prediction_log[0]
                )
            )

            if not math.isfinite(
                prediction_egp
            ):

                raise ValueError(
                    "Model produced a non-finite "
                    "EGP prediction."
                )

            model_outputs.append(
                {
                    "model": entry["model_name"],
                    "prediction_egp": round(
                        prediction_egp,
                        2,
                    ),
                    "weight": round(
                        weight,
                        4,
                    ),
                }
            )

        weights_array = np.asarray(
            weights,
            dtype=float,
        )

        if not np.all(
            np.isfinite(weights_array)
        ):

            raise ValueError(
                "Ensemble weights contain "
                "non-finite values."
            )

        if weights_array.sum() <= 0:

            raise ValueError(
                "Invalid ensemble weights."
            )

        weights_array = (
            weights_array
            / weights_array.sum()
        )

        predictions_array = np.vstack(
            predictions
        )

        if not np.all(
            np.isfinite(predictions_array)
        ):

            raise ValueError(
                "Ensemble contains non-finite "
                "model predictions."
            )

        ensemble_log = np.average(
            predictions_array,
            axis=0,
            weights=weights_array,
        )

        ensemble_log = np.asarray(
            ensemble_log,
            dtype=float,
        )

        if not np.all(
            np.isfinite(ensemble_log)
        ):

            raise ValueError(
                "Ensemble produced a non-finite prediction."
            )

        return (
            ensemble_log,
            model_outputs,
        )

    # ==================================================================
    # PROPERTY TYPE NORMALIZATION
    # ==================================================================

    @staticmethod
    def _normalize_property_type(
        property_type: str | None,
    ) -> str | None:

        if property_type is None:

            return None

        if not isinstance(
            property_type,
            str,
        ):

            raise TypeError(
                "property_type must be a string."
            )

        value = (
            property_type
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        value = " ".join(
            value.split()
        )

        return (
            value
            if value
            else None
        )

    # ==================================================================
    # MODEL RESOLUTION
    # ==================================================================

    def _resolve_model_group(
        self,
        market_type: str,
        property_type: str | None,
    ) -> tuple[
        str,
        list[dict[str, Any]],
        dict[str, Any],
    ]:

        market = (
            market_type
            .strip()
            .lower()
        )

        if market not in MARKET_VALUES:

            raise ValueError(
                "market_type must be 'sale' or 'rent'."
            )

        market_metadata = (
            self.metadata[
                "markets"
            ][market]
        )

        normalized_property = (
            self._normalize_property_type(
                property_type
            )
        )

        qualified_segments = (
            QUALIFIED_SPECIALIZED_SEGMENTS.get(
                market,
                set(),
            )
        )

        # Normalize the qualified policy itself so that
        # underscores/hyphens can never create routing mismatches.

        normalized_qualified_segments = {
            self._normalize_property_type(
                segment
            )
            for segment in qualified_segments
        }

        # --------------------------------------------------------------
        # SPECIALIZED ROUTING
        # --------------------------------------------------------------

        if (
            normalized_property
            and normalized_property
            in normalized_qualified_segments
        ):

            specialized_models = (
                market_metadata.get(
                    "specialized_models",
                    {},
                )
            )

            for key, value in (
                specialized_models.items()
            ):

                # IMPORTANT:
                # Metadata may contain "twin_house"
                # while incoming input may contain "Twin House".
                #
                # Normalize BOTH sides before comparison.

                normalized_key = (
                    self._normalize_property_type(
                        key
                    )
                )

                if (
                    normalized_key
                    == normalized_property
                ):

                    return (
                        "specialized",
                        value["models"],
                        value,
                    )

        # --------------------------------------------------------------
        # MARKET FALLBACK
        # --------------------------------------------------------------

        return (
            "market",
            market_metadata[
                "market_models"
            ],
            market_metadata,
        )

    # ==================================================================
    # MODEL QUALITY
    # ==================================================================

    @staticmethod
    def _extract_model_r2(
        metadata: dict[str, Any],
    ) -> float:

        """
        Extract the most relevant available R².

        Priority:
            test_metrics.r2
            ensemble_test_metrics.r2
            test_r2
            validation_r2
        """

        candidate_locations = [
            (
                metadata.get(
                    "test_metrics",
                    {},
                ),
                "r2",
            ),
            (
                metadata.get(
                    "ensemble_test_metrics",
                    {},
                ),
                "r2",
            ),
            (
                metadata,
                "test_r2",
            ),
            (
                metadata,
                "validation_r2",
            ),
        ]

        for container, key in candidate_locations:

            value = container.get(key)

            if value is not None:

                try:

                    value = float(value)

                    if math.isfinite(value):

                        return value

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

        return 0.0

    # ==================================================================
    # CONFIDENCE
    # ==================================================================

    @staticmethod
    def _calculate_confidence(
        model_outputs: list[dict[str, Any]],
        model_r2: float,
    ) -> dict[str, Any]:

        predictions = np.asarray(
            [
                item["prediction_egp"]
                for item in model_outputs
            ],
            dtype=float,
        )

        if len(predictions) == 0:

            return {
                "score": 0.0,
                "level": "low",
                "model_agreement": 0.0,
                "model_quality": 0.0,
                "reference_r2": round(
                    model_r2,
                    4,
                ),
            }

        if not np.all(
            np.isfinite(predictions)
        ):

            raise ValueError(
                "Confidence calculation received "
                "non-finite model predictions."
            )

        mean_prediction = float(
            predictions.mean()
        )

        if mean_prediction <= 0:

            dispersion = 1.0

        else:

            dispersion = (
                float(
                    predictions.std()
                )
                / mean_prediction
            )

        agreement_score = max(
            0.0,
            min(
                1.0,
                1.0 - dispersion,
            ),
        )

        # Convert R² into a bounded quality signal.
        #
        # R² <= -1 -> 0
        # R² =  0  -> 0.5
        # R² =  1  -> 1

        model_quality_score = max(
            0.0,
            min(
                1.0,
                (model_r2 + 1.0) / 2.0,
            ),
        )

        confidence_score = (
            0.6 * agreement_score
            + 0.4 * model_quality_score
        )

        if confidence_score >= 0.80:

            confidence_level = "high"

        elif confidence_score >= 0.60:

            confidence_level = "medium"

        else:

            confidence_level = "low"

        return {
            "score": round(
                confidence_score,
                4,
            ),
            "level": confidence_level,
            "model_agreement": round(
                agreement_score,
                4,
            ),
            "model_quality": round(
                model_quality_score,
                4,
            ),
            "reference_r2": round(
                model_r2,
                4,
            ),
        }

    # ==================================================================
    # PUBLIC PREDICTION API
    # ==================================================================

    def predict(
        self,
        property_data: dict[str, Any],
        market_type: str,
    ) -> dict[str, Any]:

        # --------------------------------------------------------------
        # MARKET VALIDATION
        # --------------------------------------------------------------

        if not isinstance(
            market_type,
            str,
        ):

            raise TypeError(
                "market_type must be a string."
            )

        market = (
            market_type
            .strip()
            .lower()
        )

        if market not in MARKET_VALUES:

            raise ValueError(
                "market_type must be 'sale' or 'rent'."
            )

        # --------------------------------------------------------------
        # INPUT VALIDATION
        # --------------------------------------------------------------

        self._validate_property_input(
            property_data=property_data,
            market=market,
        )

        # --------------------------------------------------------------
        # INPUT DATAFRAME
        # --------------------------------------------------------------

        input_df = pd.DataFrame(
            [property_data]
        )

        property_type = property_data.get(
            "property_type"
        )

        # --------------------------------------------------------------
        # MODEL RESOLUTION
        # --------------------------------------------------------------

        (
            resolution,
            model_entries,
            resolution_metadata,
        ) = self._resolve_model_group(
            market_type=market,
            property_type=property_type,
        )

        market_metadata = (
            self.metadata[
                "markets"
            ][market]
        )

        # --------------------------------------------------------------
        # FEATURES
        # --------------------------------------------------------------

        feature_columns = (
            market_metadata[
                "feature_columns"
            ]
        )

        categorical_features = (
            market_metadata[
                "categorical_features"
            ]
        )

        X = self._prepare_features(
            property_data=input_df,
            feature_columns=feature_columns,
            categorical_features=categorical_features,
        )

        # --------------------------------------------------------------
        # FEATURE SAFETY
        # --------------------------------------------------------------

        numeric_columns = X.select_dtypes(
            include=[np.number]
        ).columns

        if len(numeric_columns) > 0:

            numeric_values = (
                X[numeric_columns]
                .to_numpy(
                    dtype=float
                )
            )

            if not np.all(
                np.isfinite(numeric_values)
            ):

                raise ValueError(
                    "Engineered numeric features "
                    "contain NaN or infinite values."
                )

        # --------------------------------------------------------------
        # ENSEMBLE
        # --------------------------------------------------------------

        (
            ensemble_log,
            model_outputs,
        ) = self._ensemble_prediction(
            model_entries=model_entries,
            X=X,
            categorical_features=categorical_features,
        )

        # --------------------------------------------------------------
        # FINAL PRICE
        # --------------------------------------------------------------

        predicted_price = float(
            np.expm1(
                ensemble_log[0]
            )
        )

        if not math.isfinite(
            predicted_price
        ):

            raise ValueError(
                "Final predicted price is non-finite."
            )

        predicted_price = max(
            0.0,
            predicted_price,
        )

        # --------------------------------------------------------------
        # MODEL QUALITY
        # --------------------------------------------------------------

        model_r2 = (
            self._extract_model_r2(
                resolution_metadata
            )
        )

        # --------------------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------------------

        confidence = (
            self._calculate_confidence(
                model_outputs=model_outputs,
                model_r2=model_r2,
            )
        )

        # --------------------------------------------------------------
        # FINAL RESPONSE
        # --------------------------------------------------------------

        return {
            "market_type": market,
            "market_value": MARKET_VALUES[
                market
            ],
            "property_type": property_type,
            "model_resolution": resolution,
            "estimated_value_egp": round(
                predicted_price,
                2,
            ),
            "confidence": confidence,
            "model_predictions": model_outputs,
        }


# ======================================================================
# OPTIONAL QUICK LOCAL TEST
# ======================================================================

if __name__ == "__main__":

    engine = (
        RealEstateValuationEngine()
    )

    print(
        "Real Estate Valuation Engine loaded successfully."
    )

    print(
        "Qualified specialized routing:"
    )

    for market, segments in (
        QUALIFIED_SPECIALIZED_SEGMENTS.items()
    ):

        print(
            f"  {market}: "
            f"{sorted(segments)}"
        )

