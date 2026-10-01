from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from src.core.config import get_settings


settings = get_settings()


SALE_MODEL_PATH = settings.sale_model_path
RENT_MODEL_PATH = settings.rent_model_path


@lru_cache(maxsize=1)
def load_sale_model() -> Any:
    """
    Load and cache the sale valuation model.
    """
    if not SALE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Sale model not found: {SALE_MODEL_PATH}"
        )

    return joblib.load(SALE_MODEL_PATH)


@lru_cache(maxsize=1)
def load_rent_model() -> Any:
    """
    Load and cache the rent valuation model.
    """
    if not RENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Rent model not found: {RENT_MODEL_PATH}"
        )

    return joblib.load(RENT_MODEL_PATH)


def get_model_status() -> dict:
    """
    Return the availability status of all production models.
    """
    return {
        "sale_model": {
            "name": SALE_MODEL_PATH.name,
            "path": str(SALE_MODEL_PATH),
            "exists": SALE_MODEL_PATH.exists(),
        },
        "rent_model": {
            "name": RENT_MODEL_PATH.name,
            "path": str(RENT_MODEL_PATH),
            "exists": RENT_MODEL_PATH.exists(),
        },
    }


def validate_model_artifacts() -> None:
    """
    Validate that all critical model artifacts exist.

    This validates artifact availability without loading
    potentially expensive ML models during application startup.
    """
    missing_models = []

    if not SALE_MODEL_PATH.exists():
        missing_models.append(str(SALE_MODEL_PATH))

    if not RENT_MODEL_PATH.exists():
        missing_models.append(str(RENT_MODEL_PATH))

    if missing_models:
        raise FileNotFoundError(
            "Missing required model artifacts: "
            + ", ".join(missing_models)
        )