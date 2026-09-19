from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

SALE_MODEL_PATH = MODELS_DIR / "final_sale_model.pkl"
RENT_MODEL_PATH = MODELS_DIR / "final_rent_model.pkl"


def load_sale_model():
    if not SALE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Sale model not found: {SALE_MODEL_PATH}"
        )

    return joblib.load(SALE_MODEL_PATH)


def load_rent_model():
    if not RENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Rent model not found: {RENT_MODEL_PATH}"
        )

    return joblib.load(RENT_MODEL_PATH)


def get_model_status() -> dict:
    return {
        "sale_model": {
            "name": SALE_MODEL_PATH.name,
            "exists": SALE_MODEL_PATH.exists(),
        },
        "rent_model": {
            "name": RENT_MODEL_PATH.name,
            "exists": RENT_MODEL_PATH.exists(),
        },
    }