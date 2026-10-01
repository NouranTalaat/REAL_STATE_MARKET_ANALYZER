from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_predictions(
    y_true_log,
    y_pred_log,
) -> dict[str, float]:
    """
    Evaluate predictions after converting log targets
    back to the original EGP scale.
    """

    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    y_pred = np.maximum(
        y_pred,
        0,
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    median_absolute_error = float(
        np.median(
            np.abs(y_true - y_pred)
        )
    )

    return {
        "mae_egp": float(mae),
        "rmse_egp": float(rmse),
        "r2": float(r2),
        "median_absolute_error_egp": median_absolute_error,
    }


def format_metrics(
    metrics: dict[str, float],
) -> str:
    return (
        f"MAE: {metrics['mae_egp']:,.2f} EGP | "
        f"RMSE: {metrics['rmse_egp']:,.2f} EGP | "
        f"R²: {metrics['r2']:.4f} | "
        f"Median AE: "
        f"{metrics['median_absolute_error_egp']:,.2f} EGP"
    )