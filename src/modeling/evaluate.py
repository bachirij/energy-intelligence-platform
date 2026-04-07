"""
Evaluation — Electricity Load Forecasting (h+1)
------------------------------------------------

Metric computation and results display.
Kept separate from training logic so metrics can be reused
in notebooks, tests, or a monitoring pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# ---------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Compute MAE and RMSE between true and predicted values.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth target values.
    y_pred : np.ndarray
        Model predictions.

    Returns
    -------
    dict with keys "mae" and "rmse" (floats, rounded to 2 decimals).
    """
    return {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 2),
    }


# ---------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------

def print_summary(results: dict, best_name: str) -> None:
    """
    Print a formatted summary table of validation metrics for all models.

    Parameters
    ----------
    results : dict
        Output of train_and_compare — {model_name: {"val": {...}, "test": {...}}}.
    best_name : str
        Name of the best model (highlighted in the table).
    """
    print("\n" + "=" * 60)
    print("  Results summary")
    print("=" * 60)
    print(f"  {'Model':<12} {'Val MAE':>10} {'Val RMSE':>10}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 10}")

    for name, metrics in results.items():
        val    = metrics.get("val", {})
        marker = "  <-- best" if name == best_name else ""
        print(
            f"  {name:<12} "
            f"{val.get('mae',  '-'):>10} "
            f"{val.get('rmse', '-'):>10}"
            f"{marker}"
        )

    print("=" * 60)