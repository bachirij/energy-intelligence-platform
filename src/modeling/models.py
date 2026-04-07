"""
Model Registry — Electricity Load Forecasting (h+1)
----------------------------------------------------

Defines all models available for training and comparison.

To add a new model:
    1. Import the estimator
    2. Add an entry to MODELS dict
    3. (Optional) add its hyperparameters to config.py

Nothing else needs to change.
"""

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from config import RIDGE_PARAMS, XGBOOST_PARAMS, LIGHTGBM_PARAMS


def build_models() -> dict[str, Pipeline]:
    """
    Return a dict of named sklearn Pipelines ready to be trained.

    All models are wrapped in a Pipeline for a consistent .fit() / .predict() API.
    Ridge requires StandardScaler (linear models are sensitive to feature scale).
    XGBoost and LightGBM are tree-based — no scaling needed.

    Returns
    -------
    dict[str, Pipeline]
        Keys are model names used in results and filenames.
    """
    return {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(**RIDGE_PARAMS)),
        ]),

        "xgboost": Pipeline([
            ("model", xgb.XGBRegressor(**XGBOOST_PARAMS)),
        ]),

        "lightgbm": Pipeline([
            ("model", lgb.LGBMRegressor(**LIGHTGBM_PARAMS)),
        ]),

        # ----------------------------------------------------------------
        # To add a new model, follow this pattern:
        #
        # "random_forest": Pipeline([
        #     ("model", RandomForestRegressor(**RF_PARAMS)),
        # ]),
        #
        # "mlp": Pipeline([
        #     ("scaler", StandardScaler()),       # MLP needs scaling
        #     ("model",  MLPRegressor(**MLP_PARAMS)),
        # ]),
        # ----------------------------------------------------------------
    }