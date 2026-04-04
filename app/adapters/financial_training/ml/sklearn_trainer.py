from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from app.domain.financial_training.ports.model_training import (
    ModelTrainerPort,
)


def _wape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


class SklearnFinancialTrainer(ModelTrainerPort):
    """
    Adapter: implement sklearn model training.
    """

    def train(
        self,
        features: Any,
        labels: Any,
        dataset_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        if not isinstance(features, pd.DataFrame):
            raise ValueError("features must be a pandas DataFrame")
        if not isinstance(labels, (pd.Series, pd.DataFrame)):
            raise ValueError("labels must be pandas Series/DataFrame")

        y = labels.squeeze()
        if not isinstance(y, pd.Series):
            raise ValueError("labels should resolve to a single pandas Series")

        # Expect caller to pass split masks, otherwise fallback to random 80/20 by order.
        train_mask = None
        valid_mask = None
        if dataset_metadata:
            train_mask = dataset_metadata.get("train_mask")
            valid_mask = dataset_metadata.get("valid_mask")

        if train_mask is None or valid_mask is None:
            n = len(features)
            split_idx = int(n * 0.8)
            train_mask = np.array([i < split_idx for i in range(n)], dtype=bool)
            valid_mask = ~train_mask

        X_train = features.loc[train_mask]
        y_train = y.loc[train_mask]
        X_valid = features.loc[valid_mask]
        y_valid = y.loc[valid_mask]

        default_params = {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
        extra_params = {}
        if dataset_metadata:
            extra_params = dataset_metadata.get("xgb_params", {}) or {}
        model = XGBRegressor(**{**default_params, **extra_params})
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        pred = model.predict(X_valid)
        mae = float(mean_absolute_error(y_valid, pred))
        rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
        r2 = float(r2_score(y_valid, pred))
        wape = _wape(y_valid, pred)

        non_zero = y_valid != 0
        if non_zero.any():
            mape = float(np.mean(np.abs((y_valid[non_zero] - pred[non_zero]) / y_valid[non_zero])) * 100)
        else:
            mape = float("nan")

        return {
            "model": model,
            "feature_columns": list(features.columns),
            "metrics": {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "wape": wape,
                "mape": mape,
                "valid_rows": int(len(y_valid)),
                "train_rows": int(len(y_train)),
            },
        }

