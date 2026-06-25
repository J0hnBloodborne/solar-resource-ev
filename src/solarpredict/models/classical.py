"""Tier-1 classical ML: tabular regressors over engineered features.

Each model is a thin ``SklearnForecaster`` around a scikit-learn-style regressor.
Feature columns arrive via ``ForecastData.covariate_cols`` (the pipeline sets them
to ``features.build_features`` output). Heavy libs (xgboost/lightgbm/catboost) are
imported lazily inside the factories, so importing this module is cheap and works
without the ``classical`` extra installed (only *instantiating* those models needs
it).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .base import ForecastData, Forecaster
from .registry import register


class SklearnForecaster(Forecaster):
    """Fit any sklearn-style regressor on the engineered feature matrix."""

    tier = "tier-1"
    supports_covariates = True

    def __init__(
        self, name: str, factory: Callable[[], Any], *, scale: bool = False
    ) -> None:
        self.name = name
        self._factory = factory
        self._scale = scale
        self._model: Any = None
        self._scaler: Any = None

    def _matrix(self, data: ForecastData) -> np.ndarray:
        if not data.covariate_cols:
            raise ValueError(
                f"{self.name}: no feature columns set "
                "(run features.build_features and pass covariate_cols)"
            )
        return data.frame.loc[:, list(data.covariate_cols)].to_numpy(dtype=float)

    def fit(self, train: ForecastData) -> SklearnForecaster:
        x = self._matrix(train)
        if self._scale:
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler().fit(x)
            x = self._scaler.transform(x)
        self._model = self._factory()
        self._model.fit(x, train.y)
        return self

    def predict(self, test: ForecastData) -> np.ndarray:
        x = self._matrix(test)
        if self._scaler is not None:
            x = self._scaler.transform(x)
        return np.clip(self._model.predict(x), 0.0, None)


def _ridge() -> Any:
    from sklearn.linear_model import Ridge

    return Ridge(alpha=1.0)


def _linear() -> Any:
    from sklearn.linear_model import LinearRegression

    return LinearRegression()


def _random_forest() -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)


def _extra_trees() -> Any:
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=42)


def _xgboost() -> Any:
    import xgboost as xgb

    return xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=42,
        objective="reg:squarederror",
    )


def _lightgbm() -> Any:
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )


def _catboost() -> Any:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=400, depth=6, learning_rate=0.05, random_seed=42, verbose=False
    )


@register("ridge")
def _make_ridge() -> SklearnForecaster:
    return SklearnForecaster("ridge", _ridge, scale=True)


@register("linear")
def _make_linear() -> SklearnForecaster:
    return SklearnForecaster("linear", _linear, scale=True)


@register("random_forest")
def _make_random_forest() -> SklearnForecaster:
    return SklearnForecaster("random_forest", _random_forest)


@register("extra_trees")
def _make_extra_trees() -> SklearnForecaster:
    return SklearnForecaster("extra_trees", _extra_trees)


@register("xgboost")
def _make_xgboost() -> SklearnForecaster:
    return SklearnForecaster("xgboost", _xgboost)


@register("lightgbm")
def _make_lightgbm() -> SklearnForecaster:
    return SklearnForecaster("lightgbm", _lightgbm)


@register("catboost")
def _make_catboost() -> SklearnForecaster:
    return SklearnForecaster("catboost", _catboost)
