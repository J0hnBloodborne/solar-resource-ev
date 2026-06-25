"""Tier-0 baselines: persistence, smart (clear-sky) persistence, climatology.

Smart persistence is the **skill-score reference** the whole benchmark is judged
against: it persists the clear-sky index rather than raw GHI, removing the
deterministic diurnal cycle so the score reflects real (cloud) skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarpredict.features.clearsky import clear_sky_index

from .base import ForecastData, Forecaster
from .registry import register


@register("persistence")
class PersistenceForecaster(Forecaster):
    """Naive persistence: ``yhat(t+1) = y(t)`` within each series (trivial floor)."""

    name = "persistence"
    tier = "tier-0"
    supports_covariates = False

    def fit(self, train: ForecastData) -> PersistenceForecaster:
        return self

    def predict(self, test: ForecastData) -> np.ndarray:
        shifted = test.frame.groupby(test.series_col)[test.target_col].shift(1)
        preds = shifted.to_numpy(dtype=float).copy()
        missing = np.isnan(preds)
        preds[missing] = test.frame[test.target_col].to_numpy(dtype=float)[missing]
        return preds


@register("smart_persistence")
class SmartPersistenceForecaster(Forecaster):
    """Persist the clear-sky index: ``GHI_hat(t+1) = kt(t) * GHI_cs(t+1)``.

    Requires a ``clearsky_ghi`` column (add it via ``features.prepare_site``). This
    is the reference model for the forecast skill score.
    """

    name = "smart_persistence"
    tier = "tier-0"
    supports_covariates = False

    def fit(self, train: ForecastData) -> SmartPersistenceForecaster:
        return self

    def predict(self, test: ForecastData) -> np.ndarray:
        df = test.frame
        if "clearsky_ghi" not in df.columns:
            raise KeyError(
                "smart_persistence needs a 'clearsky_ghi' column "
                "(run features.prepare_site first)"
            )
        cs = df["clearsky_ghi"].to_numpy(dtype=float)
        kt = clear_sky_index(df[test.target_col].to_numpy(dtype=float), cs)
        # .copy() because pandas 3.0 can return a read-only array.
        kt_prev = (
            df.assign(_kt=kt).groupby(test.series_col)["_kt"].shift(1).to_numpy().copy()
        )
        missing = np.isnan(kt_prev)
        kt_prev[missing] = kt[missing]  # first step of each series -> persist itself
        return np.clip(kt_prev * cs, 0.0, None)


@register("climatology")
class ClimatologyForecaster(Forecaster):
    """Mean GHI by (hour-of-day, month), learned on train and applied to test."""

    name = "climatology"
    tier = "tier-0"
    supports_covariates = False

    def __init__(self) -> None:
        self._table: pd.Series | None = None
        self._global: float = 0.0

    def fit(self, train: ForecastData) -> ClimatologyForecaster:
        ds = pd.DatetimeIndex(train.frame[train.timestamp_col])
        tmp = pd.DataFrame(
            {
                "hour": ds.hour,
                "month": ds.month,
                "y": train.frame[train.target_col].to_numpy(dtype=float),
            }
        )
        self._table = tmp.groupby(["hour", "month"])["y"].mean()
        self._global = float(tmp["y"].mean())
        return self

    def predict(self, test: ForecastData) -> np.ndarray:
        if self._table is None:
            raise RuntimeError("ClimatologyForecaster must be fit before predict")
        ds = pd.DatetimeIndex(test.frame[test.timestamp_col])
        idx = pd.MultiIndex.from_arrays([ds.hour, ds.month])
        preds = self._table.reindex(idx).to_numpy(dtype=float)
        return np.where(np.isnan(preds), self._global, preds)
