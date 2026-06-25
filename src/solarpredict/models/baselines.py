"""Tier-0 naive baseline(s).

Persistence is the trivial floor (and a sanity check for the harness). The
*skill-score reference* is smart/clear-sky persistence, which needs pvlib
clear-sky GHI and lands in M1 alongside climatology and SARIMAX.
"""

from __future__ import annotations

import numpy as np

from .base import ForecastData, Forecaster
from .registry import register


@register("persistence")
class PersistenceForecaster(Forecaster):
    """Naive persistence: ``yhat(t+1) = y(t)`` within each series.

    Not the skill-score denominator (that is smart persistence) — this is the
    trivial floor that shows why raw RMSE/R^2 on GHI are misleading.
    """

    name = "persistence"
    tier = "tier-0"
    supports_covariates = False

    def fit(self, train: ForecastData) -> PersistenceForecaster:
        # Stateless model — nothing to fit.
        return self

    def predict(self, test: ForecastData) -> np.ndarray:
        shifted = test.frame.groupby(test.series_col)[test.target_col].shift(1)
        # .copy() because pandas 3.0 can hand back a read-only array.
        preds = shifted.to_numpy(dtype=float).copy()
        # First step of each series has no predecessor; fall back to the value
        # itself so the array stays finite (the harness masks night hours anyway).
        missing = np.isnan(preds)
        preds[missing] = test.frame[test.target_col].to_numpy(dtype=float)[missing]
        return preds
