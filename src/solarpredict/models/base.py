"""The Forecaster Strategy interface and its canonical data container.

Every model in the benchmark — naive baseline, gradient-boosted tree, torch deep
net, or pretrained foundation model — implements the same ``Forecaster`` contract
so the shared evaluation harness can score them identically. Library-specific
glue lives inside each adapter, not in the harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastData:
    """Canonical long-format data handed to every Forecaster.

    One row per (series, timestamp), sorted chronologically within each series.
    The hour-ahead benchmark predicts ``target_col`` at t+1 using only
    information available up to t (lagging is the adapter's responsibility).

    Using the Nixtla long-format column names (``unique_id``/``ds``/``y``) means
    the neuralforecast/statsforecast adapters need no renaming.
    """

    frame: pd.DataFrame
    series_col: str = "unique_id"
    timestamp_col: str = "ds"
    target_col: str = "y"
    covariate_cols: tuple[str, ...] = field(default_factory=tuple)
    freq: str = "h"
    horizon: int = 1

    @property
    def y(self) -> np.ndarray:
        """Target values as a float array, aligned 1:1 with ``frame`` rows."""
        return self.frame[self.target_col].to_numpy(dtype=float)

    def covariates(self) -> pd.DataFrame:
        """The covariate columns (possibly empty)."""
        return self.frame.loc[:, list(self.covariate_cols)]


class Forecaster(ABC):
    """Strategy interface for a single forecasting model.

    Subclasses set the class-level metadata attributes and implement ``fit`` and
    ``predict``. Zero-shot models (persistence, Chronos) implement ``fit`` as a
    no-op returning ``self``.
    """

    #: Stable identifier, also the registry key.
    name: str = "base"
    #: One of "tier-0".."tier-3" (naive/statistical, classical ML, DL, foundation).
    tier: str = "unset"
    #: Whether the model consumes exogenous weather covariates.
    supports_covariates: bool = False
    #: Whether the model emits a predictive distribution (quantiles), not just a point.
    is_probabilistic: bool = False

    @abstractmethod
    def fit(self, train: ForecastData) -> Forecaster:
        """Fit on the training window and return ``self``."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, test: ForecastData) -> np.ndarray:
        """Return hour-ahead point forecasts aligned 1:1 with ``test`` rows."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, tier={self.tier!r})"
