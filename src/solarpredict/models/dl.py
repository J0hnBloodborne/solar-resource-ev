"""Tier-2 deep learning: NHITS + LSTM via neuralforecast (covariate-aware).

neuralforecast/torch are heavy + optional (the ``dl`` extra), so they're imported
lazily inside ``predict`` — importing this module only registers the models and
stays CI-safe. Models train on GPU when available; ``num_workers=0`` avoids the
Windows DataLoader hang. 1-step-ahead predictions across the test window come from
``cross_validation`` (fit once on train, roll forward with step_size=1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarpredict.config.sites import SAFE_COVARIATES

from .base import ForecastData, Forecaster
from .registry import register

# Realized-to-t exogenous inputs (lagged history, no leakage). clearsky_ghi is
# deterministic so it's safe to feed as well.
DL_HIST_EXOG: tuple[str, ...] = (*SAFE_COVARIATES, "clearsky_ghi")

_TRAINER_KWARGS = {
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "logger": False,
    "enable_checkpointing": False,
    "gradient_clip_val": 1.0,  # GHI's night-zeros make training prone to diverging
}


class NeuralForecastAdapter(Forecaster):
    """Wrap a neuralforecast model under the Forecaster interface (hour-ahead)."""

    tier = "tier-2"
    supports_covariates = True

    def __init__(
        self,
        name: str,
        model_name: str,
        *,
        input_size: int = 48,
        max_steps: int = 300,
        hist_exog: tuple[str, ...] = DL_HIST_EXOG,
        model_kwargs: dict | None = None,
    ) -> None:
        self.name = name
        self._model_name = model_name  # e.g. "NHITS", "LSTM"
        self._input_size = input_size
        self._max_steps = max_steps
        self._hist_exog = tuple(hist_exog)
        self._model_kwargs = model_kwargs or {}
        self._train: pd.DataFrame | None = None

    def _to_long(self, data: ForecastData) -> pd.DataFrame:
        keep = [
            "unique_id",
            "ds",
            "y",
            *[c for c in self._hist_exog if c in data.frame.columns],
        ]
        return data.frame.loc[:, keep].copy()

    def fit(self, train: ForecastData) -> NeuralForecastAdapter:
        self._train = self._to_long(train)
        return self

    def predict(self, test: ForecastData) -> np.ndarray:
        from neuralforecast import NeuralForecast
        from neuralforecast import models as nf_models

        if self._train is None:
            raise RuntimeError(f"{self.name} must be fit before predict")

        h = test.horizon
        test_long = self._to_long(test)
        full = pd.concat([self._train, test_long], ignore_index=True)
        exog = [c for c in self._hist_exog if c in full.columns]

        model = getattr(nf_models, self._model_name)(
            h=h,
            input_size=self._input_size,
            hist_exog_list=exog,
            max_steps=self._max_steps,
            scaler_type="standard",
            random_seed=42,
            dataloader_kwargs={"num_workers": 0},
            **_TRAINER_KWARGS,  # forwarded to the Lightning Trainer
            **self._model_kwargs,
        )
        nf = NeuralForecast(models=[model], freq="h")
        # Roll one step at a time; each window forecasts h steps. n_windows covers the
        # whole test window plus the h-step lead so every test row has a horizon-step
        # forecast after the lead filter below.
        cv = nf.cross_validation(
            df=full,
            n_windows=len(test_long) + h - 1,
            step_size=1,
            refit=False,
            verbose=False,
        )
        pred_col = next(
            c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y")
        )
        # Keep only the h-step-ahead prediction (ds == cutoff + h) so the lead matches
        # the tabular day-ahead features.
        lead_h = (
            (pd.to_datetime(cv["ds"]) - pd.to_datetime(cv["cutoff"])).dt.total_seconds()
            / 3600.0
        ).round().astype(int) == h
        cv_h = cv[lead_h]
        merged = test.frame[["unique_id", "ds"]].merge(
            cv_h[["unique_id", "ds", pred_col]], on=["unique_id", "ds"], how="left"
        )
        preds = merged[pred_col].bfill().ffill().to_numpy(dtype=float)
        return np.clip(preds, 0.0, None)


@register("nhits")
def _make_nhits() -> NeuralForecastAdapter:
    return NeuralForecastAdapter("nhits", "NHITS", input_size=48, max_steps=300)


@register("lstm")
def _make_lstm() -> NeuralForecastAdapter:
    return NeuralForecastAdapter("lstm", "LSTM", input_size=48, max_steps=300)
