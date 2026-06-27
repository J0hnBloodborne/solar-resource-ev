"""Tier-3 foundation model: Chronos zero-shot (a pretrained transformer).

No training — for each test hour we feed Chronos the preceding context window and
take its median 1-step forecast. chronos/torch are heavy + optional (the ``fm``
extra), so they're imported lazily inside ``predict`` (CI-safe). Model weights are
downloaded once from HuggingFace and cached.
"""

from __future__ import annotations

import numpy as np

from .base import ForecastData, Forecaster
from .registry import register


class ChronosForecaster(Forecaster):
    """Rolling 1-step-ahead zero-shot forecasts from a pretrained Chronos model."""

    tier = "tier-3"
    is_probabilistic = True

    def __init__(self, name: str, model_id: str, *, context_length: int = 512) -> None:
        self.name = name
        self._model_id = model_id
        self._ctx = context_length
        self._pipe: object | None = None

    def fit(self, train: ForecastData) -> ChronosForecaster:
        return self  # zero-shot: nothing to fit

    def _pipeline(self) -> object:
        if self._pipe is None:
            import torch
            from chronos import BaseChronosPipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipe = BaseChronosPipeline.from_pretrained(
                self._model_id, device_map=device
            )
        return self._pipe

    def predict(self, test: ForecastData) -> np.ndarray:
        import torch

        pipe = self._pipeline()
        h = test.horizon
        y = test.frame[test.target_col].to_numpy(dtype=float)
        inputs = []
        for i in range(len(y)):
            end = max(0, i - h + 1)  # history available h steps before row i
            hist = y[max(0, end - self._ctx) : end]
            if hist.size == 0:  # first rows have no history; they sit in the context
                hist = y[:1]
            inputs.append(torch.tensor(hist, dtype=torch.float32))

        quantiles, _means = pipe.predict_quantiles(  # type: ignore[attr-defined]
            inputs, prediction_length=h, quantile_levels=[0.5]
        )
        # Take the h-th forecast step (index h-1). Bolt returns a stacked tensor
        # [batch, h, q]; Chronos-2 a list of [h, q].
        if isinstance(quantiles, list | tuple):
            point = np.array(
                [
                    float(np.asarray(q.detach().cpu()).reshape(h, -1)[h - 1, 0])
                    for q in quantiles
                ]
            )
        else:
            point = np.asarray(quantiles.detach().cpu())[:, h - 1, 0].astype(float)
        return np.clip(point, 0.0, None)


@register("chronos2")
def _make_chronos2() -> ChronosForecaster:
    return ChronosForecaster("chronos2", "amazon/chronos-2")


@register("chronos_bolt_small")
def _make_bolt_small() -> ChronosForecaster:
    return ChronosForecaster("chronos_bolt_small", "amazon/chronos-bolt-small")


@register("chronos_bolt_tiny")
def _make_bolt_tiny() -> ChronosForecaster:
    return ChronosForecaster("chronos_bolt_tiny", "amazon/chronos-bolt-tiny")
