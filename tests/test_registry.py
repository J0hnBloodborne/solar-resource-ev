"""Smoke tests for the Forecaster registry and the persistence baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarpredict.models import ForecastData, get_forecaster, list_forecasters


def test_persistence_is_registered() -> None:
    assert "persistence" in list_forecasters()


def test_unknown_forecaster_raises() -> None:
    try:
        get_forecaster("does-not-exist")
    except KeyError as exc:
        assert "Unknown forecaster" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")


def _toy_data() -> ForecastData:
    frame = pd.DataFrame(
        {
            "unique_id": ["s"] * 5,
            "ds": pd.date_range("2020-01-01", periods=5, freq="h"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    return ForecastData(frame=frame)


def test_persistence_predicts_previous_value() -> None:
    data = _toy_data()
    model = get_forecaster("persistence").fit(data)
    preds = model.predict(data)
    # yhat(t) = y(t-1); the first step falls back to its own value.
    np.testing.assert_array_equal(preds, np.array([1.0, 1.0, 2.0, 3.0, 4.0]))
