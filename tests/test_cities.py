"""Test the inter-city solar summary (mock repo, no network)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from solarpredict.config.sites import Location
from solarpredict.siting import city_summary


class _FakeRepo:
    def __init__(self, yields: dict[str, float]) -> None:
        self._yields = yields

    def fetch(self, location, start, end, *, force_refresh=False):
        value = self._yields[location.name]
        times = pd.date_range("2020-01-01", periods=90 * 24, freq="h", tz="UTC")
        return pd.DataFrame({"timestamp": times, "shortwave_radiation": float(value)})


def test_city_summary_ranks_by_yield() -> None:
    cities = (Location("Sunny", 30.0, 67.0), Location("Cloudy", 34.0, 71.0))
    summary = city_summary(
        _FakeRepo({"Sunny": 250.0, "Cloudy": 150.0}),
        date(2020, 1, 1),
        date(2020, 4, 1),
        cities=cities,
    )
    assert list(summary["city"]) == ["Sunny", "Cloudy"]  # higher yield first
    assert summary.iloc[0]["annual_kwh_m2"] == round(250.0 * 8.76, 0)
    assert {"latitude", "longitude", "mean_ghi_day"}.issubset(summary.columns)
