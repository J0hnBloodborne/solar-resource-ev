"""Test the data-driven city selection (mocked repo, no network)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from solarpredict.config.sites import INTRACITY_SITES
from solarpredict.siting.selection import select_city


class _FakeRepo:
    """Returns a flat GHI series whose mean is looked up per site slug."""

    def __init__(self, means: dict[str, float]) -> None:
        self._means = means

    def fetch(self, location, start, end, *, force_refresh=False):
        value = self._means[location.slug]
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=6, freq="h", tz="UTC"),
                "shortwave_radiation": [value] * 6,
            }
        )


def _means() -> dict[str, float]:
    means: dict[str, float] = {}
    for i, site in enumerate(INTRACITY_SITES["Karachi"]):
        means[site.slug] = 415.0 + 2.0 * i  # wide spread
    for i, site in enumerate(INTRACITY_SITES["Lahore"]):
        means[site.slug] = 366.0 + (i % 2)  # tiny spread
    return means


def test_select_city_prefers_higher_spread() -> None:
    chosen, ranking = select_city(
        _FakeRepo(_means()), date(2020, 1, 1), date(2020, 1, 2)
    )
    assert chosen == "Karachi"
    assert ranking.iloc[0]["city"] == "Karachi"
    k = ranking.loc[ranking["city"] == "Karachi", "spread"].iloc[0]
    lahore = ranking.loc[ranking["city"] == "Lahore", "spread"].iloc[0]
    assert k > lahore
