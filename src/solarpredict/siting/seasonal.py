"""Seasonal (4-season) within-city site suitability for Part B.

The instructor asked for the year split into the four meteorological seasons and
the sites ranked through them. We reuse the annual suitability index (yield,
consistency, coolness) but evaluate it *within each season*, so the seasonal shift
in both the solar resource and the ranking is visible. Consistency here is the
day-to-day stability of daytime GHI inside the season.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from solarpredict.config.sites import INTRACITY_SITES
from solarpredict.data.repository import DataRepository

from .suitability import DEFAULT_WEIGHTS, _unit

# Meteorological seasons (Northern Hemisphere): DJF / MAM / JJA / SON.
MONTH_TO_SEASON = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}
SEASON_ORDER = ("Winter", "Spring", "Summer", "Autumn")


def seasonal_site_suitability(
    repo: DataRepository,
    city: str,
    start: date,
    end: date,
    *,
    target: str = "shortwave_radiation",
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per (site, season) GHI, day-to-day variability, temperature, and a 0..1
    suitability score normalised across sites within each season."""
    weights = weights or DEFAULT_WEIGHTS
    rows = []
    for site in INTRACITY_SITES[city]:
        frame = repo.fetch(site, start, end)
        ts = pd.to_datetime(frame["timestamp"])
        work = pd.DataFrame(
            {
                "season": ts.dt.month.map(MONTH_TO_SEASON),
                "day": ts.dt.date,
                "ghi": frame[target].astype(float),
                "temp": (
                    frame["temperature_2m"].astype(float)
                    if "temperature_2m" in frame.columns
                    else float("nan")
                ),
            }
        )
        for season in SEASON_ORDER:
            block = work[work["season"] == season]
            daytime = block[block["ghi"] > 0]
            daily_mean = daytime.groupby("day")["ghi"].mean()
            rows.append(
                {
                    "site": site.name,
                    "latitude": site.latitude,
                    "longitude": site.longitude,
                    "season": season,
                    "mean_ghi_day": round(float(daytime["ghi"].mean()), 1),
                    "daily_cv": round(float(daily_mean.std() / daily_mean.mean()), 3),
                    "mean_temp_c": round(float(block["temp"].mean()), 1),
                }
            )

    out = pd.DataFrame(rows)
    scored = []
    for _, group in out.groupby("season", sort=False):
        group = group.copy()
        s_yield = _unit(group["mean_ghi_day"])
        s_consistency = 1.0 - _unit(group["daily_cv"])
        s_coolness = 1.0 - _unit(group["mean_temp_c"])
        group["suitability"] = (
            weights["yield"] * s_yield
            + weights["consistency"] * s_consistency
            + weights["coolness"] * s_coolness
        ).round(3)
        scored.append(group)
    return pd.concat(scored, ignore_index=True)
