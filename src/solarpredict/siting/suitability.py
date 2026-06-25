"""Within-city site suitability for solar-EV charging.

A transparent, explainable weighted index (no black box) over: annual GHI yield,
seasonal consistency, and panel-coolness (cooler air -> better PV efficiency). Yield
dominates, so the ranking is mostly resource-driven, with consistency/coolness as
tie-breakers. Reported per named district so recommendations are human-readable.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from solarpredict.config.sites import INTRACITY_SITES
from solarpredict.data.repository import DataRepository

DEFAULT_WEIGHTS = {"yield": 0.6, "consistency": 0.25, "coolness": 0.15}


def _unit(series: pd.Series) -> pd.Series:
    span = float(series.max() - series.min())
    if span == 0:
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / span


def site_suitability(
    repo: DataRepository,
    city: str,
    start: date,
    end: date,
    *,
    target: str = "shortwave_radiation",
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank a city's named districts by a 0..1 solar-EV suitability score."""
    weights = weights or DEFAULT_WEIGHTS
    rows = []
    for site in INTRACITY_SITES[city]:
        frame = repo.fetch(site, start, end)
        ghi = frame[target].astype(float)
        daytime = ghi[ghi > 0]
        months = pd.DatetimeIndex(frame["timestamp"]).month
        monthly = ghi.groupby(months).mean()
        temp = (
            float(frame["temperature_2m"].astype(float).mean())
            if "temperature_2m" in frame.columns
            else float("nan")
        )
        rows.append(
            {
                "site": site.name,
                "latitude": site.latitude,
                "longitude": site.longitude,
                "annual_kwh_m2": round(float(ghi.mean()) * 8.76, 0),
                "mean_ghi_day": round(float(daytime.mean()), 1),
                "seasonality_cv": round(float(monthly.std() / monthly.mean()), 3),
                "mean_temp_c": round(temp, 1),
            }
        )

    out = pd.DataFrame(rows)
    s_yield = _unit(out["annual_kwh_m2"])
    s_consistency = 1.0 - _unit(out["seasonality_cv"])
    s_coolness = 1.0 - _unit(out["mean_temp_c"])
    out["suitability"] = (
        weights["yield"] * s_yield
        + weights["consistency"] * s_consistency
        + weights["coolness"] * s_coolness
    ).round(3)
    return out.sort_values("suitability", ascending=False).reset_index(drop=True)
