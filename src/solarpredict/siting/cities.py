"""Inter-city solar comparison for Part B: which cities suit solar-EV charging.

Aggregates each candidate city's multi-year GHI into a daytime mean, an annual
energy yield (kWh/m^2/yr), and a seasonality measure, then ranks them. Quetta is
the standout (highest yield); even the lowest city clears typical European yields.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from solarpredict.config.sites import CANDIDATE_CITIES, Location
from solarpredict.data.repository import DataRepository

# Reference national-average GHI yields (kWh/m^2/yr) for context in the report.
REFERENCE_YIELDS = {"Germany": 1080, "United Kingdom": 1000}


def city_summary(
    repo: DataRepository,
    start: date,
    end: date,
    *,
    cities: tuple[Location, ...] = CANDIDATE_CITIES,
    target: str = "shortwave_radiation",
) -> pd.DataFrame:
    """Per-city solar summary, ranked by annual energy yield (highest first)."""
    rows = []
    for city in cities:
        frame = repo.fetch(city, start, end)
        ghi = frame[target].astype(float)
        daytime = ghi[ghi > 0]
        months = pd.DatetimeIndex(frame["timestamp"]).month
        monthly_mean = ghi.groupby(months).mean()
        rows.append(
            {
                "city": city.name,
                "latitude": city.latitude,
                "longitude": city.longitude,
                "mean_ghi": round(float(ghi.mean()), 1),
                "mean_ghi_day": round(float(daytime.mean()), 1),
                "annual_kwh_m2": round(float(ghi.mean()) * 8.76, 0),
                "seasonality_cv": round(
                    float(monthly_mean.std() / monthly_mean.mean()), 3
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("annual_kwh_m2", ascending=False)
        .reset_index(drop=True)
    )


def monthly_ghi(
    repo: DataRepository,
    start: date,
    end: date,
    *,
    cities: tuple[Location, ...] = CANDIDATE_CITIES,
    target: str = "shortwave_radiation",
) -> pd.DataFrame:
    """Long table of mean daytime GHI by (city, month) — for the seasonal figure."""
    rows = []
    for city in cities:
        frame = repo.fetch(city, start, end)
        ghi = frame[target].astype(float)
        months = pd.DatetimeIndex(frame["timestamp"]).month
        daytime = ghi.where(ghi > 0)
        for month, value in daytime.groupby(months).mean().items():
            rows.append(
                {"city": city.name, "month": int(month), "ghi_day": float(value)}
            )
    return pd.DataFrame(rows)
