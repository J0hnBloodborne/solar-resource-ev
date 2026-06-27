"""Climatological expectation of the solar resource for any day of the year.

Beyond the ~2-week limit of weather forecasting, the actual GHI on a future date is
not predictable. What a planner can use instead is the climatological expectation:
from the multi-year record, the expected daily solar energy for each day of the year
and its spread across years. This is an expected value with uncertainty bands, not a
forecast, and it is the right input for sizing a solar-EV charger months ahead.

Daily insolation is the hourly GHI summed over the day and divided by 1000, i.e.
kWh/m^2/day (equivalently peak-sun-hours). Delivered PV energy applies the same fixed
system as the rest of the report: module efficiency 0.20, performance ratio 0.80, and
a 0.90 factor for cell-heating losses.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from solarpredict.config.sites import Location
from solarpredict.data.repository import DataRepository

PV_DELIVERED = 0.20 * 0.80 * 0.90  # efficiency * performance ratio * heat-derate


def _daily_insolation(
    repo: DataRepository, location: Location, start: date, end: date, target: str
) -> pd.DataFrame:
    """Per-day insolation (kWh/m^2/day) with day-of-year and month."""
    frame = repo.fetch(location, start, end)
    ts = pd.to_datetime(frame["timestamp"])
    work = pd.DataFrame({"date": ts.dt.floor("D"), "ghi": frame[target].astype(float)})
    daily = work.groupby("date")["ghi"].sum().div(1000.0).rename("insolation")
    idx = pd.DatetimeIndex(daily.index)
    return pd.DataFrame(
        {"doy": idx.dayofyear, "month": idx.month, "insolation": daily.to_numpy()}
    )


def daily_climatology(
    repo: DataRepository,
    location: Location,
    start: date,
    end: date,
    *,
    target: str = "shortwave_radiation",
    window_days: int = 7,
) -> pd.DataFrame:
    """Expected daily solar energy by day-of-year with P10/P50/P90 across years.

    Each day-of-year pools all daily values within +/- ``window_days`` (circular), so
    the percentiles come from ~75 samples over a 5-year record rather than 5.
    """
    daily = _daily_insolation(repo, location, start, end, target)
    by_doy = {
        d: daily.loc[daily["doy"] == d, "insolation"].to_numpy() for d in range(1, 367)
    }

    rows = []
    window = range(-window_days, window_days + 1)
    for day in range(1, 366):
        pooled = np.concatenate(
            [by_doy.get(((day - 1 + off) % 365) + 1, np.empty(0)) for off in window]
        )
        if pooled.size == 0:
            continue
        rows.append(
            {
                "doy": day,
                "mean": round(float(pooled.mean()), 3),
                "p10": round(float(np.quantile(pooled, 0.1)), 3),
                "p50": round(float(np.median(pooled)), 3),
                "p90": round(float(np.quantile(pooled, 0.9)), 3),
            }
        )

    out = pd.DataFrame(rows)
    for col in ("mean", "p10", "p50", "p90"):
        out[f"{col}_pv"] = (out[col] * PV_DELIVERED).round(3)
    return out


def monthly_climatology(
    repo: DataRepository,
    location: Location,
    start: date,
    end: date,
    *,
    target: str = "shortwave_radiation",
) -> pd.DataFrame:
    """Per-month expected daily insolation (mean/P10/P90) and monthly total."""
    daily = _daily_insolation(repo, location, start, end, target)
    days_in_month = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }
    rows = []
    for month in range(1, 13):
        vals = daily.loc[daily["month"] == month, "insolation"]
        mean_daily = float(vals.mean())
        rows.append(
            {
                "month": month,
                "mean_daily_kwh_m2": round(mean_daily, 2),
                "p10_daily_kwh_m2": round(float(vals.quantile(0.1)), 2),
                "p90_daily_kwh_m2": round(float(vals.quantile(0.9)), 2),
                "expected_monthly_kwh_m2": round(mean_daily * days_in_month[month], 1),
                "expected_monthly_pv_kwh_m2": round(
                    mean_daily * days_in_month[month] * PV_DELIVERED, 1
                ),
            }
        )
    return pd.DataFrame(rows)
