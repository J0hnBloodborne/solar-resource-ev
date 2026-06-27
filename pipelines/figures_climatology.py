"""Climatological expectation of the solar resource for any day of the year.

The honest answer to "what will solar be on a given future day": not a forecast
(weather is unpredictable past ~2 weeks) but the expected daily energy and its
spread across years, which is what a charger is sized against.

python pipelines/figures_climatology.py
"""

from __future__ import annotations

from pathlib import Path

from solarpredict.config.sites import CANDIDATE_CITIES
from solarpredict.data import OpenMeteoArchiveRepository
from solarpredict.data.ingest import default_date_range
from solarpredict.siting import daily_climatology, monthly_climatology
from solarpredict.viz import climatology_band

REPORTS = Path("reports")
FIGDIR = REPORTS / "figures"
CITY = "Karachi"


def main() -> None:
    repo = OpenMeteoArchiveRepository()
    start, end = default_date_range()
    city = next(c for c in CANDIDATE_CITIES if c.name == CITY)

    daily = daily_climatology(repo, city, start, end)
    monthly = monthly_climatology(repo, city, start, end)
    REPORTS.mkdir(exist_ok=True)
    daily.to_csv(REPORTS / "climatology_daily_karachi.csv", index=False)
    monthly.to_csv(REPORTS / "climatology_monthly_karachi.csv", index=False)

    print("Expected daily insolation (kWh/m^2/day) and delivered PV by month:")
    print(
        monthly[
            [
                "month",
                "mean_daily_kwh_m2",
                "p10_daily_kwh_m2",
                "p90_daily_kwh_m2",
                "expected_monthly_pv_kwh_m2",
            ]
        ].to_string(index=False)
    )
    lo = daily.loc[daily["p50"].idxmin()]
    hi = daily.loc[daily["p50"].idxmax()]
    print(
        f"\nMedian day ranges from {lo['p50']:.1f} kWh/m^2 (doy {int(lo['doy'])}) "
        f"to {hi['p50']:.1f} (doy {int(hi['doy'])})."
    )

    climatology_band(daily, path=FIGDIR / "fig12_climatology.png", city=CITY)
    print(f"figure written to {FIGDIR}")


if __name__ == "__main__":
    main()
