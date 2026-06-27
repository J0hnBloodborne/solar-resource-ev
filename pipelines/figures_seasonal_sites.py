"""Seasonal (4-season) within-Karachi site suitability.

Splits the year into Winter/Spring/Summer/Autumn and ranks the city's districts
by solar resource + suitability in each season (the instructor's explicit ask).

python pipelines/figures_seasonal_sites.py
"""

from __future__ import annotations

from pathlib import Path

from solarpredict.data import OpenMeteoArchiveRepository
from solarpredict.data.ingest import default_date_range
from solarpredict.siting import seasonal_site_suitability
from solarpredict.viz import seasonal_site_heatmap

REPORTS = Path("reports")
FIGDIR = REPORTS / "figures"
CITY = "Karachi"


def main() -> None:
    repo = OpenMeteoArchiveRepository()
    start, end = default_date_range()

    seasonal = seasonal_site_suitability(repo, CITY, start, end)
    REPORTS.mkdir(exist_ok=True)
    seasonal.to_csv(REPORTS / "seasonal_site_suitability_karachi.csv", index=False)

    # Per-season winner (highest suitability) — quick console summary for the notes.
    for season in ("Winter", "Spring", "Summer", "Autumn"):
        block = seasonal[seasonal["season"] == season].sort_values(
            "suitability", ascending=False
        )
        top = block.iloc[0]
        print(
            f"{season:>7}: best {top['site']:<16} "
            f"GHI {top['mean_ghi_day']:.0f} W/m^2  suit {top['suitability']:.2f}"
        )

    seasonal_site_heatmap(seasonal, path=FIGDIR / "fig9_seasonal_sites.png", city=CITY)
    print(f"figure written to {FIGDIR}")


if __name__ == "__main__":
    main()
