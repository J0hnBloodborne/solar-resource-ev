"""Intra-Karachi GHI heatmap over the ~7 km sampling grid (GHI-only fetch).

python pipelines/figures_grid_heatmap.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from solarpredict.config.sites import city_grid
from solarpredict.data import OpenMeteoArchiveRepository
from solarpredict.data.ingest import default_date_range
from solarpredict.viz import grid_ghi_heatmap

REPORTS = Path("reports")
FIGDIR = REPORTS / "figures"


def main() -> None:
    repo = OpenMeteoArchiveRepository(hourly=("shortwave_radiation",))  # GHI only
    start, end = default_date_range(2)  # 2 years is plenty for the spatial pattern

    rows = []
    for loc in city_grid("Karachi"):
        ghi = repo.fetch(loc, start, end)["shortwave_radiation"].astype(float)
        daytime = ghi[ghi > 0]
        rows.append(
            {
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "mean_ghi_day": round(float(daytime.mean()), 1),
            }
        )
    grid = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    grid.to_csv(REPORTS / "karachi_grid_ghi.csv", index=False)
    print(
        f"{len(grid)} grid points; GHI {grid['mean_ghi_day'].min():.0f}"
        f"-{grid['mean_ghi_day'].max():.0f} W/m^2"
    )

    grid_ghi_heatmap(grid, path=FIGDIR / "fig2c_karachi_grid_heatmap.png")
    print(f"figure written to {FIGDIR}")


if __name__ == "__main__":
    main()
