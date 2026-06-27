"""Intra-Karachi GHI heatmap clipped to the real district boundary (GHI-only).

Samples ERA5 GHI on a ~0.1 deg grid over the Karachi district polygon, keeps the
interior points, and renders a smooth surface clipped to the city's true shape.

python pipelines/figures_grid_heatmap.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from solarpredict.config.sites import Location
from solarpredict.data import OpenMeteoArchiveRepository
from solarpredict.data.ingest import default_date_range
from solarpredict.viz import grid_ghi_heatmap
from solarpredict.viz.geo import city_outline

REPORTS = Path("reports")
FIGDIR = REPORTS / "figures"
CITY = "Karachi"
STEP = 0.1  # ~11 km; ERA5 archive (~25 km) offers no finer real resolution


def district_grid(city: str, step: float) -> list[Location]:
    """Regular grid over the district bbox, keeping only interior points."""
    geom = city_outline(city).union_all()
    minx, miny, maxx, maxy = geom.bounds
    points: list[Location] = []
    for lat in np.arange(miny, maxy + step, step):
        for lon in np.arange(minx, maxx + step, step):
            if shapely.contains_xy(geom, float(lon), float(lat)):
                points.append(
                    Location(
                        f"kgrid {round(float(lat), 3)} {round(float(lon), 3)}",
                        float(lat),
                        float(lon),
                        city=city,
                    )
                )
    return points


def main() -> None:
    repo = OpenMeteoArchiveRepository(hourly=("shortwave_radiation",))  # GHI only
    start, end = default_date_range(1)  # 1 year is plenty for a mean-GHI map

    rows = []
    for loc in district_grid(CITY, STEP):
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
        f"{len(grid)} interior grid points; GHI {grid['mean_ghi_day'].min():.0f}"
        f"-{grid['mean_ghi_day'].max():.0f} W/m^2"
    )

    grid_ghi_heatmap(grid, path=FIGDIR / "fig2c_karachi_grid_heatmap.png", city=CITY)
    print(f"figure written to {FIGDIR}")


if __name__ == "__main__":
    main()
