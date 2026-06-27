"""Within-Karachi site suitability + recommended EV-charging locations.

python pipelines/figures_part_b_sites.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from solarpredict.data import OpenMeteoArchiveRepository
from solarpredict.data.ingest import default_date_range
from solarpredict.siting import site_suitability
from solarpredict.viz import ev_locations_map, site_suitability_bar

REPORTS = Path("reports")
FIGDIR = REPORTS / "figures"


def main() -> None:
    repo = OpenMeteoArchiveRepository()
    start, end = default_date_range()

    ranked = site_suitability(repo, "Karachi", start, end)
    REPORTS.mkdir(exist_ok=True)
    ranked.to_csv(REPORTS / "site_suitability_karachi.csv", index=False)
    print(ranked.to_string(index=False))

    # The district-wide GHI grid (from figures_grid_heatmap.py) gives the EV map a
    # continuous resource surface under the recommended sites, if available.
    grid_csv = REPORTS / "karachi_grid_ghi.csv"
    grid = pd.read_csv(grid_csv) if grid_csv.exists() else None

    site_suitability_bar(ranked, path=FIGDIR / "fig4b_site_suitability.png")
    ev_locations_map(ranked, path=FIGDIR / "fig6_ev_locations.png", top=3, grid=grid)
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
