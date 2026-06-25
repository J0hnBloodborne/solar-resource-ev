"""Within-Karachi site suitability + recommended EV-charging locations.

python pipelines/figures_part_b_sites.py
"""

from __future__ import annotations

from pathlib import Path

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

    site_suitability_bar(ranked, path=FIGDIR / "fig4b_site_suitability.png")
    ev_locations_map(ranked, path=FIGDIR / "fig6_ev_locations.png", top=3)
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
