"""GHI heatmap (hour x month) + daily GHI/PV profile for Karachi.

python pipelines/figures_ghi.py
"""

from __future__ import annotations

from pathlib import Path

from solarpredict.config.sites import CANDIDATE_CITIES
from solarpredict.data import OpenMeteoArchiveRepository
from solarpredict.data.ingest import default_date_range
from solarpredict.features import prepare_site
from solarpredict.viz import daily_ghi_profile, ghi_heatmap_hour_month

FIGDIR = Path("reports/figures")


def main() -> None:
    karachi = next(c for c in CANDIDATE_CITIES if c.name == "Karachi")
    prepared = prepare_site(
        OpenMeteoArchiveRepository().fetch(karachi, *default_date_range()), karachi
    )
    ghi_heatmap_hour_month(prepared, path=FIGDIR / "fig2b_ghi_heatmap.png")
    daily_ghi_profile(prepared, path=FIGDIR / "fig3b_daily_profile.png")
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
