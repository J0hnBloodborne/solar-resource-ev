"""Generate Part-B inter-city figures: GHI map, city ranking, seasonal curves.

python pipelines/figures_part_b_cities.py
"""

from __future__ import annotations

from pathlib import Path

from solarpredict.data import OpenMeteoArchiveRepository
from solarpredict.data.ingest import default_date_range
from solarpredict.siting import city_summary, monthly_ghi
from solarpredict.viz import city_ghi_map, city_ranking_bar, seasonal_ghi_lines

REPORTS = Path("reports")
FIGDIR = REPORTS / "figures"
HIGHLIGHT = ("Quetta", "Peshawar")  # sunniest and least-sunny endpoints


def main() -> None:
    repo = OpenMeteoArchiveRepository()
    start, end = default_date_range()

    summary = city_summary(repo, start, end)
    REPORTS.mkdir(exist_ok=True)
    summary.to_csv(REPORTS / "city_summary.csv", index=False)
    print(summary.to_string(index=False))

    city_ranking_bar(
        summary, path=FIGDIR / "fig4_city_ranking.png", highlight=HIGHLIGHT
    )
    city_ghi_map(summary, path=FIGDIR / "fig2_ghi_map.png", highlight=HIGHLIGHT)

    monthly = monthly_ghi(repo, start, end)
    seasonal_ghi_lines(
        monthly,
        path=FIGDIR / "fig3_seasonal.png",
        cities=("Quetta", "Gilgit", "Karachi", "Peshawar"),
    )
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
