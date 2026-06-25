"""Data-efficiency experiment: model skill vs training-set size (fixed test).

Shows that zero-shot/foundation and deep models stay viable when data is scarce,
while trees need plenty of history to pull ahead. Run with:

    python pipelines/data_efficiency.py
"""

from __future__ import annotations

from pathlib import Path

from solarpredict.config.sites import CANDIDATE_CITIES
from solarpredict.data import OpenMeteoArchiveRepository
from solarpredict.data.ingest import default_date_range
from solarpredict.evaluation import data_efficiency
from solarpredict.features import build_features, prepare_site
from solarpredict.viz import data_efficiency_lines

REPORTS = Path("reports")
FIGDIR = REPORTS / "figures"

MODELS = ["smart_persistence", "extra_trees", "xgboost", "nhits", "lstm", "chronos2"]
# include sub-year sizes (1/3/6 months) where trained models degrade and the
# zero-shot foundation model's flat line becomes an advantage
TRAIN_YEARS = [0.083, 0.25, 0.5, 1, 2, 3, 4]


def main() -> None:
    karachi = next(c for c in CANDIDATE_CITIES if c.name == "Karachi")
    feat, cols = build_features(
        prepare_site(
            OpenMeteoArchiveRepository().fetch(karachi, *default_date_range()), karachi
        )
    )

    df = data_efficiency(feat, MODELS, TRAIN_YEARS, covariate_cols=tuple(cols))
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "data_efficiency_karachi.csv", index=False)

    print("\nForecast skill by model x training years (fixed 1-year test):\n")
    print(df.pivot(index="model", columns="train_years", values="skill").to_string())
    print("\nRMSE by model x training years:\n")
    print(df.pivot(index="model", columns="train_years", values="rmse").to_string())

    data_efficiency_lines(
        df, path=FIGDIR / "fig8_data_efficiency_skill.png", metric="skill"
    )
    data_efficiency_lines(
        df, path=FIGDIR / "fig8b_data_efficiency_rmse.png", metric="rmse"
    )
    print(f"\nfigures written to {FIGDIR}")


if __name__ == "__main__":
    main()
