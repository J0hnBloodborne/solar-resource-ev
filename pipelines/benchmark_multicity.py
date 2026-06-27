"""Multi-city GHI benchmark — all tiers across every candidate city.

Runs the full hour-ahead benchmark (naive -> classical -> deep -> foundation) for
each Pakistani city, so the model ranking can be read 'grouped by city'. Heavy
(deep + foundation models train per city); run in the background.

python pipelines/benchmark_multicity.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from solarpredict.config.sites import CANDIDATE_CITIES
from solarpredict.data import OpenMeteoArchiveRepository
from solarpredict.data.ingest import default_date_range
from solarpredict.evaluation.harness import run_benchmark
from solarpredict.features import build_features, prepare_site
from solarpredict.viz import benchmark_city_heatmap

REPORTS = Path("reports")
FIGDIR = REPORTS / "figures"

MODELS = [
    "persistence",
    "smart_persistence",
    "climatology",
    "ridge",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "catboost",
    "lstm",
    "nhits",
    "chronos2",
]


def main() -> None:
    repo = OpenMeteoArchiveRepository()
    start, end = default_date_range()

    frames = []
    for city in CANDIDATE_CITIES:
        feat, cols = build_features(prepare_site(repo.fetch(city, start, end), city))
        result = run_benchmark(feat, MODELS, covariate_cols=tuple(cols))
        tagged = result.table.copy()
        tagged.insert(0, "city", city.name)
        frames.append(tagged)
        best = tagged.sort_values("rmse").iloc[0]
        print(
            f"{city.name:>10}: best {best['model']:<16} "
            f"RMSE {best['rmse']:.1f}  skill {best['skill']:.3f}"
        )

    table = pd.concat(frames, ignore_index=True)
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / "benchmark_multicity.csv", index=False)

    benchmark_city_heatmap(
        table,
        path=FIGDIR / "fig10_multicity_skill.png",
        metric="skill",
        model_order=MODELS,
    )
    benchmark_city_heatmap(
        table,
        path=FIGDIR / "fig10b_multicity_rmse.png",
        metric="rmse",
        model_order=MODELS,
    )
    print(f"wrote benchmark_multicity.csv + figures to {FIGDIR}")


if __name__ == "__main__":
    main()
