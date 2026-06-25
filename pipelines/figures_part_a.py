"""Generate Part-A figures: model comparison, predicted-vs-actual, feature importance.

Runs the full Karachi benchmark (all tiers), writes the metrics table to
``reports/`` and the figures to ``reports/figures/``. Run with:

    python pipelines/figures_part_a.py
"""

from __future__ import annotations

from pathlib import Path

from solarpredict.config.sites import CANDIDATE_CITIES
from solarpredict.data import OpenMeteoArchiveRepository
from solarpredict.data.ingest import default_date_range
from solarpredict.evaluation.harness import run_benchmark
from solarpredict.evaluation.split import chronological_split
from solarpredict.features import build_features, prepare_site
from solarpredict.models import ForecastData, get_forecaster
from solarpredict.viz import (
    feature_importance,
    model_comparison_bar,
    predicted_vs_actual,
)

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
    karachi = next(c for c in CANDIDATE_CITIES if c.name == "Karachi")
    feat, cols = build_features(
        prepare_site(
            OpenMeteoArchiveRepository().fetch(karachi, *default_date_range()), karachi
        )
    )

    result = run_benchmark(feat, MODELS, covariate_cols=tuple(cols))
    REPORTS.mkdir(exist_ok=True)
    result.table.to_csv(REPORTS / "benchmark_karachi.csv", index=False)
    print(result.table.to_string(index=False))

    model_comparison_bar(
        result.table,
        path=FIGDIR / "fig1_model_skill.png",
        metric="skill",
        title="Hour-ahead GHI — forecast skill vs smart persistence (Karachi, 5y)",
    )
    model_comparison_bar(
        result.table,
        path=FIGDIR / "fig1b_model_rmse.png",
        metric="rmse",
        title="Hour-ahead GHI — RMSE (Karachi, 5y)",
    )

    best = str(result.table.iloc[0]["model"])
    predicted_vs_actual(
        result.y_true,
        result.predictions[best],
        path=FIGDIR / "fig5_pred_vs_actual.png",
        name=f"{best} (Karachi)",
        mask=result.mask,
    )

    # Feature importance from the best tree (refit on train).
    split = chronological_split(feat)
    tree = get_forecaster("extra_trees").fit(
        ForecastData(frame=split.train, covariate_cols=tuple(cols))
    )
    feature_importance(
        list(cols),
        tree._model.feature_importances_,
        path=FIGDIR / "fig7_feature_importance.png",
    )
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
