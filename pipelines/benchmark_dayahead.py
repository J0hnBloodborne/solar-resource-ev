"""Day-ahead (t+24h) GHI benchmark for Karachi, against the hour-ahead horizon.

Runs the full model lineup at a 24-hour lead and compares it to the hour-ahead
result on the same split. Day-ahead is the harder, operationally relevant horizon
(a charger schedules a day out), and it is where the deep/foundation models close
the gap on the trees.

python pipelines/benchmark_dayahead.py
"""

from __future__ import annotations

from pathlib import Path

from solarpredict.config.sites import CANDIDATE_CITIES
from solarpredict.data import OpenMeteoArchiveRepository
from solarpredict.data.ingest import default_date_range
from solarpredict.evaluation.harness import run_benchmark
from solarpredict.features import build_features, prepare_site
from solarpredict.viz import horizon_comparison_bar, model_comparison_bar

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
    prepared = prepare_site(
        OpenMeteoArchiveRepository().fetch(karachi, *default_date_range()), karachi
    )

    feat_h1, cols_h1 = build_features(prepared, horizon=1)
    feat_h24, cols_h24 = build_features(prepared, horizon=24)
    near = run_benchmark(feat_h1, MODELS, covariate_cols=tuple(cols_h1), horizon=1)
    far = run_benchmark(feat_h24, MODELS, covariate_cols=tuple(cols_h24), horizon=24)

    REPORTS.mkdir(exist_ok=True)
    far.table.to_csv(REPORTS / "benchmark_dayahead_karachi.csv", index=False)
    print("Day-ahead (t+24) Karachi:")
    print(far.table[["model", "tier", "rmse", "skill"]].to_string(index=False))

    horizon_comparison_bar(
        near.table, far.table, path=FIGDIR / "fig13_horizon_skill.png"
    )
    model_comparison_bar(
        far.table,
        path=FIGDIR / "fig13b_dayahead_skill.png",
        metric="skill",
        title="Day-ahead GHI — skill vs day-ahead smart persistence (Karachi, 5y)",
    )
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
