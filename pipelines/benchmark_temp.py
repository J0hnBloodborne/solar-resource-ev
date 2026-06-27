"""Second forecast target: hour-ahead 2 m air temperature (S2Cool predicted both).

Runs the same model tiers on temperature (skill vs plain persistence, since the
clear-sky reference is GHI-specific and there is no night mask for temperature),
and quantifies the heat-derating loss on PV power per Karachi site (hotter inland
sites lose more than the cooler coast).

python pipelines/benchmark_temp.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from solarpredict.config.sites import CANDIDATE_CITIES, INTRACITY_SITES
from solarpredict.data import OpenMeteoArchiveRepository
from solarpredict.data.ingest import default_date_range
from solarpredict.evaluation.harness import run_benchmark
from solarpredict.features import build_features
from solarpredict.solar import ghi_to_power_kw, ghi_to_power_kw_temp
from solarpredict.viz import model_comparison_bar

REPORTS = Path("reports")
FIGDIR = REPORTS / "figures"

# No smart_persistence/climatology-clear-sky here — temperature has no clear-sky model.
TEMP_MODELS = [
    "persistence",
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
TEMP_COVS = (
    "shortwave_radiation",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
)


def temperature_benchmark(repo: OpenMeteoArchiveRepository) -> pd.DataFrame:
    karachi = next(c for c in CANDIDATE_CITIES if c.name == "Karachi")
    raw = repo.fetch(karachi, *default_date_range())
    df = raw.rename(columns={"timestamp": "ds", "temperature_2m": "y"}).copy()
    df["unique_id"] = karachi.slug
    feat, cols = build_features(df, target="y", covariates=TEMP_COVS)
    result = run_benchmark(
        feat, TEMP_MODELS, covariate_cols=tuple(cols), reference="persistence"
    )
    return result.table


def derating_by_site(repo: OpenMeteoArchiveRepository) -> pd.DataFrame:
    """Annual PV energy lost to cell heating per Karachi site (cooler = less loss)."""
    start, end = default_date_range()
    rows = []
    for site in INTRACITY_SITES["Karachi"]:
        frame = repo.fetch(site, start, end)
        ghi = frame["shortwave_radiation"].to_numpy(float)
        temp = frame["temperature_2m"].to_numpy(float)
        e_base = float(ghi_to_power_kw(ghi).sum())
        e_temp = float(ghi_to_power_kw_temp(ghi, temp).sum())
        rows.append(
            {
                "site": site.name,
                "mean_temp_c": round(float(temp.mean()), 1),
                "heat_loss_pct": round(100.0 * (1.0 - e_temp / e_base), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("heat_loss_pct").reset_index(drop=True)


def main() -> None:
    repo = OpenMeteoArchiveRepository()
    REPORTS.mkdir(exist_ok=True)

    table = temperature_benchmark(repo)
    table.to_csv(REPORTS / "benchmark_temp_karachi.csv", index=False)
    print("Temperature (hour-ahead) benchmark:")
    print(table.to_string(index=False))

    model_comparison_bar(
        table,
        path=FIGDIR / "fig11_temp_skill.png",
        metric="skill",
        title="Hour-ahead temperature — skill vs persistence (Karachi, 5y)",
    )
    model_comparison_bar(
        table,
        path=FIGDIR / "fig11b_temp_rmse.png",
        metric="rmse",
        title="Hour-ahead temperature — RMSE degC (Karachi, 5y)",
    )

    derating = derating_by_site(repo)
    derating.to_csv(REPORTS / "temp_derating_karachi.csv", index=False)
    print("\nPV heat-derating by site (cooler sites lose less):")
    print(derating.to_string(index=False))
    print(f"\nfigures written to {FIGDIR}")


if __name__ == "__main__":
    main()
