"""Tests for the chronological split and the skill-score harness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarpredict.evaluation import DEFAULT_BASELINES, chronological_split, evaluate


def _synthetic_prepared(n_days: int = 20) -> pd.DataFrame:
    times = pd.date_range("2023-01-01", periods=n_days * 24, freq="h", tz="UTC")
    hour = times.hour.to_numpy()
    clearsky = np.clip(900 * np.sin((hour - 6) * np.pi / 12), 0, None)
    kt = 0.85 + 0.1 * np.sin(np.arange(len(times)) / 11.0)  # smooth cloud variation
    return pd.DataFrame(
        {
            "unique_id": "s",
            "ds": times,
            "y": clearsky * kt,
            "clearsky_ghi": clearsky,
            "is_day": clearsky > 5,
        }
    )


def test_chronological_split_partitions_and_orders() -> None:
    df = _synthetic_prepared(10)
    split = chronological_split(df, train_frac=0.6, val_frac=0.2)
    assert len(split.train) + len(split.val) + len(split.test) == len(df)
    assert split.train["ds"].max() <= split.val["ds"].min()
    assert split.val["ds"].max() <= split.test["ds"].min()


def test_evaluate_returns_skill_table() -> None:
    res = evaluate(_synthetic_prepared(), DEFAULT_BASELINES)
    assert {"model", "tier", "rmse", "skill"}.issubset(res.columns)
    assert set(res["model"]) == set(DEFAULT_BASELINES)
    # The reference (smart persistence) scores skill 0 against itself.
    ref_skill = res.loc[res["model"] == "smart_persistence", "skill"].iloc[0]
    assert ref_skill == 0.0
