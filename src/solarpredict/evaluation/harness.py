"""The benchmark harness: fit each model, score on test, report skill vs reference.

Every model in the registry is scored identically here — masked to daytime, with a
forecast skill score against smart (clear-sky) persistence. This is the single
place metrics are computed, so the tiers stay comparable.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from solarpredict.models import ForecastData, get_forecaster

from .metrics import mae, mbe, nrmse, r2, rmse, skill_score
from .split import Split, chronological_split

DEFAULT_BASELINES = ("persistence", "smart_persistence", "climatology")


def evaluate(
    prepared: pd.DataFrame,
    models: Sequence[str],
    *,
    split: Split | None = None,
    reference: str = "smart_persistence",
    covariate_cols: tuple[str, ...] = (),
    day_mask_col: str = "is_day",
) -> pd.DataFrame:
    """Fit/score each model on a prepared frame; return a metrics table.

    Columns: model, tier, mae, rmse, r2, nrmse, mbe, skill (vs ``reference``).
    Metrics are computed over daytime rows only (``day_mask_col``).
    """
    if split is None:
        split = chronological_split(prepared)

    train_data = ForecastData(frame=split.train, covariate_cols=covariate_cols)
    test_data = ForecastData(frame=split.test, covariate_cols=covariate_cols)
    y_true = test_data.y
    mask = (
        split.test[day_mask_col].to_numpy(dtype=bool)
        if day_mask_col in split.test.columns
        else None
    )

    names = list(dict.fromkeys(models))  # de-dup, preserve order
    if reference not in names:
        names = [reference, *names]

    preds: dict[str, np.ndarray] = {}
    tiers: dict[str, str] = {}
    for name in names:
        model = get_forecaster(name)
        model.fit(train_data)
        preds[name] = model.predict(test_data)
        tiers[name] = model.tier

    ref_rmse = rmse(y_true, preds[reference], mask)

    rows = [
        {
            "model": name,
            "tier": tiers[name],
            "mae": round(mae(y_true, preds[name], mask), 3),
            "rmse": round(rmse(y_true, preds[name], mask), 3),
            "r2": round(r2(y_true, preds[name], mask), 4),
            "nrmse": round(nrmse(y_true, preds[name], mask), 4),
            "mbe": round(mbe(y_true, preds[name], mask), 3),
            "skill": round(skill_score(rmse(y_true, preds[name], mask), ref_rmse), 4),
        }
        for name in names
    ]
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
