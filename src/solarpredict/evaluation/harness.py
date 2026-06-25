"""The benchmark harness: fit each model, score on test, report skill vs reference.

Every model is scored identically here — daytime-masked, with a forecast skill
score against smart (clear-sky) persistence. Each model receives the test rows
*preceded by a context window* of history, so sequence models (LSTM, Chronos) can
form their inputs; only the test portion is scored. Tabular/baseline models simply
predict on every row and we slice to the test portion.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from solarpredict.models import ForecastData, get_forecaster

from .metrics import mae, mbe, nrmse, r2, rmse, skill_score
from .split import Split, chronological_split

DEFAULT_BASELINES = ("persistence", "smart_persistence", "climatology")
CONTEXT_ROWS = 168  # 1 week of history prepended to the test window


def evaluate(
    prepared: pd.DataFrame,
    models: Sequence[str],
    *,
    split: Split | None = None,
    reference: str = "smart_persistence",
    covariate_cols: tuple[str, ...] = (),
    day_mask_col: str = "is_day",
    target: str = "y",
    context: int = CONTEXT_ROWS,
) -> pd.DataFrame:
    """Fit/score each model; return a metrics table sorted by RMSE.

    Columns: model, tier, mae, rmse, r2, nrmse, mbe, skill (vs ``reference``).
    Metrics use daytime rows only (``day_mask_col``).
    """
    ordered = prepared.sort_values("ds").reset_index(drop=True)
    if split is None:
        split = chronological_split(ordered)

    n_test = len(split.test)
    i_test = len(ordered) - n_test
    test_ctx = ordered.iloc[max(0, i_test - context) :].reset_index(drop=True)

    train_data = ForecastData(frame=split.train, covariate_cols=covariate_cols)
    test_data = ForecastData(frame=test_ctx, covariate_cols=covariate_cols)

    y_true = test_ctx[target].to_numpy(dtype=float)[-n_test:]
    mask = (
        test_ctx[day_mask_col].to_numpy(dtype=bool)[-n_test:]
        if day_mask_col in test_ctx.columns
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
        full = np.asarray(model.predict(test_data), dtype=float)
        preds[name] = full[-n_test:]  # score only the test portion
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
