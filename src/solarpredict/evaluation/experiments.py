"""Data-efficiency experiment: hold the test set fixed, vary the training size.

This isolates the effect of *training-data amount*. Zero-shot models (Chronos)
ignore the training set, so their accuracy is flat across sizes; trained models
(trees, deep nets) improve with more data. The cross-over — modern/foundation
models leading when data is scarce, trees pulling ahead with plenty — is the story.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .harness import CONTEXT_ROWS, run_benchmark
from .split import Split

HOURS_PER_YEAR = 8766  # 365.25 * 24


def data_efficiency(
    prepared: pd.DataFrame,
    models: Sequence[str],
    train_years: Sequence[float],
    *,
    covariate_cols: tuple[str, ...] = (),
    test_rows: int = HOURS_PER_YEAR,
    context: int = CONTEXT_ROWS,
) -> pd.DataFrame:
    """Benchmark ``models`` on a fixed test set for each training size.

    Returns a long table: train_years, train_rows, model, tier, mae, rmse, ..., skill.
    The test set is the last ``test_rows`` hours; training is the ``train_years``
    immediately before the context window (so train + test stay contiguous, no gap).
    """
    ordered = prepared.sort_values("ds").reset_index(drop=True)
    test_start = len(ordered) - test_rows
    ctx_start = max(0, test_start - context)

    frames = []
    for years in train_years:
        train_rows = round(years * HOURS_PER_YEAR)
        train_start = max(0, ctx_start - train_rows)
        split = Split(
            train=ordered.iloc[train_start:ctx_start].copy(),
            val=ordered.iloc[ctx_start:test_start].copy(),
            test=ordered.iloc[test_start:].copy(),
        )
        result = run_benchmark(
            ordered, models, split=split, covariate_cols=covariate_cols, context=context
        )
        table = result.table.copy()
        table.insert(0, "train_years", years)
        table.insert(1, "train_rows", len(split.train))
        frames.append(table)

    return pd.concat(frames, ignore_index=True)
