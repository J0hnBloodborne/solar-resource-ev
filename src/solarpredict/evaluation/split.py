"""Strictly chronological train/val/test split (never shuffle — that leaks)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    """Chronological train/val/test slices of a prepared frame."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    timestamp_col: str = "ds",
) -> Split:
    """Split ``df`` by time into train/val/test (default 60/20/20).

    For ~5 years of data this is ~3 y train / 1 y val / 1 y test.
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("require 0 < train_frac, 0 <= val_frac, train+val < 1")

    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(ordered)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return Split(
        train=ordered.iloc[:i_train].copy(),
        val=ordered.iloc[i_train:i_val].copy(),
        test=ordered.iloc[i_val:].copy(),
    )
