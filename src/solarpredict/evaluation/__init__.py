"""Evaluation: shared metrics, chronological split, and the skill-score harness."""

from __future__ import annotations

from .harness import DEFAULT_BASELINES, evaluate
from .metrics import mae, mbe, nrmse, r2, rmse, skill_score
from .split import Split, chronological_split

__all__ = [
    "DEFAULT_BASELINES",
    "Split",
    "chronological_split",
    "evaluate",
    "mae",
    "mbe",
    "nrmse",
    "r2",
    "rmse",
    "skill_score",
]
