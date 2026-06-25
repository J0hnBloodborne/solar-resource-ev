"""Tests for the figure generators (skipped on CI, which lacks the viz extra)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_model_comparison_bar_writes_png(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from solarpredict.viz import model_comparison_bar

    table = pd.DataFrame(
        {
            "model": ["smart_persistence", "extra_trees"],
            "tier": ["tier-0", "tier-1"],
            "skill": [0.0, 0.4],
            "rmse": [69.6, 41.6],
        }
    )
    out = model_comparison_bar(table, path=tmp_path / "skill.png", metric="skill")
    assert out.exists() and out.stat().st_size > 0


def test_predicted_vs_actual_writes_png(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from solarpredict.viz import predicted_vs_actual

    y = np.array([0.0, 100.0, 500.0, 900.0])
    out = predicted_vs_actual(y, y * 0.9, path=tmp_path / "scatter.png", name="m")
    assert out.exists() and out.stat().st_size > 0
