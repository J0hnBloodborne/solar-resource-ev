"""Tests for the error metrics and skill score."""

from __future__ import annotations

import numpy as np

from solarpredict.evaluation.metrics import mae, mbe, nrmse, r2, rmse, skill_score


def test_mae_and_rmse() -> None:
    y_true = np.array([0.0, 2.0, 4.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    assert mae(y_true, y_pred) == 2.0
    assert rmse(y_true, y_pred) == np.sqrt((0 + 4 + 16) / 3)


def test_mbe_sign() -> None:
    # Over-forecasting -> positive bias.
    assert mbe([1.0, 1.0], [2.0, 2.0]) == 1.0


def test_perfect_prediction_r2() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert r2(y, y) == 1.0


def test_mask_restricts_scoring() -> None:
    y_true = np.array([0.0, 10.0, 0.0])
    y_pred = np.array([5.0, 10.0, 5.0])
    # With the daytime mask only the middle (perfect) sample counts.
    assert mae(y_true, y_pred, mask=[False, True, False]) == 0.0


def test_skill_score() -> None:
    assert skill_score(0.5, 1.0) == 0.5
    assert skill_score(1.0, 1.0) == 0.0
    assert skill_score(2.0, 1.0) == -1.0


def test_nrmse_is_fraction() -> None:
    y_true = np.array([10.0, 10.0, 10.0])
    y_pred = np.array([11.0, 9.0, 10.0])
    # RMSE = sqrt((1+1+0)/3), normalized by mean 10.
    assert np.isclose(nrmse(y_true, y_pred), np.sqrt(2 / 3) / 10)
