"""quant_models：联合清洗、相关矩阵与回测（离线、无 Streamlit）。"""

import numpy as np
import pandas as pd
import pytest

from quant_models import (
    backtest_portfolio,
    clean_joint_returns,
    generate_efficient_frontier,
    returns_correlation_matrix,
)


def _sample_returns(n_rows: int = 40, n_cols: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    cols = [f"a{i}" for i in range(n_cols)]
    x = rng.standard_normal((n_rows, n_cols))
    return pd.DataFrame(x, index=idx, columns=cols)


def test_clean_joint_returns_matches_efficient_frontier_sample():
    r = _sample_returns()
    cleaned = clean_joint_returns(r, min_history=30)
    ef = generate_efficient_frontier(r, num_portfolios=20, min_history=30, random_state=0)
    assert cleaned.shape[0] == r.shape[0]
    assert ef.weights.shape[1] == cleaned.shape[1]


def test_returns_correlation_matrix_symmetric_unit_diagonal():
    r = _sample_returns()
    cm = returns_correlation_matrix(r, min_history=30)
    assert cm.shape == (r.shape[1], r.shape[1])
    assert np.allclose(cm.values, cm.values.T)
    assert np.allclose(np.diag(cm.values), 1.0)


def test_backtest_portfolio_first_day_one_and_equal_weight_manual():
    r = _sample_returns()
    cleaned = clean_joint_returns(r, min_history=30)
    w = np.array([0.5, 0.25, 0.25])
    bt = backtest_portfolio(r, w, min_history=30)
    assert np.isclose(bt["Optimal_Portfolio"].iloc[0], 1.0, rtol=1e-12)
    assert np.isclose(bt["Equal_Weight_Portfolio"].iloc[0], 1.0, rtol=1e-12)
    eq_daily = cleaned.mean(axis=1)
    raw_eq = (1 + eq_daily).cumprod()
    expected_eq = raw_eq / raw_eq.iloc[0]
    assert np.allclose(
        bt["Equal_Weight_Portfolio"].values,
        expected_eq.loc[bt.index].values,
        rtol=1e-12,
    )


def test_backtest_portfolio_idempotent_on_cleaned_frame():
    r = _sample_returns()
    cleaned = clean_joint_returns(r, min_history=30)
    w = np.ones(3) / 3
    bt1 = backtest_portfolio(r, w, min_history=30)
    bt2 = backtest_portfolio(cleaned, w, min_history=30)
    pd.testing.assert_frame_equal(bt1, bt2)


def test_backtest_portfolio_weight_length_mismatch():
    r = _sample_returns()
    with pytest.raises(ValueError, match="optimal_weights length"):
        backtest_portfolio(r, [0.5, 0.5], min_history=30)
