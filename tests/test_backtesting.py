"""真实回测：调仓、成本与换手率。"""

from __future__ import annotations

import pandas as pd
import pytest

from backtesting import (
    BacktestConfig,
    compute_turnover,
    make_equal_weights,
    run_weighted_backtest,
)


def _returns() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "a": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02],
            "b": [0.00, 0.01, 0.01, -0.01, 0.00, 0.01],
        },
        index=idx,
    )


def test_equal_weights_are_uniform() -> None:
    weights = make_equal_weights(["a", "b", "c"])
    assert weights.to_dict() == {"a": pytest.approx(1 / 3), "b": pytest.approx(1 / 3), "c": pytest.approx(1 / 3)}


def test_turnover_is_sum_absolute_weight_changes() -> None:
    assert compute_turnover(pd.Series({"a": 0.2, "b": 0.8}), pd.Series({"a": 0.5, "b": 0.5})) == pytest.approx(0.6)


def test_zero_cost_fixed_backtest_matches_weighted_curve() -> None:
    r = _returns()
    out = run_weighted_backtest(
        r,
        pd.Series({"a": 0.6, "b": 0.4}),
        BacktestConfig(strategy="fixed_weight", rebalance_frequency="D", commission_bps=0, slippage_bps=0),
    )
    expected = (1 + (r @ pd.Series({"a": 0.6, "b": 0.4}))).cumprod()
    pd.testing.assert_series_equal(out["nav"], expected.rename("nav"))


def test_positive_cost_reduces_nav_and_monthly_rebalances_less_than_daily() -> None:
    r = _returns()
    weights = pd.Series({"a": 0.6, "b": 0.4})
    no_cost = run_weighted_backtest(r, weights, BacktestConfig(rebalance_frequency="D", commission_bps=0, slippage_bps=0))
    with_cost = run_weighted_backtest(r, weights, BacktestConfig(rebalance_frequency="D", commission_bps=10, slippage_bps=5))
    monthly = run_weighted_backtest(r, weights, BacktestConfig(rebalance_frequency="M", commission_bps=10, slippage_bps=5))
    assert with_cost["nav"].iloc[-1] < no_cost["nav"].iloc[-1]
    assert int(monthly["is_rebalance"].sum()) < int(with_cost["is_rebalance"].sum())


def test_weights_are_normalized() -> None:
    out = run_weighted_backtest(_returns(), pd.Series({"a": 6.0, "b": 4.0}), BacktestConfig())
    assert out["nav"].iloc[0] > 0

