"""因子研究：基础因子、IC 与分组收益。"""

from __future__ import annotations

import pandas as pd
import pytest

from factor_research import (
    compute_information_coefficient,
    compute_quantile_returns,
    make_price_factors,
)


def _prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {
            "a": [10, 11, 12, 13, 14, 15, 16, 17],
            "b": [10, 10, 9, 9, 8, 8, 7, 7],
            "c": [5, 5.2, 5.1, 5.4, 5.5, 5.6, 5.8, 5.9],
        },
        index=idx,
    )


def test_make_price_factors_outputs_expected_names() -> None:
    factors = make_price_factors(_prices(), momentum_window=2, volatility_window=3)
    assert {"momentum_2", "volatility_3", "reversal_1"}.issubset(factors)
    assert factors["momentum_2"].shape == _prices().shape


def test_ic_uses_aligned_factor_and_forward_returns_dates() -> None:
    prices = _prices()
    factors = make_price_factors(prices, momentum_window=2)
    forward = prices.pct_change().shift(-1)
    ic = compute_information_coefficient(factors["momentum_2"], forward)
    assert not ic.empty
    assert ic.index.max() <= forward.dropna().index.max()


def test_quantile_returns_requires_enough_assets() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    one_asset = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
    with pytest.raises(ValueError, match="at least"):
        compute_quantile_returns(one_asset, one_asset, n_quantiles=5)

