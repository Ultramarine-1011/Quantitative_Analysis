"""公募基金伪 OHLCV：不依赖外网的形状与契约测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from data_fetcher import (
    REQUIRED_OHLCV_COLUMNS,
    AKShareFetcher,
    EmptySymbolDataError,
    _fund_build_pseudo_ohlcv_raw,
    _fund_raw_to_normalized_ohlcv,
)


@pytest.fixture
def fetcher() -> AKShareFetcher:
    return AKShareFetcher(max_retries=1, retry_delay=0.0, use_system_proxy=True)


def _cumulative_nav_series() -> pd.Series:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.Series([1.0, 1.01, 1.02], index=idx)


def test_fund_build_pseudo_ohlcv_raw_shape() -> None:
    raw = _fund_build_pseudo_ohlcv_raw(_cumulative_nav_series())
    assert list(raw.columns) == ["日期", "开盘", "最高", "最低", "收盘", "成交量"]
    assert raw["开盘"].equals(raw["最高"])
    assert raw["最高"].equals(raw["最低"])
    assert raw["最低"].equals(raw["收盘"])
    assert (raw["成交量"] == 0.0).all()
    assert len(raw) == 3


def test_fund_raw_to_normalized_ohlcv_contract(fetcher: AKShareFetcher) -> None:
    out = _fund_raw_to_normalized_ohlcv(fetcher, _cumulative_nav_series())
    assert list(out.columns) == list(REQUIRED_OHLCV_COLUMNS)
    assert out.index.name == "Date"
    assert str(out.index.dtype) == "datetime64[ns]"
    c = out["Close"]
    assert (out["Open"] == c).all() and (out["High"] == c).all() and (out["Low"] == c).all()
    assert (out["Volume"] == 0.0).all()
    assert (out.dtypes == "float64").all()


def test_fund_build_pseudo_ohlcv_raw_empty_raises() -> None:
    empty = pd.Series(dtype=float)
    with pytest.raises(EmptySymbolDataError):
        _fund_build_pseudo_ohlcv_raw(empty)
