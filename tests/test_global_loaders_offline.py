"""多资产 loader：mock AKShare，不访问外网。"""

from __future__ import annotations

import pandas as pd
import pytest

import data_fetcher as dfetch
from data_fetcher import (
    REQUIRED_OHLCV_COLUMNS,
    US_TREASURY_YIELD_10Y_CN_COL,
    _forex_kline_json_to_dataframe,
    load_bond_data,
    load_global_market_data,
)


class _AkUsStock:
    @staticmethod
    def stock_us_daily(symbol: str = "", adjust: str = "") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "open": [10.0, 10.5],
                "high": [10.2, 10.8],
                "low": [9.9, 10.3],
                "close": [10.1, 10.6],
                "volume": [1_000_000.0, 1_100_000.0],
            }
        )


class _AkBondUs:
    @staticmethod
    def bond_zh_us_rate(start_date: str = "") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "日期": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                US_TREASURY_YIELD_10Y_CN_COL: [4.0, 4.2],
                "美国国债收益率2年": [3.0, 3.1],
            }
        )


def test_load_global_market_us_stock_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dfetch, "_import_akshare", lambda: _AkUsStock())
    out = load_global_market_data(
        "us_stock",
        "AAPL",
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    )
    assert list(out.columns) == list(REQUIRED_OHLCV_COLUMNS)
    assert out.attrs.get("asset_type") == "us_stock"
    assert out.attrs.get("series_kind") == "price"
    assert len(out) == 2


def test_forex_kline_json_to_dataframe_shape() -> None:
    payload = {
        "data": {
            "klines": ["2024-01-02,1.0,1.1,1.2,0.9,0,0,0.1,0,0,0,0,0"],
            "code": "USDCNH",
            "name": "离岸人民币",
        }
    }
    df = _forex_kline_json_to_dataframe(payload)
    assert len(df) == 1
    assert "今开" in df.columns
    assert float(df.iloc[0]["最新价"]) == pytest.approx(1.1)


def test_load_bond_us_yield_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dfetch, "_import_akshare", lambda: _AkBondUs())
    out = load_bond_data(
        "us_yield",
        "",
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        us_yield_column=US_TREASURY_YIELD_10Y_CN_COL,
    )
    assert list(out.columns) == list(REQUIRED_OHLCV_COLUMNS)
    assert out.attrs.get("series_kind") == "yield_level"
    assert float(out.iloc[-1]["Close"]) == pytest.approx(4.2)
    assert len(out) == 2
