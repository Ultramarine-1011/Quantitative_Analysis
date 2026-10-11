"""境内 ETF / 个股：不依赖外网的清洗与入口逻辑测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from china_equity_entry import (
    format_china_equity_user_message,
    normalize_china_equity_asset_type,
    validate_analysis_date_range,
)
from data_fetcher import (
    A_SHARE_COLUMN_MAPPING,
    AKShareFetcher,
    EmptySymbolDataError,
)


def _sample_raw_ohlcv_chinese_columns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "开盘": [10.0, 10.5, 11.0],
            "最高": [10.2, 10.8, 11.2],
            "最低": [9.9, 10.3, 10.9],
            "收盘": [10.1, 10.6, 11.1],
            "成交量": [1_000_000.0, 1_100_000.0, 900_000.0],
        }
    )


@pytest.fixture
def fetcher() -> AKShareFetcher:
    return AKShareFetcher(max_retries=1, retry_delay=0.0, use_system_proxy=True)


def test_normalize_ohlcv_maps_chinese_columns_and_dtype(fetcher: AKShareFetcher) -> None:
    raw = _sample_raw_ohlcv_chinese_columns()
    out = fetcher._normalize_ohlcv(raw, A_SHARE_COLUMN_MAPPING)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.index.name == "Date"
    assert str(out.index.dtype) == "datetime64[ns]"
    assert (out.dtypes == "float64").all()
    assert len(out) == 3


def test_normalize_ohlcv_dedupes_index_keeps_last(fetcher: AKShareFetcher) -> None:
    raw = _sample_raw_ohlcv_chinese_columns()
    dup = pd.concat([raw.iloc[:1], raw], ignore_index=True)
    out = fetcher._normalize_ohlcv(dup, A_SHARE_COLUMN_MAPPING)
    assert len(out) == 3
    assert out.iloc[0]["Close"] == pytest.approx(10.1)


def test_normalize_ohlcv_accepts_datetime_index(fetcher: AKShareFetcher) -> None:
    raw = _sample_raw_ohlcv_chinese_columns().set_index("日期")
    raw = raw.rename(
        columns={
            "开盘": "Open",
            "最高": "High",
            "最低": "Low",
            "收盘": "Close",
            "成交量": "Volume",
        }
    )
    out = fetcher._normalize_ohlcv(raw, {})
    assert len(out) == 3


def test_normalize_ohlcv_empty_raises_empty_symbol(fetcher: AKShareFetcher) -> None:
    raw = _sample_raw_ohlcv_chinese_columns().iloc[:0]
    with pytest.raises(EmptySymbolDataError):
        fetcher._normalize_ohlcv(raw, A_SHARE_COLUMN_MAPPING)


def test_normalize_ohlcv_non_dataframe_raises_type_error(fetcher: AKShareFetcher) -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        fetcher._normalize_ohlcv("not-a-df", A_SHARE_COLUMN_MAPPING)  # type: ignore[arg-type]


def test_normalize_ohlcv_missing_column_raises(fetcher: AKShareFetcher) -> None:
    raw = _sample_raw_ohlcv_chinese_columns().drop(columns=["成交量"])
    with pytest.raises(ValueError, match="Missing required OHLCV"):
        fetcher._normalize_ohlcv(raw, A_SHARE_COLUMN_MAPPING)


def test_fetch_china_equity_invalid_route_raises(fetcher: AKShareFetcher) -> None:
    with pytest.raises(ValueError, match="route must be"):
        fetcher.fetch_china_equity(
            "bond",  # type: ignore[arg-type]
            "510300",
            start="2024-01-01",
            end="2024-01-31",
        )


def test_normalize_china_equity_asset_type_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="不支持的境内权益类型"):
        normalize_china_equity_asset_type("crypto")


def test_validate_analysis_date_range_rejects_inverted() -> None:
    with pytest.raises(ValueError, match="start_date"):
        validate_analysis_date_range("2024-12-31", "2024-01-01")


def test_format_china_equity_user_message_empty_symbol() -> None:
    msg = format_china_equity_user_message(EmptySymbolDataError("empty"))
    assert "未找到该代码的数据" in msg


def test_format_china_equity_user_message_generic() -> None:
    msg = format_china_equity_user_message(ValueError("bad input"))
    assert "诊断失败" in msg
    assert "bad input" in msg
