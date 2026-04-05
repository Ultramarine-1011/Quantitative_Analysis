"""asset_resolver 错误文案与 DataFetchError 分支。"""

from __future__ import annotations

import pandas as pd
import pytest

from asset_resolver import format_diagnosis_user_message
from data_fetcher import (
    ChinaEquityPipelineError,
    DataFetchError,
    EmptySymbolDataError,
    NetworkFetchError,
    VendorUpstreamError,
)


def test_format_diagnosis_empty_symbol() -> None:
    msg = format_diagnosis_user_message(EmptySymbolDataError("x"))
    assert "未找到该代码的数据" in msg


def test_format_diagnosis_data_fetch_error() -> None:
    msg = format_diagnosis_user_message(DataFetchError("timeout"))
    assert "诊断失败" in msg
    assert "数据源" in msg or "接口" in msg


def test_format_diagnosis_network_vendor_china_pipeline() -> None:
    for exc in (
        NetworkFetchError("net"),
        VendorUpstreamError("http"),
        ChinaEquityPipelineError("pipe"),
    ):
        msg = format_diagnosis_user_message(exc)
        assert "诊断失败" in msg


def test_format_diagnosis_generic_value_error() -> None:
    msg = format_diagnosis_user_message(ValueError("bad"))
    assert "诊断失败" in msg
    assert "bad" in msg
