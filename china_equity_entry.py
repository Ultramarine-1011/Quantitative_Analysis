"""兼容层：境内权益入口已迁移至 ``asset_resolver``。

新代码优先从 ``asset_resolver`` 导入；保留本模块是为了让 CLI、旧文档和旧测试
继续工作。
"""

from __future__ import annotations

from asset_resolver import (
    CHINA_EQUITY_ASSET_TYPES,
    CHINA_EQUITY_LABELS,
    DEFAULT_CHINA_EQUITY_SYMBOLS,
    EMPTY_SYMBOL_FETCH_HINT,
    GENERIC_DIAGNOSIS_FAILURE_PREFIX,
    coerce_analysis_date,
    fetch_route_for_china_equity,
    format_china_equity_user_message,
    format_diagnosis_user_message,
    load_china_equity_ohlcv,
    normalize_china_equity_asset_type,
    resolve_analysis_date_window,
    validate_analysis_date_range,
    validate_china_equity_symbol,
)

__all__ = [
    "CHINA_EQUITY_ASSET_TYPES",
    "CHINA_EQUITY_LABELS",
    "DEFAULT_CHINA_EQUITY_SYMBOLS",
    "EMPTY_SYMBOL_FETCH_HINT",
    "GENERIC_DIAGNOSIS_FAILURE_PREFIX",
    "coerce_analysis_date",
    "fetch_route_for_china_equity",
    "format_china_equity_user_message",
    "format_diagnosis_user_message",
    "load_china_equity_ohlcv",
    "normalize_china_equity_asset_type",
    "resolve_analysis_date_window",
    "validate_analysis_date_range",
    "validate_china_equity_symbol",
]

