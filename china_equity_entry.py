"""Compatibility import path. New callers should import asset_resolver."""

from asset_resolver import (
    CHINA_EQUITY_ASSET_TYPES,
    CHINA_EQUITY_LABELS,
    DEFAULT_CHINA_EQUITY_SYMBOLS,
    EMPTY_SYMBOL_FETCH_HINT,
    GENERIC_DIAGNOSIS_FAILURE_PREFIX,
    DATA_SOURCE_UNAVAILABLE_HINT,
    coerce_analysis_date,
    validate_analysis_date_range,
    resolve_analysis_date_window,
    normalize_china_equity_asset_type,
    validate_china_equity_symbol,
    fetch_route_for_china_equity,
    load_china_equity_ohlcv,
    format_diagnosis_user_message,
    format_china_equity_user_message,
)

__all__ = [
    "CHINA_EQUITY_ASSET_TYPES",
    "CHINA_EQUITY_LABELS",
    "DEFAULT_CHINA_EQUITY_SYMBOLS",
    "EMPTY_SYMBOL_FETCH_HINT",
    "GENERIC_DIAGNOSIS_FAILURE_PREFIX",
    "DATA_SOURCE_UNAVAILABLE_HINT",
    "coerce_analysis_date",
    "validate_analysis_date_range",
    "resolve_analysis_date_window",
    "normalize_china_equity_asset_type",
    "validate_china_equity_symbol",
    "fetch_route_for_china_equity",
    "load_china_equity_ohlcv",
    "format_diagnosis_user_message",
    "format_china_equity_user_message",
]
