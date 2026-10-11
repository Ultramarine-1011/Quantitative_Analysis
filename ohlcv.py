"""Shared daily dates, OHLCV cleaning and validation, independent of vendors/UI."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd

from data_sources.errors import EmptySymbolDataError

DateLike = Union[str, datetime, pd.Timestamp]
REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
A_SHARE_COLUMN_MAPPING = {
    "日期": "Date",
    "date": "Date",
    "Date": "Date",
    "开盘": "Open",
    "open": "Open",
    "Open": "Open",
    "最高": "High",
    "high": "High",
    "High": "High",
    "最低": "Low",
    "low": "Low",
    "Low": "Low",
    "收盘": "Close",
    "close": "Close",
    "Close": "Close",
    "成交量": "Volume",
    "volume": "Volume",
    "Volume": "Volume",
}


def coerce_analysis_date(value: object, field_name: str) -> pd.Timestamp:
    """Keep the supplied calendar date and remove timezone/time-of-day."""
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} 不是合法日期: {value!r}") from exc
    if pd.isna(ts):
        raise ValueError(f"{field_name} 不是合法日期: {value!r}")
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def validate_analysis_date_range(
    start_date: object, end_date: object
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = coerce_analysis_date(start_date, "start_date")
    end = coerce_analysis_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date 必须早于或等于 end_date。")
    return start, end


def filter_by_date_range(df: pd.DataFrame, start_date: object, end_date: object) -> pd.DataFrame:
    start, end = validate_analysis_date_range(start_date, end_date)
    filtered = df.loc[(df.index >= start) & (df.index <= end)].copy()
    if filtered.empty:
        raise ValueError(f"指定日期区间内没有可用数据：{start:%Y-%m-%d} 至 {end:%Y-%m-%d}。")
    return filtered


def prepare_ohlcv(df: pd.DataFrame, *, drop_invalid: bool = False) -> pd.DataFrame:
    """Copy, sort stably, keep the last duplicate, coerce numbers and forward-fill.

    Preserve extra feature columns and attrs. Vendor normalization drops leading
    incomplete rows; feature/chart inputs reject them. Never backfill prices.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input OHLCV DataFrame is empty.")
    missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")
    if not df.columns.is_unique:
        raise ValueError("OHLCV DataFrame contains duplicate column names.")
    out = df.copy()
    try:
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Input index must be a DatetimeIndex or convertible to datetime.") from exc
    if out.index.hasnans:
        raise ValueError("OHLCV DataFrame index contains invalid datetime values.")
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.as_unit("ns")
    out = out.sort_index(kind="stable")
    out = out.loc[~out.index.duplicated(keep="last")].copy()
    cols = list(REQUIRED_OHLCV_COLUMNS)
    numeric = out[cols].apply(pd.to_numeric, errors="coerce").astype("float64")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).ffill()
    if not drop_invalid and numeric.isna().any().any():
        invalid = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(
            f"OHLCV columns still contain NaN values after forward-fill: {', '.join(invalid)}"
        )
    # Column assignment replaces object dtypes; loc assignment can retain them.
    out[cols] = numeric
    if drop_invalid:
        out = out.dropna(subset=cols)
        if out.empty:
            raise EmptySymbolDataError("No valid OHLCV rows remain after preprocessing.")
    out.index.name = out.index.name or "Date"
    return out


def normalize_ohlcv(df: pd.DataFrame, column_mapping: Mapping[str, str]) -> pd.DataFrame:
    """Adapt a vendor frame to exactly five float64 columns and a Date index."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise EmptySymbolDataError("Input DataFrame is empty.")
    out = df.rename(columns=dict(column_mapping)).copy()
    if not out.columns.is_unique:
        raise ValueError("OHLCV mapping produces duplicate column names.")
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"]).set_index("Date")
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.loc[out.index.notna()]
    else:
        raise ValueError("A Date column or a DatetimeIndex is required for OHLCV normalization.")
    if out.empty:
        raise ValueError("All dates are invalid.")
    out = prepare_ohlcv(out, drop_invalid=True)
    out = out.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()
    out.index.name = "Date"
    return validate_ohlcv(out)


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Check the output contract without copying or repairing it."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")
    if df.empty or not df.columns.is_unique:
        raise ValueError("OHLCV DataFrame must be non-empty with unique columns.")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise ValueError("OHLCV DataFrame index must contain valid datetimes.")
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV DataFrame index must be unique and sorted ascending.")
    values = df.loc[:, list(REQUIRED_OHLCV_COLUMNS)]
    if not all(dtype == np.dtype("float64") for dtype in values.dtypes):
        raise TypeError("OHLCV columns must be float64.")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("OHLCV DataFrame contains non-finite values.")
    return df
