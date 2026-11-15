"""Shared daily dates, OHLCV cleaning and validation, independent of vendors/UI."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, Union

import numpy as np
import pandas as pd

from data_sources.errors import EmptySymbolDataError

DateLike = Union[str, datetime, pd.Timestamp]  # noqa: UP007 (evaluated at import time)
REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
PRICE_COLUMNS = ("Open", "High", "Low", "Close")
MissingPolicy = Literal["raise", "drop", "ffill"]
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


def _compute_ohlcv_quality(
    frame: pd.DataFrame,
    *,
    missing_policy: str,
    dropped_invalid_rows: int = 0,
    filled_price_rows: int = 0,
    missing_volume_rows: int = 0,
) -> dict[str, Any]:
    """Summarize OHLCV integrity metrics to distinguish closures, halts, and data errors."""
    if frame.empty:
        return {
            "missing_policy": missing_policy,
            "total_rows": 0,
            "dropped_invalid_rows": dropped_invalid_rows,
            "filled_price_rows": filled_price_rows,
            "missing_volume_rows": missing_volume_rows,
            "zero_volume_rows": 0,
            "non_positive_price_rows": 0,
            "negative_volume_rows": 0,
            "ohlc_anomaly_rows": 0,
            "true_zero_return_rows": 0,
            "max_calendar_gap_days": 0,
        }

    open_s = frame["Open"]
    high_s = frame["High"]
    low_s = frame["Low"]
    close_s = frame["Close"]
    vol_s = frame["Volume"]

    max_oc = np.maximum(open_s, close_s)
    min_oc = np.minimum(open_s, close_s)
    ohlc_bad = (high_s < low_s) | (high_s < max_oc) | (low_s > min_oc)
    non_pos = (frame[list(PRICE_COLUMNS)] <= 0.0).any(axis=1)
    neg_vol = vol_s < 0.0
    zero_vol = vol_s == 0.0

    # True zero returns occur on valid consecutive observations where Close matches prior Close.
    close_unchanged = close_s.diff().eq(0.0)
    true_zero_ret_count = max(0, int(close_unchanged.sum()) - int(filled_price_rows))

    if len(frame.index) >= 2:
        gaps = frame.index.to_series().diff().dropna().dt.days
        max_gap = int(gaps.max()) if not gaps.empty else 0
    else:
        max_gap = 0

    return {
        "missing_policy": missing_policy,
        "total_rows": int(len(frame)),
        "dropped_invalid_rows": int(dropped_invalid_rows),
        "filled_price_rows": int(filled_price_rows),
        "missing_volume_rows": int(missing_volume_rows),
        "zero_volume_rows": int(zero_vol.sum()),
        "non_positive_price_rows": int(non_pos.sum()),
        "negative_volume_rows": int(neg_vol.sum()),
        "ohlc_anomaly_rows": int(ohlc_bad.sum()),
        "true_zero_return_rows": int(true_zero_ret_count),
        "max_calendar_gap_days": int(max_gap),
    }


def prepare_ohlcv(
    df: pd.DataFrame,
    *,
    drop_invalid: bool = False,
    missing_policy: MissingPolicy | None = None,
) -> pd.DataFrame:
    """Copy, sort stably, deduplicate dates, coerce numbers and enforce missing-data policy.

    Missing prices/volumes are never silently masked at the analysis boundary:
    - ``missing_policy="raise"`` (default when ``drop_invalid=False``) rejects any NaN/inf.
    - ``missing_policy="drop"`` (default when ``drop_invalid=True``) drops rows with missing
      prices, never forward-filling prices or copying yesterday's volume. If only ``Volume``
      is missing on an otherwise valid price bar, ``Volume`` defaults to ``0.0`` and is counted
      in ``attrs["data_quality"]["missing_volume_rows"]``.
    - ``missing_policy="ffill"`` forward-fills missing price columns only (never ``Volume``,
      which defaults to ``0.0`` on missing bars) and drops any remaining leading NaN rows when
      ``drop_invalid=True`` (otherwise raises).
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

    if missing_policy is None:
        policy: MissingPolicy = "drop" if drop_invalid else "raise"
    elif missing_policy in ("raise", "drop", "ffill"):
        policy = missing_policy
    else:
        raise ValueError(f"Unsupported missing_policy: {missing_policy!r}")

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
    price_cols = list(PRICE_COLUMNS)
    numeric = out[cols].apply(pd.to_numeric, errors="coerce").astype("float64")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    price_na_mask = numeric[price_cols].isna().any(axis=1)
    vol_na_mask = numeric["Volume"].isna()
    dropped_invalid_rows = 0
    filled_price_rows = 0
    missing_volume_rows = 0

    if policy == "raise":
        if numeric.isna().any().any():
            invalid = numeric.columns[numeric.isna().any()].tolist()
            raise ValueError(
                f"OHLCV columns contain NaN or infinite values (missing_policy='raise'): {', '.join(invalid)}"
            )
    elif policy == "drop":
        if price_na_mask.any():
            dropped_invalid_rows = int(price_na_mask.sum())
            keep_mask = ~price_na_mask
            out = out.loc[keep_mask].copy()
            numeric = numeric.loc[keep_mask].copy()
            vol_na_mask = numeric["Volume"].isna()
        if out.empty:
            raise EmptySymbolDataError("No valid OHLCV rows remain after dropping invalid quotes.")
        if vol_na_mask.any():
            # Never clone prior day's volume onto a day with missing volume.
            missing_volume_rows = int(vol_na_mask.sum())
            numeric.loc[vol_na_mask, "Volume"] = 0.0
    else:  # policy == "ffill"
        before_price_na = numeric[price_cols].isna().any(axis=1)
        numeric[price_cols] = numeric[price_cols].ffill()
        after_price_na = numeric[price_cols].isna().any(axis=1)
        filled_price_rows = int((before_price_na & ~after_price_na).sum())
        if vol_na_mask.any():
            missing_volume_rows = int(vol_na_mask.sum())
            numeric.loc[vol_na_mask, "Volume"] = 0.0
        if drop_invalid and after_price_na.any():
            dropped_invalid_rows += int(after_price_na.sum())
            keep_mask = ~after_price_na
            out = out.loc[keep_mask].copy()
            numeric = numeric.loc[keep_mask].copy()
            if out.empty:
                raise EmptySymbolDataError("No valid OHLCV rows remain after preprocessing.")
        elif numeric.isna().any().any():
            invalid = numeric.columns[numeric.isna().any()].tolist()
            raise ValueError(
                f"OHLCV columns still contain NaN values after forward-fill: {', '.join(invalid)}"
            )

    # Column assignment replaces object dtypes; loc assignment can retain them.
    out[cols] = numeric
    out.index.name = out.index.name or "Date"
    out.attrs = dict(df.attrs)
    prev_quality = df.attrs.get("data_quality")
    recorded_policy = policy
    if isinstance(prev_quality, Mapping):
        prev_policy = str(prev_quality.get("missing_policy", "") or "").strip()
        if (
            missing_policy is None
            and not drop_invalid
            and dropped_invalid_rows == 0
            and filled_price_rows == 0
            and missing_volume_rows == 0
            and prev_policy in ("raise", "drop", "ffill")
        ):
            recorded_policy = prev_policy  # type: ignore[assignment]
        dropped_invalid_rows += int(prev_quality.get("dropped_invalid_rows", 0))
        filled_price_rows += int(prev_quality.get("filled_price_rows", 0))
        missing_volume_rows += int(prev_quality.get("missing_volume_rows", 0))
    out.attrs["data_quality"] = _compute_ohlcv_quality(
        out,
        missing_policy=recorded_policy,
        dropped_invalid_rows=dropped_invalid_rows,
        filled_price_rows=filled_price_rows,
        missing_volume_rows=missing_volume_rows,
    )
    return out


compute_ohlcv_quality = _compute_ohlcv_quality


def normalize_ohlcv(
    df: pd.DataFrame,
    column_mapping: Mapping[str, str],
    *,
    missing_policy: MissingPolicy = "drop",
) -> pd.DataFrame:
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
    out = prepare_ohlcv(out, drop_invalid=True, missing_policy=missing_policy)
    quality = dict(out.attrs.get("data_quality", {}))
    out = out.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()
    out.index.name = "Date"
    if quality:
        out.attrs["data_quality"] = quality
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
