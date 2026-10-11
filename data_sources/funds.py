"""Open-ended fund NAV parsing and cumulative-NAV OHLCV adaptation."""

from __future__ import annotations

import importlib
import inspect

import numpy as np

import pandas as pd

from ohlcv import (
    A_SHARE_COLUMN_MAPPING,
    DateLike,
    filter_by_date_range as _filter_ohlcv_by_user_date_range,
    normalize_ohlcv,
    validate_analysis_date_range,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .china_equity import AKShareFetcher
from .errors import DataFetchError, EmptySymbolDataError
from .network import apply_default_network_proxy_policy, _call_akshare


def _fund_find_date_column(df: pd.DataFrame) -> str:
    """从 AKShare 基金净值表中解析日期列名。"""
    preferred = ("日期", "净值日期", "x", "date", "Date", "时间")
    for name in preferred:
        if name in df.columns:
            return name
    best_col: str | None = None
    best_valid = 0
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid = int(parsed.notna().sum())
        if valid > best_valid:
            best_valid = valid
            best_col = str(col)
    if not best_col or best_valid == 0:
        raise ValueError("无法在基金净值数据中识别日期列。")
    return best_col


def _fund_pick_cumulative_column(df: pd.DataFrame, date_col: str) -> str | None:
    """若表中已有累计净值列则返回其列名。"""
    for col in df.columns:
        if col == date_col:
            continue
        label = str(col)
        if "累计" in label and "净值" in label:
            return str(col)
    return None


def _fund_pick_primary_value_column(df: pd.DataFrame, date_col: str) -> str:
    """在单位净值走势等表中选取主要数值列（单位净值或接口英文列）。"""
    for col in df.columns:
        if col == date_col:
            continue
        label = str(col).strip().lower()
        if label == "equityreturn":
            return str(col)
        if "单位" in str(col) and "净值" in str(col):
            return str(col)
    # 常见兜底：除日期外第一个可解析为数值的列
    for col in df.columns:
        if col == date_col:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if int(numeric.notna().sum()) > 0:
            return str(col)
    raise ValueError("无法在基金净值数据中识别净值数值列。")


def _fund_pick_cumulative_value_column(df: pd.DataFrame, date_col: str) -> str:
    """在累计净值走势表中优先选取累计净值列。"""
    picked = _fund_pick_cumulative_column(df, date_col)
    if picked is not None:
        return picked
    for col in df.columns:
        if col == date_col:
            continue
        label = str(col).strip().lower()
        if label in {"accumulatednetvalue", "accunetvalue", "totalnetvalue"}:
            return str(col)
    return _fund_pick_primary_value_column(df, date_col)


def _fund_frame_to_sorted_nav(
    df: pd.DataFrame,
    *,
    value_column: str | None = None,
    prefer_cumulative_value: bool = False,
) -> pd.DataFrame:
    """将单张 AK 返回表规范为列 ``_dt``、``nav``，按日期升序、去重。"""
    if df.empty:
        raise EmptySymbolDataError("Fund NAV DataFrame is empty.")
    date_col = _fund_find_date_column(df)
    if value_column is not None:
        val_col = value_column
    elif prefer_cumulative_value:
        val_col = _fund_pick_cumulative_value_column(df, date_col)
    else:
        val_col = _fund_pick_primary_value_column(df, date_col)
    # 基金接口返回的是净值序列，不是真正 K 线；先整理成日期 + 净值两列。
    out = pd.DataFrame(
        {
            "_dt": pd.to_datetime(df[date_col], errors="coerce"),
            "nav": pd.to_numeric(df[val_col], errors="coerce").replace([np.inf, -np.inf], np.nan),
        }
    )
    out = out.dropna(subset=["_dt", "nav"])
    if out.empty:
        raise EmptySymbolDataError("Fund NAV parsing yielded no valid rows.")
    out = out.sort_values("_dt", kind="stable").drop_duplicates(subset=["_dt"], keep="last")
    return out


def _fund_build_pseudo_ohlcv_raw(close_series: pd.Series) -> pd.DataFrame:
    """将累计净值序列构造成带中文列的伪 K 线表，供 ``A_SHARE_COLUMN_MAPPING`` + ``_normalize_ohlcv`` 使用。

    ``close_series`` 的索引须为可解析为日期的值；返回表含 ``日期`` 与 OHLCV 中文列。
    """
    idx = pd.to_datetime(close_series.index, errors="coerce")
    vals = pd.to_numeric(close_series.values, errors="coerce")
    frame = pd.DataFrame({"_dt": idx, "收盘": vals}).dropna(subset=["_dt", "收盘"])
    if frame.empty:
        raise EmptySymbolDataError("No valid cumulative NAV points for pseudo OHLCV.")
    close = frame["收盘"].astype("float64")
    zeros = pd.Series(0.0, index=frame.index, dtype="float64")
    # 公募基金没有日内开高低和成交量，这里用净值复制 O/H/L/C，成交量置 0。
    return pd.DataFrame(
        {
            "日期": frame["_dt"],
            "开盘": close,
            "最高": close,
            "最低": close,
            "收盘": close,
            "成交量": zeros,
        }
    )


def _fund_raw_to_normalized_ohlcv(
    fetcher: AKShareFetcher,
    cumulative_close: pd.Series,
) -> pd.DataFrame:
    """累计净值序列 → 伪 OHLCV → ``_normalize_ohlcv``。"""
    raw = _fund_build_pseudo_ohlcv_raw(cumulative_close)
    return fetcher._normalize_ohlcv(raw, A_SHARE_COLUMN_MAPPING)


def get_fund_data(fund_code: str, start_date: DateLike, end_date: DateLike) -> pd.DataFrame:
    """拉取开放式基金净值并输出统一 OHLCV 契约。

    使用 ``ak.fund_open_fund_info_em(symbol=..., indicator=...)``（旧版为 ``fund=``）；``Close``（及 O/H/L）对应**累计净值**。
    若「单位净值走势」结果中无累计净值列，则再请求「累计净值走势」并按日期左对齐合并。

    Parameters
    ----------
    fund_code
        基金代码，如 ``"009691"``。
    start_date, end_date
        分析区间；清洗后按该区间裁剪（与 Web 端一致）。

    Returns
    -------
    pd.DataFrame
        索引为 ``DatetimeIndex``，列为 ``Open``/``High``/``Low``/``Close``/``Volume``；
        基金场景下 O=H=L=Close（累计净值），``Volume`` 为 0。

    Raises
    ------
    ImportError
        未安装 ``akshare``。
    ValueError
        代码为空、日期非法、或区间内无数据。
    DataFetchError / EmptySymbolDataError
        抓取或清洗失败。
    """
    code = str(fund_code).strip()
    if not code:
        raise ValueError("fund_code must be a non-empty string.")
    validate_analysis_date_range(start_date, end_date)

    apply_default_network_proxy_policy()

    def _call_fund_info(indicator: str) -> pd.DataFrame:
        ak = importlib.import_module("akshare")
        fn = ak.fund_open_fund_info_em
        # Select the old parameter name before calling, so a vendor TypeError
        # is not mistaken for an API-signature change and issued twice.
        parameters = inspect.signature(fn).parameters
        symbol_key = "fund" if "fund" in parameters and "symbol" not in parameters else "symbol"
        return _call_akshare(
            "fund_open_fund_info_em", fn, **{symbol_key: code}, indicator=indicator
        )

    try:
        raw_unit = _call_fund_info("单位净值走势")
    except ImportError as exc:
        raise ImportError(
            "AKShare is required for fund data. Install it with `pip install akshare`."
        ) from exc
    except EmptySymbolDataError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataFetchError("fund_open_fund_info_em (单位净值走势) failed: %r" % (exc,)) from exc

    if not isinstance(raw_unit, pd.DataFrame) or raw_unit.empty:
        raise EmptySymbolDataError(
            "fund_open_fund_info_em returned no unit NAV data for %r." % code
        )

    date_col = _fund_find_date_column(raw_unit)
    cum_col = _fund_pick_cumulative_column(raw_unit, date_col)

    if cum_col is not None:
        # 优先使用同一张表里的累计净值列，避免多请求一次接口。
        cum_series = _fund_frame_to_sorted_nav(raw_unit, value_column=cum_col)
        cumulative_close = cum_series.set_index("_dt")["nav"]
        normalized = normalize_ohlcv(
            _fund_build_pseudo_ohlcv_raw(cumulative_close), A_SHARE_COLUMN_MAPPING
        )
    else:
        unit_sorted = _fund_frame_to_sorted_nav(raw_unit)

        try:
            raw_cum = _call_fund_info("累计净值走势")
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError(
                "fund_open_fund_info_em (累计净值走势) failed: %r" % (exc,)
            ) from exc

        if not isinstance(raw_cum, pd.DataFrame) or raw_cum.empty:
            raise EmptySymbolDataError(
                "fund_open_fund_info_em returned no cumulative NAV data for %r." % code
            )

        cum_sorted = _fund_frame_to_sorted_nav(raw_cum, prefer_cumulative_value=True)
        merged = unit_sorted.merge(cum_sorted, on="_dt", how="left", suffixes=("_unit", "_cum"))
        if "nav_cum" not in merged.columns:
            raise EmptySymbolDataError(
                "Cumulative NAV merge produced unexpected columns for %r." % code
            )
        merged = merged.assign(nav_cum=lambda m: m["nav_cum"].ffill())
        merged = merged.dropna(subset=["nav_cum"])
        if merged.empty:
            raise EmptySymbolDataError(
                "No overlapping cumulative NAV after aligning with unit NAV for %r." % code
            )
        cumulative_close = merged.set_index("_dt")["nav_cum"]
        normalized = normalize_ohlcv(
            _fund_build_pseudo_ohlcv_raw(cumulative_close), A_SHARE_COLUMN_MAPPING
        )

    out = _filter_ohlcv_by_user_date_range(normalized, start_date, end_date)
    out.attrs.update(asset_type="mutual_fund", symbol=code, series_kind="price")
    return out
