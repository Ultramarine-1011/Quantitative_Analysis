"""UI-independent asset routing, crypto loading and multi-asset alignment."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from asset_catalog import SUPPORTED_ASSET_TYPES
from asset_resolver import load_china_equity_ohlcv
from data_fetcher import (
    US_TREASURY_YIELD_10Y_CN_COL,
    get_fund_data,
    get_multiple_assets_close,
    load_bond_data,
    load_commodity_data,
    load_futures_foreign_ohlcv,
    load_fx_data,
    load_global_market_data,
    load_reit_data,
)
from ohlcv import (
    validate_analysis_date_range as _validate_date_range,
    normalize_ohlcv as _normalize_ohlcv,
    filter_by_date_range as _filter_by_date_range,
)

GOLD_SYMBOL_ALIASES = {"GC": "GC", "AU9999": "GC"}


def _normalize_proxy_url(proxy_url: str | None) -> str | None:
    """将用户输入的代理地址规整为带协议前缀的 URL。"""
    if proxy_url is None:
        return None

    normalized = str(proxy_url).strip()
    if not normalized:
        return None

    if "://" not in normalized:
        normalized = "http://%s" % normalized

    parsed = urlparse(normalized)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("proxy_url 格式无效，请使用 `http://127.0.0.1:端口`。")
    return normalized


def _resolve_gold_symbol(symbol: str) -> tuple[str, str]:
    """将展示代码映射为底层抓取所需的贵金属代码。"""
    display_symbol = str(symbol).strip().upper()
    if not display_symbol:
        raise ValueError("黄金代码不能为空。")
    vendor_symbol = GOLD_SYMBOL_ALIASES.get(display_symbol, display_symbol)
    return display_symbol, vendor_symbol


def _fetch_etf_data(symbol: str, start_date: object, end_date: object) -> pd.DataFrame:
    """通过 AKShareFetcher 路由抓取 ETF 并返回统一 OHLCV。"""
    return load_china_equity_ohlcv("etf", symbol, start_date, end_date, adjust="hfq")


def _fetch_ashare_stock_data(symbol: str, start_date: object, end_date: object) -> pd.DataFrame:
    """通过 AKShare ``stock_zh_a_hist`` 抓取 A 股个股并返回统一 OHLCV（Open/High/Low/Close/Volume）。"""
    return load_china_equity_ohlcv("ashare", symbol, start_date, end_date, adjust="hfq")


def _fetch_mutual_fund_data(symbol: str, start_date: object, end_date: object) -> pd.DataFrame:
    """通过 AKShare ``fund_open_fund_info_em`` 拉取开放式基金净值（Close 为累计净值）。"""
    fund_code = str(symbol).strip()
    if not fund_code:
        raise ValueError("公募基金代码不能为空。")
    df = get_fund_data(fund_code, start_date, end_date)
    df.attrs["asset_type"] = "mutual_fund"
    df.attrs["symbol"] = fund_code
    return df


def _fetch_gold_data(symbol: str, start_date: object, end_date: object) -> pd.DataFrame:
    """通过 ``futures_foreign_hist`` 加载贵金属期货行情。"""
    display_symbol, vendor_symbol = _resolve_gold_symbol(symbol)
    filtered = load_futures_foreign_ohlcv(
        vendor_symbol,
        start_date,
        end_date,
        asset_type="gold",
        unit_caption="USD · 贵金属外盘期货（主力连续，细则以数据源为准）",
    )
    filtered.attrs["symbol"] = display_symbol
    filtered.attrs["vendor_symbol"] = vendor_symbol
    return filtered


def _build_ccxt_proxy_config(proxy_url: str | None) -> dict[str, str] | None:
    """将单个代理地址扩展为 CCXT 期望的代理配置。"""
    normalized_proxy = _normalize_proxy_url(proxy_url)
    if normalized_proxy is None:
        return None
    return {
        "http": normalized_proxy,
        "https": normalized_proxy,
    }


def _fetch_crypto_ohlcv_rows(
    exchange: Any,
    symbol: str,
    start_date: object,
    end_date: object,
    timeframe: str = "1d",
    batch_limit: int = 1000,
) -> list[list[Any]]:
    """按日线分页拉取加密货币 K 线。"""
    start_ts, end_ts = _validate_date_range(start_date, end_date)
    if timeframe != "1d":
        raise ValueError("Only the daily 1d timeframe is supported.")
    if isinstance(batch_limit, bool) or not isinstance(batch_limit, int) or batch_limit < 1:
        raise ValueError("batch_limit must be a positive integer.")
    timeframe_ms = 24 * 60 * 60 * 1000
    end_ms = int((end_ts + pd.Timedelta(days=1)).timestamp() * 1000)
    since_ms = int(start_ts.timestamp() * 1000)
    rows: list[list[Any]] = []

    # CCXT 的日线接口通常需要通过 since + limit 分页，避免长区间直接被截断。
    while since_ms < end_ms:
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=batch_limit,
        )
        if not batch:
            break

        rows.extend(batch)
        last_ts = int(batch[-1][0])
        # 下一页从最后一根 K 线之后开始，避免重复抓取同一天。
        next_since = last_ts + timeframe_ms
        if next_since <= since_ms:
            break
        since_ms = next_since

        if last_ts >= end_ms:
            break

    return rows


def _normalize_crypto_data(
    rows: list[list[Any]],
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    """将 CCXT 返回的 OHLCV 列表标准化为 DataFrame。"""
    if not rows:
        raise ValueError("交易所未返回任何 OHLCV 数据。")

    raw_df = pd.DataFrame(
        rows,
        columns=["Date", "Open", "High", "Low", "Close", "Volume"],
    )
    raw_df["Date"] = pd.to_datetime(raw_df["Date"], unit="ms", utc=True).dt.tz_localize(None)
    normalized = _normalize_ohlcv(
        raw_df,
        {
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        },
    )
    return _filter_by_date_range(normalized, start_date, end_date)


def _fetch_crypto_data(
    symbol: str,
    start_date: object,
    end_date: object,
    proxy_url: str | None = None,
) -> pd.DataFrame:
    """通过 CCXT（默认 Binance）抓取加密货币 OHLCV。"""
    symbol_text = str(symbol).strip().upper()
    if not symbol_text:
        raise ValueError("加密货币交易对不能为空，例如 `BTC/USDT`。")

    try:
        ccxt = importlib.import_module("ccxt")
    except ImportError as exc:
        raise ImportError("缺少运行依赖 `ccxt`，请先安装：`pip install ccxt`。") from exc
    exchange_config: dict[str, Any] = {
        "timeout": 30000,
        "enableRateLimit": True,
    }

    proxies = _build_ccxt_proxy_config(proxy_url)
    if proxies is not None:
        # 代理只注入 CCXT 交易所客户端，不影响其他资产类型的数据源。
        exchange_config["proxies"] = proxies

    exchange = ccxt.binance(exchange_config)
    rows = _fetch_crypto_ohlcv_rows(
        exchange=exchange,
        symbol=symbol_text,
        start_date=start_date,
        end_date=end_date,
    )
    normalized = _normalize_crypto_data(rows, start_date=start_date, end_date=end_date)
    normalized.attrs["asset_type"] = "crypto"
    normalized.attrs["symbol"] = symbol_text
    if proxies is not None:
        normalized.attrs["proxy_url"] = proxies["https"]
    return normalized


def load_asset_data(
    asset_type: str,
    symbol: str,
    start_date: object,
    end_date: object,
    proxy_url: str | None = None,
) -> pd.DataFrame:
    """统一加载多类资产 OHLCV（契约一致）。"""
    start_date, end_date = _validate_date_range(start_date, end_date)
    asset_key = str(asset_type).strip().lower()
    if asset_key not in SUPPORTED_ASSET_TYPES:
        raise ValueError("不支持的资产类型: %s" % asset_type)

    # 这里是 Web 应用的数据总路由：不同资产来源不同，但返回值都必须是 OHLCV。
    if asset_key == "etf":
        return _fetch_etf_data(symbol=symbol, start_date=start_date, end_date=end_date)
    if asset_key == "ashare":
        return _fetch_ashare_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
    if asset_key == "mutual_fund":
        return _fetch_mutual_fund_data(symbol=symbol, start_date=start_date, end_date=end_date)
    if asset_key == "gold":
        return _fetch_gold_data(symbol=symbol, start_date=start_date, end_date=end_date)
    if asset_key == "crypto":
        return _fetch_crypto_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            proxy_url=proxy_url,
        )
    if asset_key == "us_stock":
        return load_global_market_data("us_stock", symbol, start_date, end_date)
    if asset_key == "hk_stock":
        return load_global_market_data("hk_stock", symbol, start_date, end_date)
    if asset_key == "global_index":
        return load_global_market_data("global_index", symbol, start_date, end_date)
    if asset_key == "bond_cn":
        return load_bond_data("cn_hist", symbol, start_date, end_date)
    if asset_key == "bond_us_yield":
        sym = str(symbol).strip()
        col: str | None = None
        if sym and sym.upper() not in ("10Y", "默认", "DEFAULT"):
            col = sym
        else:
            col = US_TREASURY_YIELD_10Y_CN_COL
        # 美债收益率用列名选择期限；默认 10Y，也允许用户填完整列名。
        return load_bond_data(
            "us_yield",
            "",
            start_date,
            end_date,
            us_yield_column=col,
        )
    if asset_key == "commodity":
        return load_commodity_data(symbol, start_date, end_date)
    if asset_key == "creit":
        return load_reit_data(symbol, start_date, end_date)
    return load_fx_data(symbol, start_date, end_date)


def load_multiple_etfs_close(
    symbols_tuple: tuple[str, ...],
    start_date: object,
    end_date: object,
    adjust: str = "hfq",
) -> pd.DataFrame:
    """多标的 A 股 ETF 收盘价宽表；参数为元组以便 Streamlit ``cache_data`` 键稳定。"""
    if not symbols_tuple:
        raise ValueError("symbols_tuple 不能为空。")
    return get_multiple_assets_close(list(symbols_tuple), start_date, end_date, adjust=adjust)


def multi_asset_leg_column_name(asset_type_key: str, symbol: str) -> str:
    """有效前沿宽表列名：`{资产类型}:{代码}`，避免跨类型代码冲突。"""
    return "%s:%s" % (str(asset_type_key).strip().lower(), str(symbol).strip())


def _split_asset_leg_segment(segment: str) -> tuple[str, str]:
    """解析单行 leg：`类型,代码` 或 `类型|代码`（仅拆第一处分隔符）。"""
    s = str(segment).strip()
    if not s:
        raise ValueError("空的标的行。")
    if "," in s:
        left, right = s.split(",", 1)
    elif "|" in s:
        left, right = s.split("|", 1)
    else:
        raise ValueError(
            "每行需为 `资产类型,代码` 或 `资产类型|代码`，例如 `etf,510300`；无效行：%r" % segment
        )
    asset_raw, symbol_raw = left.strip(), right.strip()
    if not asset_raw:
        raise ValueError("资产类型不能为空：%r" % segment)
    if not symbol_raw:
        raise ValueError("代码不能为空：%r" % segment)
    return asset_raw, symbol_raw


def parse_multi_asset_legs(text: str) -> tuple[tuple[str, str], ...]:
    """从多行文本解析 Tab3 有效前沿标的列表；校验类型、去重（保序）、至少 2 条。

    支持：换行分段；同一行内可用英文分号 `;` 分隔多条。空行忽略。
    返回的元组中资产类型已规范为小写键（``SUPPORTED_ASSET_TYPES`` 成员）。
    """
    segments: list[str] = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        for part in line.split(";"):
            part = part.strip()
            if part:
                segments.append(part)

    legs_ordered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for seg in segments:
        asset_raw, symbol_raw = _split_asset_leg_segment(seg)
        asset_key = asset_raw.strip().lower()
        if asset_key not in SUPPORTED_ASSET_TYPES:
            raise ValueError(
                "不支持的资产类型 %r；可选：%s" % (asset_raw, ", ".join(SUPPORTED_ASSET_TYPES))
            )
        key = (asset_key, symbol_raw)
        if key in seen:
            continue
        seen.add(key)
        legs_ordered.append((asset_key, symbol_raw))

    if len(legs_ordered) < 2:
        raise ValueError("有效前沿至少需要 2 个不重复标的（`类型,代码` 每行一条，或用 `;` 分隔）。")

    return tuple(legs_ordered)


def load_multi_leg_close_wide(
    legs_tuple: tuple[tuple[str, str], ...],
    start_date: object,
    end_date: object,
    proxy_url: str | None = None,
    *,
    loader: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """多类型标的收盘价宽表：逐 leg ``load_asset_data``，按日期 outer 合并后前向填充。"""
    if not legs_tuple:
        raise ValueError("legs_tuple 不能为空。")

    close_frames: list[pd.Series] = []
    load_leg = loader if loader is not None else load_asset_data
    for asset_key, symbol in legs_tuple:
        # 每条 leg 独立拉取，再统一成 Close 宽表，列名包含类型以避免代码冲突。
        col = multi_asset_leg_column_name(asset_key, symbol)
        df_leg = load_leg(
            asset_type=asset_key,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            proxy_url=proxy_url,
        )
        if df_leg.empty or "Close" not in df_leg.columns:
            raise ValueError(
                "标的 %r（列名 %r）在指定区间内无可用数据或缺少 Close 列。"
                % ("%s:%s" % (asset_key, symbol), col)
            )
        close = pd.to_numeric(df_leg["Close"], errors="coerce")
        if close.isna().all():
            raise ValueError(
                "标的 %r（列名 %r）在指定区间内 Close 全为缺失。"
                % ("%s:%s" % (asset_key, symbol), col)
            )
        close_frames.append(close.rename(col))

    if len({frame.name for frame in close_frames}) != len(close_frames):
        raise ValueError("多标的宽表不能包含重复标的列。")
    wide = pd.concat(close_frames, axis=1, join="outer")
    # 不同市场交易日不同，先 outer 对齐再前向填充，以保留尽可能多的联合样本。
    wide = wide.sort_index().ffill()

    for col in wide.columns:
        if wide[col].isna().all():
            raise ValueError(
                "标的列 %r 在指定区间内无有效收盘价，或与所选标的无可用重叠数据。" % str(col)
            )

    wide.index.name = "Date"
    return wide.astype("float64")
