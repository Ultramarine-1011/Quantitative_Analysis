"""Cryptocurrency CCXT fetching, normalization, and proxy configuration."""

from __future__ import annotations

import importlib
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunsplit

import pandas as pd

from ohlcv import filter_by_date_range, normalize_ohlcv, validate_analysis_date_range


def normalize_proxy_url(proxy_url: str | None) -> str | None:
    """将用户输入的代理地址规整为带协议前缀的 URL。"""
    if proxy_url is None:
        return None

    normalized = str(proxy_url).strip()
    if not normalized:
        return None

    if "://" not in normalized:
        normalized = f"http://{normalized}"

    parsed = urlparse(normalized)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("proxy_url 格式无效，请使用 `http://127.0.0.1:端口`。")
    return normalized


def redact_proxy_url(proxy_url: str) -> str:
    """Mask embedded credentials in proxy URLs before attaching metadata."""
    parsed = urlsplit(proxy_url)
    if parsed.username is None and parsed.password is None:
        return proxy_url
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port_part = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit(
        (parsed.scheme, f"***:***@{host}{port_part}", parsed.path, parsed.query, parsed.fragment)
    )


def build_ccxt_proxy_config(proxy_url: str | None) -> dict[str, str] | None:
    """将单个代理地址扩展为 CCXT 期望的代理配置。"""
    normalized_proxy = normalize_proxy_url(proxy_url)
    if normalized_proxy is None:
        return None
    return {
        "http": normalized_proxy,
        "https": normalized_proxy,
    }


def fetch_crypto_ohlcv_rows(
    exchange: Any,
    symbol: str,
    start_date: object,
    end_date: object,
    timeframe: str = "1d",
    batch_limit: int = 1000,
) -> list[list[Any]]:
    """按日线分页拉取加密货币 K 线。"""
    start_ts, end_ts = validate_analysis_date_range(start_date, end_date)
    if timeframe != "1d":
        raise ValueError("Only the daily 1d timeframe is supported.")
    if isinstance(batch_limit, bool) or not isinstance(batch_limit, int) or batch_limit < 1:
        raise ValueError("batch_limit must be a positive integer.")
    timeframe_ms = 24 * 60 * 60 * 1000
    end_ms = int((end_ts + pd.Timedelta(days=1)).tz_localize("UTC").timestamp() * 1000)
    since_ms = int(start_ts.tz_localize("UTC").timestamp() * 1000)
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


def normalize_crypto_data(
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
    normalized = normalize_ohlcv(
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
    return filter_by_date_range(normalized, start_date, end_date)


def fetch_crypto_data(
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

    proxies = build_ccxt_proxy_config(proxy_url)
    if proxies is not None:
        # 代理只注入 CCXT 交易所客户端，不影响其他资产类型的数据源。
        exchange_config["proxies"] = proxies

    exchange = ccxt.binance(exchange_config)
    rows = fetch_crypto_ohlcv_rows(
        exchange=exchange,
        symbol=symbol_text,
        start_date=start_date,
        end_date=end_date,
    )
    normalized = normalize_crypto_data(rows, start_date=start_date, end_date=end_date)
    normalized.attrs["asset_type"] = "crypto"
    normalized.attrs["symbol"] = symbol_text
    normalized.attrs["series_kind"] = "price"
    normalized.attrs["return_basis"] = "spot_price"
    normalized.attrs["data_source"] = "ccxt.binance"
    if proxies is not None:
        normalized.attrs["proxy_url"] = redact_proxy_url(proxies["https"])
    return normalized


def fetch_usdt_usd_ohlcv(
    start_date: object,
    end_date: object,
    proxy_url: str | None = None,
) -> pd.DataFrame:
    """Attempt to fetch live USDT/USD spot daily OHLCV from a USD-quoted exchange via CCXT."""
    try:
        ccxt = importlib.import_module("ccxt")
    except ImportError as exc:
        raise ValueError("获取 USDT/USD 市场汇率需要 ccxt。") from exc

    proxies = build_ccxt_proxy_config(proxy_url)
    last_exc: Exception | None = None
    for exchange_id in ("kraken", "coinbase", "bitfinex"):
        exchange_cls = getattr(ccxt, exchange_id, None)
        if exchange_cls is None:
            continue
        cfg: dict[str, Any] = {"timeout": 15000, "enableRateLimit": True}
        if proxies is not None:
            cfg["proxies"] = proxies
        try:
            ex = exchange_cls(cfg)
            rows = fetch_crypto_ohlcv_rows(ex, "USDT/USD", start_date, end_date)
            if rows:
                return normalize_crypto_data(rows, start_date, end_date)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    raise ValueError(f"无法获取真实的 USDT/USD 市场汇率: {last_exc}")


__all__ = [
    "build_ccxt_proxy_config",
    "fetch_crypto_data",
    "fetch_crypto_ohlcv_rows",
    "fetch_usdt_usd_ohlcv",
    "normalize_crypto_data",
    "normalize_proxy_url",
    "redact_proxy_url",
]
