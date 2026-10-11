"""East Money forex parsing and robust transport."""

from __future__ import annotations

import importlib
import time
from typing import Any

import pandas as pd

from .eastmoney import _EASTMONEY_DEFAULT_HEADERS, _EASTMONEY_KLINE_APIS
from .errors import DataFetchError, EmptySymbolDataError
from .network import apply_default_network_proxy_policy, _import_akshare, _call_akshare

_EASTMONEY_FOREX_KLINE_URLS = _EASTMONEY_KLINE_APIS


def _load_httpx_module() -> Any:
    try:
        return importlib.import_module("httpx")
    except ImportError as exc:
        raise ImportError("外汇历史行情直连需要 httpx，请执行：`pip install httpx`。") from exc


def _forex_kline_json_to_dataframe(data_json: dict[str, Any]) -> pd.DataFrame:
    """解析东财外汇 K 线 JSON，列结构与 AKShare ``forex_hist_em`` 一致。"""
    if not isinstance(data_json, dict):
        return pd.DataFrame()
    block = data_json.get("data")
    if not isinstance(block, dict) or not block.get("klines"):
        return pd.DataFrame()
    rows = [str(item).split(",") for item in block["klines"]]
    if any(len(row) < 8 for row in rows):
        raise DataFetchError("East Money forex kline row is missing required fields.")
    temp_df = pd.DataFrame(
        {
            "日期": [row[0] for row in rows],
            "代码": block.get("code", ""),
            "名称": block.get("name", ""),
            "今开": [row[1] for row in rows],
            "最新价": [row[2] for row in rows],
            "最高": [row[3] for row in rows],
            "最低": [row[4] for row in rows],
            "振幅": [row[7] for row in rows],
        }
    )
    temp_df["日期"] = pd.to_datetime(temp_df["日期"], errors="coerce").dt.date
    columns = ["今开", "最新价", "最高", "最低", "振幅"]
    temp_df[columns] = temp_df[columns].apply(pd.to_numeric, errors="coerce")
    return temp_df


def _fetch_forex_hist_em_robust(pair: str) -> pd.DataFrame:
    """东财外汇日线：多节点 + 浏览器头 + ``trust_env=False``，避免裸请求被对端直接断开。"""
    cons = importlib.import_module("akshare.forex.cons")
    symbol_map: dict[str, int] = getattr(cons, "symbol_market_map", {})
    if pair not in symbol_map:
        raise ValueError("不支持的外汇品种代码: %r（请使用东财代码如 USDCNH）。" % pair)

    market_code = int(symbol_map[pair])
    params: dict[str, Any] = {
        "secid": "%d.%s" % (market_code, pair),
        "klt": "101",
        "fqt": "1",
        "lmt": "50000",
        "end": "20500000",
        "iscca": "1",
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64",
        "ut": "f057cbcbce2a86e2866ab8877db1d059",
        "forcect": 1,
    }

    headers = dict(_EASTMONEY_DEFAULT_HEADERS)
    headers["Referer"] = "https://quote.eastmoney.com/center/gridlist.html#forex_all"
    headers["Accept"] = "application/json, text/plain, */*"

    apply_default_network_proxy_policy()
    httpx = _load_httpx_module()
    last_error: BaseException | None = None

    for round_idx in range(3):
        for api_url in _EASTMONEY_FOREX_KLINE_URLS:
            try:
                with httpx.Client(
                    timeout=45.0,
                    trust_env=False,
                    headers=headers,
                    follow_redirects=True,
                ) as client:
                    response = client.get(api_url, params=params)
                    response.raise_for_status()
                    payload = response.json()
                frame = _forex_kline_json_to_dataframe(payload)
                if not frame.empty:
                    return frame
                last_error = EmptySymbolDataError(
                    "East Money forex kline returned empty klines for %r." % pair
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        if round_idx < 2:
            time.sleep(0.4 * (round_idx + 1))

    ak = _import_akshare()
    try:
        return _call_akshare("forex_hist_em(%r) fallback" % pair, ak.forex_hist_em, symbol=pair)
    except Exception as exc:  # noqa: BLE001
        detail: BaseException = last_error if isinstance(last_error, BaseException) else exc
        err = DataFetchError(
            "外汇历史行情抓取失败（已尝试多东财节点与 AKShare 回退）: %r" % (detail,)
        )
        raise err from detail
