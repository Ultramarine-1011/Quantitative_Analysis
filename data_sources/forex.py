"""East Money forex parsing and robust transport without hard private-module coupling."""

from __future__ import annotations

import importlib
import time
from typing import Any

import pandas as pd

from .eastmoney import _EASTMONEY_DEFAULT_HEADERS, _EASTMONEY_KLINE_APIS
from .errors import DataFetchError, EmptySymbolDataError
from .network import (
    RequestBudget,
    _call_akshare,
    _import_akshare,
    apply_default_network_proxy_policy,
)

_EASTMONEY_FOREX_KLINE_URLS = _EASTMONEY_KLINE_APIS

# 常用外汇品种东财 market code，作为脱离 AKShare 私有常量的内置兜底映射。
_BUILTIN_FOREX_MARKET_MAP: dict[str, int] = {
    "USDCNH": 133,
    "CNHUSD": 133,
    "EURCNH": 133,
    "CNHEUR": 133,
    "GBPCNH": 133,
    "CNHGBP": 133,
    "JPYCNH": 133,
    "CNHJPY": 133,
    "HKDCNH": 133,
    "CNHHKD": 133,
    "AUDCNH": 133,
    "CNHAUD": 133,
    "CADCNH": 133,
    "CNHCAD": 133,
    "CHFCNH": 133,
    "CNHCHF": 133,
    "SGDCNH": 133,
    "CNHSGD": 133,
    "NZDCNH": 133,
    "CNHNZD": 133,
    "EURUSD": 119,
    "USDEUR": 119,
    "GBPUSD": 119,
    "USDGBP": 119,
    "USDJPY": 119,
    "JPYUSD": 119,
    "USDHKD": 119,
    "HKDUSD": 119,
    "AUDUSD": 119,
    "USDCAD": 119,
    "USDCHF": 119,
    "NZDUSD": 119,
    "EURJPY": 119,
    "GBPJPY": 119,
    "EURGBP": 119,
    "USDSGD": 119,
}


def _resolve_forex_market_map() -> dict[str, int]:
    """Build market-code lookup from local defaults + optional AKShare symbol table."""
    merged = dict(_BUILTIN_FOREX_MARKET_MAP)
    try:
        cons = importlib.import_module("akshare.forex.cons")
        ext_map = getattr(cons, "symbol_market_map", None)
        if isinstance(ext_map, dict):
            for k, v in ext_map.items():
                try:
                    merged[str(k)] = int(v)
                except (TypeError, ValueError):
                    continue
    except Exception:  # noqa: BLE001
        # Keep working seamlessly even if AKShare renames internal forex.cons.
        pass
    return merged


def _infer_forex_market_code(pair: str, market_map: dict[str, int]) -> int:
    if pair in market_map:
        return int(market_map[pair])
    if len(pair) == 6 and pair.isalpha():
        if "CNH" in pair:
            return 133
        return 119
    if len(pair) == 7 and pair.endswith("CNYC"):
        return 120
    raise ValueError("不支持的外汇品种代码: %r（请使用标准代码如 USDCNH / EURUSD）。" % pair)


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


def _fetch_forex_hist_em_robust(
    pair: str,
    *,
    budget: RequestBudget | None = None,
) -> pd.DataFrame:
    """东财外汇日线：连接复用 + 请求预算 + 浏览器头 + 降级至 AKShare。"""
    market_map = _resolve_forex_market_map()
    market_code = _infer_forex_market_code(pair, market_map)
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
    active_budget = (
        budget
        if budget is not None
        else RequestBudget(
            total_timeout_seconds=30.0, per_request_timeout_seconds=10.0, max_attempts=6
        )
    )
    last_error: BaseException | None = None
    saw_empty_payload = False

    with httpx.Client(
        timeout=min(active_budget.per_request_timeout_seconds, 12.0),
        trust_env=False,
        headers=headers,
        follow_redirects=True,
    ) as client:
        for round_idx in range(2):
            if saw_empty_payload or active_budget.remaining_seconds() <= 0.2:
                break
            for api_url in _EASTMONEY_FOREX_KLINE_URLS:
                if (
                    active_budget.remaining_seconds() <= 0.2
                    or active_budget.attempts >= active_budget.max_attempts
                ):
                    break
                try:
                    req_timeout = active_budget.consume_attempt(f"EastMoney forex {pair}")
                    response = client.get(api_url, params=params, timeout=req_timeout)
                    response.raise_for_status()
                    payload = response.json()
                    frame = _forex_kline_json_to_dataframe(payload)
                    if not frame.empty:
                        frame.attrs["data_source"] = "eastmoney_forex_direct"
                        return frame
                    saw_empty_payload = True
                    last_error = EmptySymbolDataError(
                        "East Money forex kline returned empty klines for %r." % pair
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    continue
            if not saw_empty_payload and round_idx == 0 and active_budget.remaining_seconds() > 0.5:
                time.sleep(0.4)

    ak = _import_akshare()
    try:
        frame = _call_akshare(
            "forex_hist_em(%r) fallback" % pair,
            ak.forex_hist_em,
            symbol=pair,
            max_retries=2,
            budget=active_budget,
        )
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.attrs["data_source"] = "akshare.forex_hist_em"
        return frame
    except Exception as exc:  # noqa: BLE001
        if isinstance(last_error, EmptySymbolDataError):
            raise last_error from exc
        detail: BaseException = last_error if isinstance(last_error, BaseException) else exc
        err = DataFetchError(
            "外汇历史行情抓取失败（已尝试多东财节点与 AKShare 回退）: %r" % (detail,)
        )
        raise err from detail
