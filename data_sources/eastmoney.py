"""East Money daily-kline transport and market-code resolution."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
import requests

from .errors import NetworkFetchError, VendorUpstreamError
from .network import _REQUESTS_NO_PROXY

# 东方财富 push2his 直连：浏览器风格头可降低被边缘节点直接断连的概率；
# 可被 ``EastMoneyDailyKlineClient`` / ``AKShareFetcher(..., eastmoney_headers=...)`` 覆盖或追加。
_EASTMONEY_DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "application/json, text/plain, */*",
}

# 东方财富 K 线接口多节点（与 AKShare ``stock_hist_em`` 中常见 host 对齐）；单节点被墙或限流时轮询可提高成功率。
_EASTMONEY_KLINE_APIS: tuple[str, ...] = (
    "https://push2his.eastmoney.com/api/qt/stock/kline/get",
    "https://push2.eastmoney.com/api/qt/stock/kline/get",
    "https://33.push2his.eastmoney.com/api/qt/stock/kline/get",
    "https://63.push2his.eastmoney.com/api/qt/stock/kline/get",
)


def _eastmoney_sec_market_id(symbol: str) -> int:
    """东方财富 K 线接口 ``secid`` 的市场前缀：0=深、1=沪（不依赖 AKShare 内部实现）。

    规则覆盖常见 A 股与 ETF/LOF；存疑代码由调用方再尝试另一市场前缀。
    """
    code = str(symbol).strip()
    if not code.isdigit():
        raise ValueError("East Money kline API expects a numeric security code, got %r." % symbol)
    if code.startswith(("0", "3")):
        return 0
    if code.startswith("6"):
        return 1
    if code.startswith("9"):
        return 1
    if code.startswith("2"):
        return 0
    if code.startswith(("15", "16", "18")):
        return 0
    if code.startswith(("51", "56", "58")):
        return 1
    if code.startswith("50"):
        return 1
    if code.startswith("5"):
        return 1
    if code.startswith("1"):
        return 0
    return 0


def parse_daily_kline_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse core fields while tolerating extra fields appended by the vendor."""
    if not isinstance(payload, dict):
        raise VendorUpstreamError("East Money kline payload must be a JSON object.")
    block = payload.get("data")
    if not block:
        return pd.DataFrame()
    if not isinstance(block, dict):
        raise VendorUpstreamError("East Money kline data must be a JSON object.")
    rows = [str(item).split(",") for item in block.get("klines", [])]
    if not rows:
        return pd.DataFrame()
    if any(len(row) < 6 for row in rows):
        raise VendorUpstreamError("East Money kline row is missing core OHLCV fields.")
    columns = [
        "日期",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌幅",
        "涨跌额",
        "换手率",
    ]
    rows = [row[:11] + [None] * max(0, 11 - len(row)) for row in rows]
    frame = pd.DataFrame(rows, columns=columns)
    frame["日期"] = pd.to_datetime(frame["日期"], errors="coerce")
    frame[columns[1:]] = frame[columns[1:]].apply(pd.to_numeric, errors="coerce")
    return frame


class EastMoneyDailyKlineClient:
    """东方财富 push2his 日线 K 线直连；不引用 AKShare 内部符号。"""

    def __init__(self, extra_headers: Mapping[str, str] | None = None) -> None:
        self._headers = dict(_EASTMONEY_DEFAULT_HEADERS)
        if extra_headers:
            self._headers.update(dict(extra_headers))

    @staticmethod
    def _load_httpx() -> Any:
        try:
            return importlib.import_module("httpx")
        except ImportError as exc:
            raise ImportError(
                "httpx is required for East Money direct fetch. Install it with `pip install httpx`."
            ) from exc

    @staticmethod
    def _recoverable_httpx_transport_error(exc: BaseException) -> bool:
        mod = type(exc).__module__ or ""
        name = type(exc).__name__
        if not mod.startswith("httpx") and not mod.startswith("httpcore"):
            return False
        return name in (
            "RemoteProtocolError",
            "ConnectError",
            "ConnectTimeout",
            "ProxyError",
            "ReadTimeout",
            "UnsupportedProtocol",
        )

    def fetch_daily_klines(
        self,
        symbol: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        adjust: str,
        *,
        trust_env: bool,
    ) -> pd.DataFrame:
        """多域名轮询东方财富 K 线（在新浪/腾讯与 AKShare 主源之后调用）。"""
        last_exc: BaseException | None = None
        # 东财同一个接口有多个可用域名；逐个尝试可绕过单个节点抖动或限流。
        for api_url in _EASTMONEY_KLINE_APIS:
            try:
                df = self._fetch_daily_klines_one_host(
                    api_url, symbol, start_ts, end_ts, adjust, trust_env=trust_env
                )
                if not df.empty:
                    return df
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        return pd.DataFrame()

    def _fetch_daily_klines_one_host(
        self,
        api_url: str,
        symbol: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        adjust: str,
        *,
        trust_env: bool,
    ) -> pd.DataFrame:
        """单节点东方财富 K 线；列名与 ``stock_zh_a_hist`` / ``fund_etf_hist_em`` 中文列一致。"""
        import json
        from urllib.parse import urlencode, urlparse

        httpx = self._load_httpx()
        adjust_dict = {"qfq": "1", "hfq": "2", "": "0"}
        if adjust not in adjust_dict:
            raise ValueError("adjust must be one of {'qfq', 'hfq', ''}.")

        parsed = urlparse(api_url)
        em_host = parsed.netloc
        path_base = parsed.path or "/api/qt/stock/kline/get"
        url = api_url
        params_base: dict[str, str] = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "klt": "101",
            "fqt": adjust_dict[adjust],
            "beg": start_ts.strftime("%Y%m%d"),
            "end": end_ts.strftime("%Y%m%d"),
        }
        sym = str(symbol).strip()
        headers = dict(self._headers)

        def _pull_via_http_response(response: Any) -> pd.DataFrame:
            try:
                response.raise_for_status()
            except Exception as exc:
                raise VendorUpstreamError(
                    "East Money kline HTTP status error (%s)."
                    % getattr(response, "status_code", "?")
                ) from exc
            try:
                payload = response.json()
            except Exception as exc:
                raise VendorUpstreamError("East Money kline response is not valid JSON.") from exc
            return parse_daily_kline_payload(payload)

        try:
            primary_id = _eastmoney_sec_market_id(sym)
        except ValueError as exc:
            raise VendorUpstreamError("%s" % exc) from exc

        market_ids: list[int] = [primary_id]
        alternate_id = 1 - primary_id if primary_id in (0, 1) else None
        if alternate_id is not None:
            # 本地规则可能判断错沪/深市场，失败时再用另一个市场前缀试一次。
            market_ids.append(alternate_id)

        def pull_market_ids(get_with_params: Callable[[dict[str, str]], Any]) -> pd.DataFrame:
            for mid in market_ids:
                params = dict(params_base)
                params["secid"] = "%d.%s" % (mid, sym)
                frame = _pull_via_http_response(get_with_params(params))
                if not frame.empty:
                    return frame
            return pd.DataFrame()

        def pull_market_ids_stdlib() -> tuple[pd.DataFrame, int | None]:
            import http.client
            import ssl

            host = em_host
            ctx = ssl.create_default_context()
            ctx.set_alpn_protocols(["http/1.1"])
            last_status: int | None = None
            for mid in market_ids:
                params = dict(params_base)
                params["secid"] = "%d.%s" % (mid, sym)
                path = path_base + "?" + urlencode(params)
                conn = http.client.HTTPSConnection(host, 443, context=ctx, timeout=30)
                try:
                    conn.putrequest("GET", path)
                    conn.putheader("Host", host)
                    for hk, hv in headers.items():
                        if hk.lower() == "host":
                            continue
                        conn.putheader(hk, hv)
                    conn.putheader("Connection", "close")
                    conn.endheaders()
                    resp = conn.getresponse()
                    body = resp.read()
                finally:
                    conn.close()
                last_status = resp.status
                if resp.status != 200:
                    continue
                try:
                    payload = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                frame = parse_daily_kline_payload(payload)
                if not frame.empty:
                    return frame, last_status
            return pd.DataFrame(), last_status

        last_error: Exception | None = None
        trust_sequence: list[bool] = []
        for te in (trust_env, not trust_env):
            if te not in trust_sequence:
                trust_sequence.append(te)

        for te in trust_sequence:
            try:
                # trust_env 控制是否读取系统代理；两种模式都试，兼容直连和必须代理的网络。
                client_kwargs: dict[str, Any] = {
                    "timeout": 30.0,
                    "trust_env": te,
                    "headers": headers,
                }
                if not te:
                    client_kwargs["proxy"] = None
                with httpx.Client(**client_kwargs) as client:
                    out = pull_market_ids(lambda p: client.get(url, params=p))
                if not out.empty:
                    return out
            except VendorUpstreamError:
                raise
            except Exception as exc:
                if self._recoverable_httpx_transport_error(exc):
                    last_error = exc
                else:
                    if type(exc).__module__.startswith(("httpx", "httpcore")):
                        raise NetworkFetchError(
                            "East Money kline request failed (httpx/httpcore): %s" % exc
                        ) from exc
                    raise

            try:
                # httpx 失败后再用 requests 试一次，减少单一 HTTP 客户端兼容性问题。
                with requests.Session() as session:
                    session.trust_env = te
                    session.headers.clear()
                    session.headers.update(headers)
                    req_proxies = None if te else _REQUESTS_NO_PROXY
                    out = pull_market_ids(
                        lambda p: session.get(url, timeout=30, params=p, proxies=req_proxies)
                    )
                if not out.empty:
                    return out
            except VendorUpstreamError:
                raise
            except requests.RequestException as exc:
                last_error = exc

        stdlib_status: int | None = None
        try:
            # 最后用标准库 http.client 兜底，避开第三方 HTTP 库的部分协议差异。
            out, stdlib_status = pull_market_ids_stdlib()
            if not out.empty:
                return out
        except Exception as exc:
            last_error = exc

        if last_error is not None:
            if stdlib_status is not None and stdlib_status != 200:
                raise VendorUpstreamError(
                    "East Money kline API rejected the request (last HTTP status=%s from stdlib, host=%s). "
                    "Try another network (e.g. mobile hotspot), disable interfering security software, "
                    "or ensure the host is reachable." % (stdlib_status, em_host)
                ) from last_error
            if isinstance(last_error, requests.RequestException):
                raise NetworkFetchError(
                    "East Money kline transport failed after retries (requests/urllib3)."
                ) from last_error
            raise NetworkFetchError(
                "East Money kline transport failed after retries: %r" % (last_error,)
            ) from last_error
        return pd.DataFrame()
