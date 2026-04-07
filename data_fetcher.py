"""量化分析系统的数据获取层。

本模块实现了面向 A 股日线数据的稳健 ETL 流程，重点处理以下问题：

- 网络抖动导致的临时抓取失败
- 第三方接口返回列名不统一
- 数值列 dtype 混乱
- 日期列未标准化
- 缺失值与重复索引
- ETF/个股：优先新浪 / 腾讯，再 AKShare（东财接口）与东方财富多节点直连；``secid`` 市场前缀本地推导；WinINet 代理策略在首个 ``AKShareFetcher`` 构造时解析
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
import importlib
import os
import time
from typing import TYPE_CHECKING, Any, Union

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_float_dtype


import requests

# 全进程仅解析一次：首个 ``AKShareFetcher`` 构造时的参数优先，避免导入本模块即改写全局代理行为。
_PROXY_POLICY_RESOLVED: bool = False


def _configure_windows_proxy_behavior(*, use_system_proxy: bool | None = None) -> None:
    """按配置决定是否屏蔽 Windows「Internet 设置」里的系统代理（仍尊重显式 ``HTTP_PROXY`` 等）。

    默认（``use_system_proxy is None``）读取环境变量 ``QUANT_USE_SYSTEM_PROXY``：
    若为 ``1``/``true``/``yes`` 则保留系统代理；否则将 ``urllib.request.getproxies`` 置为空映射，
    使 AKShare/requests 在典型环境下直连行情源，规避损坏的本地系统代理。

    Parameters
    ----------
    use_system_proxy
        ``True`` 强制走系统代理；``False`` 强制忽略 WinINet 代理；``None`` 由环境变量决定。

    Notes
    -----
    首次调用后策略即固定，同一进程内后续 ``AKShareFetcher`` 无法改变，以免全局状态不一致。
    """
    global _PROXY_POLICY_RESOLVED
    if _PROXY_POLICY_RESOLVED:
        return
    _PROXY_POLICY_RESOLVED = True
    if use_system_proxy is None:
        use_system_proxy = os.environ.get("QUANT_USE_SYSTEM_PROXY", "").lower() in {
            "1",
            "true",
            "yes",
        }
    if use_system_proxy:
        return
    import urllib.request as _urllib_request

    _urllib_request.getproxies = lambda: {}  # type: ignore[method-assign]


def apply_default_network_proxy_policy() -> None:
    """幂等地根据 ``QUANT_USE_SYSTEM_PROXY`` 应用 WinINet 代理策略（全进程仅生效一次）。

    Web 等入口若在未构造 ``AKShareFetcher`` 前直接调用 akshare（如贵金属），宜在启动时调用本函数，
    使与 ETF/个股链路共享同一代理语义。
    """
    _configure_windows_proxy_behavior(use_system_proxy=None)


# 直连东方财富 / 临时禁用 AKShare 所继承的坏代理时，显式关掉 requests 的代理（trust_env=False 仍可能被环境或适配器带上代理）。
_REQUESTS_NO_PROXY: dict[str, None] = {"http": None, "https": None}


@contextmanager
def _without_http_proxy_env() -> Iterator[None]:
    """临时屏蔽代理，使 AKShare 内建的 ``requests.get`` 能直连东方财富。

    仅清理环境变量不够：``requests`` 在 Windows 上会通过 ``get_environ_proxies`` 读取系统代理，
    且该函数往往在补丁 ``urllib.request.getproxies`` 之前已绑定旧实现，故在此临时将
    ``requests.utils.get_environ_proxies`` 置为返回空映射。
    """
    keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    )
    saved: dict[str, str] = {}
    for k in keys:
        if k in os.environ:
            saved[k] = os.environ.pop(k)

    import requests.sessions as _requests_sessions
    import requests.utils as _requests_utils

    _orig_utils_gep = _requests_utils.get_environ_proxies
    _orig_sessions_gep = _requests_sessions.get_environ_proxies

    def _no_environ_proxies(url: str, no_proxy: Any = None) -> dict[str, str]:
        return {}

    _requests_utils.get_environ_proxies = _no_environ_proxies  # type: ignore[method-assign]
    _requests_sessions.get_environ_proxies = _no_environ_proxies  # type: ignore[method-assign]
    try:
        yield
    finally:
        _requests_utils.get_environ_proxies = _orig_utils_gep  # type: ignore[method-assign]
        _requests_sessions.get_environ_proxies = _orig_sessions_gep  # type: ignore[method-assign]
        for k, v in saved.items():
            os.environ[k] = v


if TYPE_CHECKING:
    from typing import Literal


DateLike = Union[str, datetime, pd.Timestamp]
REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")

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

A_SHARE_COLUMN_MAPPING: dict[str, str] = {
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


class DataFetchError(RuntimeError):
    """数据抓取或清洗失败时抛出的异常。"""


class EmptySymbolDataError(DataFetchError):
    """标的无行情、空表或清洗后无有效 OHLCV 行时抛出，供 Web 端展示统一提示。"""


class NetworkFetchError(DataFetchError):
    """传输层失败：超时、TLS、连接重置、代理错误等。"""


class VendorUpstreamError(DataFetchError):
    """上游 HTTP 非预期状态、JSON 无法解析或接口明确拒绝等。"""


class ChinaEquityPipelineError(DataFetchError):
    """AKShare 主源与东方财富直连回退均未能返回有效 K 线时的汇总错误。"""


def _eastmoney_sec_market_id(symbol: str) -> int:
    """东方财富 K 线接口 ``secid`` 的市场前缀：0=深、1=沪（不依赖 AKShare 内部实现）。

    规则覆盖常见 A 股与 ETF/LOF；存疑代码由调用方再尝试另一市场前缀。
    """
    code = str(symbol).strip()
    if not code.isdigit():
        raise ValueError(
            "East Money kline API expects a numeric security code, got %r." % symbol
        )
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


class BaseDataFetcher(ABC):
    """数据获取层抽象基类。

    Parameters
    ----------
    max_retries : int, default 3
        抓取失败时的最大重试次数。
    retry_delay : float, default 1.0
        重试间隔秒数。未来如需更复杂的退避策略，可在此基础上扩展。

    Notes
    -----
    所有面向下游特征工程层和可视化层的标准输出都应满足以下统一契约：

    - 类型为 ``pd.DataFrame``
    - 索引为 ``pd.DatetimeIndex``
    - 至少包含 ``Open``、``High``、``Low``、``Close``、``Volume`` 五列
    - 上述 OHLCV 列 dtype 为 ``float64``
    - 索引按时间升序排列，且不存在重复索引
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative.")

        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _run_with_retry(
        self,
        operation: Callable[[], pd.DataFrame],
        operation_name: str,
    ) -> pd.DataFrame:
        """以固定重试机制执行数据抓取操作。

        Parameters
        ----------
        operation : Callable[[], pd.DataFrame]
            不带参数的可调用对象，执行一次实际的数据抓取请求并返回原始
            ``pd.DataFrame``。
        operation_name : str
            操作描述文本，用于异常信息拼接。

        Returns
        -------
        pd.DataFrame
            抓取成功后的原始数据表。

        Raises
        ------
        DataFetchError
            若在 ``max_retries`` 次尝试后仍无法成功抓取，则抛出该异常。

        Notes
        -----
        该方法主要用于吸收网络抖动带来的瞬时失败，不负责数据清洗。
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                data = operation()
                if not isinstance(data, pd.DataFrame):
                    raise TypeError(
                        "%s must return a pandas DataFrame, got %s."
                        % (operation_name, type(data).__name__)
                    )
                if data.empty:
                    raise EmptySymbolDataError(
                        "%s returned an empty DataFrame." % operation_name
                    )
                return data
            except EmptySymbolDataError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_delay)

        raise DataFetchError(
            "%s failed after %d attempts." % (operation_name, self.max_retries)
        ) from last_error

    def _coerce_date(self, value: DateLike, field_name: str) -> pd.Timestamp:
        """将外部日期输入标准化为 ``pd.Timestamp``。"""
        try:
            ts = pd.Timestamp(value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("%s is not a valid date: %r" % (field_name, value)) from exc

        if pd.isna(ts):
            raise ValueError("%s is not a valid date: %r" % (field_name, value))
        return ts

    def _format_vendor_date(self, value: DateLike) -> str:
        """将日期格式化为 AKShare 常用的 ``YYYYMMDD`` 字符串。"""
        return self._coerce_date(value, "date").strftime("%Y%m%d")

    def _normalize_ohlcv(
        self,
        df: pd.DataFrame,
        column_mapping: Mapping[str, str],
    ) -> pd.DataFrame:
        """将原始字段映射为统一 OHLCV 结构并做基础清洗。

        Parameters
        ----------
        df : pd.DataFrame
            原始行情数据表。
            该对象通常直接来自第三方数据接口，列名和索引结构可能尚未标准化。
        column_mapping : Mapping[str, str]
            原始列名到标准列名的映射。
            目标列应至少覆盖 ``Date``、``Open``、``High``、``Low``、``Close``、
            ``Volume`` 中的数据来源。

        Returns
        -------
        pd.DataFrame
            标准化后的 OHLCV 数据表。
            返回值索引为 ``pd.DatetimeIndex``，列严格为 ``Open``、``High``、
            ``Low``、``Close``、``Volume``，五列 dtype 均为 ``float64``。

        Raises
        ------
        TypeError
            若 ``df`` 不是 ``pd.DataFrame``，则抛出该异常。
        ValueError
            若日期列缺失、关键 OHLCV 列缺失、日期无法解析或清洗后无有效数据，
            则抛出该异常。

        Notes
        -----
        清洗流程包括：

        - 重命名原始列
        - 将日期标准化为 ``pd.DatetimeIndex``
        - 将 OHLCV 列转为数值并强制为 ``float64``
        - 按时间排序并去除重复索引
        - 对缺失值执行前向填充 ``ffill``
        - 删除前向填充后仍残留缺失值的行
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise EmptySymbolDataError("Input DataFrame is empty.")

        normalized = df.rename(columns=dict(column_mapping)).copy()

        if "Date" in normalized.columns:
            normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
            normalized = normalized.dropna(subset=["Date"])
            if normalized.empty:
                raise ValueError("All values in the Date column are invalid.")
            normalized = normalized.set_index("Date")
        elif isinstance(normalized.index, pd.DatetimeIndex):
            normalized.index = pd.to_datetime(normalized.index, errors="coerce")
            normalized = normalized[normalized.index.notna()]
        else:
            raise ValueError(
                "A Date column or a DatetimeIndex is required for OHLCV normalization."
            )

        missing_columns = [
            column for column in REQUIRED_OHLCV_COLUMNS if column not in normalized.columns
        ]
        if missing_columns:
            raise ValueError(
                "Missing required OHLCV columns after normalization: %s"
                % ", ".join(missing_columns)
            )

        normalized = normalized.loc[:, REQUIRED_OHLCV_COLUMNS].copy()
        for column in REQUIRED_OHLCV_COLUMNS:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

        normalized = normalized.sort_index()
        normalized = normalized[~normalized.index.duplicated(keep="last")]
        normalized = normalized.ffill()
        normalized = normalized.dropna(subset=list(REQUIRED_OHLCV_COLUMNS))
        if normalized.empty:
            raise EmptySymbolDataError("No valid OHLCV rows remain after preprocessing.")

        normalized = normalized.astype("float64")
        normalized.index.name = "Date"

        return self._validate_ohlcv(normalized)

    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """校验数据是否满足统一 OHLCV 契约。

        Parameters
        ----------
        df : pd.DataFrame
            待校验的数据表。
            预期索引为 ``pd.DatetimeIndex``，并严格包含 ``Open``、``High``、
            ``Low``、``Close``、``Volume`` 五列，且列 dtype 为 ``float64``。

        Returns
        -------
        pd.DataFrame
            通过校验后的数据表本身。

        Raises
        ------
        TypeError
            若输入对象不是 ``pd.DataFrame``，或关键列不是 ``float64``，
            则抛出该异常。
        ValueError
            若索引类型不正确、存在重复索引、缺失列、空表或残留缺失值，
            则抛出该异常。
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("OHLCV DataFrame is empty.")

        missing_columns = [
            column for column in REQUIRED_OHLCV_COLUMNS if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                "OHLCV DataFrame is missing columns: %s"
                % ", ".join(missing_columns)
            )

        if not is_datetime64_any_dtype(df.index):
            raise ValueError("OHLCV DataFrame index must be a pandas DatetimeIndex.")
        if df.index.hasnans:
            raise ValueError("OHLCV DataFrame index contains invalid datetime values.")
        if df.index.duplicated().any():
            raise ValueError("OHLCV DataFrame index contains duplicate timestamps.")
        if not df.index.is_monotonic_increasing:
            raise ValueError("OHLCV DataFrame index must be sorted ascending.")
        if df.loc[:, REQUIRED_OHLCV_COLUMNS].isna().any().any():
            raise ValueError("OHLCV DataFrame still contains NaN values.")

        invalid_dtypes = [
            column for column in REQUIRED_OHLCV_COLUMNS if not is_float_dtype(df[column])
        ]
        if invalid_dtypes:
            raise TypeError(
                "OHLCV columns must be float64-compatible. Invalid columns: %s"
                % ", ".join(invalid_dtypes)
            )

        return df

    @abstractmethod
    def _fetch_raw(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """执行底层原始数据抓取。"""


class AKShareChinaPrimarySource:
    """AKShare 沪深股票 / ETF 主路径，与东方财富直连回退解耦。"""

    def __init__(self, ak_loader: Callable[[], Any]) -> None:
        self._ak_loader = ak_loader

    def fetch_stock_hist(
        self,
        symbol: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        adjust: str,
    ) -> pd.DataFrame:
        ak = self._ak_loader()
        return ak.stock_zh_a_hist(
            symbol=str(symbol).strip(),
            period="daily",
            start_date=start_ts.strftime("%Y%m%d"),
            end_date=end_ts.strftime("%Y%m%d"),
            adjust=adjust,
        )

    def fetch_etf_hist_em(
        self,
        symbol: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        adjust: str,
    ) -> pd.DataFrame:
        ak = self._ak_loader()
        return ak.fund_etf_hist_em(
            symbol=str(symbol).strip(),
            period="daily",
            start_date=start_ts.strftime("%Y%m%d"),
            end_date=end_ts.strftime("%Y%m%d"),
            adjust=adjust,
        )


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
        for api_url in _EASTMONEY_KLINE_APIS:
            try:
                df = self._fetch_daily_klines_one_host(
                    api_url, symbol, start_ts, end_ts, adjust, trust_env=trust_env
                )
                if not df.empty:
                    return df
            except ValueError:
                raise
            except BaseException as exc:  # noqa: BLE001
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

        def _dataframe_from_kline_payload(payload: dict[str, Any]) -> pd.DataFrame:
            if not (payload.get("data") and payload["data"].get("klines")):
                return pd.DataFrame()
            rows = [item.split(",") for item in payload["data"]["klines"]]
            temp_df = pd.DataFrame(rows)
            temp_df.columns = [
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
            temp_df.index = pd.to_datetime(temp_df["日期"], errors="coerce")
            temp_df.reset_index(inplace=True, drop=True)
            temp_df["开盘"] = pd.to_numeric(temp_df["开盘"], errors="coerce")
            temp_df["收盘"] = pd.to_numeric(temp_df["收盘"], errors="coerce")
            temp_df["最高"] = pd.to_numeric(temp_df["最高"], errors="coerce")
            temp_df["最低"] = pd.to_numeric(temp_df["最低"], errors="coerce")
            temp_df["成交量"] = pd.to_numeric(temp_df["成交量"], errors="coerce")
            temp_df["成交额"] = pd.to_numeric(temp_df["成交额"], errors="coerce")
            temp_df["振幅"] = pd.to_numeric(temp_df["振幅"], errors="coerce")
            temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"], errors="coerce")
            temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"], errors="coerce")
            temp_df["换手率"] = pd.to_numeric(temp_df["换手率"], errors="coerce")
            return temp_df

        def _pull_via_http_response(response: Any) -> pd.DataFrame:
            try:
                response.raise_for_status()
            except Exception as exc:
                raise VendorUpstreamError(
                    "East Money kline HTTP status error (%s)." % getattr(response, "status_code", "?")
                ) from exc
            try:
                payload = response.json()
            except Exception as exc:
                raise VendorUpstreamError("East Money kline response is not valid JSON.") from exc
            return _dataframe_from_kline_payload(payload)

        try:
            primary_id = _eastmoney_sec_market_id(sym)
        except ValueError as exc:
            raise VendorUpstreamError("%s" % exc) from exc

        market_ids: list[int] = [primary_id]
        alternate_id = 1 - primary_id if primary_id in (0, 1) else None
        if alternate_id is not None:
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
                frame = _dataframe_from_kline_payload(payload)
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
                session = requests.Session()
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
                    "or ensure the host is reachable."
                    % (stdlib_status, em_host)
                ) from last_error
            if isinstance(last_error, requests.RequestException):
                raise NetworkFetchError(
                    "East Money kline transport failed after retries (requests/urllib3)."
                ) from last_error
            raise NetworkFetchError(
                "East Money kline transport failed after retries: %r" % (last_error,)
            ) from last_error
        return pd.DataFrame()


class AKShareFetcher(BaseDataFetcher):
    """基于 AKShare 的 A 股与 ETF 数据获取器。"""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        *,
        eastmoney_headers: Mapping[str, str] | None = None,
        use_system_proxy: bool | None = None,
    ) -> None:
        _configure_windows_proxy_behavior(use_system_proxy=use_system_proxy)
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        if use_system_proxy is True:
            self._neutralize_env_for_ak_primary = False
        elif use_system_proxy is False:
            self._neutralize_env_for_ak_primary = True
        else:
            self._neutralize_env_for_ak_primary = not (
                os.environ.get("QUANT_USE_SYSTEM_PROXY", "").lower() in {"1", "true", "yes"}
            )
        self._eastmoney = EastMoneyDailyKlineClient(
            None if eastmoney_headers is None else dict(eastmoney_headers)
        )
        self._primary = AKShareChinaPrimarySource(self._load_akshare)

    def _load_akshare(self) -> Any:
        """按需导入 AKShare，避免模块导入阶段产生硬依赖。"""
        try:
            return importlib.import_module("akshare")
        except ImportError as exc:
            raise ImportError(
                "AKShare is required for AKShareFetcher. Install it with `pip install akshare`."
            ) from exc

    def _fetch_sina_tencent_china_equity_raw(
        self,
        equity_route: str,
        sym: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        adjust: str,
    ) -> pd.DataFrame:
        """新浪 / 腾讯日线（非东方财富域名，作为境内权益首选数据源）。"""
        code = str(sym).strip()
        start_s = start_ts.strftime("%Y%m%d")
        end_s = end_ts.strftime("%Y%m%d")
        try:
            prefix = "sh" if _eastmoney_sec_market_id(code) == 1 else "sz"
        except ValueError:
            return pd.DataFrame()
        qq = "%s%s" % (prefix, code)
        adj = adjust if adjust in {"qfq", "hfq", ""} else "hfq"

        def _to_em_style(df: pd.DataFrame, vol_col: str) -> pd.DataFrame:
            if df.empty or "date" not in df.columns:
                return pd.DataFrame()
            out = pd.DataFrame(
                {
                    "日期": pd.to_datetime(df["date"], errors="coerce"),
                    "开盘": pd.to_numeric(df["open"], errors="coerce"),
                    "收盘": pd.to_numeric(df["close"], errors="coerce"),
                    "最高": pd.to_numeric(df["high"], errors="coerce"),
                    "最低": pd.to_numeric(df["low"], errors="coerce"),
                    "成交量": pd.to_numeric(df[vol_col], errors="coerce"),
                }
            )
            out = out.dropna(subset=["日期"])
            lo = start_ts.normalize()
            hi = end_ts.normalize()
            mask = (out["日期"] >= lo) & (out["日期"] <= hi)
            return out.loc[mask].copy()

        try:
            if equity_route == "etf":
                try:
                    sina = importlib.import_module("akshare.fund.fund_etf_sina")
                    raw_s = sina.fund_etf_hist_sina(symbol=qq)
                    if not raw_s.empty and "volume" in raw_s.columns:
                        conv = _to_em_style(raw_s, "volume")
                        if not conv.empty:
                            return conv
                except Exception:  # noqa: BLE001
                    pass
            mod_tx = importlib.import_module("akshare.stock_feature.stock_hist_tx")
            raw_tx = mod_tx.stock_zh_a_hist_tx(
                symbol=qq,
                start_date=start_s,
                end_date=end_s,
                adjust=adj,
            )
            return _to_em_style(raw_tx, "amount")
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def _fetch_via_primary_then_eastmoney(
        self,
        *,
        sym: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        adjust: str,
        label_zh: str,
        equity_route: str,
        fetch_primary: Callable[[], pd.DataFrame],
        empty_primary_message: str,
    ) -> pd.DataFrame:
        def _pipeline() -> pd.DataFrame:
            # 1) 新浪 / 腾讯（不依赖东方财富域名）
            alt = self._fetch_sina_tencent_china_equity_raw(
                equity_route, sym, start_ts, end_ts, adjust
            )
            if not alt.empty:
                return alt

            first_error: BaseException | None = None
            try:
                via_ak = fetch_primary()
                if not via_ak.empty:
                    return via_ak
                first_error = ValueError(empty_primary_message)
            except EmptySymbolDataError:
                raise
            except Exception as exc:  # noqa: BLE001
                first_error = exc

            em_fail: BaseException | None = None
            try:
                direct = self._eastmoney.fetch_daily_klines(
                    sym,
                    start_ts,
                    end_ts,
                    adjust,
                    trust_env=False,
                )
                if not direct.empty:
                    return direct
            except (NetworkFetchError, VendorUpstreamError) as exc:
                em_fail = exc
            except Exception as exc:  # noqa: BLE001
                em_fail = exc

            if em_fail is not None:
                raise ChinaEquityPipelineError(
                    "%s：新浪/腾讯无有效数据；AKShare 主源未成功 (%r)；东方财富直连失败 (%s)。"
                    % (label_zh, first_error, em_fail)
                ) from em_fail
            raise ChinaEquityPipelineError(
                "%s：新浪/腾讯无有效数据；AKShare 未返回有效数据 (%r)；东方财富直连返回空 K 线。"
                % (label_zh, first_error)
            ) from first_error

        if self._neutralize_env_for_ak_primary:
            try:
                with _without_http_proxy_env():
                    return _pipeline()
            except ChinaEquityPipelineError:
                # 直连不可达时（例如必须经本机 Clash / 公司代理才能访问外网），再按系统与环境变量走代理重试一轮
                return _pipeline()
        return _pipeline()

    def _fetch_raw(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """抓取 AKShare 原始 A 股行情数据。

        Parameters
        ----------
        symbol : str
            A 股股票代码，例如 ``"600519"``。
        **kwargs : Any
            额外抓取参数，当前实现使用 ``start_date``、``end_date``、``adjust``。

        Returns
        -------
        pd.DataFrame
            AKShare 返回的原始数据表。

        Raises
        ------
        DataFetchError
            若多次尝试后仍抓取失败，则抛出该异常。
        """
        if not symbol or not str(symbol).strip():
            raise ValueError("symbol must be a non-empty stock code.")

        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        adjust = kwargs.get("adjust", "hfq")

        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required for A-share fetching.")

        start_ts = self._coerce_date(start_date, "start_date")
        end_ts = self._coerce_date(end_date, "end_date")
        if start_ts > end_ts:
            raise ValueError("start_date must be earlier than or equal to end_date.")

        sym = str(symbol).strip()

        def operation() -> pd.DataFrame:
            return self._fetch_via_primary_then_eastmoney(
                sym=sym,
                start_ts=start_ts,
                end_ts=end_ts,
                adjust=adjust,
                label_zh="A 股日线抓取",
                equity_route="stock",
                fetch_primary=lambda: self._primary.fetch_stock_hist(sym, start_ts, end_ts, adjust),
                empty_primary_message="AKShare stock_zh_a_hist returned an empty DataFrame.",
            )

        return self._run_with_retry(
            operation=operation,
            operation_name="AKShare A-share historical data fetch",
        )

    def _fetch_etf_raw(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """抓取 AKShare 原始 ETF 历史行情数据。"""
        if not symbol or not str(symbol).strip():
            raise ValueError("symbol must be a non-empty ETF code.")

        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        adjust = kwargs.get("adjust", "hfq")

        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required for ETF fetching.")

        start_ts = self._coerce_date(start_date, "start_date")
        end_ts = self._coerce_date(end_date, "end_date")
        if start_ts > end_ts:
            raise ValueError("start_date must be earlier than or equal to end_date.")

        sym = str(symbol).strip()

        def operation() -> pd.DataFrame:
            return self._fetch_via_primary_then_eastmoney(
                sym=sym,
                start_ts=start_ts,
                end_ts=end_ts,
                adjust=adjust,
                label_zh="ETF 日线抓取",
                equity_route="etf",
                fetch_primary=lambda: self._primary.fetch_etf_hist_em(sym, start_ts, end_ts, adjust),
                empty_primary_message="AKShare fund_etf_hist_em returned an empty DataFrame.",
            )

        return self._run_with_retry(
            operation=operation,
            operation_name="AKShare ETF historical data fetch",
        )

    def fetch_ashare_data(
        self,
        symbol: str,
        start_date: DateLike,
        end_date: DateLike,
    ) -> pd.DataFrame:
        """抓取沪深 A 股日线后复权数据并完成预处理。

        Parameters
        ----------
        symbol : str
            沪深 A 股股票代码，例如 ``"600519"``、``"000001"``。
        start_date : str | datetime | pd.Timestamp
            开始日期。
            若为字符串，推荐使用 ``"YYYY-MM-DD"`` 格式；内部会转换为
            AKShare 所需的 ``YYYYMMDD``。
        end_date : str | datetime | pd.Timestamp
            结束日期。
            若为字符串，推荐使用 ``"YYYY-MM-DD"`` 格式；内部会转换为
            AKShare 所需的 ``YYYYMMDD``。

        Returns
        -------
        pd.DataFrame
            预处理后的标准 OHLCV 数据表。
            返回值索引为 ``pd.DatetimeIndex``，索引名为 ``Date``，列严格为
            ``Open``、``High``、``Low``、``Close``、``Volume``，五列 dtype
            全部为 ``float64``。

        Raises
        ------
        ImportError
            若运行环境未安装 ``akshare``，则抛出该异常。
        ValueError
            若股票代码为空、日期非法、日期区间倒置、AKShare 返回结构不含必需列，
            或清洗后无有效数据，则抛出该异常。
        DataFetchError
            若网络抖动或第三方接口异常导致连续 3 次抓取失败，则抛出该异常。

        Notes
        -----
        本方法实现的 ETL 流程如下：

        - 使用 AKShare 获取 A 股日线后复权数据
        - 抓取失败时自动重试 3 次
        - 将列名重命名为标准的 ``Date``、``Open``、``High``、``Low``、
          ``Close``、``Volume``
        - 将 ``Date`` 转换为 ``datetime64`` 并设置为索引
        - 将 OHLCV 五列统一转换为 ``float64``
        - 对缺失值执行前向填充 ``ffill``
        - 删除前向填充后仍残留缺失值的行
        - 输出连续、标准化、可直接进入特征工程层的时间序列向量
        """
        raw_df = self._fetch_raw(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="hfq",
        )
        return self._normalize_ohlcv(raw_df, A_SHARE_COLUMN_MAPPING)

    def fetch_china_equity(
        self,
        route: Literal["etf", "stock"],
        symbol: str,
        start: DateLike,
        end: DateLike,
        adjust: Literal["qfq", "hfq"] | None = None,
    ) -> pd.DataFrame:
        """境内权益数据路由：ETF 走 ``fund_etf_hist_em``，A 股个股走 ``stock_zh_a_hist``。

        两类数据经统一 ``_normalize_ohlcv`` 后均为 ``Open``/``High``/``Low``/``Close``/``Volume``
        与 ``DatetimeIndex``，可直接对接 ``feature_engineering``。
        """
        route_key = str(route).strip().lower()
        if route_key == "etf":
            return self.fetch_etf(symbol, start=start, end=end, adjust=adjust)
        if route_key == "stock":
            return self.fetch_stock(symbol, start=start, end=end, adjust=adjust)
        raise ValueError("fetch_china_equity route must be 'etf' or 'stock', got %r." % (route,))

    def fetch_stock(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        adjust: Literal["qfq", "hfq"] | None = None,
    ) -> pd.DataFrame:
        """抓取 A 股股票行情并输出统一 OHLCV 结构。

        Parameters
        ----------
        symbol : str
            A 股股票代码。
        start : str | datetime | pd.Timestamp | None, default None
            开始日期。当前实现要求显式传入。
        end : str | datetime | pd.Timestamp | None, default None
            结束日期。当前实现要求显式传入。
        adjust : {"qfq", "hfq"} | None, default None
            复权方式。若为 ``None``，默认使用 ``"hfq"``。

        Returns
        -------
        pd.DataFrame
            标准化后的 A 股 OHLCV 数据表。

        Raises
        ------
        ValueError
            若 ``start`` 或 ``end`` 缺失，或参数不合法，则抛出该异常。
        """
        if start is None or end is None:
            raise ValueError("fetch_stock currently requires both start and end.")

        selected_adjust = adjust or "hfq"
        raw_df = self._fetch_raw(
            symbol=symbol,
            start_date=start,
            end_date=end,
            adjust=selected_adjust,
        )
        return self._normalize_ohlcv(raw_df, A_SHARE_COLUMN_MAPPING)

    def fetch_etf(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        adjust: Literal["qfq", "hfq"] | None = None,
    ) -> pd.DataFrame:
        """抓取 ETF 历史行情并输出统一 OHLCV 结构。

        Parameters
        ----------
        symbol : str
            ETF 代码，例如 ``"510300"``。
        start : str | datetime | pd.Timestamp | None, default None
            开始日期。当前实现要求显式传入。
        end : str | datetime | pd.Timestamp | None, default None
            结束日期。当前实现要求显式传入。
        adjust : {"qfq", "hfq"} | None, default None
            复权方式。若为 ``None``，默认使用 ``"hfq"``。

        Returns
        -------
        pd.DataFrame
            标准化后的 ETF OHLCV 数据表。
            返回值索引为 ``pd.DatetimeIndex``，列严格为 ``Open``、``High``、
            ``Low``、``Close``、``Volume``，五列 dtype 全部为 ``float64``。

        Raises
        ------
        ImportError
            若运行环境未安装 ``akshare``，则抛出该异常。
        ValueError
            若 ``start`` 或 ``end`` 缺失、日期非法、日期区间倒置，或返回数据
            结构缺少必需列，则抛出该异常。
        DataFetchError
            若第三方接口连续重试后仍抓取失败，则抛出该异常。
        """
        if start is None or end is None:
            raise ValueError("fetch_etf currently requires both start and end.")

        selected_adjust = adjust or "hfq"
        raw_df = self._fetch_etf_raw(
            symbol=symbol,
            start_date=start,
            end_date=end,
            adjust=selected_adjust,
        )
        return self._normalize_ohlcv(raw_df, A_SHARE_COLUMN_MAPPING)


class YFinanceFetcher(BaseDataFetcher):
    """基于 yfinance 的美股数据获取器。"""

    def _fetch_raw(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """抓取 yfinance 原始行情数据。"""
        raise NotImplementedError("yfinance fetching is not implemented yet.")

    def fetch_us_stock(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        interval: str = "1d",
        auto_adjust: bool = False,
    ) -> pd.DataFrame:
        """抓取美股行情并输出统一 OHLCV 结构。"""
        raise NotImplementedError("US stock fetching is not implemented yet.")


def _coerce_analysis_timestamp(value: DateLike, field_name: str) -> pd.Timestamp:
    """将区间边界转为日级、无时区的时间戳（与 ``app._coerce_date`` 语义对齐）。"""
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("%s is not a valid date: %r" % (field_name, value)) from exc
    if pd.isna(ts):
        raise ValueError("%s is not a valid date: %r" % (field_name, value))
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _filter_ohlcv_by_user_date_range(
    df: pd.DataFrame,
    start_date: DateLike,
    end_date: DateLike,
) -> pd.DataFrame:
    """按用户区间裁剪 OHLCV；空区间抛 ``ValueError``（与 ``app._filter_by_date_range`` 文案一致）。"""
    start_ts = _coerce_analysis_timestamp(start_date, "start_date")
    end_ts = _coerce_analysis_timestamp(end_date, "end_date")
    if start_ts > end_ts:
        raise ValueError("start_date must be earlier than or equal to end_date.")
    filtered = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if filtered.empty:
        raise ValueError(
            "指定日期区间内没有可用数据：%s 至 %s。"
            % (start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        )
    return filtered


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
    out = pd.DataFrame(
        {
            "_dt": pd.to_datetime(df[date_col], errors="coerce"),
            "nav": pd.to_numeric(df[val_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["_dt", "nav"])
    if out.empty:
        raise EmptySymbolDataError("Fund NAV parsing yielded no valid rows.")
    out = out.sort_values("_dt").drop_duplicates(subset=["_dt"], keep="last")
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


def _akshare_neutralize_proxy_like_fetcher() -> bool:
    """与 ``AKShareFetcher(use_system_proxy=None)`` 一致：默认忽略 WinINet 代理。"""
    return not (os.environ.get("QUANT_USE_SYSTEM_PROXY", "").lower() in {"1", "true", "yes"})


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

    apply_default_network_proxy_policy()

    def _call_fund_info(indicator: str) -> pd.DataFrame:
        ak = importlib.import_module("akshare")
        fn = ak.fund_open_fund_info_em
        try:
            return fn(symbol=code, indicator=indicator)
        except TypeError:
            # 旧版 AKShare 使用参数名 ``fund``；新版为 ``symbol``。
            return fn(fund=code, indicator=indicator)

    def _fetch_with_proxy_policy() -> pd.DataFrame:
        if _akshare_neutralize_proxy_like_fetcher():
            with _without_http_proxy_env():
                return _call_fund_info("单位净值走势")
        return _call_fund_info("单位净值走势")

    try:
        raw_unit = _fetch_with_proxy_policy()
    except ImportError as exc:
        raise ImportError(
            "AKShare is required for fund data. Install it with `pip install akshare`."
        ) from exc
    except EmptySymbolDataError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataFetchError("fund_open_fund_info_em (单位净值走势) failed: %r" % (exc,)) from exc

    if not isinstance(raw_unit, pd.DataFrame) or raw_unit.empty:
        raise EmptySymbolDataError("fund_open_fund_info_em returned no unit NAV data for %r." % code)

    date_col = _fund_find_date_column(raw_unit)
    cum_col = _fund_pick_cumulative_column(raw_unit, date_col)

    fetcher = AKShareFetcher(max_retries=3, retry_delay=1.0)

    if cum_col is not None:
        cum_series = _fund_frame_to_sorted_nav(raw_unit, value_column=cum_col)
        cumulative_close = cum_series.set_index("_dt")["nav"]
        normalized = _fund_raw_to_normalized_ohlcv(fetcher, cumulative_close)
    else:
        unit_sorted = _fund_frame_to_sorted_nav(raw_unit)

        def _fetch_cumulative() -> pd.DataFrame:
            if _akshare_neutralize_proxy_like_fetcher():
                with _without_http_proxy_env():
                    return _call_fund_info("累计净值走势")
            return _call_fund_info("累计净值走势")

        try:
            raw_cum = _fetch_cumulative()
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError("fund_open_fund_info_em (累计净值走势) failed: %r" % (exc,)) from exc

        if not isinstance(raw_cum, pd.DataFrame) or raw_cum.empty:
            raise EmptySymbolDataError("fund_open_fund_info_em returned no cumulative NAV data for %r." % code)

        cum_sorted = _fund_frame_to_sorted_nav(raw_cum, prefer_cumulative_value=True)
        merged = unit_sorted.merge(cum_sorted, on="_dt", how="left", suffixes=("_unit", "_cum"))
        if "nav_cum" not in merged.columns:
            raise EmptySymbolDataError("Cumulative NAV merge produced unexpected columns for %r." % code)
        merged = merged.assign(nav_cum=lambda m: m["nav_cum"].ffill().bfill())
        merged = merged.dropna(subset=["nav_cum"])
        if merged.empty:
            raise EmptySymbolDataError("No overlapping cumulative NAV after aligning with unit NAV for %r." % code)
        cumulative_close = merged.set_index("_dt")["nav_cum"]
        normalized = _fund_raw_to_normalized_ohlcv(fetcher, cumulative_close)

    return _filter_ohlcv_by_user_date_range(normalized, start_date, end_date)


# ---------------------------------------------------------------------------
# 全球多资产：AKShare 路由与统一 OHLCV（与 app / 特征工程契约一致）
# ---------------------------------------------------------------------------

FOREIGN_FUTURES_COLUMN_MAPPING: dict[str, str] = {
    "date": "Date",
    "Date": "Date",
    "open": "Open",
    "Open": "Open",
    "high": "High",
    "High": "High",
    "low": "Low",
    "Low": "Low",
    "close": "Close",
    "Close": "Close",
    "volume": "Volume",
    "Volume": "Volume",
}

# 东方财富 K 线类接口常见列（指数 / 外汇 / REITs 历史）
_EM_KLINE_OHLCV_MAPPING: dict[str, str] = {
    "日期": "Date",
    "今开": "Open",
    "最高": "High",
    "最低": "Low",
    "最新价": "Close",
    "成交量": "Volume",
}

# 全球指数（东财）中文名称；用户可输入简码或中文
GLOBAL_INDEX_EM_ALIASES: dict[str, str] = {
    "HSI": "恒生指数",
    "HSCEI": "国企指数",
    "SPX": "标普500",
    "INX": "标普500",
    "GSPC": "标普500",
    "NDX": "纳斯达克",
    "IXIC": "纳斯达克",
    "DJI": "道琼斯",
    "DJIA": "道琼斯",
    "UDI": "美元指数",
    "DXY": "美元指数",
    "N225": "日经225",
    "FTSE": "英国富时100",
    "GDAXI": "德国DAX30",
    "FCHI": "法国CAC40",
    "SX5E": "欧洲斯托克50",
}

# 美债收益率表默认列（bond_zh_us_rate）
US_TREASURY_YIELD_10Y_CN_COL = "美国国债收益率10年"

FX_PAIR_ALIASES: dict[str, str] = {
    "USD/CNY": "USDCNH",
    "USD/CNH": "USDCNH",
    "USDCNY": "USDCNH",
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
}

# 与 A 股 K 线类似：东财 ``push2his`` 单节点易被 TLS/中间设备断开；多域名轮询并带浏览器头可显著降低
# ``RemoteDisconnected`` / ``Connection aborted`` 概率。AKShare ``forex_hist_em`` 使用裸 ``requests.get`` 无 Referer，
# 在部分网络环境下会被服务端直接掐断连接。
_EASTMONEY_FOREX_KLINE_URLS: tuple[str, ...] = (
    "https://push2his.eastmoney.com/api/qt/stock/kline/get",
    "https://push2.eastmoney.com/api/qt/stock/kline/get",
    "https://33.push2his.eastmoney.com/api/qt/stock/kline/get",
    "https://63.push2his.eastmoney.com/api/qt/stock/kline/get",
)


def _load_httpx_module() -> Any:
    try:
        return importlib.import_module("httpx")
    except ImportError as exc:
        raise ImportError(
            "外汇历史行情直连需要 httpx，请执行：`pip install httpx`。"
        ) from exc


def _forex_kline_json_to_dataframe(data_json: dict[str, Any]) -> pd.DataFrame:
    """解析东财外汇 K 线 JSON，列结构与 AKShare ``forex_hist_em`` 一致。"""
    if not isinstance(data_json, dict):
        return pd.DataFrame()
    block = data_json.get("data")
    if not block or not block.get("klines"):
        return pd.DataFrame()
    klines = block["klines"]
    temp_df = pd.DataFrame([str(item).split(",") for item in klines])
    n_split = int(temp_df.shape[1])
    temp_df["code"] = block.get("code", "")
    temp_df["name"] = block.get("name", "")
    # 东财 klines 常见为 13 段（与当前 AKShare 解析一致）；少数环境为 14 段，按列数分支命名。
    if n_split == 13:
        split_names = [
            "日期",
            "今开",
            "最新价",
            "最高",
            "最低",
            "-",
            "-",
            "振幅",
            "-",
            "-",
            "-",
            "-",
            "-",
        ]
    elif n_split == 14:
        split_names = [
            "日期",
            "今开",
            "最新价",
            "最高",
            "最低",
            "-",
            "-",
            "振幅",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
        ]
    else:
        split_names = ["c%d" % i for i in range(n_split)]
    temp_df.columns = list(split_names) + ["代码", "名称"]
    temp_df = temp_df[
        [
            "日期",
            "代码",
            "名称",
            "今开",
            "最新价",
            "最高",
            "最低",
            "振幅",
        ]
    ]
    temp_df["日期"] = pd.to_datetime(temp_df["日期"], errors="coerce").dt.date
    temp_df["今开"] = pd.to_numeric(temp_df["今开"], errors="coerce")
    temp_df["最新价"] = pd.to_numeric(temp_df["最新价"], errors="coerce")
    temp_df["最高"] = pd.to_numeric(temp_df["最高"], errors="coerce")
    temp_df["最低"] = pd.to_numeric(temp_df["最低"], errors="coerce")
    temp_df["振幅"] = pd.to_numeric(temp_df["振幅"], errors="coerce")
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
                with _without_http_proxy_env():
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


def _import_akshare() -> Any:
    try:
        return importlib.import_module("akshare")
    except ImportError as exc:
        raise ImportError("运行该数据源需要安装 akshare：`pip install akshare`。") from exc


def _call_akshare(
    op_label: str,
    fn: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """在默认代理策略下调用 AKShare，失败时抛出带上下文的 ``DataFetchError``。"""
    apply_default_network_proxy_policy()
    try:
        if _akshare_neutralize_proxy_like_fetcher():
            with _without_http_proxy_env():
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)
    except EmptySymbolDataError:
        raise
    except DataFetchError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataFetchError("%s 失败: %r" % (op_label, exc)) from exc


def _em_kline_frame_to_ohlcv_raw(df: pd.DataFrame) -> pd.DataFrame:
    """为缺少成交量列的东财 K 线表补 ``成交量``，再交给 ``A_SHARE_COLUMN_MAPPING`` 清洗。"""
    work = df.copy()
    if "成交量" not in work.columns:
        work["成交量"] = 0.0
    return work


def _normalize_hk_stock_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    if not text:
        raise ValueError("港股代码不能为空。")
    if text.isdigit():
        return text.zfill(5)
    return text


def _resolve_global_index_em_name(symbol: str) -> str:
    raw = str(symbol).strip()
    if not raw:
        raise ValueError("指数代码或名称不能为空。")
    if raw in GLOBAL_INDEX_EM_ALIASES:
        return GLOBAL_INDEX_EM_ALIASES[raw]
    upper = raw.upper()
    if upper in GLOBAL_INDEX_EM_ALIASES:
        return GLOBAL_INDEX_EM_ALIASES[upper]
    return raw


def load_futures_foreign_ohlcv(
    symbol: str,
    start_date: DateLike,
    end_date: DateLike,
    *,
    asset_type: str,
    unit_caption: str = "",
    series_kind: str = "price",
) -> pd.DataFrame:
    """外盘期货日线（新浪全球期货），输出统一 OHLCV。"""
    display = str(symbol).strip().upper()
    if not display:
        raise ValueError("期货品种代码不能为空。")
    ak = _import_akshare()
    raw_df = _call_akshare(
        "futures_foreign_hist(%r)" % display,
        ak.futures_foreign_hist,
        symbol=display,
    )
    fetcher = AKShareFetcher(max_retries=1, retry_delay=0.0)
    normalized = fetcher._normalize_ohlcv(raw_df, FOREIGN_FUTURES_COLUMN_MAPPING)
    out = _filter_ohlcv_by_user_date_range(normalized, start_date, end_date)
    out.attrs["asset_type"] = str(asset_type)
    out.attrs["symbol"] = display
    out.attrs["series_kind"] = str(series_kind)
    if unit_caption:
        out.attrs["unit_caption"] = str(unit_caption)
    return out


def load_global_market_data(
    route: str,
    symbol: str,
    start_date: DateLike,
    end_date: DateLike,
) -> pd.DataFrame:
    """全球权益/指数：``route`` 为 ``us_stock`` | ``hk_stock`` | ``global_index``。"""
    route_key = str(route).strip().lower()
    ak = _import_akshare()
    fetcher = AKShareFetcher(max_retries=1, retry_delay=0.0)
    sym = str(symbol).strip()
    if not sym:
        raise ValueError("代码不能为空。")

    if route_key == "us_stock":
        raw = _call_akshare(
            "stock_us_daily(%r)" % sym,
            ak.stock_us_daily,
            symbol=sym,
            adjust="",
        )
        out = fetcher._normalize_ohlcv(raw, FOREIGN_FUTURES_COLUMN_MAPPING)
        atype = "us_stock"
        caption = "USD 计价 · 美股日线"
    elif route_key == "hk_stock":
        hk_sym = _normalize_hk_stock_symbol(sym)
        raw = _call_akshare(
            "stock_hk_daily(%r)" % hk_sym,
            ak.stock_hk_daily,
            symbol=hk_sym,
            adjust="",
        )
        out = fetcher._normalize_ohlcv(raw, FOREIGN_FUTURES_COLUMN_MAPPING)
        out.attrs["symbol"] = hk_sym
        atype = "hk_stock"
        caption = "HKD 计价 · 港股日线"
    elif route_key == "global_index":
        display_sym = sym.strip()
        if display_sym.startswith("."):
            raw = _call_akshare(
                "index_us_stock_sina(%r)" % display_sym,
                ak.index_us_stock_sina,
                symbol=display_sym,
            )
            out = fetcher._normalize_ohlcv(raw, FOREIGN_FUTURES_COLUMN_MAPPING)
            atype = "global_index"
            caption = "美股指数（新浪）· 点"
        else:
            em_name = _resolve_global_index_em_name(display_sym)
            raw = _call_akshare(
                "index_global_hist_em(%r)" % em_name,
                ak.index_global_hist_em,
                symbol=em_name,
            )
            raw2 = _em_kline_frame_to_ohlcv_raw(raw)
            out = fetcher._normalize_ohlcv(raw2, {**A_SHARE_COLUMN_MAPPING, **_EM_KLINE_OHLCV_MAPPING})
            out.attrs["symbol"] = em_name
            atype = "global_index"
            caption = "全球指数（东财）· 点"
    else:
        raise ValueError("load_global_market_data: 未知 route %r。" % route)

    out = _filter_ohlcv_by_user_date_range(out, start_date, end_date)
    out.attrs["asset_type"] = atype
    if "symbol" not in out.attrs or not str(out.attrs.get("symbol", "")).strip():
        out.attrs["symbol"] = sym
    out.attrs["series_kind"] = "price"
    out.attrs["unit_caption"] = caption
    return out


def load_bond_data(
    kind: str,
    symbol: str,
    start_date: DateLike,
    end_date: DateLike,
    *,
    us_yield_column: str | None = None,
) -> pd.DataFrame:
    """债券：``cn_hist`` 中国国债 K 线；``us_yield`` 美债收益率时间序列（伪 OHLCV）。"""
    kind_key = str(kind).strip().lower()
    ak = _import_akshare()
    fetcher = AKShareFetcher(max_retries=1, retry_delay=0.0)
    start_ts = _coerce_analysis_timestamp(start_date, "start_date")
    end_ts = _coerce_analysis_timestamp(end_date, "end_date")

    if kind_key == "cn_hist":
        code = str(symbol).strip()
        if not code:
            raise ValueError("国债代码不能为空，例如 sh010107。")
        raw = _call_akshare("bond_zh_hs_daily(%r)" % code, ak.bond_zh_hs_daily, symbol=code)
        if "volume" not in raw.columns:
            raw = raw.copy()
            raw["volume"] = 0.0
        out = fetcher._normalize_ohlcv(raw, FOREIGN_FUTURES_COLUMN_MAPPING)
        out = _filter_ohlcv_by_user_date_range(out, start_date, end_date)
        out.attrs["asset_type"] = "bond_cn"
        out.attrs["symbol"] = code
        out.attrs["series_kind"] = "price"
        out.attrs["unit_caption"] = "人民币计价 · 国债现货/债券 K 线（列含义以数据源为准）"
        return out

    if kind_key == "us_yield":
        start_arg = start_ts.strftime("%Y%m%d")
        raw = _call_akshare("bond_zh_us_rate", ak.bond_zh_us_rate, start_date=start_arg)
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raise EmptySymbolDataError("bond_zh_us_rate returned empty.")

        date_col = raw.columns[0]
        col_name = us_yield_column or US_TREASURY_YIELD_10Y_CN_COL
        if col_name not in raw.columns:
            candidates = [c for c in raw.columns if c != date_col]
            raise ValueError(
                "未找到收益率列 %r，可选列示例: %s"
                % (col_name, ", ".join(str(c) for c in candidates[:8]))
            )

        work = pd.DataFrame(
            {
                "日期": pd.to_datetime(raw[date_col], errors="coerce"),
                "收盘": pd.to_numeric(raw[col_name], errors="coerce"),
            }
        ).dropna(subset=["日期", "收盘"])
        if work.empty:
            raise EmptySymbolDataError("美债收益率解析后无有效行。")
        close = work["收盘"].astype("float64")
        zeros = pd.Series(0.0, index=work.index, dtype="float64")
        pseudo = pd.DataFrame(
            {
                "日期": work["日期"],
                "开盘": close,
                "最高": close,
                "最低": close,
                "收盘": close,
                "成交量": zeros,
            }
        )
        out = fetcher._normalize_ohlcv(pseudo, A_SHARE_COLUMN_MAPPING)
        out = _filter_ohlcv_by_user_date_range(out, start_date, end_date)
        out.attrs["asset_type"] = "bond_us_yield"
        out.attrs["symbol"] = col_name
        out.attrs["series_kind"] = "yield_level"
        out.attrs["unit_caption"] = "收益率水平 %% · %s（非持有期回报，指标为水平序列统计）" % col_name
        out.attrs["yield_column"] = col_name
        return out

    raise ValueError("load_bond_data: 未知 kind %r。" % kind)


def load_commodity_data(symbol: str, start_date: DateLike, end_date: DateLike) -> pd.DataFrame:
    """工业商品等外盘期货（与贵金属共用 ``futures_foreign_hist``）。"""
    return load_futures_foreign_ohlcv(
        symbol,
        start_date,
        end_date,
        asset_type="commodity",
        unit_caption="USD 等 · 外盘期货主力连续（合约细则以数据源为准）",
    )


def load_reit_data(symbol: str, start_date: DateLike, end_date: DateLike) -> pd.DataFrame:
    """沪深 C-REITs 日线（东财 K 线）。"""
    code = str(symbol).strip()
    if not code:
        raise ValueError("REITs 基金代码不能为空，例如 508097。")
    ak = _import_akshare()
    raw = _call_akshare("reits_hist_em(%r)" % code, ak.reits_hist_em, symbol=code)
    raw2 = _em_kline_frame_to_ohlcv_raw(raw)
    fetcher = AKShareFetcher(max_retries=1, retry_delay=0.0)
    out = fetcher._normalize_ohlcv(raw2, {**A_SHARE_COLUMN_MAPPING, **_EM_KLINE_OHLCV_MAPPING})
    out = _filter_ohlcv_by_user_date_range(out, start_date, end_date)
    out.attrs["asset_type"] = "creit"
    out.attrs["symbol"] = code
    out.attrs["series_kind"] = "price"
    out.attrs["unit_caption"] = "CNY 计价 · 基础设施 REITs"
    return out


def load_fx_data(symbol: str, start_date: DateLike, end_date: DateLike) -> pd.DataFrame:
    """外汇即期日线（东财 ``forex_hist_em``）。"""
    raw_key = str(symbol).strip().upper().replace(" ", "")
    if not raw_key:
        raise ValueError("外汇代码不能为空，例如 USDCNH。")
    pair = FX_PAIR_ALIASES.get(raw_key, raw_key)
    raw = _fetch_forex_hist_em_robust(pair)
    raw2 = _em_kline_frame_to_ohlcv_raw(raw)
    fetcher = AKShareFetcher(max_retries=1, retry_delay=0.0)
    out = fetcher._normalize_ohlcv(raw2, {**A_SHARE_COLUMN_MAPPING, **_EM_KLINE_OHLCV_MAPPING})
    out = _filter_ohlcv_by_user_date_range(out, start_date, end_date)
    out.attrs["asset_type"] = "fx"
    out.attrs["symbol"] = pair
    out.attrs["series_kind"] = "price"
    out.attrs["unit_caption"] = "汇率 · %s（间接标价含义以数据源为准）" % pair
    return out


def get_multiple_assets_close(
    symbols_list: list[str] | tuple[str, ...],
    start_date: DateLike,
    end_date: DateLike,
    *,
    adjust: str = "hfq",
) -> pd.DataFrame:
    """拉取多只 A 股 ETF 的收盘价宽表（按日期 outer 对齐后前向填充）。

    假设 ``symbols_list`` 中均为沪深 ETF 代码。通过本模块内 ``AKShareFetcher`` 逐只拉取并裁剪区间，
    逻辑上等价于单标的 ETF 拉取的组合；**不**依赖 ``asset_resolver``，以避免循环导入。

    Parameters
    ----------
    symbols_list
        ETF 代码列表；元素会先 ``strip``、去重（保序），且不得包含去空白后的空串。
    start_date, end_date
        分析区间；与 ``_filter_ohlcv_by_user_date_range`` 使用相同的日级时间戳与空区间语义。
    adjust
        复权方式，``"qfq"`` 或 ``"hfq"``，默认后复权。

    Returns
    -------
    pd.DataFrame
        索引为 ``DatetimeIndex``（升序），每列名为对应 ETF 代码，值为 ``Close``（``float64``）。

    Raises
    ------
    ValueError
        代码列表非法、复权参数非法、日期区间非法，或合并后某代码列全为缺失（与用户区间无重叠或无有效数据）。
    """
    if not isinstance(symbols_list, (list, tuple)):
        raise TypeError("symbols_list must be a list or tuple of ETF codes.")

    stripped = [str(s).strip() for s in symbols_list]
    if any(not s for s in stripped):
        raise ValueError("symbols_list must not contain empty codes after stripping whitespace.")

    symbols_ordered: list[str] = []
    seen: set[str] = set()
    for s in stripped:
        if s not in seen:
            seen.add(s)
            symbols_ordered.append(s)

    if not symbols_ordered:
        raise ValueError("symbols_list must be a non-empty list of ETF codes after normalization.")

    adjust_key = str(adjust).strip().lower()
    if adjust_key not in {"qfq", "hfq"}:
        raise ValueError("adjust must be 'qfq' or 'hfq', got %r." % (adjust,))

    start_ts = _coerce_analysis_timestamp(start_date, "start_date")
    end_ts = _coerce_analysis_timestamp(end_date, "end_date")
    if start_ts > end_ts:
        raise ValueError("start_date must be earlier than or equal to end_date.")

    fetcher = AKShareFetcher(max_retries=3, retry_delay=1.0)
    close_frames: list[pd.Series] = []
    for sym in symbols_ordered:
        ohlcv = fetcher.fetch_china_equity(
            "etf",
            sym,
            start=start_ts,
            end=end_ts,
            adjust=adjust_key,
        )
        clipped = _filter_ohlcv_by_user_date_range(ohlcv, start_date, end_date)
        close_frames.append(clipped["Close"].rename(sym))

    wide = pd.concat(close_frames, axis=1, join="outer")
    wide = wide.sort_index().ffill()

    for col in wide.columns:
        if wide[col].isna().all():
            raise ValueError(
                "标的 %r 在指定区间内无有效收盘价，或与所选标的无可用重叠数据。" % str(col)
            )

    wide.index.name = "Date"
    return wide.astype("float64")


__all__ = [
    "DateLike",
    "REQUIRED_OHLCV_COLUMNS",
    "A_SHARE_COLUMN_MAPPING",
    "DataFetchError",
    "EmptySymbolDataError",
    "NetworkFetchError",
    "VendorUpstreamError",
    "ChinaEquityPipelineError",
    "apply_default_network_proxy_policy",
    "BaseDataFetcher",
    "AKShareFetcher",
    "AKShareChinaPrimarySource",
    "EastMoneyDailyKlineClient",
    "YFinanceFetcher",
    "get_fund_data",
    "FOREIGN_FUTURES_COLUMN_MAPPING",
    "GLOBAL_INDEX_EM_ALIASES",
    "US_TREASURY_YIELD_10Y_CN_COL",
    "FX_PAIR_ALIASES",
    "load_futures_foreign_ohlcv",
    "load_global_market_data",
    "load_bond_data",
    "load_commodity_data",
    "load_reit_data",
    "load_fx_data",
    "get_multiple_assets_close",
]
