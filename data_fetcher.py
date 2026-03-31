"""量化分析系统的数据获取层。

本模块实现了面向 A 股日线数据的稳健 ETL 流程，重点处理以下问题：

- 网络抖动导致的临时抓取失败
- 第三方接口返回列名不统一
- 数值列 dtype 混乱
- 日期列未标准化
- 缺失值与重复索引
- ETF/个股：AKShare 主源 → 东方财富多节点 K 线 → 腾讯/新浪备用源；``secid`` 市场前缀本地推导；WinINet 代理策略在首个 ``AKShareFetcher`` 构造时解析
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
        """多域名轮询直连 K 线；若全部失败由调用方再走腾讯/新浪等备用源。"""
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

    def _fetch_tertiary_china_equity_raw(
        self,
        equity_route: str,
        sym: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        adjust: str,
    ) -> pd.DataFrame:
        """AKShare 腾讯 / 新浪接口备用（域名与东方财富不同，部分网络环境下可兜底）。"""
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

            backup = self._fetch_tertiary_china_equity_raw(
                equity_route, sym, start_ts, end_ts, adjust
            )
            if not backup.empty:
                return backup

            if em_fail is not None:
                raise ChinaEquityPipelineError(
                    "%s：AKShare 主源未成功 (%r)；东方财富直连失败 (%s)。"
                    % (label_zh, first_error, em_fail)
                ) from em_fail
            raise ChinaEquityPipelineError(
                "%s：AKShare 未返回有效数据 (%r)；东方财富直连返回空 K 线。"
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
]
