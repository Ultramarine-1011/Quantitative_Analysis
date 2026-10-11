"""Network/proxy policy shared by vendor adapters. Importing has no side effects."""

from __future__ import annotations

import importlib
import os
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from .errors import DataFetchError

# 全进程仅解析一次：首个 ``AKShareFetcher`` 构造时的参数优先，避免导入本模块即改写全局代理行为。
_PROXY_POLICY_RESOLVED: bool = False


def _configure_windows_proxy_behavior(*, use_system_proxy: bool | None = None) -> None:
    """按配置决定是否屏蔽 Windows「Internet 设置」里的系统代理（仍尊重显式 ``HTTP_PROXY`` 等）。

    默认（``use_system_proxy is None``）读取环境变量 ``QUANT_USE_SYSTEM_PROXY``：
    若为 ``1``/``true``/``yes`` 则保留系统代理；否则仅在 Windows 使用环境变量代理解析，
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
    if use_system_proxy or sys.platform != "win32":
        return
    import urllib.request as _urllib_request

    _urllib_request.getproxies = _urllib_request.getproxies_environment


def apply_default_network_proxy_policy() -> None:
    """幂等地根据 ``QUANT_USE_SYSTEM_PROXY`` 应用 WinINet 代理策略（全进程仅生效一次）。

    Web 等入口若在未构造 ``AKShareFetcher`` 前直接调用 akshare（如贵金属），宜在启动时调用本函数，
    使与 ETF/个股链路共享同一代理语义。
    """
    _configure_windows_proxy_behavior(use_system_proxy=None)


# 直连东方财富 / 临时禁用 AKShare 所继承的坏代理时，显式关掉 requests 的代理（trust_env=False 仍可能被环境或适配器带上代理）。
_REQUESTS_NO_PROXY: dict[str, None] = {"http": None, "https": None}


_PROXY_LOCK = threading.RLock()
_PROXY_CONTEXT = threading.local()


@contextmanager
def _without_http_proxy_env() -> Iterator[None]:
    """Bypass requests' inherited proxies only in this thread.

    Never mutate os.environ: concurrent Streamlit sessions and explicit crypto
    proxies must keep their own configuration. Serialize patch installation;
    other threads still call the original requests proxy resolver.
    """
    import requests.sessions as sessions
    import requests.utils as utils

    with _PROXY_LOCK:
        original_utils = utils.get_environ_proxies
        original_sessions = sessions.get_environ_proxies
        previous = getattr(_PROXY_CONTEXT, "direct", False)

        def utils_proxies(url: str, no_proxy: Any = None) -> dict[str, str]:
            return {} if getattr(_PROXY_CONTEXT, "direct", False) else original_utils(url, no_proxy)

        def session_proxies(url: str, no_proxy: Any = None) -> dict[str, str]:
            return (
                {} if getattr(_PROXY_CONTEXT, "direct", False) else original_sessions(url, no_proxy)
            )

        _PROXY_CONTEXT.direct = True
        utils.get_environ_proxies = utils_proxies
        sessions.get_environ_proxies = session_proxies
        try:
            yield
        finally:
            utils.get_environ_proxies = original_utils
            sessions.get_environ_proxies = original_sessions
            _PROXY_CONTEXT.direct = previous


def _akshare_neutralize_proxy_like_fetcher() -> bool:
    """与 ``AKShareFetcher(use_system_proxy=None)`` 一致：默认忽略 WinINet 代理。"""
    return not (os.environ.get("QUANT_USE_SYSTEM_PROXY", "").lower() in {"1", "true", "yes"})


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
            # 与境内权益链路保持一致：默认临时屏蔽环境代理，降低坏代理导致的失败率。
            with _without_http_proxy_env():
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)
    except (ImportError, DataFetchError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataFetchError("%s 失败: %r" % (op_label, exc)) from exc
