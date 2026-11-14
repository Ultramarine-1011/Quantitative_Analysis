"""Network/proxy policy and bounded request budgets shared by vendor adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import os
import sys
import threading
import time
from typing import Any

from .errors import DataFetchError, EmptySymbolDataError, NetworkFetchError, VendorUpstreamError

# 全进程仅解析一次：首个 ``AKShareFetcher`` 构造时的参数优先，避免导入本模块即改写全局代理行为。
_PROXY_POLICY_RESOLVED: bool = False
_POLICY_LOCK = threading.Lock()


def _configure_windows_proxy_behavior(*, use_system_proxy: bool | None = None) -> None:
    """按配置决定是否屏蔽 Windows「Internet 设置」里的系统代理（仍尊重显式 ``HTTP_PROXY`` 等）。"""
    global _PROXY_POLICY_RESOLVED
    with _POLICY_LOCK:
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
    """幂等地根据 ``QUANT_USE_SYSTEM_PROXY`` 应用 WinINet 代理策略（全进程仅生效一次）。"""
    _configure_windows_proxy_behavior(use_system_proxy=None)


# 直连东方财富 / 显式关闭 requests 会话的代理继承。
_REQUESTS_NO_PROXY: dict[str, None] = {"http": None, "https": None}

_PROXY_LOCK = threading.RLock()
_PROXY_CONTEXT = threading.local()
_PATCH_REFCOUNT: int = 0
_ORIGINAL_UTILS_GETPROXIES: Any = None
_ORIGINAL_SESSIONS_GETPROXIES: Any = None


@dataclass
class RequestBudget:
    """Bounded wall-clock and attempt budget to prevent fallback retry storms."""

    total_timeout_seconds: float = 45.0
    per_request_timeout_seconds: float = 12.0
    max_attempts: int = 8
    _started_at: float = field(default_factory=time.monotonic, init=False)
    attempts: int = field(default=0, init=False)

    def remaining_seconds(self) -> float:
        elapsed = time.monotonic() - self._started_at
        return max(0.0, float(self.total_timeout_seconds) - elapsed)

    def effective_timeout(self) -> float:
        rem = self.remaining_seconds()
        if rem <= 0.0:
            raise NetworkFetchError(f"请求总超时预算已耗尽（{self.total_timeout_seconds:.1f}s）。")
        return min(float(self.per_request_timeout_seconds), rem)

    def consume_attempt(self, label: str = "request") -> float:
        rem = self.remaining_seconds()
        if rem <= 0.0:
            raise NetworkFetchError(
                f"{label} 已达到总耗时上限（{self.total_timeout_seconds:.1f}s）。"
            )
        if self.attempts >= self.max_attempts:
            raise NetworkFetchError(f"{label} 已达到最大尝试次数上限（{self.max_attempts} 次）。")
        self.attempts += 1
        return min(float(self.per_request_timeout_seconds), rem)


def classify_network_exception(exc: BaseException) -> str:
    """Classify a fetch failure into 'empty', 'transport', 'upstream', or 'caller'."""
    if isinstance(exc, EmptySymbolDataError):
        return "empty"
    if isinstance(exc, NetworkFetchError):
        return "transport"
    if isinstance(exc, VendorUpstreamError):
        return "upstream"
    if isinstance(exc, (ValueError, TypeError, ImportError)):
        return "caller"
    mod = (type(exc).__module__ or "").lower()
    name = type(exc).__name__.lower()
    if any(
        token in name
        for token in (
            "timeout",
            "connect",
            "proxy",
            "protocol",
            "ssl",
            "socket",
            "dns",
            "reset",
            "eof",
            "brokenpipe",
            "requestexception",
            "networkerror",
        )
    ):
        return "transport"
    if any(token in name for token in ("httpstatuserror", "jsondecodeerror")):
        return "upstream"
    if mod.startswith(("httpx", "httpcore", "requests", "urllib3", "ccxt")):
        return "transport"
    return "upstream"


@contextmanager
def _without_http_proxy_env() -> Iterator[None]:
    """Reference-counted thread-local proxy bypass for third-party libs that read env proxies.

    The lock is held ONLY while installing/removing the dispatcher and updating the reference
    count—NEVER across blocking network I/O—so concurrent fetches execute in parallel.
    """
    global _PATCH_REFCOUNT, _ORIGINAL_UTILS_GETPROXIES, _ORIGINAL_SESSIONS_GETPROXIES
    import requests.sessions as sessions
    import requests.utils as utils

    previous = getattr(_PROXY_CONTEXT, "direct", False)
    with _PROXY_LOCK:
        if _PATCH_REFCOUNT == 0:
            orig_utils = utils.get_environ_proxies
            orig_sessions = sessions.get_environ_proxies
            _ORIGINAL_UTILS_GETPROXIES = orig_utils
            _ORIGINAL_SESSIONS_GETPROXIES = orig_sessions

            def utils_proxies(url: str, no_proxy: Any = None) -> dict[str, str]:
                return {} if getattr(_PROXY_CONTEXT, "direct", False) else orig_utils(url, no_proxy)

            def session_proxies(url: str, no_proxy: Any = None) -> dict[str, str]:
                return (
                    {} if getattr(_PROXY_CONTEXT, "direct", False) else orig_sessions(url, no_proxy)
                )

            utils.get_environ_proxies = utils_proxies
            sessions.get_environ_proxies = session_proxies
        _PATCH_REFCOUNT += 1
        _PROXY_CONTEXT.direct = True

    try:
        yield
    finally:
        _PROXY_CONTEXT.direct = previous
        with _PROXY_LOCK:
            _PATCH_REFCOUNT = max(0, _PATCH_REFCOUNT - 1)
            if _PATCH_REFCOUNT == 0:
                if _ORIGINAL_UTILS_GETPROXIES is not None:
                    utils.get_environ_proxies = _ORIGINAL_UTILS_GETPROXIES
                if _ORIGINAL_SESSIONS_GETPROXIES is not None:
                    sessions.get_environ_proxies = _ORIGINAL_SESSIONS_GETPROXIES
                _ORIGINAL_UTILS_GETPROXIES = None
                _ORIGINAL_SESSIONS_GETPROXIES = None


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
    max_retries: int = 2,
    retry_delay: float = 0.4,
    budget: RequestBudget | None = None,
    **kwargs: Any,
) -> Any:
    """在默认代理策略与有限重试预算下调用 AKShare，按故障类别抛出清晰异常。"""
    apply_default_network_proxy_policy()
    active_budget = budget if budget is not None else RequestBudget()
    last_exc: Exception | None = None
    attempts = max(1, int(max_retries))

    for attempt in range(attempts):
        try:
            active_budget.consume_attempt(op_label)
            if _akshare_neutralize_proxy_like_fetcher():
                with _without_http_proxy_env():
                    return fn(*args, **kwargs)
            return fn(*args, **kwargs)
        except (ImportError, ValueError, TypeError, EmptySymbolDataError):
            raise
        except DataFetchError as exc:
            last_exc = exc
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if classify_network_exception(exc) == "caller":
                raise
        if attempt + 1 < attempts and active_budget.remaining_seconds() > retry_delay:
            time.sleep(retry_delay)

    if isinstance(last_exc, DataFetchError):
        raise last_exc
    if last_exc is not None and classify_network_exception(last_exc) == "transport":
        raise NetworkFetchError("%s 网络传输失败: %r" % (op_label, last_exc)) from last_exc
    raise DataFetchError("%s 失败: %r" % (op_label, last_exc)) from last_exc


__all__ = [
    "RequestBudget",
    "apply_default_network_proxy_policy",
    "classify_network_exception",
    "_without_http_proxy_env",
    "_import_akshare",
    "_call_akshare",
    "_REQUESTS_NO_PROXY",
    "_configure_windows_proxy_behavior",
]
