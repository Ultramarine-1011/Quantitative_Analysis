"""Deterministic daily OHLCV fixtures; tests never need vendor services."""

import http.client
import os
import socket
import urllib.error
import urllib.parse
import urllib.request

import httpx
import numpy as np
import pandas as pd
import pytest
import requests
import urllib3

os.environ.setdefault("MPLBACKEND", "Agg")

_ORIG_SOCKET_CREATE_CONNECTION = socket.create_connection
_ORIG_SOCKET_CONNECT = socket.socket.connect
_ORIG_SOCKET_CONNECT_EX = socket.socket.connect_ex
_ORIG_SOCKET_GETADDRINFO = socket.getaddrinfo
_ORIG_REQUESTS_SEND = requests.sessions.Session.send
_ORIG_URLLIB3_URLOPEN = urllib3.connectionpool.HTTPConnectionPool.urlopen
_ORIG_HTTPX_SEND = httpx.Client.send
_ORIG_HTTPX_ASYNC_SEND = httpx.AsyncClient.send
_ORIG_URLLIB_OPEN = urllib.request.OpenerDirector.open
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_INET_FAMILIES = frozenset(
    getattr(socket, name) for name in ("AF_INET", "AF_INET6") if hasattr(socket, name)
)


def _is_loopback_host(host: object) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="ignore")
    if isinstance(host, str):
        return host.strip().lower() in _LOOPBACK_HOSTS
    return False


def _extract_address_host(address: object) -> object:
    if isinstance(address, tuple) and address:
        return address[0]
    return address


@pytest.fixture(autouse=True)
def _block_outbound_network(monkeypatch):
    """Fail fast if an offline test attempts a real outbound network connection."""

    def _deny_http_connect(self, *args, **kwargs):
        host = getattr(self, "host", None)
        if _is_loopback_host(host):
            raise OSError(f"Loopback HTTP connect unexpected in mocked test ({host!r})")
        raise OSError(f"Offline test guard blocked outbound HTTP connection to {host!r}")

    def _deny_create_connection(address, *args, **kwargs):
        host = _extract_address_host(address)
        if _is_loopback_host(host):
            return _ORIG_SOCKET_CREATE_CONNECTION(address, *args, **kwargs)
        raise OSError(f"Offline test guard blocked outbound socket connection to {address!r}")

    def _deny_socket_connect(self, address, *args, **kwargs):
        if getattr(self, "family", None) in _INET_FAMILIES:
            host = _extract_address_host(address)
            if not _is_loopback_host(host):
                raise OSError(f"Offline test guard blocked outbound socket connect to {address!r}")
        return _ORIG_SOCKET_CONNECT(self, address, *args, **kwargs)

    def _deny_socket_connect_ex(self, address, *args, **kwargs):
        if getattr(self, "family", None) in _INET_FAMILIES:
            host = _extract_address_host(address)
            if not _is_loopback_host(host):
                raise OSError(
                    f"Offline test guard blocked outbound socket connect_ex to {address!r}"
                )
        return _ORIG_SOCKET_CONNECT_EX(self, address, *args, **kwargs)

    def _deny_getaddrinfo(host, port, *args, **kwargs):
        if host is None or host == "" or _is_loopback_host(host):
            return _ORIG_SOCKET_GETADDRINFO(host, port, *args, **kwargs)
        raise OSError(f"Offline test guard blocked outbound DNS resolution for {host!r}")

    def _deny_requests_send(self, request, *args, **kwargs):
        host = urllib.parse.urlsplit(str(getattr(request, "url", "") or "")).hostname
        if not _is_loopback_host(host):
            raise requests.exceptions.ConnectionError(
                f"Offline test guard blocked outbound requests URL {getattr(request, 'url', '')!r}"
            )
        return _ORIG_REQUESTS_SEND(self, request, *args, **kwargs)

    def _deny_urllib3_urlopen(self, method, url, *args, **kwargs):
        url_host = urllib.parse.urlsplit(str(url or "")).hostname
        target_host = url_host or getattr(self, "host", None)
        if not _is_loopback_host(target_host):
            raise urllib3.exceptions.HTTPError(
                f"Offline test guard blocked outbound urllib3 request to {target_host!r}"
            )
        return _ORIG_URLLIB3_URLOPEN(self, method, url, *args, **kwargs)

    def _deny_httpx_send(self, request, *args, **kwargs):
        host = getattr(getattr(request, "url", None), "host", None)
        if not _is_loopback_host(host):
            raise httpx.ConnectError(
                f"Offline test guard blocked outbound httpx URL {getattr(request, 'url', '')!r}",
                request=request,
            )
        return _ORIG_HTTPX_SEND(self, request, *args, **kwargs)

    async def _deny_httpx_async_send(self, request, *args, **kwargs):
        host = getattr(getattr(request, "url", None), "host", None)
        if not _is_loopback_host(host):
            raise httpx.ConnectError(
                f"Offline test guard blocked outbound httpx URL {getattr(request, 'url', '')!r}",
                request=request,
            )
        return await _ORIG_HTTPX_ASYNC_SEND(self, request, *args, **kwargs)

    def _deny_urllib_open(self, fullurl, *args, **kwargs):
        raw_url = getattr(fullurl, "full_url", fullurl)
        host = urllib.parse.urlsplit(str(raw_url or "")).hostname
        if not _is_loopback_host(host):
            raise urllib.error.URLError(
                f"Offline test guard blocked outbound urllib URL {raw_url!r}"
            )
        return _ORIG_URLLIB_OPEN(self, fullurl, *args, **kwargs)

    monkeypatch.setattr(http.client.HTTPConnection, "connect", _deny_http_connect)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", _deny_http_connect)
    monkeypatch.setattr(socket, "create_connection", _deny_create_connection)
    monkeypatch.setattr(socket.socket, "connect", _deny_socket_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _deny_socket_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", _deny_getaddrinfo)
    monkeypatch.setattr(requests.sessions.Session, "send", _deny_requests_send)
    monkeypatch.setattr(
        urllib3.connectionpool.HTTPConnectionPool,
        "urlopen",
        _deny_urllib3_urlopen,
    )
    monkeypatch.setattr(httpx.Client, "send", _deny_httpx_send)
    monkeypatch.setattr(httpx.AsyncClient, "send", _deny_httpx_async_send)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", _deny_urllib_open)


@pytest.fixture(autouse=True)
def _isolate_disk_cache(monkeypatch):
    """Disable persistent disk cache by default so offline unit tests stay isolated."""
    monkeypatch.setenv("QUANT_DISABLE_CACHE", "1")


@pytest.fixture
def sample_ohlcv():
    close = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0.0004, 0.01, 120)))
    frame = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(close), 1000.0),
        },
        index=pd.date_range("2024-01-02", periods=len(close), freq="B"),
    )
    frame.index = frame.index.as_unit("ns")
    frame.index.name = "Date"
    frame.attrs.update(symbol="510300", asset_type="etf", series_kind="price")
    return frame
