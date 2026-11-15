"""Deterministic daily OHLCV fixtures; tests never need vendor services."""

import http.client
import os
import socket

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

_ORIG_SOCKET_CREATE_CONNECTION = socket.create_connection


@pytest.fixture(autouse=True)
def _block_outbound_network(monkeypatch):
    """Fail fast if an offline test attempts a real outbound network connection."""

    def _deny_http_connect(self, *args, **kwargs):
        host = getattr(self, "host", None)
        raise OSError(f"Offline test guard blocked outbound HTTP connection to {host!r}")

    def _deny_create_connection(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) and address else address
        if host in {"127.0.0.1", "::1", "localhost"}:
            return _ORIG_SOCKET_CREATE_CONNECTION(address, *args, **kwargs)
        raise OSError(f"Offline test guard blocked outbound socket connection to {address!r}")

    monkeypatch.setattr(http.client.HTTPConnection, "connect", _deny_http_connect)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", _deny_http_connect)
    monkeypatch.setattr(socket, "create_connection", _deny_create_connection)


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
