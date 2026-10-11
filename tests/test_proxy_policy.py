"""Proxy isolation across concurrent Streamlit sessions and nested failures."""

from concurrent.futures import ThreadPoolExecutor
import os
import urllib.request

import pytest
import requests.sessions
import requests.utils

from data_sources import network


def test_proxy_context_is_thread_local_and_restores_after_error(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9999")
    resolver = lambda url, no_proxy=None: {"http": os.environ["HTTP_PROXY"]}
    monkeypatch.setattr(requests.sessions, "get_environ_proxies", resolver)
    monkeypatch.setattr(requests.utils, "get_environ_proxies", resolver)
    original = requests.sessions.get_environ_proxies
    with pytest.raises(RuntimeError, match="stop"):
        with network._without_http_proxy_env():
            assert requests.sessions.get_environ_proxies("https://example.org") == {}
            assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:9999"
            with ThreadPoolExecutor(max_workers=1) as pool:
                assert pool.submit(
                    requests.sessions.get_environ_proxies, "https://example.org"
                ).result(timeout=5) == {"http": "http://127.0.0.1:9999"}
            with network._without_http_proxy_env():
                assert requests.utils.get_environ_proxies("https://example.org") == {}
            assert requests.sessions.get_environ_proxies("https://example.org") == {}
            raise RuntimeError("stop")
    assert requests.sessions.get_environ_proxies is original
    assert requests.utils.get_environ_proxies is resolver


def test_windows_proxy_policy_never_changes_other_platforms(monkeypatch):
    monkeypatch.setattr(network, "_PROXY_POLICY_RESOLVED", False)
    monkeypatch.setattr(network.sys, "platform", "darwin")
    before = urllib.request.getproxies
    network._configure_windows_proxy_behavior(use_system_proxy=False)
    assert urllib.request.getproxies is before


def test_windows_policy_preserves_explicit_environment_proxy(monkeypatch):
    monkeypatch.setattr(network, "_PROXY_POLICY_RESOLVED", False)
    monkeypatch.setattr(network.sys, "platform", "win32")
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"https": "system"})
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1234")
    network._configure_windows_proxy_behavior(use_system_proxy=False)
    assert urllib.request.getproxies()["https"] == "http://127.0.0.1:1234"
