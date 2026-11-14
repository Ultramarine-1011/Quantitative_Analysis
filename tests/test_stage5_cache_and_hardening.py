"""Stage 5 tests: shared Parquet+JSON cache, CLI --refresh, UI bypass, & hardening."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app
import asset_loader
import main
from data_fetcher import YFinanceFetcher
from data_sources.cache import (
    build_cache_key,
    clear_ohlcv_cache,
    get_cache_dir,
    get_ohlcv_cache_stats,
    is_cache_enabled,
    load_cached_ohlcv,
    save_cached_ohlcv,
)


def _enable_tmp_cache(monkeypatch, tmp_path: Path) -> Path:
    cache_dir = tmp_path / "ohlcv_cache"
    monkeypatch.delenv("QUANT_DISABLE_CACHE", raising=False)
    monkeypatch.setenv("QUANT_CACHE_DIR", str(cache_dir))
    return cache_dir


def test_pytest_isolation_disables_disk_cache_by_default():
    assert not is_cache_enabled()


def test_disk_cache_roundtrip_attrs_and_subset_slicing(monkeypatch, tmp_path, sample_ohlcv):
    cache_dir = _enable_tmp_cache(monkeypatch, tmp_path)
    calls: list[tuple[str, object]] = []

    def fake_fetch(**kwargs):
        calls.append((kwargs["symbol"], kwargs.get("adjust")))
        df = sample_ohlcv.copy()
        df.attrs["data_source"] = "mock.etf"
        df.attrs["adjust"] = kwargs.get("adjust", "hfq")
        df.attrs["quality"] = {"dropped_rows": 0, "ffilled_cells": 0, "missing_policy": "drop"}
        return df

    monkeypatch.setattr(asset_loader, "_fetch_etf_data", fake_fetch)

    start_full = sample_ohlcv.index[0].strftime("%Y-%m-%d")
    end_full = sample_ohlcv.index[-1].strftime("%Y-%m-%d")
    first = asset_loader.load_asset_data("etf", "510300", start_full, end_full)
    assert len(calls) == 1
    assert first.attrs["cache_hit"] is False

    stats = get_ohlcv_cache_stats()
    assert stats["enabled"] is True
    assert stats["entries"] == 1
    key = build_cache_key("etf", "510300", adjust=None)
    assert key == build_cache_key("etf", "510300", adjust="hfq")
    assert (cache_dir / f"{key}.parquet").is_file()
    assert (cache_dir / f"{key}.meta.json").is_file()

    # Sub-window request with explicit adjust='hfq' hits the same cache and slices rows.
    sub_start = sample_ohlcv.index[10].strftime("%Y-%m-%d")
    sub_end = sample_ohlcv.index[40].strftime("%Y-%m-%d")
    second = asset_loader.load_asset_data("etf", "510300", sub_start, sub_end, adjust="hfq")
    assert len(calls) == 1
    assert second.attrs["cache_hit"] is True
    assert second.attrs["data_source"] == "mock.etf"
    assert second.attrs["adjust"] == "hfq"
    assert second.index[0] >= pd.Timestamp(sub_start)
    assert second.index[-1] <= pd.Timestamp(sub_end)
    assert len(second) == 31

    # Wider window exceeds cached boundaries and triggers a clean upstream fetch.
    wider_start = (sample_ohlcv.index[0] - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    third = asset_loader.load_asset_data("etf", "510300", wider_start, end_full)
    assert len(calls) == 2
    assert third.attrs["cache_hit"] is False


def test_disk_cache_ttl_expiration_and_force_refresh(monkeypatch, tmp_path, sample_ohlcv):
    cache_dir = _enable_tmp_cache(monkeypatch, tmp_path)
    calls = 0

    def fake_fetch(**kwargs):
        nonlocal calls
        calls += 1
        return sample_ohlcv.copy()

    monkeypatch.setattr(asset_loader, "_fetch_etf_data", fake_fetch)
    start_s = sample_ohlcv.index[0].strftime("%Y-%m-%d")
    end_s = sample_ohlcv.index[-1].strftime("%Y-%m-%d")

    asset_loader.load_asset_data("etf", "510300", start_s, end_s)
    assert calls == 1

    # force_refresh=True bypasses valid cache and overwrites entry.
    refreshed = asset_loader.load_asset_data("etf", "510300", start_s, end_s, force_refresh=True)
    assert calls == 2
    assert refreshed.attrs["cache_hit"] is False

    # Backdate cached_at in sidecar JSON to simulate TTL expiry.
    key = build_cache_key("etf", "510300", adjust="hfq")
    meta_path = cache_dir / f"{key}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["cached_at"] = float(meta["cached_at"]) - 7200.0
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    expired = asset_loader.load_asset_data("etf", "510300", start_s, end_s)
    assert calls == 3
    assert expired.attrs["cache_hit"] is False


def test_disk_cache_shared_between_cli_and_loader_and_cli_refresh(
    monkeypatch, tmp_path, sample_ohlcv
):
    _enable_tmp_cache(monkeypatch, tmp_path)
    calls = 0

    def fake_fetch(**kwargs):
        nonlocal calls
        calls += 1
        df = sample_ohlcv.copy()
        df.attrs["data_source"] = "cli.shared.mock"
        return df

    monkeypatch.setattr(asset_loader, "_fetch_etf_data", fake_fetch)
    monkeypatch.setattr(main.MplfinanceVisualizer, "plot", lambda self, df, **kwargs: None)

    start_s = sample_ohlcv.index[0].strftime("%Y-%m-%d")
    end_s = sample_ohlcv.index[-1].strftime("%Y-%m-%d")
    rc = main.main(
        [
            "--asset-type",
            "etf",
            "--symbol",
            "510300",
            "--start",
            start_s,
            "--end",
            end_s,
            "--no-show",
        ]
    )
    assert rc == 0
    assert calls == 1

    # Subsequent loader call reuses CLI-populated disk cache across entrypoints.
    hit = asset_loader.load_asset_data("etf", "510300", start_s, end_s)
    assert calls == 1
    assert hit.attrs["cache_hit"] is True
    assert hit.attrs["data_source"] == "cli.shared.mock"

    # CLI --refresh forces re-fetch despite valid cache.
    rc2 = main.main(
        [
            "--asset-type",
            "etf",
            "--symbol",
            "510300",
            "--start",
            start_s,
            "--end",
            end_s,
            "--refresh",
            "--no-show",
        ]
    )
    assert rc2 == 0
    assert calls == 2


def test_disk_cache_handles_corrupt_files_and_skips_empty(monkeypatch, tmp_path, sample_ohlcv):
    cache_dir = _enable_tmp_cache(monkeypatch, tmp_path)
    start_s = sample_ohlcv.index[0].strftime("%Y-%m-%d")
    end_s = sample_ohlcv.index[-1].strftime("%Y-%m-%d")

    assert save_cached_ohlcv(sample_ohlcv.iloc[0:0], "etf", "510300", start_s, end_s) is None

    saved = save_cached_ohlcv(sample_ohlcv, "etf", "510300", start_s, end_s)
    assert saved is not None
    meta_path = saved.with_suffix(".meta.json")
    meta_path.write_text("{corrupt json", encoding="utf-8")
    assert load_cached_ohlcv("etf", "510300", start_s, end_s) is None

    # Rewrite valid entry then corrupt parquet bytes.
    save_cached_ohlcv(sample_ohlcv, "etf", "510300", start_s, end_s)
    saved.write_bytes(b"not a parquet file")
    assert load_cached_ohlcv("etf", "510300", start_s, end_s) is None

    assert clear_ohlcv_cache() == 2
    assert get_cache_dir() == cache_dir
    assert get_ohlcv_cache_stats()["entries"] == 0


def test_app_force_refresh_checkbox_bypasses_streamlit_and_disk_cache(
    monkeypatch, tmp_path, sample_ohlcv
):
    from streamlit.testing.v1 import AppTest

    _enable_tmp_cache(monkeypatch, tmp_path)
    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()

    # Build a synthetic ETF covering the app's default 3-year window up to today so cache covers it.
    today = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=today, periods=900)
    close = 100.0 * np.exp(np.cumsum(np.random.default_rng(7).normal(0.0005, 0.01, len(dates))))
    wide_sample = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(dates), 1000.0),
        },
        index=dates,
    )
    wide_sample.index.name = "Date"

    fetches = 0

    def fake_fetch(**kwargs):
        nonlocal fetches
        fetches += 1
        df = wide_sample.copy()
        df.attrs["data_source"] = "app.test.etf"
        return df

    monkeypatch.setattr(asset_loader, "_fetch_etf_data", fake_fetch)

    page1 = AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()
    page1.sidebar.button[0].click().run()
    assert fetches == 1
    assert not page1.exception

    # Fresh UI session clears @st.cache_data but still hits shared Parquet disk cache.
    app.load_asset_data.clear()
    page2 = AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()
    page2.sidebar.button[0].click().run()
    assert fetches == 1
    assert any("本地缓存命中" in cap.value for cap in page2.caption)

    # Checking force_refresh_input forces upstream re-fetch even with warm disk + memory cache.
    page2.checkbox("force_refresh_input").set_value(True).run()
    page2.sidebar.button[0].click().run()
    assert fetches == 2
    assert not any("本地缓存命中" in cap.value for cap in page2.caption)

    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()


def test_yfinance_fetcher_deprecated_and_pyproject_requires_py311():
    with pytest.warns(DeprecationWarning, match="YFinanceFetcher is deprecated"):
        client = YFinanceFetcher()
    with pytest.raises(NotImplementedError):
        client.fetch_us_stock("AAPL")

    pyproject_text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert 'requires-python = ">=3.11"' in pyproject_text
