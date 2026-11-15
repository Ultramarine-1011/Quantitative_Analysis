"""M2 tests: Agent CLI (`quant`), structured outputs, exit codes, and hardening regressions."""

from __future__ import annotations

import errno
import json
import os
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
import requests
import urllib3

import quant_cli
from asset_catalog import SUPPORTED_ASSET_TYPES
from contracts import SCHEMA_VERSION, SOFTWARE_VERSION, AnalysisRequest, build_quality_report
from data_sources import cache as cache_mod
from data_sources.errors import (
    ChinaEquityPipelineError,
    DataFetchError,
    EmptySymbolDataError,
    NetworkFetchError,
    VendorUpstreamError,
)
from data_sources.network import classify_network_exception
from ohlcv import compute_ohlcv_quality
from services.analysis_service import analyze_single_asset, fetch_single_asset


def _make_ohlcv(
    dates: pd.DatetimeIndex,
    closes: np.ndarray | list[float],
    *,
    asset_type: str = "etf",
    symbol: str = "510300",
    adjust: str = "hfq",
    currency: str | None = "CNY",
    data_source: str = "mock:source",
    extra_attrs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    df = pd.DataFrame(
        {
            "Open": c * 0.995,
            "High": c * 1.01,
            "Low": c * 0.99,
            "Close": c,
            "Volume": np.full(len(c), 12000.0),
        },
        index=dates,
    )
    df.index = pd.DatetimeIndex(df.index).as_unit("ns")
    df.index.name = "Date"
    df.attrs.update(
        {
            "asset_type": asset_type,
            "symbol": symbol,
            "canonical_symbol": symbol,
            "adjust": adjust,
            "currency": currency,
            "data_source": data_source,
            "fetched_at": "2026-01-15T00:00:00+00:00",
            "cache_hit": False,
        }
    )
    df.attrs["data_quality"] = compute_ohlcv_quality(df, missing_policy="drop")
    if extra_attrs:
        df.attrs.update(extra_attrs)
    return df


# ---------------------------------------------------------------------------
# 1. Agent CLI (`quant fetch` / `quant analyze`) + structured output artifacts
# ---------------------------------------------------------------------------


def test_quant_analyze_writes_result_json_parquet_and_csv_idempotently(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=35)
    closes = 3.5 + np.linspace(0.0, 0.7, len(dates))
    raw_df = _make_ohlcv(dates, closes, asset_type="etf", symbol="510300", adjust="hfq")
    loader_calls: list[dict[str, Any]] = []

    def _loader(
        asset_type: str, symbol: str, start: object, end: object, **kwargs: Any
    ) -> pd.DataFrame:
        loader_calls.append({"asset_type": asset_type, "symbol": symbol, **kwargs})
        return raw_df.copy()

    out_dir = tmp_path / "run_out"
    code, res1 = quant_cli.run_cli_command(
        [
            "analyze",
            "etf:510300",
            "--start",
            "2024-01-02",
            "--end",
            "2024-02-20",
            "--adjust",
            "hfq",
            "--rf",
            "0.02",
            "--trading-days",
            "252",
            "--return-convention",
            "simple",
            "--sma-short",
            "3",
            "--sma-long",
            "6",
            "--bb-window",
            "5",
            "--include-momentum",
            "--csv",
            "-o",
            str(out_dir),
        ],
        loader=_loader,
    )
    assert code == quant_cli.EXIT_OK
    assert res1 is not None
    assert len(loader_calls) == 1

    captured1 = capsys.readouterr()
    assert "[quant] analyze etf:510300" in captured1.err
    assert "[quant] done status=ok" in captured1.err
    stdout_payload = json.loads(captured1.out)
    assert "data" not in stdout_payload

    result_path = out_dir / "result.json"
    parquet_path = out_dir / "data.parquet"
    csv_path = out_dir / "data.csv"
    assert result_path.is_file()
    assert parquet_path.is_file()
    assert csv_path.is_file()

    disk_payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert disk_payload == stdout_payload
    assert disk_payload["schema_version"] == SCHEMA_VERSION
    assert disk_payload["software_version"] == SOFTWARE_VERSION
    assert disk_payload["command"] == "analyze"
    assert disk_payload["status"] == "ok"
    assert disk_payload["parameters"]["return_convention"] == "simple"
    assert disk_payload["parameters"]["risk_free_rate"] == pytest.approx(0.02)
    assert disk_payload["parameters"]["sma_short"] == 3
    assert disk_payload["parameters"]["include_momentum"] is True
    assert disk_payload["metric_units"]["cumulative_return"] == "ratio_decimal"
    assert disk_payload["metric_units"]["sharpe_ratio"] == "dimensionless_ratio"
    assert disk_payload["metric_units"]["latest_close"] == "currency:CNY"
    assert disk_payload["data_summary"]["actual_start"] == "2024-01-02"
    assert disk_payload["data_summary"]["actual_end"] == "2024-02-19"
    assert disk_payload["data_summary"]["row_count"] == len(dates)
    assert disk_payload["artifacts"] == {
        "output_dir": str(out_dir.resolve()),
        "result_json": str(result_path.resolve()),
        "data_parquet": str(parquet_path.resolve()),
        "data_csv": str(csv_path.resolve()),
    }

    loaded_parquet = pd.read_parquet(parquet_path)
    assert loaded_parquet.index.name == "Date"
    assert isinstance(loaded_parquet.index, pd.DatetimeIndex)
    assert len(loaded_parquet) == len(dates)
    assert {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "sma_3",
        "sma_6",
        "bollinger_upper_5",
        "bollinger_lower_5",
        "rsi_14",
        "macd",
    }.issubset(loaded_parquet.columns)

    loaded_csv = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert len(loaded_csv) == len(dates)

    # Repeat execution in the same output directory must cleanly overwrite atomically.
    code2, _ = quant_cli.run_cli_command(
        [
            "analyze",
            "-t",
            "etf",
            "-s",
            "510300",
            "--start",
            "2024-01-02",
            "--end",
            "2024-02-20",
            "--refresh",
            "--quiet",
            "-o",
            str(out_dir),
        ],
        loader=_loader,
    )
    assert code2 == quant_cli.EXIT_OK
    assert loader_calls[1]["force_refresh"] is True
    captured2 = capsys.readouterr()
    assert captured2.err == ""
    refreshed_payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert "data_csv" not in refreshed_payload["artifacts"]
    json.dumps(refreshed_payload, allow_nan=False)


def test_quant_fetch_stdout_and_disk_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dates = pd.bdate_range("2024-03-01", periods=10)
    closes = np.linspace(100.0, 105.0, len(dates))
    raw_df = _make_ohlcv(
        dates, closes, asset_type="us_stock", symbol="AAPL", currency="USD", adjust=""
    )

    # Without -o, `quant fetch` embeds row-level OHLCV under `data` on stdout.
    code, _ = quant_cli.run_cli_command(
        [
            "fetch",
            "-t",
            "us_stock",
            "-s",
            "AAPL",
            "--start",
            "2024-03-01",
            "--end",
            "2024-03-15",
            "--compact",
            "--quiet",
        ],
        loader=lambda *a, **kw: raw_df.copy(),
    )
    assert code == quant_cli.EXIT_OK
    out_inline = capsys.readouterr()
    assert out_inline.err == ""
    assert "\n" not in out_inline.out.strip()
    payload_inline = json.loads(out_inline.out)
    assert payload_inline["command"] == "fetch"
    assert payload_inline["metrics"] == {}
    assert payload_inline["metric_units"] == {}
    assert payload_inline["artifacts"] == {}
    assert len(payload_inline["data"]) == len(dates)

    # With -o, `quant fetch` writes data.parquet + result.json without duplicating rows into result.json.
    fetch_dir = tmp_path / "fetch_out"
    code_disk, _ = quant_cli.run_cli_command(
        [
            "fetch",
            "us_stock:AAPL",
            "--start",
            "2024-03-01",
            "--end",
            "2024-03-15",
            "-o",
            str(fetch_dir),
            "--quiet",
        ],
        loader=lambda *a, **kw: raw_df.copy(),
    )
    assert code_disk == quant_cli.EXIT_OK
    capsys.readouterr()
    disk_payload = json.loads((fetch_dir / "result.json").read_text(encoding="utf-8"))
    assert "data" not in disk_payload
    assert Path(disk_payload["artifacts"]["data_parquet"]).is_file()
    parquet_df = pd.read_parquet(fetch_dir / "data.parquet")
    assert list(parquet_df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_quant_subprocess_end_to_end_via_installed_entrypoint(tmp_path: Path) -> None:
    cache_dir = tmp_path / "quant_cache"
    out_dir = tmp_path / "cli_out"
    dates = pd.bdate_range("2024-05-01", periods=30)
    sample = _make_ohlcv(
        dates,
        np.linspace(10.0, 12.5, len(dates)),
        asset_type="ashare",
        symbol="600519",
        adjust="qfq",
        currency="CNY",
    )

    env = dict(os.environ)
    env["QUANT_DISABLE_CACHE"] = "0"
    env["QUANT_CACHE_DIR"] = str(cache_dir)
    cache_dir_prev = os.environ.get(cache_mod.ENV_CACHE_DIR)
    disable_prev = os.environ.get(cache_mod.ENV_DISABLE_CACHE)
    try:
        os.environ[cache_mod.ENV_CACHE_DIR] = str(cache_dir)
        os.environ[cache_mod.ENV_DISABLE_CACHE] = "0"
        saved_path = cache_mod.save_cached_ohlcv(
            sample,
            "ashare",
            "600519",
            "2024-05-01",
            "2024-06-15",
            adjust="qfq",
        )
        assert saved_path is not None
    finally:
        if cache_dir_prev is None:
            os.environ.pop(cache_mod.ENV_CACHE_DIR, None)
        else:
            os.environ[cache_mod.ENV_CACHE_DIR] = cache_dir_prev
        if disable_prev is None:
            os.environ.pop(cache_mod.ENV_DISABLE_CACHE, None)
        else:
            os.environ[cache_mod.ENV_DISABLE_CACHE] = disable_prev

    quant_bin = Path(sys.executable).with_name("quant")
    proc = subprocess.run(
        [
            str(quant_bin),
            "analyze",
            "ashare:600519",
            "--start",
            "2024-05-01",
            "--end",
            "2024-06-15",
            "--adjust",
            "qfq",
            "-o",
            str(out_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[quant] done status=ok" in proc.stderr
    parsed = json.loads(proc.stdout)
    assert parsed["asset"]["cache_hit"] is True
    assert parsed["asset"]["canonical_symbol"] == "600519"
    assert parsed["software_version"] == SOFTWARE_VERSION
    assert SOFTWARE_VERSION == "0.4.0"
    assert (out_dir / "result.json").is_file()
    assert (out_dir / "data.parquet").is_file()

    # Verify in a clean subprocess that quant_cli never pulls matplotlib or mplfinance into sys.modules.
    import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, quant_cli; "
                "assert 'matplotlib' not in sys.modules and 'mplfinance' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert import_check.returncode == 0, import_check.stderr


def test_quant_cli_yield_level_degraded_and_currency_conversion(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    yield_dates = pd.bdate_range("2024-01-02", periods=25)
    yield_df = _make_ohlcv(
        yield_dates,
        np.linspace(4.0, 4.5, len(yield_dates)),
        asset_type="bond_us_yield",
        symbol="10Y",
        adjust="",
        currency="USD",
        extra_attrs={"return_basis": "yield_level", "series_kind": "yield_level"},
    )
    code, res = quant_cli.run_cli_command(
        [
            "analyze",
            "-t",
            "bond_us_yield",
            "-s",
            "10Y",
            "--start",
            "2024-01-02",
            "--end",
            "2024-02-05",
            "--quiet",
        ],
        loader=lambda *a, **kw: yield_df.copy(),
    )
    assert code == quant_cli.EXIT_OK
    assert res is not None
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "degraded"
    assert payload["metrics"]["cumulative_return"] is None
    assert payload["metric_reasons"]["sharpe_ratio"] == "yield_level_not_return"
    assert payload["metric_units"]["latest_level"] == "percentage_points_yield"

    crypto_dates = pd.bdate_range("2024-01-02", periods=30)
    crypto_df = _make_ohlcv(
        crypto_dates,
        np.linspace(40000.0, 45000.0, len(crypto_dates)),
        asset_type="crypto",
        symbol="BTC/USDT",
        adjust="",
        currency="USDT",
    )
    fx_df = _make_ohlcv(
        crypto_dates,
        np.full(len(crypto_dates), 7.2),
        asset_type="fx",
        symbol="USD/CNY",
        adjust="",
        currency="CNY",
    )
    code_fx, res_fx = quant_cli.run_cli_command(
        [
            "analyze",
            "-t",
            "crypto",
            "-s",
            "BTC/USDT",
            "--start",
            "2024-01-02",
            "--end",
            "2024-02-12",
            "--base-currency",
            "CNY",
            "--allow-usdt-peg",
            "-o",
            str(tmp_path / "fx_out"),
        ],
        loader=lambda *a, **kw: crypto_df.copy(),
        fx_loader=lambda *a, **kw: fx_df.copy(),
    )
    assert code_fx == quant_cli.EXIT_OK
    assert res_fx is not None
    assert res_fx.asset.output_currency == "CNY"
    capsys.readouterr()


@pytest.mark.parametrize(
    ("exc_factory", "expected_code", "expected_cat"),
    [
        (lambda: ValueError("bad flag"), quant_cli.EXIT_USAGE_OR_CALLER, "caller"),
        (lambda: ImportError("missing dep"), quant_cli.EXIT_USAGE_OR_CALLER, "caller"),
        (lambda: EmptySymbolDataError("empty"), quant_cli.EXIT_EMPTY_DATA, "empty"),
        (lambda: NetworkFetchError("tls timeout"), quant_cli.EXIT_NETWORK, "transport"),
        (
            lambda: ChinaEquityPipelineError("both failed", network_related=True),
            quant_cli.EXIT_NETWORK,
            "transport",
        ),
        (lambda: httpx.ConnectTimeout("timed out"), quant_cli.EXIT_NETWORK, "transport"),
        (lambda: VendorUpstreamError("http 503"), quant_cli.EXIT_UPSTREAM, "upstream"),
        (
            lambda: ChinaEquityPipelineError("bad schema", network_related=False),
            quant_cli.EXIT_UPSTREAM,
            "upstream",
        ),
        (lambda: RuntimeError("internal bug"), quant_cli.EXIT_UNEXPECTED, "unexpected"),
    ],
)
def test_quant_cli_error_classification_and_exit_codes(
    exc_factory: Any,
    expected_code: int,
    expected_cat: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise exc_factory()

    code, res = quant_cli.run_cli_command(
        ["analyze", "-t", "etf", "-s", "510300", "--start", "2024-01-01", "--end", "2024-02-01"],
        loader=_boom,
    )
    assert res is None
    assert code == expected_code
    captured = capsys.readouterr()
    assert captured.out == ""
    err_lines = [line for line in captured.err.strip().splitlines() if line.strip()]
    err_obj = json.loads(err_lines[-1])
    assert err_obj["status"] == "error"
    assert err_obj["error"]["category"] == expected_cat
    assert err_obj["error"]["exit_code"] == expected_code


def test_quant_cli_usage_errors_emit_structured_json_on_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Missing subcommand
    code_no_cmd, _ = quant_cli.run_cli_command([])
    assert code_no_cmd == quant_cli.EXIT_USAGE_OR_CALLER
    err = json.loads(capsys.readouterr().err)
    assert err["error"]["category"] == "caller"

    # Conflicting positional and --symbol
    code_conflict, _ = quant_cli.run_cli_command(["fetch", "510300", "-s", "510500"])
    assert code_conflict == quant_cli.EXIT_USAGE_OR_CALLER
    assert json.loads(capsys.readouterr().err)["error"]["category"] == "caller"

    # --csv without --output
    code_csv, _ = quant_cli.run_cli_command(["fetch", "510300", "--csv"])
    assert code_csv == quant_cli.EXIT_USAGE_OR_CALLER
    assert json.loads(capsys.readouterr().err)["error"]["category"] == "caller"

    # --output pointing to an existing regular file
    file_target = tmp_path / "not_a_dir"
    file_target.write_text("x", encoding="utf-8")
    code_file, _ = quant_cli.run_cli_command(["fetch", "510300", "-o", str(file_target)])
    assert code_file == quant_cli.EXIT_USAGE_OR_CALLER
    assert json.loads(capsys.readouterr().err)["error"]["category"] == "caller"


def test_classify_network_exception_unwraps_causes_and_china_equity_flag() -> None:
    try:
        try:
            raise httpx.ConnectError("reset")
        except httpx.ConnectError as cause:
            raise DataFetchError("wrapped fetch failure") from cause
    except DataFetchError as wrapped:
        assert classify_network_exception(wrapped) == "transport"

    assert (
        classify_network_exception(ChinaEquityPipelineError("net", network_related=True))
        == "transport"
    )
    assert (
        classify_network_exception(ChinaEquityPipelineError("proto", network_related=False))
        == "upstream"
    )


# ---------------------------------------------------------------------------
# 2. Sample range clipping, QualityReport bounds, & data_summary accuracy
# ---------------------------------------------------------------------------


def test_service_clips_wider_loader_data_and_updates_quality_bounds() -> None:
    all_dates = pd.bdate_range("2024-01-02", "2024-03-29")
    closes = np.linspace(10.0, 15.0, len(all_dates))
    wider_df = _make_ohlcv(all_dates, closes, asset_type="etf", symbol="510300")
    # Inject a zero-volume anomaly outside February (in January) and a large gap outside February
    wider_df.iloc[0, wider_df.columns.get_loc("Volume")] = 0.0
    wider_df.attrs["data_quality"] = compute_ohlcv_quality(wider_df, missing_policy="drop")
    wider_df.attrs["data_quality"]["quality_scope"] = "full"

    req = AnalysisRequest(
        asset_type="etf",
        symbol="510300",
        start_date="2024-02-01",
        end_date="2024-02-29",
    )
    fetched = fetch_single_asset(req, raw_df=wider_df)
    analyzed = analyze_single_asset(req, raw_df=wider_df)

    for res in (fetched, analyzed):
        assert res.data_frame is not None
        assert res.data_frame.index.min() >= pd.Timestamp("2024-02-01")
        assert res.data_frame.index.max() <= pd.Timestamp("2024-02-29")
        assert res.quality.start_date == "2024-02-01"
        assert res.quality.end_date == "2024-02-29"
        assert res.quality.total_rows == len(res.data_frame)
        assert res.quality.zero_volume_rows == 0
        assert res.quality.quality_scope == "slice"
        assert res.data_summary["requested_start"] == "2024-02-01"
        assert res.data_summary["requested_end"] == "2024-02-29"
        assert res.data_summary["actual_start"] == "2024-02-01"
        assert res.data_summary["actual_end"] == "2024-02-29"
        assert res.data_summary["row_count"] == len(res.data_frame)


def test_build_quality_report_overrides_stale_attr_dates_and_non_overlapping_window() -> None:
    dates = pd.bdate_range("2024-02-01", "2024-02-20")
    df = _make_ohlcv(dates, np.linspace(100.0, 105.0, len(dates)))
    # Simulate stale source metadata bounds that disagree with active df index
    df.attrs["data_quality"] = {
        **compute_ohlcv_quality(df, missing_policy="ffill"),
        "start_date": "2020-01-01",
        "end_date": "2025-12-31",
        "quality_scope": "full",
    }
    report = build_quality_report(df)
    assert report.start_date == "2024-02-01"
    assert report.end_date == "2024-02-20"
    assert report.quality_scope == "slice"

    non_overlap_req = AnalysisRequest(
        asset_type="etf",
        symbol="510300",
        start_date="2024-06-01",
        end_date="2024-06-30",
    )
    with pytest.raises(EmptySymbolDataError):
        analyze_single_asset(non_overlap_req, raw_df=df)


def test_fx_inner_join_truncation_updates_quality_report_row_bounds() -> None:
    asset_dates = pd.bdate_range("2024-01-02", periods=30)
    # FX starts 3 business days after asset_dates[0], dropping the 3 leading unconvertible bars
    # while forward-filling the trailing FX observation across asset_dates[-2:].
    fx_dates = asset_dates[3:-2]
    expected_dates = asset_dates[3:]
    asset_df = _make_ohlcv(
        asset_dates,
        np.linspace(100.0, 120.0, len(asset_dates)),
        asset_type="us_stock",
        symbol="AAPL",
        currency="USD",
        adjust="",
    )
    fx_df = _make_ohlcv(
        fx_dates,
        np.full(len(fx_dates), 7.1),
        asset_type="fx",
        symbol="USD/CNY",
        currency="CNY",
        adjust="",
    )
    req = AnalysisRequest(
        asset_type="us_stock",
        symbol="AAPL",
        start_date="2024-01-02",
        end_date="2024-02-12",
        base_currency="CNY",
    )
    res = analyze_single_asset(req, raw_df=asset_df, fx_loader=lambda *a, **kw: fx_df)
    assert res.quality.total_rows == len(expected_dates)
    assert res.quality.start_date == expected_dates.min().strftime("%Y-%m-%d")
    assert res.quality.end_date == expected_dates.max().strftime("%Y-%m-%d")
    assert res.data_summary["row_count"] == len(expected_dates)


# ---------------------------------------------------------------------------
# 3. Cache concurrency, torn-pair protection, and Windows msvcrt lock path
# ---------------------------------------------------------------------------


def test_cache_concurrent_threads_preserve_locks_and_atomicity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cache_mod.ENV_DISABLE_CACHE, "0")
    monkeypatch.setenv(cache_mod.ENV_CACHE_DIR, str(tmp_path))

    dates = pd.bdate_range("2024-01-02", periods=20)
    df = _make_ohlcv(dates, np.linspace(10.0, 12.0, len(dates)), asset_type="etf", symbol="510300")
    key = cache_mod.build_cache_key("etf", "510300", adjust="hfq")
    lock_file = tmp_path / ".locks" / f"{key}.lock"

    def _worker(idx: int) -> None:
        if idx % 5 == 0:
            cache_mod.clear_ohlcv_cache()
        elif idx % 2 == 0:
            cache_mod.save_cached_ohlcv(
                df, "etf", "510300", "2024-01-02", "2024-01-29", adjust="hfq"
            )
        else:
            hit = cache_mod.load_cached_ohlcv(
                "etf", "510300", "2024-01-02", "2024-01-29", adjust="hfq"
            )
            if hit is not None:
                assert not hit.empty
        stats = cache_mod.get_ohlcv_cache_stats()
        assert stats["entries"] >= 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_worker, range(40)))

    # Saving afterwards ensures .locks/<key>.lock exists, and clear keeps the lock file intact.
    saved = cache_mod.save_cached_ohlcv(
        df, "etf", "510300", "2024-01-02", "2024-01-29", adjust="hfq"
    )
    assert saved is not None and lock_file.exists()
    cache_mod.clear_ohlcv_cache()
    assert lock_file.exists()


def test_cache_save_rolls_back_parquet_if_meta_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cache_mod.ENV_DISABLE_CACHE, "0")
    monkeypatch.setenv(cache_mod.ENV_CACHE_DIR, str(tmp_path))

    dates = pd.bdate_range("2024-01-02", periods=10)
    df = _make_ohlcv(dates, np.linspace(10.0, 11.0, len(dates)), asset_type="etf", symbol="510300")
    real_replace = os.replace

    def _fail_meta_replace(src: Any, dst: Any) -> None:
        if str(dst).endswith(".meta.json"):
            raise OSError("simulated disk failure on meta replace")
        real_replace(src, dst)

    monkeypatch.setattr(cache_mod.os, "replace", _fail_meta_replace)
    res = cache_mod.save_cached_ohlcv(df, "etf", "510300", "2024-01-02", "2024-01-15", adjust="hfq")
    assert res is None
    assert list(tmp_path.glob("*.parquet")) == []
    assert list(tmp_path.glob("*.meta.json")) == []


def test_cache_windows_msvcrt_lock_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[int] = []
    calls = {"count": 0}

    def _fake_locking(fd: int, mode: int, nbytes: int) -> None:
        events.append(mode)
        if mode == 2 and calls["count"] == 0:
            calls["count"] += 1
            raise OSError(errno.EACCES, "temporarily locked")

    fake_msvcrt = SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, locking=_fake_locking)
    monkeypatch.setattr(cache_mod, "fcntl", None)
    monkeypatch.setattr(cache_mod, "msvcrt", fake_msvcrt)

    with cache_mod._cache_entry_lock(tmp_path, "win_key"):
        pass

    assert events == [2, 2, 0]


# ---------------------------------------------------------------------------
# 4. Offline network guard & backward compatibility checks
# ---------------------------------------------------------------------------


def test_offline_guard_blocks_requests_urllib3_dns_and_raw_socket_while_allowing_loopback() -> None:
    with pytest.raises(OSError, match="Offline test guard"):
        socket.getaddrinfo("example.com", 443)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(OSError, match="Offline test guard"):
            sock.connect(("93.184.216.34", 80))

    with pytest.raises(urllib3.exceptions.HTTPError, match="Offline test guard"):
        urllib3.PoolManager().request("GET", "http://example.com", retries=False, timeout=1.0)

    with pytest.raises(requests.exceptions.RequestException, match="Offline test guard"):
        requests.get("http://example.com", timeout=1.0)

    with pytest.raises(httpx.ConnectError, match="Offline test guard"):
        httpx.get("http://example.com", timeout=1.0)

    with pytest.raises(urllib.error.URLError, match="Offline test guard"):
        urllib.request.urlopen("http://example.com", timeout=1.0)

    # Loopback DNS & socket calls bypass the offline guard and reach the OS stack cleanly.
    addrs = socket.getaddrinfo("127.0.0.1", 0)
    assert addrs
    with pytest.raises((ConnectionRefusedError, PermissionError)) as excinfo:
        socket.create_connection(("127.0.0.1", 1), timeout=0.5)
    assert "Offline test guard" not in str(excinfo.value)


def test_legacy_cli_and_all_supported_asset_types_remain_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import main as legacy_main

    dates = pd.bdate_range("2024-01-02", periods=30)
    sample = _make_ohlcv(
        dates, np.linspace(100.0, 110.0, len(dates)), asset_type="etf", symbol="510300"
    )
    monkeypatch.setattr(legacy_main, "load_asset_data", lambda *a, **kw: sample.copy())

    csv_out = tmp_path / "legacy.csv"
    res = legacy_main.run_analysis(
        asset_type="etf",
        symbol="510300",
        start_date="2024-01-02",
        end_date="2024-02-12",
        export_csv=str(csv_out),
        show=False,
    )
    assert csv_out.is_file()
    assert res["analysis_result"].asset.symbol == "510300"

    parser = quant_cli._build_parser()
    # Verify quant subcommands accept every supported asset type cleanly.
    for asset_type in SUPPORTED_ASSET_TYPES:
        parsed = parser.parse_args(["fetch", "-t", asset_type])
        assert parsed.asset_type == asset_type
