"""M4 tests: independent report/chart renderers, CLI integration, parity & atomic commits."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import quant_cli
from contracts import AnalysisResult, MultiAssetResult
from data_sources import cache as cache_mod
from ohlcv import compute_ohlcv_quality
from reporting import (
    build_plotly_figures,
    format_drawdown_percent,
    format_percent,
    format_ratio,
    format_yield_percent,
    load_result_from_artifacts,
    parse_chart_formats,
    parse_chart_types,
    parse_report_formats,
    render_charts,
    render_report,
)


def _make_ohlcv(
    dates: pd.DatetimeIndex,
    closes: np.ndarray | list[float],
    *,
    asset_type: str = "etf",
    symbol: str = "510300",
    adjust: str = "hfq",
    currency: str | None = "CNY",
    data_source: str = "mock:source",
    volume: float = 15000.0,
    extra_attrs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    df = pd.DataFrame(
        {
            "Open": c * 0.996,
            "High": c * 1.009,
            "Low": c * 0.991,
            "Close": c,
            "Volume": np.full(len(c), volume, dtype=float),
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
            "fetched_at": "2026-06-01T00:00:00+00:00",
            "cache_hit": False,
        }
    )
    df.attrs["data_quality"] = compute_ohlcv_quality(df, missing_policy="drop")
    if extra_attrs:
        df.attrs.update(extra_attrs)
    return df


def _make_multi_asset_loader() -> tuple[Any, Any]:
    dates = pd.bdate_range("2024-01-02", periods=95)
    rng = np.random.default_rng(7)
    shocks_a = rng.normal(0.0008, 0.011, size=len(dates))
    shocks_b = rng.normal(0.0004, 0.015, size=len(dates))
    shocks_c = -0.35 * shocks_a + rng.normal(0.0003, 0.008, size=len(dates))
    shocks_bench = 0.6 * shocks_a + 0.4 * shocks_b + rng.normal(0.0, 0.004, size=len(dates))

    px_a = 100.0 * np.exp(np.cumsum(shocks_a))
    px_b = 50.0 * np.exp(np.cumsum(shocks_b))
    px_c = 120.0 * np.exp(np.cumsum(shocks_c))
    px_bench = 300.0 * np.exp(np.cumsum(shocks_bench))
    fx_series = np.linspace(7.10, 7.25, len(dates))

    frames = {
        ("etf", "510300"): _make_ohlcv(
            dates, px_a, asset_type="etf", symbol="510300", currency="CNY"
        ),
        ("us_stock", "SPY"): _make_ohlcv(
            dates, px_b, asset_type="us_stock", symbol="SPY", adjust="qfq", currency="USD"
        ),
        ("bond_cn", "TLT"): _make_ohlcv(
            dates, px_c, asset_type="bond_cn", symbol="TLT", adjust="none", currency="USD"
        ),
        ("global_index", "000300"): _make_ohlcv(
            dates,
            px_bench,
            asset_type="global_index",
            symbol="000300",
            adjust="none",
            currency="CNY",
        ),
    }

    def loader(asset_type: str, symbol: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return frames[(asset_type, symbol)].copy()

    def fx_loader(pair: str, start: Any, end: Any) -> pd.DataFrame:
        rates = fx_series if pair == "USDCNY" else 1.0 / fx_series
        return _make_ohlcv(
            dates,
            rates,
            asset_type="fx",
            symbol=pair,
            adjust="none",
            currency=pair[3:],
            volume=0.0,
        )

    return loader, fx_loader


def _assert_valid_chart_files(artifacts: dict[str, str], keys: list[str]) -> None:
    for key in keys:
        assert key in artifacts, f"Missing artifact key {key}"
        path = Path(artifacts[key])
        assert path.is_file(), f"Artifact file missing on disk: {path}"
        raw = path.read_bytes()
        assert len(raw) > 100
        if key.endswith("_png"):
            assert raw.startswith(b"\x89PNG\r\n\x1a\n")
        elif key.endswith("_svg"):
            text = raw.decode("utf-8")
            assert "<svg" in text
        elif key.endswith("_html"):
            text = raw.decode("utf-8")
            assert "<!DOCTYPE html>" in text
            assert "plotly" in text.lower()


# ---------------------------------------------------------------------------
# 1. Option parsing & caller validation
# ---------------------------------------------------------------------------


def test_parse_report_and_chart_options_validation() -> None:
    assert parse_report_formats(None) == ()
    assert parse_report_formats("none") == ()
    assert parse_report_formats("markdown,html,md") == ("md", "html")
    assert parse_report_formats("all") == ("md", "html")

    assert parse_chart_formats(None) == ()
    assert parse_chart_formats("none") == ()
    assert parse_chart_formats("png,html,png") == ("png", "html")
    assert parse_chart_formats("all") == ("html", "png", "svg")

    assert parse_chart_types("analyze", None) == ("price", "normalized", "drawdown", "rolling")
    assert parse_chart_types("analyze", "drawdown,price,drawdown") == ("drawdown", "price")
    assert parse_chart_types("compare", "auto") == (
        "normalized",
        "drawdown",
        "rolling",
        "correlation",
    )

    with pytest.raises(ValueError):
        parse_report_formats("pdf")
    with pytest.raises(ValueError):
        parse_report_formats("none,md")
    with pytest.raises(ValueError):
        parse_chart_formats("jpg")
    with pytest.raises(ValueError):
        parse_chart_formats("none,png")
    with pytest.raises(ValueError):
        parse_chart_types("analyze", "frontier")
    with pytest.raises(ValueError):
        parse_chart_types("compare", "weight_history")


def test_cli_requires_output_for_reports_or_charts_and_rejects_bad_types(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, res = quant_cli.run_cli_command(["analyze", "etf:510300", "--report", "md"])
    capsys.readouterr()
    assert code == quant_cli.EXIT_USAGE_OR_CALLER
    assert res is None

    code, res = quant_cli.run_cli_command(
        ["compare", "etf:510300", "us_stock:SPY", "--charts", "png"]
    )
    capsys.readouterr()
    assert code == quant_cli.EXIT_USAGE_OR_CALLER
    assert res is None

    code, res = quant_cli.run_cli_command(
        [
            "portfolio",
            "etf:510300",
            "us_stock:SPY",
            "--chart-types",
            "bogus_chart",
            "-o",
            "/tmp/unused",
        ]
    )
    capsys.readouterr()
    assert code == quant_cli.EXIT_USAGE_OR_CALLER
    assert res is None


# ---------------------------------------------------------------------------
# 2. Single-asset E2E: equity, yield_level edge case, fx_rate, short slice
# ---------------------------------------------------------------------------


def test_analyze_report_and_all_chart_formats_numerical_parity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=65)
    closes = np.concatenate(
        [
            np.linspace(100.0, 128.0, 35),
            np.linspace(125.0, 108.0, 15),
            np.linspace(110.0, 134.0, 15),
        ]
    )
    df = _make_ohlcv(
        dates, closes, asset_type="ashare", symbol="600519", adjust="hfq", currency="CNY"
    )

    out_dir = tmp_path / "analyze_full"
    code, res = quant_cli.run_cli_command(
        [
            "analyze",
            "ashare:600519",
            "--start",
            "2024-01-02",
            "--end",
            "2024-04-30",
            "--include-momentum",
            "-o",
            str(out_dir),
            "--report",
            "all",
            "--charts",
            "all",
        ],
        loader=lambda *a, **kw: df.copy(),
    )
    captured = capsys.readouterr()
    assert code == quant_cli.EXIT_OK, captured.err
    assert isinstance(res, AnalysisResult)

    payload = json.loads(captured.out)
    artifacts = payload["artifacts"]
    assert "report_md" in artifacts and Path(artifacts["report_md"]).is_file()
    assert "report_html" in artifacts and Path(artifacts["report_html"]).is_file()
    assert "result.json" in Path(artifacts["result_json"]).name

    expected_chart_keys = [
        f"chart_{ct}_{ext}"
        for ct in ("price", "normalized", "drawdown", "rolling")
        for ext in ("html", "png", "svg")
    ]
    _assert_valid_chart_files(artifacts, expected_chart_keys)

    md_text = Path(artifacts["report_md"]).read_text(encoding="utf-8")
    html_text = Path(artifacts["report_html"]).read_text(encoding="utf-8")

    # Numerical parity with result.json (including underwater non-positive drawdown display)
    m = payload["metrics"]
    expected_strs = [
        format_percent(m["cumulative_return"], signed=True),
        format_percent(m["cagr"], signed=True),
        format_percent(m["annualized_volatility"]),
        format_drawdown_percent(m["max_drawdown"]),
        format_ratio(m["sharpe_ratio"]),
    ]
    for s in expected_strs:
        assert s in md_text, f"Expected {s!r} in report.md"
        assert s in html_text, f"Expected {s!r} in report.html"

    # Non-negative JSON max_drawdown renders cleanly as non-positive in report
    assert float(m["max_drawdown"]) > 0.0
    assert format_drawdown_percent(m["max_drawdown"]).startswith("-")

    # Chart links embedded relatively
    assert "![价格与技术指标走势](charts/price.png)" in md_text
    assert "[交互式图表 (HTML)](charts/price.html)" in md_text
    assert 'src="charts/price.png"' in html_text
    assert 'href="charts/price.html"' in html_text
    assert str(m["latest_close"] and f"{float(m['latest_close']):,.2f}") in md_text


def test_analyze_yield_level_blocks_return_charts_and_avoids_percent_scaling(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=45)
    yields = np.linspace(4.15, 4.48, len(dates))
    df = _make_ohlcv(
        dates,
        yields,
        asset_type="bond_us_yield",
        symbol="US10Y",
        adjust="none",
        currency="USD",
        volume=0.0,
    )

    out_dir = tmp_path / "analyze_yield"
    code, res = quant_cli.run_cli_command(
        [
            "analyze",
            "bond_us_yield:US10Y",
            "--start",
            "2024-01-02",
            "--end",
            "2024-03-31",
            "-o",
            str(out_dir),
            "--report",
            "all",
            "--charts",
            "html,png",
        ],
        loader=lambda *a, **kw: df.copy(),
    )
    captured = capsys.readouterr()
    assert code == quant_cli.EXIT_OK, captured.err
    assert isinstance(res, AnalysisResult)
    payload = json.loads(captured.out)
    artifacts = payload["artifacts"]

    _assert_valid_chart_files(artifacts, ["chart_price_html", "chart_price_png"])
    for blocked_ct in ("normalized", "drawdown", "rolling"):
        assert f"chart_{blocked_ct}_html" not in artifacts
        assert f"chart_{blocked_ct}_png" not in artifacts
    assert any("yield_level_not_return" in w for w in payload["warnings"])

    md_text = Path(artifacts["report_md"]).read_text(encoding="utf-8")
    html_text = Path(artifacts["report_html"]).read_text(encoding="utf-8")
    latest_lvl_str = format_yield_percent(payload["metrics"]["latest_level"])
    assert latest_lvl_str in md_text
    assert latest_lvl_str in html_text
    assert "448.00%" not in md_text
    assert "yield_level_not_return" in md_text
    assert "N/A" in md_text


def test_analyze_short_window_and_degraded_quality_renders_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dates = pd.bdate_range("2024-02-01", periods=10)
    df = _make_ohlcv(
        dates,
        np.linspace(7.15, 7.22, len(dates)),
        asset_type="fx",
        symbol="USDCNY",
        adjust="none",
        currency="CNY",
        volume=0.0,
    )
    # Trigger ohlc anomaly to mark degraded quality cleanly
    df.iloc[2, df.columns.get_loc("High")] = df.iloc[2]["Low"] * 0.99
    df.attrs["data_quality"] = compute_ohlcv_quality(df, missing_policy="drop")

    out_dir = tmp_path / "short_fx"
    code, res = quant_cli.run_cli_command(
        [
            "analyze",
            "fx:USDCNY",
            "--start",
            "2024-02-01",
            "--end",
            "2024-02-20",
            "-o",
            str(out_dir),
            "--report",
            "html",
            "--charts",
            "png,svg",
        ],
        loader=lambda *a, **kw: df.copy(),
    )
    captured = capsys.readouterr()
    assert code == quant_cli.EXIT_OK, captured.err
    assert res is not None and res.status == "degraded"
    artifacts = json.loads(captured.out)["artifacts"]
    _assert_valid_chart_files(
        artifacts, ["chart_price_png", "chart_rolling_png", "chart_rolling_svg"]
    )
    html_text = Path(artifacts["report_html"]).read_text(encoding="utf-8")
    assert "Degraded Result" in html_text
    assert "fx_rate" in html_text


# ---------------------------------------------------------------------------
# 3. Compare E2E: cross-market, benchmark, Memmel/JB tests, mixed LOCAL warning
# ---------------------------------------------------------------------------


def test_compare_report_and_charts_benchmark_and_mixed_currency_warnings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader, fx_loader = _make_multi_asset_loader()
    out_cny = tmp_path / "compare_cny"
    code, res = quant_cli.run_cli_command(
        [
            "compare",
            "etf:510300",
            "us_stock:SPY",
            "bond_cn:TLT",
            "--benchmark",
            "global_index:000300",
            "--base-currency",
            "CNY",
            "--calendar-mode",
            "mtm_outer",
            "--rolling-window",
            "15",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-31",
            "-o",
            str(out_cny),
            "--report",
            "all",
            "--charts",
            "html,png",
        ],
        loader=loader,
        fx_loader=fx_loader,
    )
    captured = capsys.readouterr()
    assert code == quant_cli.EXIT_OK, captured.err
    assert isinstance(res, MultiAssetResult)
    payload = json.loads(captured.out)
    artifacts = payload["artifacts"]

    expected_charts = [
        f"chart_{ct}_{ext}"
        for ct in ("normalized", "drawdown", "rolling", "correlation")
        for ext in ("html", "png")
    ]
    _assert_valid_chart_files(artifacts, expected_charts)

    md_text = Path(artifacts["report_md"]).read_text(encoding="utf-8")
    html_text = Path(artifacts["report_html"]).read_text(encoding="utf-8")

    # Verify compare drawdown (stored <=0 in JSON) displays cleanly as non-positive underwater %
    spy_dd = payload["metrics"]["per_asset"]["us_stock:SPY"]["max_drawdown"]
    assert float(spy_dd) <= 0.0
    assert format_drawdown_percent(spy_dd) in md_text
    assert "Jarque-Bera" in md_text
    assert "Memmel (2003)" in md_text
    assert "mtm_outer" in html_text

    # Verify LOCAL mixed-currency run flags danger callout in report
    out_local = tmp_path / "compare_local"
    code2, _ = quant_cli.run_cli_command(
        [
            "compare",
            "etf:510300",
            "us_stock:SPY",
            "--base-currency",
            "LOCAL",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-31",
            "-o",
            str(out_local),
            "--report",
            "md",
        ],
        loader=loader,
        fx_loader=fx_loader,
    )
    capsys.readouterr()
    assert code2 == quant_cli.EXIT_OK
    local_md = (out_local / "report.md").read_text(encoding="utf-8")
    assert "Mixed Local Currencies" in local_md


# ---------------------------------------------------------------------------
# 4. Portfolio E2E: train_test, walk_forward, in_sample lookahead, frontier=0
# ---------------------------------------------------------------------------


def test_portfolio_train_test_full_reports_and_charts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader, fx_loader = _make_multi_asset_loader()
    out_dir = tmp_path / "port_tt"
    code, res = quant_cli.run_cli_command(
        [
            "portfolio",
            "etf:510300",
            "us_stock:SPY",
            "bond_cn:TLT",
            "--benchmark",
            "global_index:000300",
            "--strategy",
            "risk_parity",
            "--validation-mode",
            "train_test",
            "--rebalance-freq",
            "monthly",
            "--tx-bps",
            "10",
            "--base-currency",
            "CNY",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-31",
            "-o",
            str(out_dir),
            "--report",
            "all",
            "--charts",
            "all",
        ],
        loader=loader,
        fx_loader=fx_loader,
    )
    captured = capsys.readouterr()
    assert code == quant_cli.EXIT_OK, captured.err
    assert isinstance(res, MultiAssetResult)
    payload = json.loads(captured.out)
    artifacts = payload["artifacts"]

    expected = [
        f"chart_{ct}_{ext}"
        for ct in ("backtest", "weights", "frontier", "correlation", "normalized", "drawdown")
        for ext in ("html", "png", "svg")
    ]
    _assert_valid_chart_files(artifacts, expected)
    # In train_test mode, weight_history is not present and should not clutter warnings unless asked
    assert "chart_weight_history_html" not in artifacts

    md_text = Path(artifacts["report_md"]).read_text(encoding="utf-8")
    html_text = Path(artifacts["report_html"]).read_text(encoding="utf-8")
    oos_cagr = payload["metrics"]["backtest"]["OOS_Portfolio"]["cagr"]
    oos_mdd = payload["metrics"]["backtest"]["OOS_Portfolio"]["max_drawdown"]
    assert format_percent(oos_cagr, signed=True) in md_text
    assert format_drawdown_percent(oos_mdd) in md_text
    assert "Euler 风险贡献" in md_text
    assert "SLSQP 凸优化精确前沿与随机模拟对照" in md_text
    assert "双边交易成本敏感性" in html_text
    assert format_percent(oos_cagr, signed=True) in html_text


def test_portfolio_walk_forward_and_in_sample_hindsight_callouts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader, fx_loader = _make_multi_asset_loader()
    out_wf = tmp_path / "port_wf"
    code, res_wf = quant_cli.run_cli_command(
        [
            "portfolio",
            "etf:510300",
            "us_stock:SPY",
            "bond_cn:TLT",
            "--strategy",
            "min_variance",
            "--validation-mode",
            "walk_forward_rolling",
            "--train-window",
            "45",
            "--rebalance-freq",
            "monthly",
            "--frontier-points",
            "0",
            "--no-sensitivity",
            "--base-currency",
            "CNY",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-31",
            "-o",
            str(out_wf),
            "--report",
            "md",
            "--charts",
            "html,png",
            "--chart-types",
            "backtest,weight_history,frontier",
        ],
        loader=loader,
        fx_loader=fx_loader,
    )
    captured = capsys.readouterr()
    assert code == quant_cli.EXIT_OK, captured.err
    assert isinstance(res_wf, MultiAssetResult)
    artifacts = json.loads(captured.out)["artifacts"]
    _assert_valid_chart_files(
        artifacts,
        [
            "chart_backtest_html",
            "chart_backtest_png",
            "chart_weight_history_html",
            "chart_weight_history_png",
        ],
    )
    assert "chart_frontier_png" not in artifacts
    wf_md = (out_wf / "report.md").read_text(encoding="utf-8")
    assert "In-Sample Reference [后视全样本参考-不可实盘]" in wf_md
    assert "`frontier`" in wf_md

    # In-sample mode with HTML report & HTML-only charts embeds <iframe>
    out_is = tmp_path / "port_is"
    code_is, _ = quant_cli.run_cli_command(
        [
            "portfolio",
            "etf:510300",
            "us_stock:SPY",
            "--strategy",
            "max_sharpe",
            "--validation-mode",
            "in_sample",
            "--no-sensitivity",
            "--frontier-points",
            "10",
            "--num-random-portfolios",
            "20",
            "--base-currency",
            "CNY",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-31",
            "-o",
            str(out_is),
            "--report",
            "html",
            "--charts",
            "html",
            "--chart-types",
            "backtest",
        ],
        loader=loader,
        fx_loader=fx_loader,
    )
    capsys.readouterr()
    assert code_is == quant_cli.EXIT_OK
    is_html = (out_is / "report.html").read_text(encoding="utf-8")
    assert "Lookahead Bias Warning" in is_html
    assert '<iframe src="charts/backtest.html"' in is_html


# ---------------------------------------------------------------------------
# 5. Compute/Render separation, M5 Plotly API, Stale Cleanup & Rollback
# ---------------------------------------------------------------------------


def test_load_result_from_artifacts_renders_without_recomputing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader, fx_loader = _make_multi_asset_loader()
    compute_dir = tmp_path / "compute_only"
    code, _ = quant_cli.run_cli_command(
        [
            "portfolio",
            "etf:510300",
            "us_stock:SPY",
            "bond_cn:TLT",
            "--strategy",
            "max_sharpe",
            "--validation-mode",
            "train_test",
            "--base-currency",
            "CNY",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-31",
            "-o",
            str(compute_dir),
        ],
        loader=loader,
        fx_loader=fx_loader,
    )
    captured = capsys.readouterr()
    assert code == quant_cli.EXIT_OK, captured.err
    disk_payload = json.loads((compute_dir / "result.json").read_text(encoding="utf-8"))
    # Backward compatibility: omitting --report / --charts leaves artifacts untouched
    assert not any(
        k.startswith("report_") or k.startswith("chart_") for k in disk_payload["artifacts"]
    )
    assert not (compute_dir / "report.md").exists()
    assert not (compute_dir / "charts").exists()

    # Load offline from artifacts only and render into a separate directory
    loaded = load_result_from_artifacts(compute_dir)
    assert isinstance(loaded, MultiAssetResult)

    figs, skipped = build_plotly_figures(loaded, chart_types=("backtest", "weights", "frontier"))
    assert set(figs.keys()) == {"backtest", "weights", "frontier"}
    assert skipped == {}

    render_dir = tmp_path / "rendered_offline"
    chart_out = render_charts(loaded, render_dir, formats=("html", "png"))
    rep_out = render_report(
        loaded,
        render_dir,
        formats=("md", "html"),
        charts=chart_out.charts,
        skipped_charts=chart_out.skipped,
    )
    assert len(chart_out.charts) == 12
    assert len(rep_out.reports) == 2
    assert (render_dir / "report.md").is_file()
    assert (render_dir / "charts" / "frontier.png").is_file()

    # Also verify single-asset analyze disk round-trip (result.json + data.parquet -> report & charts)
    analyze_dir = tmp_path / "analyze_compute_only"
    code_a, _ = quant_cli.run_cli_command(
        [
            "analyze",
            "etf:510300",
            "--include-momentum",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-31",
            "-o",
            str(analyze_dir),
        ],
        loader=loader,
    )
    capsys.readouterr()
    assert code_a == quant_cli.EXIT_OK
    loaded_single = load_result_from_artifacts(analyze_dir)
    assert isinstance(loaded_single, AnalysisResult)
    single_out = tmp_path / "analyze_rendered_offline"
    s_charts = render_charts(loaded_single, single_out, formats=("html", "png", "svg"))
    s_reps = render_report(
        loaded_single, single_out, formats=("md", "html"), charts=s_charts.charts
    )
    assert len(s_charts.charts) == 12
    assert len(s_reps.reports) == 2
    assert (single_out / "charts" / "rolling.svg").is_file()
    assert format_drawdown_percent(loaded_single.metrics["max_drawdown"]) in (
        single_out / "report.md"
    ).read_text(encoding="utf-8")


def test_stale_managed_reports_and_charts_cleanup_preserves_user_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader, fx_loader = _make_multi_asset_loader()
    out_dir = tmp_path / "reuse_out"

    code1, _ = quant_cli.run_cli_command(
        [
            "compare",
            "etf:510300",
            "us_stock:SPY",
            "--base-currency",
            "CNY",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-31",
            "-o",
            str(out_dir),
            "--report",
            "all",
            "--charts",
            "html,png",
        ],
        loader=loader,
        fx_loader=fx_loader,
    )
    capsys.readouterr()
    assert code1 == quant_cli.EXIT_OK
    assert (out_dir / "report.html").is_file()
    assert (out_dir / "charts" / "correlation.png").is_file()

    # User places non-managed custom files both in out_dir and out_dir/charts
    user_root = out_dir / "user_notes.txt"
    user_chart = out_dir / "charts" / "custom_user_plot.png"
    user_root.write_text("keep root note", encoding="utf-8")
    user_chart.write_bytes(b"custom-png-data")

    # Rerun with only Markdown report and NO charts -> removes report.html and managed charts only
    code2, _ = quant_cli.run_cli_command(
        [
            "compare",
            "etf:510300",
            "us_stock:SPY",
            "--base-currency",
            "CNY",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-31",
            "-o",
            str(out_dir),
            "--report",
            "md",
        ],
        loader=loader,
        fx_loader=fx_loader,
    )
    capsys.readouterr()
    assert code2 == quant_cli.EXIT_OK
    assert (out_dir / "report.md").is_file()
    assert not (out_dir / "report.html").exists()
    assert not (out_dir / "charts" / "correlation.png").exists()
    assert user_root.read_text(encoding="utf-8") == "keep root note"
    assert user_chart.read_bytes() == b"custom-png-data"


def test_atomic_rollback_restores_prior_reports_and_charts_on_commit_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=35)
    df1 = _make_ohlcv(dates, np.linspace(10.0, 12.0, len(dates)), asset_type="etf", symbol="510300")
    out_dir = tmp_path / "rollback_out"

    code1, _ = quant_cli.run_cli_command(
        [
            "analyze",
            "etf:510300",
            "--start",
            "2024-01-02",
            "--end",
            "2024-03-01",
            "-o",
            str(out_dir),
            "--report",
            "md",
            "--charts",
            "png",
            "--chart-types",
            "price",
        ],
        loader=lambda *a, **kw: df1.copy(),
    )
    capsys.readouterr()
    assert code1 == quant_cli.EXIT_OK
    orig_report = (out_dir / "report.md").read_text(encoding="utf-8")
    orig_chart = (out_dir / "charts" / "price.png").read_bytes()
    orig_result = (out_dir / "result.json").read_text(encoding="utf-8")

    # Second run changes prices and injects failure right after report.md/charts replace
    df2 = _make_ohlcv(dates, np.linspace(20.0, 26.0, len(dates)), asset_type="etf", symbol="510300")
    real_replace = os.replace

    def flaky_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(dst).name == "result.json":
            raise OSError("Simulated disk failure on result.json commit")
        real_replace(src, dst)

    monkeypatch.setattr(quant_cli.os, "replace", flaky_replace)
    code2, res2 = quant_cli.run_cli_command(
        [
            "analyze",
            "etf:510300",
            "--start",
            "2024-01-02",
            "--end",
            "2024-03-01",
            "-o",
            str(out_dir),
            "--report",
            "md",
            "--charts",
            "png",
            "--chart-types",
            "price",
        ],
        loader=lambda *a, **kw: df2.copy(),
    )
    capsys.readouterr()
    assert code2 == quant_cli.EXIT_UNEXPECTED
    assert res2 is None
    assert (out_dir / "report.md").read_text(encoding="utf-8") == orig_report
    assert (out_dir / "charts" / "price.png").read_bytes() == orig_chart
    assert (out_dir / "result.json").read_text(encoding="utf-8") == orig_result


def test_subprocess_lazy_imports_and_out_of_tree_cli_report_generation(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "subproc_out"
    dates = pd.bdate_range("2024-05-01", periods=35)
    sample = _make_ohlcv(
        dates,
        np.linspace(100.0, 118.0, len(dates)),
        asset_type="ashare",
        symbol="600519",
        adjust="qfq",
        currency="CNY",
    )

    prev_dir = os.environ.get(cache_mod.ENV_CACHE_DIR)
    prev_dis = os.environ.get(cache_mod.ENV_DISABLE_CACHE)
    try:
        os.environ[cache_mod.ENV_CACHE_DIR] = str(cache_dir)
        os.environ[cache_mod.ENV_DISABLE_CACHE] = "0"
        cache_mod.save_cached_ohlcv(
            sample, "ashare", "600519", "2024-05-01", "2024-06-30", adjust="qfq"
        )
    finally:
        if prev_dir is None:
            os.environ.pop(cache_mod.ENV_CACHE_DIR, None)
        else:
            os.environ[cache_mod.ENV_CACHE_DIR] = prev_dir
        if prev_dis is None:
            os.environ.pop(cache_mod.ENV_DISABLE_CACHE, None)
        else:
            os.environ[cache_mod.ENV_DISABLE_CACHE] = prev_dis

    env = dict(os.environ)
    env["QUANT_DISABLE_CACHE"] = "0"
    env["QUANT_CACHE_DIR"] = str(cache_dir)

    # 1. Running report-only in subprocess from /tmp never loads matplotlib or plotly
    report_only_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, quant_cli; "
                "code, _ = quant_cli.run_cli_command(["
                "'analyze', 'ashare:600519', '--start', '2024-05-01', '--end', '2024-06-30', "
                "'--adjust', 'qfq', '-o', sys.argv[1], '--report', 'all']); "
                "assert code == 0; "
                "assert 'matplotlib' not in sys.modules and 'plotly' not in sys.modules"
            ),
            str(out_dir / "report_only"),
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert report_only_check.returncode == 0, report_only_check.stderr
    assert (out_dir / "report_only" / "report.md").is_file()
    assert (out_dir / "report_only" / "report.html").is_file()

    # 2. Installed `quant` console script works from outside repo root with --report & --charts
    quant_bin = Path(sys.executable).with_name("quant")
    proc = subprocess.run(
        [
            str(quant_bin),
            "analyze",
            "ashare:600519",
            "--start",
            "2024-05-01",
            "--end",
            "2024-06-30",
            "--adjust",
            "qfq",
            "-o",
            str(out_dir / "with_charts"),
            "--report",
            "md",
            "--charts",
            "html,png",
            "--chart-types",
            "price,drawdown",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    parsed = json.loads(proc.stdout)
    _assert_valid_chart_files(
        parsed["artifacts"],
        ["chart_price_html", "chart_price_png", "chart_drawdown_html", "chart_drawdown_png"],
    )
    assert Path(parsed["artifacts"]["report_md"]).is_file()
