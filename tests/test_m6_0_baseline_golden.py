"""M6-0 baseline golden tests, cross-entry parity, dependency audit, and P0 characterizations."""

from __future__ import annotations

import ast
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import quant_cli
from contracts import (
    AnalysisRequest,
    AnalysisResult,
    AssetSpec,
    CompareRequest,
    MultiAssetResult,
    PortfolioRequest,
    build_series_metadata,
)
from ohlcv import compute_ohlcv_quality
from reporting import (
    build_html_report,
    build_markdown_report,
    format_drawdown_percent,
    load_result_from_artifacts,
)
from services.analysis_service import analyze_single_asset, fetch_single_asset
from services.multi_asset_service import analyze_portfolio, compare_assets

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_ohlcv(
    dates: pd.DatetimeIndex,
    closes: np.ndarray | list[float],
    *,
    asset_type: str = "etf",
    symbol: str = "510300",
    adjust: str | None = "hfq",
    currency: str | None = "CNY",
    data_source: str = "mock:golden",
    volume: float = 10_000.0,
    extra_attrs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    df = pd.DataFrame(
        {
            "Open": c * 0.998,
            "High": c * 1.006,
            "Low": c * 0.992,
            "Close": c,
            "Volume": np.full(len(c), volume, dtype=float),
        },
        index=pd.DatetimeIndex(dates).as_unit("ns"),
    )
    df.index.name = "Date"
    attrs: dict[str, Any] = {
        "asset_type": asset_type,
        "symbol": symbol,
        "canonical_symbol": symbol,
        "currency": currency,
        "data_source": data_source,
        "fetched_at": "2026-06-01T00:00:00+00:00",
        "cache_hit": False,
    }
    if adjust is not None:
        attrs["adjust"] = adjust
    df.attrs.update(attrs)
    df.attrs["data_quality"] = compute_ohlcv_quality(df, missing_policy="drop")
    if extra_attrs:
        df.attrs.update(extra_attrs)
    return df


def _make_multi_market_bundle() -> tuple[Any, Any, pd.DatetimeIndex]:
    dates = pd.bdate_range("2024-01-02", periods=90)
    rng = np.random.default_rng(123)
    shocks_etf = 0.0007 + 0.010 * rng.normal(size=len(dates))
    shocks_spy = 0.0005 + 0.014 * rng.normal(size=len(dates))
    shocks_gld = 0.0003 + 0.009 * rng.normal(size=len(dates))
    shocks_bench = 0.55 * shocks_etf + 0.45 * shocks_spy + 0.002 * rng.normal(size=len(dates))

    frames = {
        ("etf", "510300"): _make_ohlcv(
            dates,
            100.0 * np.exp(np.cumsum(shocks_etf)),
            asset_type="etf",
            symbol="510300",
            adjust="hfq",
            currency="CNY",
        ),
        ("us_stock", "SPY"): _make_ohlcv(
            dates,
            400.0 * np.exp(np.cumsum(shocks_spy)),
            asset_type="us_stock",
            symbol="SPY",
            adjust="qfq",
            currency="USD",
        ),
        ("gold", "GC"): _make_ohlcv(
            dates,
            2000.0 * np.exp(np.cumsum(shocks_gld)),
            asset_type="gold",
            symbol="GC",
            adjust="",
            currency="USD",
        ),
        ("etf", "510500"): _make_ohlcv(
            dates,
            5.0 * np.exp(np.cumsum(shocks_bench)),
            asset_type="etf",
            symbol="510500",
            adjust="hfq",
            currency="CNY",
        ),
    }

    fx_series = 7.15 + 0.05 * np.sin(np.linspace(0.0, 2.0 * np.pi, len(dates)))

    def loader(asset_type: str, symbol: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return frames[(asset_type, symbol)].copy()

    def fx_loader(pair: str, start: Any, end: Any) -> pd.DataFrame:
        rates = fx_series if pair == "USDCNY" or pair == "USDCNH" else 1.0 / fx_series
        return _make_ohlcv(
            dates,
            rates,
            asset_type="fx",
            symbol=pair,
            adjust="",
            currency=pair[3:],
            volume=0.0,
        )

    return loader, fx_loader, dates


# ---------------------------------------------------------------------------
# 1. Single-asset deterministic golden tests & service/CLI parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("convention", ["log", "simple"])
def test_single_asset_analytic_golden_and_cli_parity(
    convention: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Deterministic trajectory: 100 -> 125 (+25%) -> 100 (-20% drawdown) -> 130 (+30% total)
    dates = pd.bdate_range("2024-01-02", periods=25)
    closes = np.concatenate(
        [
            np.linspace(100.0, 125.0, 10),
            np.linspace(120.0, 100.0, 7),
            np.linspace(105.0, 130.0, 8),
        ]
    )
    df = _make_ohlcv(dates, closes, asset_type="etf", symbol="510300", adjust="hfq", currency="CNY")

    req = AnalysisRequest(
        asset_type="etf",
        symbol="510300",
        start_date="2024-01-02",
        end_date="2024-02-05",
        adjust="hfq",
        risk_free_rate=0.02,
        trading_days=252,
        return_convention=convention,  # type: ignore[arg-type]
        sma_short=3,
        sma_long=6,
        bb_window=5,
        include_momentum=True,
    )
    svc_res = analyze_single_asset(req, loader=lambda *a, **kw: df.copy())

    code, cli_res = quant_cli.run_cli_command(
        [
            "analyze",
            "etf:510300",
            "--start",
            "2024-01-02",
            "--end",
            "2024-02-05",
            "--adjust",
            "hfq",
            "--rf",
            "0.02",
            "--trading-days",
            "252",
            "--return-convention",
            convention,
            "--sma-short",
            "3",
            "--sma-long",
            "6",
            "--bb-window",
            "5",
            "--include-momentum",
            "-q",
        ],
        loader=lambda *a, **kw: df.copy(),
    )
    stdout_payload = json.loads(capsys.readouterr().out)
    assert code == quant_cli.EXIT_OK
    assert isinstance(cli_res, AnalysisResult)

    # Analytic reference checks
    elapsed_days = float((dates[-1] - dates[0]).days)
    expected_cum = 130.0 / 100.0 - 1.0
    expected_cagr = (130.0 / 100.0) ** (365.25 / elapsed_days) - 1.0
    expected_mdd = 1.0 - 100.0 / 125.0  # 0.20

    simple_r = closes[1:] / closes[:-1] - 1.0
    period_r = np.log(closes[1:] / closes[:-1]) if convention == "log" else simple_r
    rf_daily = np.log1p(0.02) / 252.0 if convention == "log" else 0.02 / 252.0
    vol_expected = float(np.std(period_r, ddof=1) * math.sqrt(252.0))
    ex = period_r - rf_daily
    sharpe_expected = float(math.sqrt(252.0) * np.mean(ex) / np.std(period_r, ddof=1))
    down = np.minimum(ex, 0.0)
    down_dev = float(np.sqrt(np.mean(down * down)))
    sortino_expected = float(math.sqrt(252.0) * np.mean(ex) / down_dev)
    calmar_expected = expected_cagr / expected_mdd
    var95_expected = float(np.quantile(simple_r, 0.05))
    cvar95_expected = float(np.mean(simple_r[simple_r <= var95_expected]))

    m = svc_res.metrics
    assert m["cumulative_return"] == pytest.approx(expected_cum, rel=1e-12, abs=1e-12)
    assert m["cagr"] == pytest.approx(expected_cagr, rel=1e-12, abs=1e-12)
    assert m["max_drawdown"] == pytest.approx(expected_mdd, rel=1e-12, abs=1e-12)
    assert m["annualized_volatility"] == pytest.approx(vol_expected, rel=1e-12, abs=1e-12)
    assert m["sharpe_ratio"] == pytest.approx(sharpe_expected, rel=1e-12, abs=1e-12)
    assert m["sortino_ratio"] == pytest.approx(sortino_expected, rel=1e-12, abs=1e-12)
    assert m["calmar_ratio"] == pytest.approx(calmar_expected, rel=1e-12, abs=1e-12)
    assert m["var_95"] == pytest.approx(var95_expected, rel=1e-12, abs=1e-12)
    assert m["cvar_95"] == pytest.approx(cvar95_expected, rel=1e-12, abs=1e-12)
    assert m["latest_close"] == pytest.approx(130.0, rel=1e-12)
    assert m["trading_days"] == 252
    assert m["return_convention"] == convention

    # Service <-> CLI JSON strict parity
    for k in (
        "cumulative_return",
        "cagr",
        "max_drawdown",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "var_95",
        "cvar_95",
        "latest_close",
    ):
        assert stdout_payload["metrics"][k] == pytest.approx(m[k], rel=1e-12, abs=1e-12)
    assert stdout_payload["quality"] == svc_res.quality.to_dict()
    assert stdout_payload["data_summary"] == svc_res.data_summary
    assert stdout_payload["metric_units"] == svc_res.metric_units


# ---------------------------------------------------------------------------
# 2. Multi-asset compare golden tests & CLI/Service parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("calendar_mode", ["common_trading_dates", "mtm_outer"])
def test_compare_cross_market_golden_and_cli_parity(
    calendar_mode: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader, fx_loader, _ = _make_multi_market_bundle()
    req = CompareRequest(
        assets=(
            AssetSpec("etf", "510300", adjust="hfq"),
            AssetSpec("us_stock", "SPY", adjust="qfq"),
            AssetSpec("gold", "GC"),
        ),
        start_date="2024-01-02",
        end_date="2024-05-06",
        benchmark=AssetSpec("etf", "510500", adjust="hfq"),
        calendar_mode=calendar_mode,  # type: ignore[arg-type]
        base_currency="CNY",
        risk_free_rate=0.016,
        return_convention="simple",
        rolling_window=20,
        cov_method="ledoit_wolf",
    )
    svc_res = compare_assets(req, loader=loader, fx_loader=fx_loader)

    code, cli_res = quant_cli.run_cli_command(
        [
            "compare",
            "etf:510300",
            "us_stock:SPY",
            "gold:GC",
            "--benchmark",
            "etf:510500",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-06",
            "--calendar-mode",
            calendar_mode,
            "--base-currency",
            "CNY",
            "--rf",
            "0.016",
            "--return-convention",
            "simple",
            "--rolling-window",
            "20",
            "--cov-method",
            "ledoit_wolf",
            "-q",
        ],
        loader=loader,
        fx_loader=fx_loader,
    )
    cli_payload = json.loads(capsys.readouterr().out)
    assert code == quant_cli.EXIT_OK
    assert isinstance(cli_res, MultiAssetResult)
    assert svc_res.status == "ok"
    assert svc_res.assets["us_stock:SPY"].fx_converted is True
    assert svc_res.assets["us_stock:SPY"].output_currency == "CNY"
    assert svc_res.assets["gold:GC"].fx_converted is True

    for leg in svc_res.asset_order:
        svc_m = svc_res.metrics["per_asset"][leg]
        cli_m = cli_payload["metrics"]["per_asset"][leg]
        for k in (
            "cumulative_return",
            "cagr",
            "max_drawdown",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "var_95",
            "cvar_95",
        ):
            assert cli_m[k] == pytest.approx(svc_m[k], rel=1e-12, abs=1e-12)
        assert cli_payload["metrics"]["distribution"][leg]["jarque_bera_p_value"] == pytest.approx(
            svc_res.metrics["distribution"][leg]["jarque_bera_p_value"], rel=1e-12, abs=1e-12
        )
        assert cli_payload["metrics"]["benchmark_comparison"][leg]["memmel_sharpe_test"][
            "p_value"
        ] == pytest.approx(
            svc_res.metrics["benchmark_comparison"][leg]["memmel_sharpe_test"]["p_value"],
            rel=1e-12,
            abs=1e-12,
        )


# ---------------------------------------------------------------------------
# 3. Portfolio optimization/OOS/seeded reproducibility golden tests
# ---------------------------------------------------------------------------


def test_portfolio_train_test_walk_forward_custom_and_seed_reproducibility(
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader, fx_loader, _ = _make_multi_market_bundle()
    assets = (
        AssetSpec("etf", "510300", adjust="hfq"),
        AssetSpec("us_stock", "SPY", adjust="qfq"),
        AssetSpec("gold", "GC"),
    )
    bench = AssetSpec("etf", "510500", adjust="hfq")

    # 3a. max_sharpe + train_test + seeded random cloud reproducibility & CLI parity
    req_tt = PortfolioRequest(
        assets=assets,
        start_date="2024-01-02",
        end_date="2024-05-06",
        strategy="max_sharpe",
        validation_mode="train_test",
        benchmark=bench,
        base_currency="CNY",
        rebalance_freq="monthly",
        transaction_cost_bps=10.0,
        frontier_points=12,
        num_random_portfolios=25,
        random_seed=42,
        include_sensitivity=True,
    )
    svc_tt_1 = analyze_portfolio(req_tt, loader=loader, fx_loader=fx_loader)
    svc_tt_2 = analyze_portfolio(req_tt, loader=loader, fx_loader=fx_loader)
    code_tt, _ = quant_cli.run_cli_command(
        [
            "portfolio",
            "etf:510300",
            "us_stock:SPY",
            "gold:GC",
            "--benchmark",
            "etf:510500",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-06",
            "--strategy",
            "max_sharpe",
            "--validation-mode",
            "train_test",
            "--base-currency",
            "CNY",
            "--rebalance-freq",
            "monthly",
            "--tx-bps",
            "10",
            "--frontier-points",
            "12",
            "--num-random-portfolios",
            "25",
            "--seed",
            "42",
            "-q",
        ],
        loader=loader,
        fx_loader=fx_loader,
    )
    cli_tt = json.loads(capsys.readouterr().out)
    assert code_tt == quant_cli.EXIT_OK
    assert svc_tt_1.methodology["is_out_of_sample"] is True
    assert svc_tt_1.methodology["has_lookahead_bias"] is False

    # Seed reproducibility & Euler RC sum invariant
    rc = svc_tt_1.metrics["risk_contributions"]
    total_rc = sum(rc["risk_contribution"].values())
    assert total_rc == pytest.approx(rc["portfolio_volatility"], rel=1e-10, abs=1e-10)
    assert sum(rc["pct_risk_contribution"].values()) == pytest.approx(1.0, rel=1e-10, abs=1e-10)
    assert svc_tt_1.metrics["efficient_frontier"]["random_sampling"]["best_sampled_sharpe"][
        "sharpe"
    ] == pytest.approx(
        svc_tt_2.metrics["efficient_frontier"]["random_sampling"]["best_sampled_sharpe"]["sharpe"],
        abs=1e-12,
    )
    for leg in svc_tt_1.asset_order:
        assert cli_tt["metrics"]["portfolio"]["weights"][leg] == pytest.approx(
            svc_tt_1.metrics["portfolio"]["weights"][leg], rel=1e-12, abs=1e-12
        )
    for k in ("cumulative_return", "cagr", "max_drawdown", "sharpe_ratio", "total_turnover"):
        assert cli_tt["metrics"]["backtest"]["OOS_Portfolio"][k] == pytest.approx(
            svc_tt_1.metrics["backtest"]["OOS_Portfolio"][k], rel=1e-12, abs=1e-12
        )

    # 3b. risk_parity + walk_forward_rolling
    req_wf = PortfolioRequest(
        assets=assets,
        start_date="2024-01-02",
        end_date="2024-05-06",
        strategy="risk_parity",
        validation_mode="walk_forward_rolling",
        train_window=45,
        rebalance_freq="monthly",
        base_currency="CNY",
        frontier_points=0,
        include_sensitivity=False,
    )
    svc_wf = analyze_portfolio(req_wf, loader=loader, fx_loader=fx_loader)
    assert "weight_history" in svc_wf.frames
    assert not svc_wf.frames["weight_history"].empty
    assert svc_wf.metrics["backtest"]["In_Sample_Portfolio"]["has_lookahead_bias"] is True

    # 3c. custom weights in train_test & in_sample
    req_custom_tt = PortfolioRequest(
        assets=assets[:2],
        start_date="2024-01-02",
        end_date="2024-05-06",
        strategy="custom",
        validation_mode="train_test",
        custom_weights=(0.6, 0.4),
        base_currency="CNY",
        frontier_points=0,
        include_sensitivity=False,
    )
    svc_custom_tt = analyze_portfolio(req_custom_tt, loader=loader, fx_loader=fx_loader)
    assert svc_custom_tt.metrics["portfolio"]["weights"] == pytest.approx(
        {"etf:510300": 0.6, "us_stock:SPY": 0.4}, abs=1e-12
    )
    assert svc_custom_tt.methodology["is_out_of_sample"] is True
    assert svc_custom_tt.methodology["has_lookahead_bias"] is False

    req_custom_is = PortfolioRequest(
        assets=assets[:2],
        start_date="2024-01-02",
        end_date="2024-05-06",
        strategy="custom",
        validation_mode="in_sample",
        custom_weights=(0.6, 0.4),
        base_currency="CNY",
        frontier_points=0,
        include_sensitivity=False,
    )
    svc_custom_is = analyze_portfolio(req_custom_is, loader=loader, fx_loader=fx_loader)
    assert svc_custom_is.methodology["is_out_of_sample"] is False
    assert svc_custom_is.methodology["has_lookahead_bias"] is False


# ---------------------------------------------------------------------------
# 4. Special series & mathematical edge-case golden tests
# ---------------------------------------------------------------------------


def test_special_series_and_edge_cases_golden(capsys: pytest.CaptureFixture[str]) -> None:
    dates = pd.bdate_range("2024-01-02", periods=30)

    # 4a. yield_level semantic block & JSON null serialization
    yield_df = _make_ohlcv(
        dates,
        np.linspace(4.0, 4.35, len(dates)),
        asset_type="bond_us_yield",
        symbol="10Y",
        adjust="",
        currency="USD",
        volume=0.0,
        extra_attrs={"series_kind": "yield_level", "return_basis": "yield_level"},
    )
    code_y, res_y = quant_cli.run_cli_command(
        [
            "analyze",
            "bond_us_yield:10Y",
            "--start",
            "2024-01-02",
            "--end",
            "2024-02-12",
            "-q",
        ],
        loader=lambda *a, **kw: yield_df.copy(),
    )
    payload_y = json.loads(capsys.readouterr().out)
    assert code_y == quant_cli.EXIT_OK
    assert isinstance(res_y, AnalysisResult) and res_y.status == "degraded"
    for blocked in (
        "cumulative_return",
        "cagr",
        "max_drawdown",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "var_95",
        "cvar_95",
    ):
        assert math.isnan(res_y.metrics[blocked])
        assert payload_y["metrics"][blocked] is None
        assert payload_y["metric_reasons"][blocked] == "yield_level_not_return"
    assert payload_y["metrics"]["latest_level"] == pytest.approx(4.35, abs=1e-12)
    assert payload_y["metrics"]["level_change"] == pytest.approx(0.35, abs=1e-12)

    with pytest.raises(ValueError, match="yield_level"):
        analyze_single_asset(
            AnalysisRequest(
                asset_type="bond_us_yield",
                symbol="10Y",
                start_date="2024-01-02",
                end_date="2024-02-12",
                base_currency="CNY",
            ),
            raw_df=yield_df,
        )

    # 4b. fx_rate blocks non-LOCAL conversion, succeeds on LOCAL with caveat
    fx_df = _make_ohlcv(
        dates,
        np.linspace(7.10, 7.25, len(dates)),
        asset_type="fx",
        symbol="USDCNH",
        adjust="",
        currency="CNY",
        volume=0.0,
        extra_attrs={"series_kind": "fx_rate", "return_basis": "fx_rate"},
    )
    with pytest.raises(ValueError, match="fx_rate"):
        analyze_single_asset(
            AnalysisRequest(
                asset_type="fx",
                symbol="USDCNH",
                start_date="2024-01-02",
                end_date="2024-02-12",
                base_currency="USD",
            ),
            raw_df=fx_df,
        )
    fx_local = analyze_single_asset(
        AnalysisRequest(
            asset_type="fx",
            symbol="USDCNH",
            start_date="2024-01-02",
            end_date="2024-02-12",
            base_currency="LOCAL",
        ),
        raw_df=fx_df,
    )
    assert any("Carry" in w or "汇率" in w for w in fx_local.warnings)

    # 4c. Constant / flat series
    flat_df = _make_ohlcv(dates, np.full(len(dates), 100.0), asset_type="etf", symbol="510300")
    flat_res = analyze_single_asset(
        AnalysisRequest("etf", "510300", "2024-01-02", "2024-02-12"),
        raw_df=flat_df,
    )
    assert flat_res.metrics["cumulative_return"] == pytest.approx(0.0, abs=1e-12)
    assert flat_res.metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    assert flat_res.metrics["annualized_volatility"] == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(flat_res.metrics["sharpe_ratio"])
    assert math.isnan(flat_res.metrics["sortino_ratio"])
    assert math.isnan(flat_res.metrics["calmar_ratio"])
    assert flat_res.metric_reasons["calmar_ratio"] == "zero_max_drawdown"

    # 4d. Monotonic rising series (zero drawdown, positive CAGR -> calmar NaN with zero_max_drawdown)
    rising_df = _make_ohlcv(
        dates, np.linspace(100.0, 120.0, len(dates)), asset_type="etf", symbol="510300"
    )
    rising_res = analyze_single_asset(
        AnalysisRequest("etf", "510300", "2024-01-02", "2024-02-12"),
        raw_df=rising_df,
    )
    assert rising_res.metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(rising_res.metrics["calmar_ratio"])
    assert rising_res.metric_reasons["calmar_ratio"] == "zero_max_drawdown"

    # 4e. Very short 3-bar series
    short_dates = pd.bdate_range("2024-01-02", periods=3)
    short_df = _make_ohlcv(short_dates, [100.0, 102.0, 101.0], asset_type="etf", symbol="510300")
    short_res = fetch_single_asset(
        AnalysisRequest("etf", "510300", "2024-01-02", "2024-01-04"),
        raw_df=short_df,
    )
    assert short_res.quality.total_rows == 3


# ---------------------------------------------------------------------------
# 5. Offline Parquet & report replay golden test
# ---------------------------------------------------------------------------


def test_offline_parquet_and_report_replay_without_loaders(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader, fx_loader, _ = _make_multi_market_bundle()

    for cmd_name, argv in (
        (
            "analyze",
            [
                "analyze",
                "etf:510300",
                "--start",
                "2024-01-02",
                "--end",
                "2024-05-06",
                "-o",
                str(tmp_path / "out_analyze"),
                "-q",
            ],
        ),
        (
            "compare",
            [
                "compare",
                "etf:510300",
                "us_stock:SPY",
                "--benchmark",
                "etf:510500",
                "--start",
                "2024-01-02",
                "--end",
                "2024-05-06",
                "-o",
                str(tmp_path / "out_compare"),
                "-q",
            ],
        ),
        (
            "portfolio",
            [
                "portfolio",
                "etf:510300",
                "us_stock:SPY",
                "--start",
                "2024-01-02",
                "--end",
                "2024-05-06",
                "--no-sensitivity",
                "--frontier-points",
                "6",
                "-o",
                str(tmp_path / "out_portfolio"),
                "-q",
            ],
        ),
    ):
        code, res = quant_cli.run_cli_command(argv, loader=loader, fx_loader=fx_loader)
        capsys.readouterr()
        assert code == quant_cli.EXIT_OK and res is not None

        reloaded = load_result_from_artifacts(tmp_path / f"out_{cmd_name}")
        assert reloaded.command == cmd_name
        md_text = build_markdown_report(reloaded)
        html_text = build_html_report(reloaded)
        assert md_text and "<!DOCTYPE html>" in html_text
        if isinstance(reloaded, AnalysisResult):
            assert format_drawdown_percent(reloaded.metrics["max_drawdown"]) in md_text


# ---------------------------------------------------------------------------
# 6. AST-based dependency direction audit
# ---------------------------------------------------------------------------


def _collect_internal_imports() -> dict[str, set[str]]:
    modules: dict[str, set[str]] = {}
    all_py: list[tuple[str, Path]] = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if any(
            part in {".venv", "venv", "tests", "scripts", "__pycache__", "build", "dist"}
            for part in path.parts
        ):
            continue
        rel = path.relative_to(REPO_ROOT).with_suffix("")
        mod_name = ".".join(rel.parts)
        if mod_name.endswith(".__init__"):
            mod_name = mod_name[:-9]
        all_py.append((mod_name, path))

    top_names = {name.split(".")[0] for name, _ in all_py}
    for mod_name, path in all_py:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        deps: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in top_names:
                        deps.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in top_names:
                    deps.add(node.module)
        modules[mod_name] = deps
    return modules


def test_module_dependency_direction_audit() -> None:
    graph = _collect_internal_imports()

    def _top_deps(mod: str) -> set[str]:
        return {d.split(".")[0] for d in graph[mod]}

    # Domain/contract layer must remain decoupled from services, UI, CLI, rendering, and network fetchers.
    assert _top_deps("contracts").isdisjoint(
        {
            "services",
            "quant_cli",
            "app",
            "main",
            "reporting",
            "data_sources",
            "data_fetcher",
            "asset_loader",
        }
    )
    assert _top_deps("asset_catalog") == set()
    assert _top_deps("ohlcv").isdisjoint({"services", "quant_cli", "app", "main", "reporting"})

    # Quantitative computation core must remain decoupled from UI, CLI, reporting, services, and loaders.
    for core_mod in ("analysis_metrics", "feature_engineering", "quant_models"):
        assert _top_deps(core_mod).isdisjoint(
            {
                "streamlit",
                "app",
                "main",
                "quant_cli",
                "reporting",
                "services",
                "data_fetcher",
                "asset_loader",
            }
        )

    # Core backend modules must import cleanly in fresh Python without circular ImportError.
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asset_catalog, ohlcv, contracts, analysis_metrics, quant_models, services.analysis_service, services.multi_asset_service, reporting",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# 7. P0 behavior characterizations (expected failures BEFORE M6-1 fixes)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="A1 baseline defect: compare stores max_drawdown <= 0 while analyze and portfolio store >= 0.",
)
def test_p0_a1_cross_command_max_drawdown_sign_consistency() -> None:
    dates = pd.bdate_range("2024-01-02", periods=25)
    # Deterministic path with peak=125 and trough=100 => 20% drawdown depth (+0.20)
    closes_a = np.concatenate(
        [
            np.linspace(100.0, 125.0, 10),
            np.linspace(120.0, 100.0, 7),
            np.linspace(105.0, 130.0, 8),
        ]
    )
    closes_b = np.linspace(50.0, 60.0, len(dates))
    df_a = _make_ohlcv(
        dates, closes_a, asset_type="etf", symbol="510300", adjust="hfq", currency="CNY"
    )
    df_b = _make_ohlcv(
        dates, closes_b, asset_type="etf", symbol="510500", adjust="hfq", currency="CNY"
    )

    def loader(asset_type: str, symbol: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return df_a.copy() if "510300" in symbol else df_b.copy()

    an_res = analyze_single_asset(
        AnalysisRequest("etf", "510300", "2024-01-02", "2024-02-05", adjust="hfq"),
        loader=loader,
    )
    cmp_res = compare_assets(
        CompareRequest(
            assets=(
                AssetSpec("etf", "510300", adjust="hfq"),
                AssetSpec("etf", "510500", adjust="hfq"),
            ),
            start_date="2024-01-02",
            end_date="2024-02-05",
            base_currency="CNY",
        ),
        loader=loader,
    )
    port_res = analyze_portfolio(
        PortfolioRequest(
            assets=(
                AssetSpec("etf", "510300", adjust="hfq"),
                AssetSpec("etf", "510500", adjust="hfq"),
            ),
            start_date="2024-01-02",
            end_date="2024-02-05",
            strategy="custom",
            validation_mode="in_sample",
            rebalance_freq="buy_and_hold",
            transaction_cost_bps=0.0,
            custom_weights=(1.0, 0.0),
            base_currency="CNY",
            frontier_points=0,
            include_sensitivity=False,
        ),
        loader=loader,
    )

    mdd_analyze = an_res.metrics["max_drawdown"]
    mdd_compare = cmp_res.metrics["per_asset"]["etf:510300"]["max_drawdown"]
    mdd_portfolio = port_res.metrics["backtest"]["Portfolio"]["max_drawdown"]

    assert mdd_analyze == pytest.approx(0.20, rel=1e-10)
    assert mdd_portfolio == pytest.approx(0.20, rel=1e-10)
    assert mdd_compare == pytest.approx(0.20, rel=1e-10)


def test_p0_a2_series_metadata_rejects_missing_adjust_contradiction() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10)
    raw = _make_ohlcv(
        dates,
        np.linspace(100.0, 110.0, len(dates)),
        asset_type="ashare",
        symbol="600519",
        adjust=None,
        currency="CNY",
    )
    with pytest.raises(ValueError, match="adjust"):
        build_series_metadata(raw)


@pytest.mark.xfail(
    strict=True,
    reason="A3 baseline defect: result.json emits absolute artifact paths and copied directory reads mutated original directory.",
)
def test_p0_a3_copied_artifact_dir_reads_local_parquet_and_uses_relative_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=30)
    closes_a = np.linspace(100.0, 120.0, len(dates))
    df_a = _make_ohlcv(
        dates, closes_a, asset_type="etf", symbol="510300", adjust="hfq", currency="CNY"
    )

    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    code, _ = quant_cli.run_cli_command(
        [
            "analyze",
            "etf:510300",
            "--start",
            "2024-01-02",
            "--end",
            "2024-02-12",
            "-o",
            str(dir_a),
            "-q",
        ],
        loader=lambda *a, **kw: df_a.copy(),
    )
    capsys.readouterr()
    assert code == quant_cli.EXIT_OK

    shutil.copytree(dir_a, dir_b)

    # Mutate dir_a/data.parquet after copying to dir_b
    df_mutated = df_a.copy()
    df_mutated["Close"] = df_mutated["Close"] * 10.0
    df_mutated.attrs = {}
    df_mutated.to_parquet(dir_a / "data.parquet", engine="pyarrow", index=True)

    payload_b = json.loads((dir_b / "result.json").read_text(encoding="utf-8"))
    assert payload_b["artifacts"]["data_parquet"] == "data.parquet"

    loaded_b = load_result_from_artifacts(dir_b)
    assert loaded_b.data_frame is not None
    np.testing.assert_allclose(loaded_b.data_frame["Close"].to_numpy(), closes_a)
