"""M3 tests: multi-asset comparison, portfolio optimization/backtesting, and M2 hardening fixes."""

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
from scipy import stats as scipy_stats

import data_sources.cache as cache_mod
import quant_cli
from analysis_metrics import (
    compute_benchmark_relative_metrics,
    compute_memmel_sharpe_test,
    compute_return_distribution_stats,
    compute_rolling_comparison_stats,
)
from contracts import SCHEMA_VERSION, SOFTWARE_VERSION
from quant_models import (
    compute_risk_contributions,
    solve_inverse_volatility_portfolio,
    solve_risk_parity_portfolio,
)


def _make_ohlcv(
    dates: pd.DatetimeIndex,
    close: np.ndarray | list[float],
    *,
    asset_type: str = "etf",
    symbol: str = "510300",
    currency: str | None = None,
    data_source: str = "mock",
) -> pd.DataFrame:
    arr = np.asarray(close, dtype=float)
    df = pd.DataFrame(
        {
            "Open": arr * 0.999,
            "High": arr * 1.005,
            "Low": arr * 0.995,
            "Close": arr,
            "Volume": np.full(len(arr), 100000.0),
        },
        index=dates,
    )
    df.index.name = "Date"
    df.attrs["asset_type"] = asset_type
    df.attrs["symbol"] = symbol
    df.attrs["data_source"] = data_source
    if currency is not None:
        df.attrs["currency"] = currency
    return df


# ---------------------------------------------------------------------------
# 1. M2 legacy fixes: cache overwrite rollback, multi-file CLI atomicity & parsing
# ---------------------------------------------------------------------------


def test_cache_overwrite_preserves_prior_valid_entry_when_meta_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cache_mod.ENV_DISABLE_CACHE, "0")
    monkeypatch.setenv(cache_mod.ENV_CACHE_DIR, str(tmp_path))

    dates = pd.bdate_range("2024-01-02", periods=10)
    old_df = _make_ohlcv(
        dates, np.linspace(10.0, 11.0, len(dates)), asset_type="etf", symbol="510300"
    )
    new_df = _make_ohlcv(
        dates, np.linspace(20.0, 25.0, len(dates)), asset_type="etf", symbol="510300"
    )

    first_path = cache_mod.save_cached_ohlcv(
        old_df, "etf", "510300", "2024-01-02", "2024-01-15", adjust="hfq"
    )
    assert first_path is not None and first_path.is_file()

    real_replace = os.replace

    def _fail_meta_replace(src: Any, dst: Any) -> None:
        if str(dst).endswith(".meta.json"):
            raise OSError("simulated disk failure on meta replace during overwrite")
        real_replace(src, dst)

    with monkeypatch.context() as mp:
        mp.setattr(cache_mod.os, "replace", _fail_meta_replace)
        res = cache_mod.save_cached_ohlcv(
            new_df,
            "etf",
            "510300",
            "2024-01-02",
            "2024-01-15",
            adjust="hfq",
        )
        assert res is None

    # No stray .tmp / .bak files and prior valid cache remains readable with old values.
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name or ".bak" in p.name]
    assert leftovers == []

    restored = cache_mod.load_cached_ohlcv(
        "etf",
        "510300",
        "2024-01-02",
        "2024-01-15",
        adjust="hfq",
    )
    assert restored is not None
    np.testing.assert_allclose(restored["Close"].to_numpy(), old_df["Close"].to_numpy())


def test_cli_multi_file_output_atomic_rollback_and_stale_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=60)
    rng = np.random.default_rng(7)
    r_a = 0.0005 + 0.01 * rng.normal(size=len(dates))
    r_b = 0.0004 + 0.015 * rng.normal(size=len(dates))
    df_a = _make_ohlcv(
        dates, 100.0 * np.cumprod(1.0 + r_a), asset_type="etf", symbol="510300", currency="CNY"
    )
    df_b = _make_ohlcv(
        dates, 100.0 * np.cumprod(1.0 + r_b), asset_type="etf", symbol="510500", currency="CNY"
    )

    def fake_loader(
        *, asset_type: str, symbol: str, start_date: str, end_date: str, proxy_url=None, **kwargs
    ):
        return df_a if "510300" in symbol else df_b

    out_dir = tmp_path / "artifacts"
    out_dir.mkdir()
    user_note = out_dir / "research_notes.md"
    user_note.write_text("unrelated user file\n", encoding="utf-8")
    extra_csv = tmp_path / "standalone.csv"

    # Run 1: portfolio with --csv and --export-csv
    code1, _ = quant_cli.run_cli_command(
        [
            "portfolio",
            "etf:510300",
            "etf:510500",
            "--start",
            "2024-01-02",
            "--end",
            "2024-03-29",
            "--no-sensitivity",
            "-o",
            str(out_dir),
            "--csv",
            "--export-csv",
            str(extra_csv),
            "-q",
        ],
        loader=fake_loader,
    )
    assert code1 == 0
    assert (out_dir / "weights.parquet").is_file()
    assert (out_dir / "weights.csv").is_file()
    assert (out_dir / "frontier.parquet").is_file()
    assert (out_dir / "backtest.parquet").is_file()
    assert extra_csv.is_file()

    prior_result_bytes = (out_dir / "result.json").read_bytes()
    prior_parquet_bytes = (out_dir / "data.parquet").read_bytes()
    prior_extra_csv_bytes = extra_csv.read_bytes()

    # Run 2: simulate failure midway when replacing result.json (after data.parquet & extra.csv replaced)
    real_replace = os.replace

    def _fail_result_json_replace(src: Any, dst: Any) -> None:
        if Path(str(dst)).name == "result.json":
            raise OSError("simulated IO error on result.json commit")
        real_replace(src, dst)

    with monkeypatch.context() as mp:
        mp.setattr(quant_cli.os, "replace", _fail_result_json_replace)
        code_fail, res_fail = quant_cli.run_cli_command(
            [
                "compare",
                "etf:510300",
                "etf:510500",
                "--start",
                "2024-01-02",
                "--end",
                "2024-03-29",
                "-o",
                str(out_dir),
                "--export-csv",
                str(extra_csv),
                "-q",
            ],
            loader=fake_loader,
        )
    assert code_fail == quant_cli.EXIT_UNEXPECTED
    assert res_fail is None
    # Exact byte-for-byte rollback across all files & stale portfolio artifacts preserved
    assert (out_dir / "result.json").read_bytes() == prior_result_bytes
    assert (out_dir / "data.parquet").read_bytes() == prior_parquet_bytes
    assert extra_csv.read_bytes() == prior_extra_csv_bytes
    assert (out_dir / "weights.parquet").is_file()
    assert user_note.read_text(encoding="utf-8") == "unrelated user file\n"
    assert [p for p in out_dir.iterdir() if ".tmp" in p.name] == []

    # Run 3: successful compare without --csv cleanly purges stale portfolio & CSV artifacts
    code3, res3 = quant_cli.run_cli_command(
        [
            "compare",
            "etf:510300",
            "etf:510500",
            "--start",
            "2024-01-02",
            "--end",
            "2024-03-29",
            "-o",
            str(out_dir),
            "-q",
        ],
        loader=fake_loader,
    )
    assert code3 == 0 and res3 is not None
    assert not (out_dir / "weights.parquet").exists()
    assert not (out_dir / "frontier.parquet").exists()
    assert not (out_dir / "backtest.parquet").exists()
    assert not (out_dir / "data.csv").exists()
    assert not (out_dir / "prices.csv").exists()
    assert (out_dir / "prices.parquet").is_file()
    assert (out_dir / "correlation.parquet").is_file()
    assert (out_dir / "rolling_metrics.parquet").is_file()
    assert user_note.read_text(encoding="utf-8") == "unrelated user file\n"
    for art_key, art_val in res3.artifacts.items():
        if art_key == "output_dir":
            assert Path(art_val).is_dir()
        else:
            assert Path(art_val).is_file()


@pytest.mark.parametrize(
    "argv",
    [
        ["fetch", "AAPL"],
        ["analyze", "600519"],
        ["fetch", "foo:510300"],
        ["fetch", "ashare:"],
        ["fetch", ":510300"],
        ["analyze", "-t", "etf", "ashare:600519"],
        ["compare", "510300", "510500"],
        ["compare", "badtype:510300", "etf:510500"],
        ["compare", "hk_stock:700", "hk_stock:00700"],
        ["compare", "ashare:sh600519", "ashare:600519"],
        ["portfolio", "etf:510300"],
        ["portfolio", "etf:510300", "etf:510500", "--strategy", "custom"],
        ["portfolio", "etf:510300", "etf:510500", "--weights", "0.5,0.5"],
        [
            "portfolio",
            "etf:510300",
            "etf:510500",
            "--strategy",
            "custom",
            "--weights",
            "0.5,0.5",
            "--validation-mode",
            "walk_forward_rolling",
        ],
        ["portfolio", "etf:510300", "etf:510500", "--min-weight", "0.6"],
    ],
)
def test_cli_asset_type_and_caller_validation_rejects_ambiguous_inputs(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, res = quant_cli.run_cli_command(argv)
    captured = capsys.readouterr()
    assert code == quant_cli.EXIT_USAGE_OR_CALLER
    assert res is None
    assert captured.out == ""
    err = json.loads(captured.err.strip().splitlines()[-1])
    assert err["error"]["category"] == "caller"


def test_cli_parses_legacy_comma_pipe_and_shared_asset_type(tmp_path: Path) -> None:
    dates = pd.bdate_range("2024-01-02", periods=30)
    df_a = _make_ohlcv(
        dates,
        np.linspace(10.0, 12.0, len(dates)),
        asset_type="etf",
        symbol="510300",
        currency="CNY",
    )
    df_b = _make_ohlcv(
        dates,
        np.linspace(20.0, 22.0, len(dates)),
        asset_type="etf",
        symbol="510500",
        currency="CNY",
    )

    def fake_loader(
        *, asset_type: str, symbol: str, start_date: str, end_date: str, proxy_url=None, **kwargs
    ):
        return df_a if "510300" in symbol else df_b

    code, res = quant_cli.run_cli_command(
        [
            "compare",
            "--assets",
            "etf,510300;etf|510500",
            "--start",
            "2024-01-02",
            "--end",
            "2024-02-15",
            "-q",
        ],
        loader=fake_loader,
    )
    assert code == 0 and res is not None
    assert res.asset_order == ("etf:510300", "etf:510500")

    code_shared, res_shared = quant_cli.run_cli_command(
        [
            "compare",
            "-t",
            "etf",
            "510300",
            "510500",
            "--start",
            "2024-01-02",
            "--end",
            "2024-02-15",
            "-q",
        ],
        loader=fake_loader,
    )
    assert code_shared == 0 and res_shared is not None
    assert res_shared.asset_order == ("etf:510300", "etf:510500")


# ---------------------------------------------------------------------------
# 2. Numerical correctness & edge cases for M3 statistical / risk helpers
# ---------------------------------------------------------------------------


def test_euler_risk_contributions_and_risk_parity_convergence() -> None:
    cov = np.array(
        [
            [0.0100, 0.0050, 0.0040],
            [0.0050, 0.0625, 0.0150],
            [0.0040, 0.0150, 0.1600],
        ],
        dtype=float,
    )
    w = np.array([0.5, 0.3, 0.2], dtype=float)
    rc = compute_risk_contributions(w, cov, ["A", "B", "C"])
    total_rc = sum(rc["risk_contribution"].values())
    total_prc = sum(rc["pct_risk_contribution"].values())
    assert total_rc == pytest.approx(rc["portfolio_volatility"], rel=1e-10)
    assert total_prc == pytest.approx(1.0, rel=1e-10)
    assert rc["diversification_ratio"] > 1.0

    # Build synthetic returns having roughly distinct volatilities + nonzero correlation
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2023-01-02", periods=200)
    chol = np.linalg.cholesky(cov / 252.0)
    z = rng.normal(size=(len(dates), 3)) @ chol.T
    ret_df = pd.DataFrame(z, index=dates, columns=["A", "B", "C"])

    rp_sol = solve_risk_parity_portfolio(ret_df, cov_method="sample", trading_days=252)
    est_cov = ret_df.cov(ddof=1).to_numpy() * 252.0
    rp_rc = compute_risk_contributions(rp_sol.weights, est_cov, ["A", "B", "C"])
    assert rp_sol.converged is True
    assert rp_sol.weights.sum() == pytest.approx(1.0, rel=1e-8)
    for prc_val in rp_rc["pct_risk_contribution"].values():
        assert prc_val == pytest.approx(1.0 / 3.0, abs=1e-4)

    iv_sol = solve_inverse_volatility_portfolio(ret_df, cov_method="sample", trading_days=252)
    assert iv_sol.weights.sum() == pytest.approx(1.0, rel=1e-8)
    # Lower vol asset A should receive strictly larger weight than higher vol asset C
    assert iv_sol.weights[0] > iv_sol.weights[1] > iv_sol.weights[2]

    # Constrained bounds check
    rp_capped = solve_risk_parity_portfolio(
        ret_df,
        cov_method="ledoit_wolf",
        min_weight=0.15,
        max_weight=0.50,
    )
    assert np.all(rp_capped.weights >= 0.15 - 1e-6)
    assert np.all(rp_capped.weights <= 0.50 + 1e-6)
    assert rp_capped.weights.sum() == pytest.approx(1.0, rel=1e-8)


def test_distribution_stats_memmel_and_benchmark_metrics_exact() -> None:
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2023-01-02", periods=120)
    rb = pd.Series(0.0005 + 0.012 * rng.normal(size=len(dates)), index=dates)
    # Exact linear excess response: rp = alpha_d + 1.5 * rb
    rp = 0.0004 + 1.5 * rb

    dist = compute_return_distribution_stats(rb)
    assert dist["observations"] == 120
    assert dist["skewness"] == pytest.approx(
        float(scipy_stats.skew(rb.to_numpy(), bias=False)), rel=1e-10
    )
    assert dist["excess_kurtosis"] == pytest.approx(
        float(scipy_stats.kurtosis(rb.to_numpy(), fisher=True, bias=False)), rel=1e-10
    )
    assert dist["mean_t_stat"] == pytest.approx(
        float(scipy_stats.ttest_1samp(rb.to_numpy(), 0.0).statistic)
    )

    const_dist = compute_return_distribution_stats(pd.Series(np.full(20, 0.01), index=dates[:20]))
    assert np.isnan(const_dist["jarque_bera_stat"])
    assert const_dist["metric_reasons"]["skewness"] == "zero_volatility"

    rel = compute_benchmark_relative_metrics(
        rp,
        rb,
        periods_per_year=252.0,
        risk_free_rate=0.0,
        return_convention="simple",
    )
    assert rel["beta"] == pytest.approx(1.5, rel=1e-6)
    assert rel["jensen_alpha"] == pytest.approx(0.0004 * 252.0, rel=1e-6)
    assert rel["correlation"] == pytest.approx(1.0, abs=1e-10)
    assert rel["up_capture_ratio"] > 1.4
    assert rel["down_capture_ratio"] < 1.6

    # Memmel Sharpe difference test: superior alpha-shifted portfolio vs benchmark
    mem_diff = compute_memmel_sharpe_test(
        rp + 0.003 * pd.Series(rng.normal(size=len(dates)), index=dates),
        rb,
        periods_per_year=252.0,
        risk_free_rate=0.0,
    )
    assert np.isfinite(mem_diff["z_stat"])
    assert 0.0 <= mem_diff["p_value"] <= 1.0

    mem_ident = compute_memmel_sharpe_test(rb, rb)
    assert np.isnan(mem_ident["z_stat"])
    assert mem_ident["reason"] == "zero_difference_variance"


def test_rolling_comparison_avoids_nan_gap_distortion() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(11)
    btc = pd.Series(0.001 + 0.02 * rng.normal(size=len(dates)), index=dates)
    etf = pd.Series(0.0005 + 0.01 * rng.normal(size=len(dates)), index=dates)
    etf.iloc[5::7] = np.nan
    etf.iloc[6::7] = np.nan  # weekend NaNs

    wide = pd.DataFrame({"crypto:BTC/USDT": btc, "etf:510300": etf})
    rolling_df, summary = compute_rolling_comparison_stats(wide, window=15, periods_per_year=252.0)

    valid_etf = etf.dropna()
    expected_latest_vol = float(valid_etf.iloc[-15:].std(ddof=1) * np.sqrt(252.0))
    assert summary["assets"]["etf:510300"]["volatility_ann"]["latest"] == pytest.approx(
        expected_latest_vol, rel=1e-10
    )
    assert "etf:510300__rolling_vol_ann" in rolling_df.columns


# ---------------------------------------------------------------------------
# 3. End-to-end quant compare & quant portfolio tests
# ---------------------------------------------------------------------------


def test_quant_compare_cross_market_fx_benchmark_and_artifacts(tmp_path: Path) -> None:
    dates = pd.bdate_range("2024-01-02", periods=90)
    rng = np.random.default_rng(42)
    base_mkt = 0.0006 + 0.012 * rng.normal(size=len(dates))
    etf_ret = 0.8 * base_mkt + 0.004 * rng.normal(size=len(dates))
    spy_ret = 1.1 * base_mkt + 0.005 * rng.normal(size=len(dates))

    etf_df = _make_ohlcv(
        dates, 100.0 * np.cumprod(1.0 + etf_ret), asset_type="etf", symbol="510300", currency="CNY"
    )
    spy_df = _make_ohlcv(
        dates,
        400.0 * np.cumprod(1.0 + spy_ret),
        asset_type="us_stock",
        symbol="SPY",
        currency="USD",
    )
    bench_df = _make_ohlcv(
        dates, 300.0 * np.cumprod(1.0 + base_mkt), asset_type="etf", symbol="510500", currency="CNY"
    )
    fx_df = _make_ohlcv(dates, np.full(len(dates), 7.2), asset_type="fx", symbol="USDCNH")

    def fake_loader(
        *, asset_type: str, symbol: str, start_date: str, end_date: str, proxy_url=None, **kwargs
    ):
        if "510300" in symbol:
            return etf_df
        if "SPY" in symbol:
            return spy_df
        return bench_df

    out_dir = tmp_path / "compare_out"
    code, res = quant_cli.run_cli_command(
        [
            "compare",
            "etf:510300",
            "us_stock:SPY",
            "--benchmark",
            "etf:510500",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-10",
            "--base-currency",
            "CNY",
            "--rolling-window",
            "20",
            "-o",
            str(out_dir),
            "--csv",
            "-q",
        ],
        loader=fake_loader,
        fx_loader=lambda pair, s, e: fx_df,
    )
    assert code == 0 and res is not None
    assert res.command == "compare"
    assert res.status == "ok"
    assert res.assets["us_stock:SPY"].fx_converted is True
    assert res.assets["us_stock:SPY"].output_currency == "CNY"
    assert res.benchmark is not None and res.benchmark["leg_name"] == "etf:510500"
    assert "etf:510300" in res.metrics["benchmark_comparison"]
    assert "us_stock:SPY" in res.metrics["benchmark_comparison"]

    # Normalized data.parquet starts strictly at 1.0
    norm_df = pd.read_parquet(out_dir / "data.parquet")
    assert norm_df.iloc[0]["etf:510300"] == pytest.approx(1.0)
    assert norm_df.iloc[0]["us_stock:SPY"] == pytest.approx(1.0)

    corr_df = pd.read_parquet(out_dir / "correlation.parquet")
    assert corr_df.loc["etf:510300", "etf:510300"] == pytest.approx(1.0)
    assert corr_df.loc["etf:510300", "us_stock:SPY"] > 0.7

    parsed_json = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert parsed_json["schema_version"] == SCHEMA_VERSION
    assert parsed_json["software_version"] == SOFTWARE_VERSION
    assert parsed_json["methodology"]["base_currency"] == "CNY"


def test_quant_compare_mtm_outer_and_semantic_guards() -> None:
    cal_dates = pd.date_range("2024-01-01", periods=60, freq="D")
    b_dates = pd.bdate_range("2024-01-01", "2024-02-29")
    rng = np.random.default_rng(55)
    btc_df = _make_ohlcv(
        cal_dates,
        40000.0 * np.cumprod(1.0 + 0.001 + 0.02 * rng.normal(size=len(cal_dates))),
        asset_type="crypto",
        symbol="BTC/USDT",
        currency="USDT",
    )
    etf_df = _make_ohlcv(
        b_dates,
        10.0 * np.cumprod(1.0 + 0.0005 + 0.01 * rng.normal(size=len(b_dates))),
        asset_type="etf",
        symbol="510300",
        currency="CNY",
    )
    yield_df = _make_ohlcv(
        b_dates,
        np.linspace(4.0, 4.3, len(b_dates)),
        asset_type="bond_us_yield",
        symbol="10Y",
        currency="USD",
    )
    yield_df.attrs["series_kind"] = "yield_level"
    fx_cny_usd = _make_ohlcv(
        cal_dates, np.full(len(cal_dates), 7.2), asset_type="fx", symbol="USDCNH"
    )

    def fake_loader(
        *, asset_type: str, symbol: str, start_date: str, end_date: str, proxy_url=None, **kwargs
    ):
        if asset_type == "crypto":
            return btc_df
        if asset_type == "bond_us_yield":
            return yield_df
        if asset_type == "fx":
            return fx_cny_usd
        return etf_df

    # mtm_outer uses concurrent quotes for vol/correlation while preserving continuous NAV grid
    code, res = quant_cli.run_cli_command(
        [
            "compare",
            "etf:510300",
            "crypto:BTC/USDT",
            "--start",
            "2024-01-01",
            "--end",
            "2024-02-29",
            "--calendar-mode",
            "mtm_outer",
            "--base-currency",
            "CNY",
            "--allow-usdt-peg",
            "-q",
        ],
        loader=fake_loader,
        fx_loader=lambda pair, s, e: fx_cny_usd,
    )
    assert code == 0 and res is not None
    assert res.methodology["mtm_correlation_uses_concurrent_quotes"] is True
    assert res.data_summary["concurrent_return_periods"] < res.data_summary["return_periods"]

    # Mixed LOCAL currencies trigger explicit degraded warning
    code_mix, res_mix = quant_cli.run_cli_command(
        [
            "compare",
            "etf:510300",
            "crypto:BTC/USDT",
            "--start",
            "2024-01-01",
            "--end",
            "2024-02-29",
            "--base-currency",
            "LOCAL",
            "-q",
        ],
        loader=fake_loader,
    )
    assert code_mix == 0 and res_mix is not None
    assert res_mix.status == "degraded"
    assert res_mix.methodology["mixed_local_currencies"] is True
    assert any("LOCAL" in w for w in res_mix.warnings)

    # yield_level must be rejected cleanly with EXIT_USAGE_OR_CALLER
    code_yld, _ = quant_cli.run_cli_command(
        [
            "compare",
            "etf:510300",
            "bond_us_yield:10Y",
            "--start",
            "2024-01-01",
            "--end",
            "2024-02-29",
            "-q",
        ],
        loader=fake_loader,
    )
    assert code_yld == quant_cli.EXIT_USAGE_OR_CALLER

    # fx_rate must be rejected when base_currency != LOCAL
    code_fx, _ = quant_cli.run_cli_command(
        [
            "compare",
            "etf:510300",
            "fx:USDCNH",
            "--start",
            "2024-01-01",
            "--end",
            "2024-02-29",
            "--base-currency",
            "CNY",
            "-q",
        ],
        loader=fake_loader,
    )
    assert code_fx == quant_cli.EXIT_USAGE_OR_CALLER


def test_quant_portfolio_train_test_zero_lookahead_and_exact_vs_random_frontier(
    tmp_path: Path,
) -> None:
    dates = pd.bdate_range("2023-01-02", periods=120)
    rng = np.random.default_rng(17)
    r1 = 0.0009 + 0.008 * rng.normal(size=len(dates))
    r2 = 0.0005 + 0.015 * rng.normal(size=len(dates))
    r3 = 0.0003 + 0.022 * rng.normal(size=len(dates))

    p1 = 100.0 * np.cumprod(1.0 + r1)
    p2 = 100.0 * np.cumprod(1.0 + r2)
    p3 = 100.0 * np.cumprod(1.0 + r3)

    # Second universe identical on first 85 rows (train) and drastically inverted on last 35 rows (test)
    p1_pert = p1.copy()
    p2_pert = p2.copy()
    p3_pert = p3.copy()
    p1_pert[85:] = p1_pert[84] * np.cumprod(1.0 + np.full(len(dates) - 85, -0.02))
    p2_pert[85:] = p2_pert[84] * np.cumprod(1.0 + np.full(len(dates) - 85, 0.015))
    p3_pert[85:] = p3_pert[84] * np.cumprod(1.0 + np.full(len(dates) - 85, 0.03))

    def _loader_builder(arr1: np.ndarray, arr2: np.ndarray, arr3: np.ndarray):
        f1 = _make_ohlcv(dates, arr1, asset_type="etf", symbol="510300", currency="CNY")
        f2 = _make_ohlcv(dates, arr2, asset_type="etf", symbol="510500", currency="CNY")
        f3 = _make_ohlcv(dates, arr3, asset_type="etf", symbol="510880", currency="CNY")

        def _fn(
            *,
            asset_type: str,
            symbol: str,
            start_date: str,
            end_date: str,
            proxy_url=None,
            **kwargs,
        ):
            if "510300" in symbol:
                return f1
            if "510500" in symbol:
                return f2
            return f3

        return _fn

    out1 = tmp_path / "run1"
    code1, res1 = quant_cli.run_cli_command(
        [
            "portfolio",
            "etf:510300",
            "etf:510500",
            "etf:510880",
            "--start",
            "2023-01-02",
            "--end",
            "2023-06-30",
            "--strategy",
            "max_sharpe",
            "--validation-mode",
            "train_test",
            "--train-ratio",
            "0.7",
            "--frontier-points",
            "15",
            "--num-random-portfolios",
            "40",
            "-o",
            str(out1),
            "-q",
        ],
        loader=_loader_builder(p1, p2, p3),
    )
    assert code1 == 0 and res1 is not None
    assert res1.methodology["is_out_of_sample"] is True
    assert res1.methodology["has_lookahead_bias"] is False

    # Frontier Parquet clearly separates exact SLSQP frontier from Dirichlet scatter
    frontier_df = pd.read_parquet(out1 / "frontier.parquet")
    assert set(frontier_df["frontier_type"].unique()) == {"slsqp_exact", "dirichlet_random"}
    assert res1.metrics["efficient_frontier"]["exact_slsqp"]["algorithm"] == "slsqp_exact_frontier"
    assert (
        res1.metrics["efficient_frontier"]["random_sampling"]["algorithm"]
        == "dirichlet_random_sampling"
    )
    assert len(res1.metrics["sensitivity"]["covariance_method"]) == 3
    assert len(res1.metrics["sensitivity"]["transaction_cost_bps"]) >= 5

    # Run with perturbed test window -> weights & frontier & train cov MUST remain 100% invariant
    code2, res2 = quant_cli.run_cli_command(
        [
            "portfolio",
            "etf:510300",
            "etf:510500",
            "etf:510880",
            "--start",
            "2023-01-02",
            "--end",
            "2023-06-30",
            "--strategy",
            "max_sharpe",
            "--validation-mode",
            "train_test",
            "--train-ratio",
            "0.7",
            "--frontier-points",
            "15",
            "--num-random-portfolios",
            "40",
            "--no-sensitivity",
            "-q",
        ],
        loader=_loader_builder(p1_pert, p2_pert, p3_pert),
    )
    assert code2 == 0 and res2 is not None
    for leg in res1.asset_order:
        assert res1.metrics["portfolio"]["weights"][leg] == pytest.approx(
            res2.metrics["portfolio"]["weights"][leg], abs=1e-12
        )
    # OOS backtest performance DOES change when test data changed
    assert res1.metrics["backtest"]["OOS_Portfolio"]["cumulative_return"] != pytest.approx(
        res2.metrics["backtest"]["OOS_Portfolio"]["cumulative_return"], abs=1e-6
    )


@pytest.mark.parametrize("validation_mode", ["walk_forward_rolling", "walk_forward_expanding"])
def test_quant_portfolio_walk_forward_and_oos_benchmark_slice(
    validation_mode: str,
    tmp_path: Path,
) -> None:
    dates = pd.bdate_range("2023-01-02", periods=100)
    rng = np.random.default_rng(77)
    r_a = 0.0005 + 0.01 * rng.normal(size=len(dates))
    r_b = 0.0006 + 0.018 * rng.normal(size=len(dates))
    # Benchmark falls sharply in first 35 days (pre-OOS), rises steadily in OOS span [40:]
    r_bm = np.where(np.arange(len(dates)) < 35, -0.02, 0.003)

    df_a = _make_ohlcv(
        dates, 100.0 * np.cumprod(1.0 + r_a), asset_type="etf", symbol="510300", currency="CNY"
    )
    df_b = _make_ohlcv(
        dates, 100.0 * np.cumprod(1.0 + r_b), asset_type="us_stock", symbol="SPY", currency="USD"
    )
    df_bm = _make_ohlcv(
        dates, 100.0 * np.cumprod(1.0 + r_bm), asset_type="etf", symbol="510500", currency="CNY"
    )
    fx_df = _make_ohlcv(dates, np.full(len(dates), 7.2), asset_type="fx", symbol="USDCNH")

    def fake_loader(
        *, asset_type: str, symbol: str, start_date: str, end_date: str, proxy_url=None, **kwargs
    ):
        if "510300" in symbol:
            return df_a
        if "SPY" in symbol:
            return df_b
        return df_bm

    out_dir = tmp_path / validation_mode
    code, res = quant_cli.run_cli_command(
        [
            "portfolio",
            "etf:510300",
            "us_stock:SPY",
            "--benchmark",
            "etf:510500",
            "--start",
            "2023-01-02",
            "--end",
            "2023-06-01",
            "--base-currency",
            "CNY",
            "--strategy",
            "risk_parity",
            "--validation-mode",
            validation_mode,
            "--train-window",
            "40",
            "--rebalance-freq",
            "monthly",
            "--tx-bps",
            "10",
            "--no-sensitivity",
            "-o",
            str(out_dir),
            "-q",
        ],
        loader=fake_loader,
        fx_loader=lambda pair, s, e: fx_df,
    )
    assert code == 0 and res is not None
    assert (out_dir / "weight_history.parquet").is_file()
    assert (out_dir / "backtest.parquet").is_file()
    assert res.metrics["backtest"]["In_Sample_Portfolio"]["has_lookahead_bias"] is True

    # Crucial: Benchmark must only be evaluated on the OOS span (so positive cumulative return!), not pre-OOS slump
    assert res.benchmark is not None
    assert res.benchmark["performance"]["cumulative_return"] > 0.0
    bt_df = pd.read_parquet(out_dir / "backtest.parquet")
    assert bt_df["Benchmark"].iloc[0] == pytest.approx(1.0)
    assert bt_df["OOS_Portfolio"].iloc[0] == pytest.approx(1.0)


def test_quant_portfolio_in_sample_warns_and_custom_strategy_works(tmp_path: Path) -> None:
    dates = pd.bdate_range("2024-01-02", periods=65)
    rng = np.random.default_rng(9)
    df_a = _make_ohlcv(
        dates,
        100.0 * np.cumprod(1.0 + 0.001 + 0.01 * rng.normal(size=len(dates))),
        asset_type="etf",
        symbol="510300",
        currency="CNY",
    )
    df_b = _make_ohlcv(
        dates,
        100.0 * np.cumprod(1.0 + 0.0005 + 0.015 * rng.normal(size=len(dates))),
        asset_type="etf",
        symbol="510500",
        currency="CNY",
    )

    def fake_loader(
        *, asset_type: str, symbol: str, start_date: str, end_date: str, proxy_url=None, **kwargs
    ):
        return df_a if "510300" in symbol else df_b

    code_is, res_is = quant_cli.run_cli_command(
        [
            "portfolio",
            "etf:510300",
            "etf:510500",
            "--start",
            "2024-01-02",
            "--end",
            "2024-04-10",
            "--strategy",
            "inverse_volatility",
            "--validation-mode",
            "in_sample",
            "--no-sensitivity",
            "-q",
        ],
        loader=fake_loader,
    )
    assert code_is == 0 and res_is is not None
    assert res_is.methodology["is_out_of_sample"] is False
    assert res_is.methodology["has_lookahead_bias"] is True
    assert any("lookahead" in w.lower() or "前视" in w for w in res_is.warnings)

    # Custom weights normalized cleanly from e.g. 60,40 -> 0.6,0.4
    code_c, res_c = quant_cli.run_cli_command(
        [
            "portfolio",
            "etf:510300",
            "etf:510500",
            "--start",
            "2024-01-02",
            "--end",
            "2024-04-10",
            "--strategy",
            "custom",
            "--weights",
            "60,40",
            "--validation-mode",
            "train_test",
            "--no-sensitivity",
            "-q",
        ],
        loader=fake_loader,
    )
    assert code_c == 0 and res_c is not None
    assert res_c.metrics["portfolio"]["weights"]["etf:510300"] == pytest.approx(0.6)
    assert res_c.metrics["portfolio"]["weights"]["etf:510500"] == pytest.approx(0.4)


def test_quant_cli_version_and_no_matplotlib_import() -> None:
    ver = subprocess.run(
        [sys.executable, "-m", "quant_cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ver.returncode == 0
    assert SOFTWARE_VERSION in ver.stdout
    assert SCHEMA_VERSION in ver.stdout

    imp = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, quant_cli; assert 'matplotlib' not in sys.modules and 'mplfinance' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert imp.returncode == 0, imp.stderr
