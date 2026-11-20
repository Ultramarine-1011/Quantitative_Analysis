"""Unit tests for the live asset audit helpers and classification rules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.live_asset_audit import (
    ALIAS_TARGETS,
    PRIMARY_TARGETS,
    build_audit_plan,
    classify_audit_status,
    compute_category_summaries,
    detect_rate_limit_signals,
    generate_reports_and_summaries,
    inspect_ohlcv_dataframe,
    safe_symbol_dirname,
)


def _make_ohlcv(
    dates: list[str],
    *,
    open_vals: list[float],
    high_vals: list[float],
    low_vals: list[float],
    close_vals: list[float],
    vol_vals: list[float],
) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    return pd.DataFrame(
        {
            "Open": pd.Series(open_vals, index=idx, dtype="float64"),
            "High": pd.Series(high_vals, index=idx, dtype="float64"),
            "Low": pd.Series(low_vals, index=idx, dtype="float64"),
            "Close": pd.Series(close_vals, index=idx, dtype="float64"),
            "Volume": pd.Series(vol_vals, index=idx, dtype="float64"),
        }
    )


def test_build_audit_plan_counts_and_leading_zeros() -> None:
    primaries, aliases = build_audit_plan()
    assert len(PRIMARY_TARGETS) == 13
    assert len(primaries) == 65
    assert len(aliases) == len(ALIAS_TARGETS)
    hk_syms = [t.symbol for t in primaries if t.asset_type == "hk_stock"]
    assert hk_syms == ["00700", "09988", "01810", "03690", "00941"]
    assert safe_symbol_dirname("BTC/USDT") == "BTC_USDT"
    assert safe_symbol_dirname("00700") == "00700"


def test_inspect_detects_ohlc_anomalies_with_dates_and_reasons() -> None:
    df = _make_ohlcv(
        ["2025-11-17", "2026-11-16"],
        open_vals=[10.0, 10.5],
        high_vals=[10.2, 10.3],  # row 1: high < open
        low_vals=[9.9, 10.6],  # row 1: low > open & low > close & high < low
        close_vals=[10.1, 10.4],
        vol_vals=[1000.0, 2000.0],
    )
    insp = inspect_ohlcv_dataframe(
        df,
        asset_type="etf",
        symbol="510300",
        start_date="2025-11-17",
        end_date="2026-11-16",
        data_source="akshare.stock_zh_a_hist_tx",
    )
    assert insp["ohlc_anomaly_rows"] == 1
    anom = insp["ohlc_anomalies"][0]
    assert anom["date"] == "2026-11-16"
    assert set(anom["reasons"]) == {
        "high_lt_low",
        "high_lt_open",
        "high_lt_close",
        "low_gt_open",
        "low_gt_close",
    }


def test_pseudo_ohlcv_semantics_for_mutual_fund_and_bond_us_yield() -> None:
    nav_df = _make_ohlcv(
        ["2025-11-17", "2026-11-16"],
        open_vals=[1.2, 1.3],
        high_vals=[1.2, 1.3],
        low_vals=[1.2, 1.3],
        close_vals=[1.2, 1.3],
        vol_vals=[0.0, 0.0],
    )
    insp_nav = inspect_ohlcv_dataframe(
        nav_df,
        asset_type="mutual_fund",
        symbol="009691",
        start_date="2025-11-17",
        end_date="2026-11-16",
        data_source="akshare.fund_open_fund_info_em",
    )
    # Note: 2025-11-17 -> 2026-11-16 in 2 rows creates an intentional calendar gap;
    # test with consecutive dates to verify zero-volume pseudo-OHLCV passes cleanly.
    nav_df2 = _make_ohlcv(
        ["2025-11-17", "2025-11-18", "2026-11-16"],
        open_vals=[1.2, 1.21, 1.3],
        high_vals=[1.2, 1.21, 1.3],
        low_vals=[1.2, 1.21, 1.3],
        close_vals=[1.2, 1.21, 1.3],
        vol_vals=[0.0, 0.0, 0.0],
    )
    insp_clean = inspect_ohlcv_dataframe(
        nav_df2,
        asset_type="mutual_fund",
        symbol="009691",
        start_date="2025-11-17",
        end_date="2026-11-16",
    )
    insp_clean["abnormal_gaps"] = []
    status, issues, _, info = classify_audit_status(
        asset_type="mutual_fund",
        symbol="009691",
        fetch_ok=True,
        result_payload={
            "status": "ok",
            "asset": {"output_currency": "CNY"},
            "quality": {},
            "warnings": [],
        },
        inspection=insp_clean,
    )
    assert status == "PASS"
    assert issues == []
    assert "zero_volume_expected" in info
    assert insp_nav["pseudo_ohlc_mismatch_rows"] == 0

    yield_df = _make_ohlcv(
        ["2025-11-17", "2026-11-16"],
        open_vals=[4.2, 4.5],
        high_vals=[4.2, 4.5],
        low_vals=[4.2, 4.5],
        close_vals=[4.2, 4.5],
        vol_vals=[0.0, 0.0],
    )
    insp_yield = inspect_ohlcv_dataframe(
        yield_df,
        asset_type="bond_us_yield",
        symbol="10Y",
        start_date="2025-11-17",
        end_date="2026-11-16",
    )
    insp_yield["abnormal_gaps"] = []
    status_y, issues_y, _, _ = classify_audit_status(
        asset_type="bond_us_yield",
        symbol="10Y",
        fetch_ok=True,
        result_payload={
            "status": "ok",
            "asset": {"output_currency": "USD"},
            "quality": {},
            "warnings": [],
        },
        inspection=insp_yield,
    )
    assert status_y == "PASS"
    assert issues_y == []


def test_category_summary_pass_and_usable_rates() -> None:
    records = [
        {
            "asset_type": "etf",
            "audit_status": "PASS",
            "data_source": "src_a",
            "elapsed_seconds": 1.0,
            "ohlc_anomaly_rows": 0,
            "fallback_used": False,
        },
        {
            "asset_type": "etf",
            "audit_status": "WARN",
            "data_source": "src_b",
            "elapsed_seconds": 2.0,
            "ohlc_anomaly_rows": 3,
            "fallback_used": True,
        },
        {
            "asset_type": "etf",
            "audit_status": "FAIL",
            "data_source": "",
            "elapsed_seconds": 3.0,
            "ohlc_anomaly_rows": 0,
            "fallback_used": False,
        },
    ]
    rows = compute_category_summaries(records)
    etf_row = next(r for r in rows if r["asset_type"] == "etf")
    assert etf_row["pass_count"] == 1
    assert etf_row["warn_count"] == 1
    assert etf_row["fail_count"] == 1
    assert etf_row["pass_rate_pct"] == 33.33
    assert etf_row["usable_rate_pct"] == 66.67
    assert etf_row["total_ohlc_anomaly_rows"] == 3


def test_detect_rate_limit_signals_handles_chinese_and_avoids_timestamps() -> None:
    assert detect_rate_limit_signals("started=2026-11-18T16:25:22.429904+00:00") == []
    assert detect_rate_limit_signals("HTTP 429 Too Many Requests") == ["http_429"]
    assert detect_rate_limit_signals(
        "上游限流/拦截: 东方财富 push2 行情服务器断开了连接，当前 IP 被临时限流"
    ) == ["push2_rate_limit"]


def test_generate_reports_and_summaries_writes_expected_artifacts(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    report_root = tmp_path / "reports"
    primary = [
        {
            "item_id": "P01",
            "group": "primary",
            "asset_type": "etf",
            "symbol": "510300",
            "canonical_symbol": "510300",
            "alias_of": "",
            "audit_status": "PASS",
            "cli_status": "ok",
            "exit_code": 0,
            "attempts": 1,
            "elapsed_seconds": 0.9,
            "data_source": "akshare.stock_zh_a_hist_tx",
            "fallback_used": False,
            "fallback_count": 0,
            "fallback_warnings": [],
            "rate_limit_signals": [],
            "adjust": "hfq",
            "return_basis": "adjusted_price",
            "series_kind": "price",
            "native_currency": "CNY",
            "output_currency": "CNY",
            "requested_start": "2025-11-17",
            "requested_end": "2026-11-16",
            "actual_start": "2025-11-17",
            "actual_end": "2026-11-16",
            "start_lag_days": 0,
            "end_lag_days": 0,
            "row_count": 242,
            "duplicate_dates": 0,
            "ohlc_anomaly_rows": 0,
            "ohlc_anomalies": [],
            "pseudo_ohlc_mismatch_rows": 0,
            "non_positive_price_rows": 0,
            "negative_volume_rows": 0,
            "missing_volume_rows": 0,
            "zero_volume_rows": 0,
            "dropped_invalid_rows": 0,
            "filled_price_rows": 0,
            "max_calendar_gap_days": 11,
            "abnormal_gaps": [],
            "extreme_jumps": [],
            "vol_min": 100.0,
            "vol_median": 200.0,
            "vol_max": 300.0,
            "volume_unit_note": "shares_or_units",
            "latest_close": 1.83,
            "issue_codes": [],
            "info_notes": [],
            "warnings": [],
            "error_category": "",
            "error_type": "",
            "error_message": "",
            "artifact_dir": "etf/510300",
        }
    ]
    summary = generate_reports_and_summaries(
        data_root=data_root,
        report_root=report_root,
        primary_records=primary,
        alias_records=[],
        audit_metadata={
            "audit_date": "2026-11-17",
            "requested_start": "2025-11-17",
            "requested_end": "2026-11-16",
            "base_currency": "LOCAL",
            "refresh": True,
            "run_started_at": "2026-11-18T16:24:10+00:00",
            "run_finished_at": "2026-11-18T16:26:40+00:00",
            "git_branch": "codex/frontend-qa",
            "git_commit": "a322f3e",
            "versions": {"python": "3.14.5"},
        },
    )
    assert summary["overall_summary"]["pass_count"] == 1
    assert (data_root / "manifest.csv").is_file()
    assert (data_root / "summary.csv").is_file()
    assert (data_root / "summary.json").is_file()
    assert (report_root / "REPORT.md").is_file()
    assert (report_root / "failures.md").is_file()
