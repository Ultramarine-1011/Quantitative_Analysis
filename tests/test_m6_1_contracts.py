"""M6-1a tests: unified domain metric definitions, strong contract validation, and metadata consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contracts import (
    METRIC_DEFINITIONS,
    SCHEMA_VERSION,
    AnalysisRequest,
    AnalysisResult,
    MultiAssetResult,
    QualityReport,
    SeriesMetadata,
    build_metric_units,
    build_multi_asset_metric_units,
    build_series_metadata,
    get_metric_definition,
    sync_metadata_to_frame,
    validate_frame_metadata_consistency,
    validate_metric_value,
)
from ohlcv import compute_ohlcv_quality
from services.analysis_service import analyze_single_asset


def _sample_df(
    *,
    asset_type: str = "etf",
    symbol: str = "510300",
    adjust: str | None = "hfq",
    currency: str | None = "CNY",
    extra_attrs: dict[str, object] | None = None,
) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=25)
    closes = np.linspace(100.0, 115.0, len(dates))
    df = pd.DataFrame(
        {
            "Open": closes * 0.998,
            "High": closes * 1.005,
            "Low": closes * 0.994,
            "Close": closes,
            "Volume": np.full(len(dates), 10_000.0),
        },
        index=dates.as_unit("ns"),
    )
    df.index.name = "Date"
    df.attrs["asset_type"] = asset_type
    df.attrs["symbol"] = symbol
    df.attrs["canonical_symbol"] = symbol
    if adjust is not None:
        df.attrs["adjust"] = adjust
    if currency is not None:
        df.attrs["currency"] = currency
    if extra_attrs:
        df.attrs.update(extra_attrs)
    df.attrs["data_quality"] = compute_ohlcv_quality(df, missing_policy="drop")
    return df


def _valid_meta(**overrides: object) -> SeriesMetadata:
    base: dict[str, object] = {
        "asset_type": "etf",
        "symbol": "510300",
        "canonical_symbol": "510300",
        "adjust": "hfq",
        "return_basis": "adjusted_price",
        "series_kind": "price",
        "native_currency": "CNY",
        "output_currency": "CNY",
        "fx_converted": False,
        "fx_pair": None,
        "trading_days": 252,
        "data_source": "mock",
        "fetched_at": "2026-06-01T00:00:00+00:00",
        "cache_hit": False,
    }
    base.update(overrides)
    return SeriesMetadata(**base)  # type: ignore[arg-type]


def _valid_quality(**overrides: object) -> QualityReport:
    base: dict[str, object] = {
        "missing_policy": "drop",
        "total_rows": 25,
        "start_date": "2024-01-02",
        "end_date": "2024-02-05",
    }
    base.update(overrides)
    return QualityReport(**base)  # type: ignore[arg-type]


def test_metric_definitions_registry_and_unit_builders() -> None:
    mdd = get_metric_definition("max_drawdown")
    assert mdd.unit == "ratio_decimal"
    assert mdd.sign_convention == "non_negative_depth"
    assert mdd.display_convention == "underwater_percent"
    assert mdd.yield_level_allowed is False

    uw = get_metric_definition("underwater_drawdown")
    assert uw.sign_convention == "underwater_non_positive"

    with pytest.raises(KeyError):
        get_metric_definition("nonexistent_metric")

    assert "sharpe_ratio" in METRIC_DEFINITIONS
    single_units = build_metric_units(
        {"cumulative_return": 0.1, "latest_close": 100.0},
        series_kind="price",
        output_currency="CNY",
    )
    assert single_units == {
        "cumulative_return": "ratio_decimal",
        "latest_close": "currency:CNY",
    }
    multi_units = build_multi_asset_metric_units()
    assert multi_units["max_drawdown"] == "ratio_decimal"
    assert multi_units["tracking_error"] == "ratio_decimal_per_year"
    assert multi_units["effective_n_risk"] == "effective_asset_count"


def test_series_metadata_validation_and_contradiction_guards() -> None:
    # Legacy "none" normalizes to "" on non-adjustable asset types, rejected on equities
    gold_meta = _valid_meta(
        asset_type="gold",
        symbol="GC",
        canonical_symbol="GC",
        adjust="none",
        return_basis="futures_continuous",
        series_kind="futures_continuous",
        native_currency="USD",
        output_currency="USD",
    )
    assert gold_meta.adjust == ""

    with pytest.raises(ValueError, match="复权口径"):
        _valid_meta(asset_type="us_stock", adjust="hfq")
    with pytest.raises(ValueError, match="复权口径"):
        _valid_meta(asset_type="ashare", adjust="none")
    with pytest.raises(ValueError, match="语义矛盾"):
        _valid_meta(asset_type="ashare", adjust="hfq", return_basis="unadjusted_price")
    with pytest.raises(ValueError, match="语义矛盾"):
        _valid_meta(asset_type="ashare", adjust="", return_basis="adjusted_price")
    with pytest.raises(ValueError, match="series_kind"):
        _valid_meta(series_kind="invalid_kind")
    with pytest.raises(ValueError, match="fx_converted"):
        _valid_meta(
            asset_type="bond_us_yield",
            adjust="",
            return_basis="yield_level",
            series_kind="yield_level",
            fx_converted=True,
        )
    with pytest.raises(ValueError, match="trading_days"):
        _valid_meta(trading_days=0)
    with pytest.raises(ValueError, match="trading_days"):
        _valid_meta(trading_days=True)


def test_build_series_metadata_explicit_checks_and_frame_sync() -> None:
    df_missing_adj = _sample_df(asset_type="ashare", symbol="600519", adjust=None)
    with pytest.raises(ValueError, match="adjust"):
        build_series_metadata(df_missing_adj)

    df_no_ccy = _sample_df(asset_type="etf", symbol="510300", adjust="hfq", currency=None)
    with pytest.raises(ValueError, match="strict"):
        build_series_metadata(df_no_ccy, strict=True)

    df_etf = _sample_df(asset_type="etf", symbol="510300", adjust="hfq", currency="CNY")
    with pytest.raises(ValueError, match="asset_type 冲突"):
        build_series_metadata(df_etf, asset_type="ashare")
    with pytest.raises(ValueError, match="symbol 冲突"):
        build_series_metadata(df_etf, symbol="510500")
    with pytest.raises(ValueError, match="adjust 冲突"):
        build_series_metadata(df_etf, adjust="qfq")
    with pytest.raises(ValueError, match="output_currency 冲突"):
        build_series_metadata(df_etf, output_currency="USD")

    meta = build_series_metadata(df_etf, strict=True)
    sync_metadata_to_frame(df_etf, meta)
    validate_frame_metadata_consistency(df_etf, meta)

    df_etf.attrs["currency"] = "USD"
    with pytest.raises(ValueError, match="不一致"):
        validate_frame_metadata_consistency(df_etf, meta)


def test_service_honors_loader_adjust_attr_over_default_request_adjust() -> None:
    # Loader explicitly returns unadjusted prices (adjust="") despite default request adjust="hfq"
    raw = _sample_df(asset_type="etf", symbol="510300", adjust="", currency="CNY")
    req = AnalysisRequest(
        asset_type="etf",
        symbol="510300",
        start_date="2024-01-02",
        end_date="2024-02-05",
    )
    res = analyze_single_asset(req, raw_df=raw)
    assert res.asset.adjust == ""
    assert res.asset.return_basis == "unadjusted_price"
    assert res.data_frame is not None
    validate_frame_metadata_consistency(res.data_frame, res.asset)


def test_quality_report_and_result_invariants() -> None:
    with pytest.raises(ValueError, match="不能为负数"):
        _valid_quality(dropped_invalid_rows=-1)
    with pytest.raises(ValueError, match="不能为布尔值"):
        _valid_quality(total_rows=True)
    with pytest.raises(ValueError, match="quality_scope"):
        _valid_quality(quality_scope="")

    yield_meta = _valid_meta(
        asset_type="bond_us_yield",
        symbol="10Y",
        canonical_symbol="10Y",
        adjust="",
        return_basis="yield_level",
        series_kind="yield_level",
        native_currency="USD",
        output_currency="USD",
    )
    with pytest.raises(ValueError, match="yield_level"):
        AnalysisResult(
            schema_version=SCHEMA_VERSION,
            status="degraded",
            asset=yield_meta,
            quality=_valid_quality(),
            parameters={},
            metrics={"cagr": 0.05},
            metric_reasons={"cagr": "yield_level_not_return"},
            latest_snapshot={},
            data_frame=None,
            command="analyze",
        )

    with pytest.raises(ValueError, match="schema_version"):
        AnalysisResult(
            schema_version="bad",
            status="ok",
            asset=_valid_meta(),
            quality=_valid_quality(),
            parameters={},
            metrics={},
            metric_reasons={},
            latest_snapshot={},
            data_frame=None,
            command="fetch",
        )

    with pytest.raises(ValueError, match="至少需要 2 个标的"):
        MultiAssetResult(
            schema_version=SCHEMA_VERSION,
            command="compare",
            status="ok",
            asset_order=("etf:510300",),
            assets={"etf:510300": _valid_meta()},
            quality={"etf:510300": _valid_quality()},
            data_summary={},
            parameters={},
            methodology={},
            metrics={},
        )

    with pytest.raises(ValueError, match="non-negative"):
        validate_metric_value("annualized_volatility", -0.05)
    with pytest.raises(ValueError, match="Correlation"):
        validate_metric_value("correlation", 1.25)
    with pytest.raises(ValueError, match="R-squared"):
        validate_metric_value("r_squared", 1.1)
