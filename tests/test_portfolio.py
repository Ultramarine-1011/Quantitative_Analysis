"""家庭资产台账：导入、估值、汇总与导出。"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from portfolio import (
    PORTFOLIO_COLUMNS,
    compute_holdings_values,
    holdings_to_csv_bytes,
    normalize_holdings,
    read_holdings_csv,
    summarize_assets,
)


def _sample_holdings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "account": "家庭账户",
                "item_type": "asset",
                "asset_class": "权益",
                "asset_type": "etf",
                "symbol": "510300",
                "name": "沪深300ETF",
                "quantity": "100",
                "current_price": "4",
                "currency": "CNY",
                "is_risk_asset": "true",
                "target_weight": "0.5",
                "notes": "",
            },
            {
                "account": "家庭账户",
                "item_type": "asset",
                "asset_class": "现金",
                "asset_type": "cash",
                "symbol": "",
                "name": "现金",
                "quantity": "1",
                "current_price": "200",
                "currency": "CNY",
                "is_risk_asset": "false",
                "target_weight": "0.3",
                "notes": "",
            },
            {
                "account": "房贷",
                "item_type": "liability",
                "asset_class": "负债",
                "asset_type": "loan",
                "symbol": "",
                "name": "房贷",
                "quantity": "1",
                "current_price": "150",
                "currency": "CNY",
                "is_risk_asset": "false",
                "target_weight": "0",
                "notes": "",
            },
        ]
    )


def test_normalize_holdings_missing_required_column_raises() -> None:
    with pytest.raises(ValueError, match="缺少必填列"):
        normalize_holdings(pd.DataFrame({"item_type": ["asset"]}))


def test_summarize_assets_net_worth_and_risk_ratio() -> None:
    valued = compute_holdings_values(normalize_holdings(_sample_holdings()))
    summary = summarize_assets(valued)
    assert summary["gross_asset_value"] == pytest.approx(600.0)
    assert summary["total_liability_value"] == pytest.approx(150.0)
    assert summary["net_asset_value"] == pytest.approx(450.0)
    assert summary["risk_asset_ratio"] == pytest.approx(400.0 / 600.0)
    by_class = summary["asset_class_summary"]
    assert by_class.loc["权益", "market_value"] == pytest.approx(400.0)


def test_csv_roundtrip_preserves_core_columns() -> None:
    normalized = normalize_holdings(_sample_holdings())
    payload = holdings_to_csv_bytes(normalized)
    loaded = read_holdings_csv(io.BytesIO(payload))
    assert list(loaded.columns) == PORTFOLIO_COLUMNS
    assert loaded.loc[0, "symbol"] == "510300"
    assert loaded.loc[2, "item_type"] == "liability"

