"""Historical console diagnostic report format."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis_metrics import (
    RISK_FREE_RATE,
    format_percent_metric as _format_percent,
    format_ratio_metric as _format_ratio,
    get_quant_metrics,
)
from asset_catalog import (
    ASSET_LABELS,
    ASSET_RETURN_BASIS_NOTES,
    get_asset_trading_days,
)
from feature_engineering import OHLCVFeatureEngineer
from reporting.formatting import format_drawdown_percent as _format_drawdown_percent


def build_diagnostic_report(
    symbol: str,
    df: pd.DataFrame,
    engineer: OHLCVFeatureEngineer,
    risk_free_rate: float = RISK_FREE_RATE,
    *,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Build the comprehensive mathematical diagnostic report for console output."""
    del (
        engineer
    )  # Kept in signature for caller compatibility; shared get_quant_metrics runs pipeline.
    if metrics is None:
        metrics = get_quant_metrics(df, risk_free_rate=risk_free_rate)
    asset_key = str(df.attrs.get("asset_type", "") or "").strip().lower()
    asset_label = ASSET_LABELS.get(asset_key, asset_key or "未知资产")
    currency = str(metrics.get("currency") or df.attrs.get("currency") or "").strip()
    native_ccy = str(df.attrs.get("native_currency", "") or "").strip()
    ccy_text = (
        f"{currency}（原币种: {native_ccy}）"
        if native_ccy and currency and native_ccy != currency
        else (currency or "N/A")
    )
    data_source = str(metrics.get("data_source") or "N/A")
    adjust = str(df.attrs.get("adjust", "") or "")
    adjust_text = adjust if adjust else "未指定/不适用"
    return_basis = str(metrics.get("return_basis") or "")
    basis_note = ASSET_RETURN_BASIS_NOTES.get(return_basis, "")
    return_convention = str(metrics.get("return_convention") or "log")
    trading_days = metrics.get("trading_days") or get_asset_trading_days(asset_key or None)

    lines = [
        "【资产数学诊断报告】",
        f"资产类别: {asset_label}",
        f"标的代码: {symbol}",
        f"数据源 / 计价币种 / 复权口径: {data_source} | {ccy_text} | {adjust_text}",
        f"计算口径: return_convention={return_convention} | 年化交易日数={trading_days}",
        "样本区间: %s 至 %s (%d 条观测)"
        % (
            df.index[0].strftime("%Y-%m-%d"),
            df.index[-1].strftime("%Y-%m-%d"),
            len(df),
        ),
    ]
    if metrics["series_kind"] == "yield_level":
        latest_lvl = metrics.get("latest_level", float("nan"))
        lvl_chg = metrics.get("level_change", float("nan"))
        if pd.notna(latest_lvl) and pd.notna(lvl_chg):
            lines.append(
                f"收益率水平（非总回报）: 最新 {float(latest_lvl):.4f}% | 区间点差变化 {float(lvl_chg):+.4f}%"
            )
    lines.extend(
        [
            "%s: %s"
            % (
                "水平变化（禁用收益率）"
                if metrics["series_kind"] == "yield_level"
                else "期间累计收益率",
                _format_percent(metrics["cumulative_return"], signed=True),
            ),
            "年化收益率 (CAGR): %s" % _format_percent(metrics["cagr"]),
            "年化波动率: %s" % _format_percent(metrics["annualized_volatility"]),
            "最大回撤: %s" % _format_drawdown_percent(metrics["max_drawdown"]),
            "夏普比率 (Rf=%.1f%%): %s"
            % (risk_free_rate * 100.0, _format_ratio(metrics["sharpe_ratio"])),
            "索提诺比率 (Sortino): %s" % _format_ratio(metrics["sortino_ratio"]),
            "卡玛比率 (Calmar): %s" % _format_ratio(metrics["calmar_ratio"]),
            "95%% 日 VaR / CVaR: %s / %s"
            % (
                _format_percent(metrics["var_95"], signed=True),
                _format_percent(metrics["cvar_95"], signed=True),
            ),
            "当前布林带位置: %s" % metrics["latest_signal"],
        ]
    )

    quality = metrics.get("data_quality") or {}
    if isinstance(quality, dict):
        dropped = int(quality.get("dropped_invalid_rows", 0) or 0)
        filled = int(quality.get("filled_price_rows", 0) or 0)
        missing_vol = int(quality.get("missing_volume_rows", 0) or 0)
        non_pos = int(quality.get("non_positive_price_rows", 0) or 0)
        ohlc_bad = int(quality.get("ohlc_anomaly_rows", 0) or 0)
        if dropped or filled or missing_vol or non_pos or ohlc_bad:
            lines.append(
                "数据质量: dropped=%d, filled=%d, missing_vol=%d, non_positive_price=%d, ohlc_anomaly=%d (policy=%s)"
                % (
                    dropped,
                    filled,
                    missing_vol,
                    non_pos,
                    ohlc_bad,
                    quality.get("missing_policy", "drop"),
                )
            )
    for fx_warn in df.attrs.get("fx_warnings") or ():
        lines.append(f"汇率提示: {fx_warn}")
    for fb_warn in df.attrs.get("fallback_warnings") or ():
        lines.append(f"数据源回退提示: {fb_warn}")
    if basis_note:
        lines.append(f"口径说明: {basis_note}")
    if metrics.get("metric_disclaimer"):
        lines.append(f"风险提示: {metrics['metric_disclaimer']}")
    return "\n".join(lines)
