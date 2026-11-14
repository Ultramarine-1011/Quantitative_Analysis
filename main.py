"""Quantitative analysis system CLI entrypoint covering all 13 asset classes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from analysis_metrics import (
    RISK_FREE_RATE,
    TRADING_DAYS as TRADING_DAYS,
    compute_cagr as _compute_cagr,  # noqa: F401 - re-exported for parity tests
    format_percent_metric as _format_percent,
    format_ratio_metric as _format_ratio,
    get_quant_metrics,
)
from asset_catalog import (
    ASSET_LABELS,
    ASSET_RETURN_BASIS_NOTES,
    DEFAULT_SYMBOLS,
    SUPPORTED_ASSET_TYPES,
    get_asset_trading_days,
    resolve_adjustment,
)
from asset_loader import load_asset_data
from asset_resolver import (
    CHINA_EQUITY_ASSET_TYPES,
    format_diagnosis_user_message,
    load_china_equity_ohlcv,
    resolve_analysis_date_window,
)
from currency import (
    convert_prices_to_base_currency,
    infer_asset_currency,
    normalize_currency_code,
)
from data_fetcher import DataFetchError, apply_default_network_proxy_policy
from feature_engineering import OHLCVFeatureEngineer
from ohlcv import REQUIRED_OHLCV_COLUMNS
from visualizer import MplfinanceVisualizer

LOOKBACK_YEARS = 3


def _convert_ohlcv_to_base_currency(
    df: pd.DataFrame,
    asset_key: str,
    symbol: str,
    target_currency: str,
    *,
    proxy_url: str | None = None,
) -> pd.DataFrame:
    """将单标的 OHLCV 价格列按对齐汇率同比例折算为目标基准货币。"""
    series_kind = str(df.attrs.get("series_kind", "price") or "price").strip().lower()
    if series_kind == "yield_level" or asset_key == "bond_us_yield":
        raise ValueError("收益率水平序列（yield_level）不是货币计价资产价格，不支持基准货币换算。")
    native_ccy = infer_asset_currency(asset_key, symbol, df.attrs)
    target_norm = normalize_currency_code(target_currency)
    if target_norm == native_ccy:
        out = df.copy()
        out.attrs["native_currency"] = native_ccy
        out.attrs["currency"] = target_norm
        return out

    col_key = f"{asset_key}:{symbol}"
    conv = convert_prices_to_base_currency(
        pd.DataFrame({col_key: df["Close"]}),
        {col_key: native_ccy},
        base_currency=target_norm,
        proxy_url=proxy_url,
    )
    aligned_idx = conv.base_prices.index
    local_close = conv.local_prices[col_key]
    base_close = conv.base_prices[col_key]
    fx_ratio = base_close / local_close

    out = df.reindex(aligned_idx).copy()
    for price_col in ("Open", "High", "Low", "Close"):
        out[price_col] = pd.to_numeric(out[price_col], errors="coerce") * fx_ratio
    out.attrs = dict(df.attrs)
    out.attrs["native_currency"] = native_ccy
    out.attrs["currency"] = conv.base_currency
    if conv.warnings:
        out.attrs["fx_warnings"] = conv.warnings
    return out


def _export_ohlcv_csv(df: pd.DataFrame, export_path: str) -> Path:
    """导出标准化 UTF-8-SIG OHLCV CSV 文件。"""
    target = Path(export_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    subset = df.loc[:, list(REQUIRED_OHLCV_COLUMNS)].sort_index().copy()
    if not isinstance(subset.index, pd.DatetimeIndex):
        subset.index = pd.to_datetime(subset.index, errors="coerce")
    subset.index.name = "Date"
    subset.to_csv(target, index=True, date_format="%Y-%m-%d", encoding="utf-8-sig")
    return target


def build_diagnostic_report(
    symbol: str,
    df: pd.DataFrame,
    engineer: OHLCVFeatureEngineer,
    risk_free_rate: float = RISK_FREE_RATE,
) -> str:
    """Build the comprehensive mathematical diagnostic report for console output."""
    del (
        engineer
    )  # Kept in signature for caller compatibility; shared get_quant_metrics runs pipeline.
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

    lines = [
        "【资产数学诊断报告】",
        f"资产类别: {asset_label}",
        f"标的代码: {symbol}",
        f"数据源 / 计价币种 / 复权口径: {data_source} | {ccy_text} | {adjust_text}",
        "样本区间: %s 至 %s (%d 条观测)"
        % (
            df.index[0].strftime("%Y-%m-%d"),
            df.index[-1].strftime("%Y-%m-%d"),
            len(df),
        ),
        "%s: %s"
        % (
            "水平相对变化" if metrics["series_kind"] == "yield_level" else "期间累计收益率",
            _format_percent(metrics["cumulative_return"], signed=True),
        ),
        "年化收益率 (CAGR): %s" % _format_percent(metrics["cagr"]),
        "年化波动率: %s" % _format_percent(metrics["annualized_volatility"]),
        "最大回撤: %s" % _format_percent(-abs(float(metrics["max_drawdown"]))),
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
    if basis_note:
        lines.append(f"口径说明: {basis_note}")
    if metrics.get("metric_disclaimer"):
        lines.append(f"风险提示: {metrics['metric_disclaimer']}")
    return "\n".join(lines)


def run_analysis(
    asset_type: str = "etf",
    symbol: str | None = None,
    start_date: object | None = None,
    end_date: object | None = None,
    *,
    adjust: str | None = None,
    proxy_url: str | None = None,
    base_currency: str | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
    save_chart: str | None = None,
    export_csv: str | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Run the end-to-end quantitative analysis workflow across all supported asset types."""
    apply_default_network_proxy_policy()
    asset_key = str(asset_type).strip().lower()
    if asset_key not in SUPPORTED_ASSET_TYPES:
        raise ValueError(
            "Unsupported asset_type: %r. Choose from: %s"
            % (asset_type, ", ".join(SUPPORTED_ASSET_TYPES))
        )

    sym = (
        str(symbol).strip()
        if symbol is not None and str(symbol).strip()
        else DEFAULT_SYMBOLS[asset_key]
    )
    resolved_adjust = resolve_adjustment(asset_key, adjust)
    start_ts, end_ts = resolve_analysis_date_window(
        start_date,
        end_date,
        lookback_years=LOOKBACK_YEARS,
    )

    if asset_key in CHINA_EQUITY_ASSET_TYPES:
        raw_df = load_china_equity_ohlcv(
            asset_key,
            sym,
            start_ts,
            end_ts,
            adjust=resolved_adjust,
        )
    else:
        raw_df = load_asset_data(
            asset_key,
            sym,
            start_ts,
            end_ts,
            proxy_url=proxy_url,
            adjust=resolved_adjust if adjust is not None else None,
        )

    raw_df.attrs.setdefault("asset_type", asset_key)
    raw_df.attrs.setdefault("symbol", sym)
    if base_currency is not None:
        raw_df = _convert_ohlcv_to_base_currency(
            raw_df,
            asset_key,
            sym,
            base_currency,
            proxy_url=proxy_url,
        )

    trading_days = get_asset_trading_days(asset_key)
    engineer = OHLCVFeatureEngineer()
    featured_df = engineer.transform(raw_df, trading_days=trading_days)
    featured_df.attrs["symbol"] = sym
    featured_df.attrs["asset_type"] = asset_key
    featured_df.attrs["analysis_start"] = start_ts.strftime("%Y-%m-%d")
    featured_df.attrs["analysis_end"] = end_ts.strftime("%Y-%m-%d")
    if "adjust" not in featured_df.attrs and resolved_adjust:
        featured_df.attrs["adjust"] = resolved_adjust

    visualizer = MplfinanceVisualizer()
    figure = visualizer.plot(featured_df, show=show, savepath=save_chart)
    csv_path = _export_ohlcv_csv(raw_df, export_csv) if export_csv is not None else None
    report = build_diagnostic_report(
        symbol=sym,
        df=featured_df,
        engineer=engineer,
        risk_free_rate=risk_free_rate,
    )
    print(report)

    return {
        "raw_df": raw_df,
        "featured_df": featured_df,
        "figure": figure,
        "report": report,
        "saved_csv": str(csv_path) if csv_path is not None else None,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="全资产定量分析 CLI：支持 13 类资产的数据抓取、口径切换、指标诊断与图表导出。",
    )
    parser.add_argument(
        "--asset-type",
        choices=SUPPORTED_ASSET_TYPES,
        default="etf",
        metavar="TYPE",
        help=f"资产类型（默认 etf）；可选: {', '.join(SUPPORTED_ASSET_TYPES)}",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="标的代码（缺省使用对应资产类别默认示例，如 510300 / AAPL / GC / BTC/USDT）",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="开始日期 YYYY-MM-DD；默认约为结束日期往前 3 年",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="结束日期 YYYY-MM-DD；默认今天",
    )
    parser.add_argument(
        "--adjust",
        default=None,
        help="历史行情复权口径（如 hfq / qfq / 空字符串 '' 表示不复权；按资产接口实际支持校验）",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="网络代理地址（如 http://127.0.0.1:7890），用于加密货币或跨市场数据源",
    )
    parser.add_argument(
        "--base-currency",
        choices=("CNY", "USD"),
        default=None,
        help="可选单标的计价折算目标货币（CNY 或 USD）",
    )
    parser.add_argument(
        "--save-chart",
        default=None,
        help="可选图表输出路径（如 outputs/chart.png）",
    )
    parser.add_argument(
        "--export-csv",
        default=None,
        help="可选标准化 OHLCV CSV 导出路径（如 outputs/ohlcv.csv）",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="仅生成/保存图表而不阻塞弹窗显示（适合无头脚本与批处理）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _parse_args(argv)
    try:
        run_analysis(
            asset_type=args.asset_type,
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            adjust=args.adjust,
            proxy_url=args.proxy,
            base_currency=args.base_currency,
            save_chart=args.save_chart,
            export_csv=args.export_csv,
            show=not args.no_show,
        )
        return 0
    except (ImportError, DataFetchError, ValueError, RuntimeError, TypeError) as exc:
        print(format_diagnosis_user_message(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
