"""Agent-friendly CLI (`quant`) for automated financial retrieval, comparison, and portfolio research."""

from __future__ import annotations

import argparse
import json
import os as os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from artifacts.store import collect_stale_managed_artifacts as _collect_stale_managed_artifacts  # noqa: F401 - compatibility
from artifacts.transaction import ArtifactIOError
from asset_catalog import (
    DEFAULT_SYMBOLS,
    SUPPORTED_ASSET_TYPES,
    get_asset_trading_days,
    resolve_adjustment,
)
from asset_inputs import parse_asset_list, parse_asset_token, parse_weights
from asset_loader import DEFAULT_REPOSITORY
from asset_resolver import format_diagnosis_user_message, resolve_analysis_date_window
from contracts import (
    SCHEMA_VERSION,
    SOFTWARE_VERSION,
    SUPPORTED_CALENDAR_MODES,
    SUPPORTED_COV_METHODS,
    SUPPORTED_PORTFOLIO_STRATEGIES,
    SUPPORTED_REBALANCE_FREQS,
    SUPPORTED_RETURN_CONVENTIONS,
    SUPPORTED_VALIDATION_MODES,
    AnalysisRequest,
    AnalysisResult,
    AssetSpec,
    CompareRequest,
    MultiAssetResult,
    PortfolioRequest,
)
from data_sources.errors import DataFetchError, SnapshotError
from data_sources.network import apply_default_network_proxy_policy, classify_network_exception
from data_sources.repository import QuoteRepository
from data_sources.research import ResearchRepository
from reporting.output import write_outputs
from services.analysis_service import AssetLoaderFn, analyze_single_asset, fetch_single_asset
from services.multi_asset_service import analyze_portfolio, compare_assets

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_USAGE_OR_CALLER = 2
EXIT_EMPTY_DATA = 3
EXIT_NETWORK = 4
EXIT_UPSTREAM = 5

_CATEGORY_TO_EXIT_CODE: dict[str, int] = {
    "caller": EXIT_USAGE_OR_CALLER,
    "empty": EXIT_EMPTY_DATA,
    "transport": EXIT_NETWORK,
    "upstream": EXIT_UPSTREAM,
    "io_error": EXIT_UNEXPECTED,
    "unexpected": EXIT_UNEXPECTED,
}

_TRANSPORT_MODULES = (
    "httpx",
    "httpcore",
    "requests",
    "urllib3",
    "ccxt",
    "yfinance",
    "akshare",
    "curl_cffi",
    "aiohttp",
    "urllib.error",
    "http.client",
    "ssl",
    "socket",
)


class _CliUsageError(ValueError):
    """Raised when CLI syntax or flag parsing fails so stderr can emit structured JSON."""


class _QuantArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that raises ``_CliUsageError`` instead of emitting unstructured exits."""

    def error(self, message: str) -> Any:  # type: ignore[override]
        raise _CliUsageError(message)


def _add_shared_runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_base_currency: str,
) -> None:
    parser.add_argument(
        "-t",
        "--asset-type",
        choices=SUPPORTED_ASSET_TYPES,
        default=None,
        metavar="TYPE",
        help=f"Asset type ({', '.join(SUPPORTED_ASSET_TYPES)})",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date YYYY-MM-DD (default: ~3 years before --end)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--adjust",
        default=None,
        help="Price adjustment policy (hfq, qfq, or '' where supported)",
    )
    parser.add_argument(
        "--base-currency",
        choices=("LOCAL", "CNY", "USD"),
        default=default_base_currency,
        help=f"Reporting base currency (LOCAL, CNY, USD; default: {default_base_currency})",
    )
    parser.add_argument(
        "--allow-usdt-peg",
        action="store_true",
        help="Allow explicit 1:1 USDT=USD peg when converting crypto quotes",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="HTTP/SOCKS proxy URL (e.g. http://127.0.0.1:7890)",
    )
    parser.add_argument(
        "--data-mode",
        choices=("online", "prefer_local", "offline"),
        help="Use fresh quotes, prefer a published snapshot, or replay without network access",
    )
    parser.add_argument(
        "--snapshot-dir", type=Path, metavar="DIR", help="Portable research input store"
    )
    parser.add_argument("--snapshot-id", help="Pin an immutable input manifest for offline replay")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass local persistent disk cache and force fresh upstream fetch",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="DIR",
        help="Output directory where transactional result.json and Parquet artifacts are written",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write companion CSV files alongside Parquet outputs in --output directory",
    )
    parser.add_argument(
        "--export-csv",
        default=None,
        metavar="PATH",
        help="Optional standalone CSV export path for primary output series",
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--include-data",
        dest="include_data",
        action="store_true",
        default=None,
        help="Embed primary row-level records in JSON output under 'data'",
    )
    data_group.add_argument(
        "--no-data",
        dest="include_data",
        action="store_false",
        help="Omit inline row-level 'data' records in JSON output",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit single-line compact JSON instead of pretty-printed JSON",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-error progress and warning lines on stderr",
    )


def _add_reporting_arguments(parser: argparse.ArgumentParser, *, command: str) -> None:
    """Register on-demand report and chart rendering flags on an analysis subcommand."""
    parser.add_argument(
        "--report",
        default=None,
        metavar="FORMATS",
        help="Generate readable report(s): md, html, all, none (requires -o/--output)",
    )
    parser.add_argument(
        "--charts",
        default=None,
        metavar="FORMATS",
        help="Generate charts under <output>/charts: html, png, svg, all, none (requires -o/--output)",
    )
    parser.add_argument(
        "--chart-types",
        default=None,
        metavar="TYPES",
        help=f"Comma-separated chart types for {command} (default: auto; e.g. price,drawdown)",
    )


def _add_single_asset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "symbol_pos",
        nargs="?",
        default=None,
        metavar="SYMBOL",
        help="Optional TYPE:SYMBOL shorthand (e.g. ashare:600519) or bare SYMBOL when -t is set",
    )
    parser.add_argument(
        "-s",
        "--symbol",
        dest="symbol_opt",
        default=None,
        help="Target symbol code (requires -t/--asset-type unless prefixed with TYPE:)",
    )
    _add_shared_runtime_arguments(parser, default_base_currency="LOCAL")


def _add_multi_asset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "assets_pos",
        nargs="*",
        default=(),
        metavar="ASSET",
        help="Target assets (e.g. etf:510300 us_stock:SPY, or bare SYMBOL with -t/--asset-type)",
    )
    parser.add_argument(
        "-a",
        "--assets",
        dest="assets_opts",
        action="append",
        default=[],
        metavar="ASSETS",
        help="Comma/semicolon-separated asset specs (repeatable, e.g. --assets etf:510300,us_stock:SPY)",
    )
    _add_shared_runtime_arguments(parser, default_base_currency="CNY")
    parser.add_argument(
        "--calendar-mode",
        choices=SUPPORTED_CALENDAR_MODES,
        default="common_trading_dates",
        help="Cross-market calendar alignment mode (default: common_trading_dates)",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        metavar="BENCHMARK",
        help="Optional benchmark asset spec (e.g. etf:510300 or us_stock:SPY)",
    )
    parser.add_argument(
        "--rf",
        "--risk-free-rate",
        dest="risk_free_rate",
        type=float,
        default=0.016,
        help="Annual risk-free rate in decimal (e.g. 0.02 for 2%%; default: 0.016)",
    )
    parser.add_argument(
        "--trading-days",
        type=int,
        default=None,
        help="Explicit annualization periods per year (defaults to aligned calendar frequency)",
    )
    parser.add_argument(
        "--cov-method",
        choices=SUPPORTED_COV_METHODS,
        default="ledoit_wolf",
        help="Covariance estimator (ledoit_wolf, sample, diagonal; default: ledoit_wolf)",
    )


def _build_parser() -> _QuantArgumentParser:
    parser = _QuantArgumentParser(
        prog="quant",
        description="Automated quantitative data retrieval, multi-asset comparison, and portfolio CLI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"quant {SOFTWARE_VERSION} (schema {SCHEMA_VERSION})",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch, validate, currency-normalize, and export clean OHLCV series + metadata",
        description="Retrieve clean OHLCV data, provenance, and QualityReport without charts.",
    )
    _add_single_asset_arguments(fetch_parser)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Fetch, clean, compute indicators & risk/return metrics, and export structured results",
        description="Run single-asset quantitative analysis and emit result.json + data.parquet.",
    )
    _add_single_asset_arguments(analyze_parser)
    analyze_parser.add_argument(
        "--rf",
        "--risk-free-rate",
        dest="risk_free_rate",
        type=float,
        default=0.016,
        help="Annual risk-free rate in decimal (e.g. 0.02 for 2%%; default: 0.016)",
    )
    analyze_parser.add_argument(
        "--trading-days",
        type=int,
        default=None,
        help="Annualization trading days per year (defaults to asset_type convention)",
    )
    analyze_parser.add_argument(
        "--return-convention",
        choices=SUPPORTED_RETURN_CONVENTIONS,
        default="log",
        help="Return calculation convention: log or simple (default: log)",
    )
    analyze_parser.add_argument(
        "--sma-short",
        type=int,
        default=20,
        help="Short simple moving average window >= 2 (default: 20)",
    )
    analyze_parser.add_argument(
        "--sma-long",
        type=int,
        default=60,
        help="Long simple moving average window > --sma-short (default: 60)",
    )
    analyze_parser.add_argument(
        "--bb-window",
        type=int,
        default=20,
        help="Bollinger Bands rolling window >= 2 (default: 20)",
    )
    analyze_parser.add_argument(
        "--include-momentum",
        action="store_true",
        help="Include RSI-14 and MACD momentum indicator columns in output frame",
    )
    _add_reporting_arguments(analyze_parser, command="analyze")

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple assets across normalized NAV, risk-return, correlation & rolling stats",
        description="Align multiple heterogeneous assets and compute cross-asset statistical comparison.",
    )
    _add_multi_asset_arguments(compare_parser)
    compare_parser.add_argument(
        "--return-convention",
        choices=SUPPORTED_RETURN_CONVENTIONS,
        default="simple",
        help="Return calculation convention for comparison stats & moments (default: simple)",
    )
    compare_parser.add_argument(
        "--rolling-window",
        type=int,
        default=20,
        help="Rolling window length in aligned observations >= 2 (default: 20)",
    )
    _add_reporting_arguments(compare_parser, command="compare")

    portfolio_parser = subparsers.add_parser(
        "portfolio",
        help="Optimize multi-asset portfolios, compute risk contributions, frontier & OOS backtests",
        description="Run causal multi-asset portfolio optimization, SLSQP efficient frontier, and backtesting.",
    )
    _add_multi_asset_arguments(portfolio_parser)
    portfolio_parser.add_argument(
        "--strategy",
        "--objective",
        dest="strategy",
        choices=SUPPORTED_PORTFOLIO_STRATEGIES,
        default="max_sharpe",
        help="Allocation/optimization strategy (default: max_sharpe)",
    )
    portfolio_parser.add_argument(
        "--validation-mode",
        choices=SUPPORTED_VALIDATION_MODES,
        default="train_test",
        help="Validation/backtest protocol (default: train_test)",
    )
    portfolio_parser.add_argument(
        "--rebalance-freq",
        choices=SUPPORTED_REBALANCE_FREQS,
        default="monthly",
        help="Portfolio rebalance frequency (default: monthly)",
    )
    portfolio_parser.add_argument(
        "--tx-bps",
        "--transaction-cost-bps",
        dest="transaction_cost_bps",
        type=float,
        default=10.0,
        help="One-way turnover transaction cost in basis points (default: 10.0)",
    )
    portfolio_parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Lower per-asset long-only weight bound in [0, 1] (default: 0.0)",
    )
    portfolio_parser.add_argument(
        "--max-weight",
        type=float,
        default=1.0,
        help="Upper per-asset weight bound in [0, 1] (default: 1.0)",
    )
    portfolio_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training sample ratio in (0, 1) for train_test split (default: 0.7)",
    )
    portfolio_parser.add_argument(
        "--split-date",
        default=None,
        help="Explicit OOS split date YYYY-MM-DD for train_test mode (overrides --train-ratio)",
    )
    portfolio_parser.add_argument(
        "--train-window",
        type=int,
        default=60,
        help="Training lookback window in periods >= 5 for walk_forward_* modes (default: 60)",
    )
    portfolio_parser.add_argument(
        "--weights",
        default=None,
        metavar="W1,W2,...",
        help="Comma-separated non-negative custom weights matching asset order (requires --strategy custom)",
    )
    portfolio_parser.add_argument(
        "--frontier-points",
        type=int,
        default=25,
        help="Number of exact SLSQP efficient frontier points (0 disables frontier; default: 25)",
    )
    portfolio_parser.add_argument(
        "--num-random-portfolios",
        type=int,
        default=0,
        help="Optional Dirichlet random cloud count cleanly separated from SLSQP frontier (default: 0)",
    )
    portfolio_parser.add_argument(
        "--seed",
        "--random-seed",
        dest="random_seed",
        type=int,
        default=42,
        help="Random seed when --num-random-portfolios > 0 (default: 42)",
    )
    sens_group = portfolio_parser.add_mutually_exclusive_group()
    sens_group.add_argument(
        "--sensitivity",
        dest="include_sensitivity",
        action="store_true",
        default=True,
        help="Include covariance, transaction-cost, and rebalance sensitivity grids (default)",
    )
    sens_group.add_argument(
        "--no-sensitivity",
        dest="include_sensitivity",
        action="store_false",
        help="Skip sensitivity table sweeps for faster single-pass execution",
    )
    _add_reporting_arguments(portfolio_parser, command="portfolio")

    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.command:
        raise _CliUsageError(
            "Subcommand required: choose 'fetch', 'analyze', 'compare', or 'portfolio'."
        )
    return args


def _resolve_asset_and_symbol(
    *,
    asset_type: str | None,
    symbol_pos: str | None,
    symbol_opt: str | None,
) -> tuple[str, str]:
    if symbol_pos is not None and symbol_opt is not None:
        raise ValueError("Cannot specify both positional SYMBOL and --symbol.")

    raw_symbol = symbol_opt if symbol_opt is not None else symbol_pos
    if raw_symbol is None:
        resolved_type = asset_type or "etf"
        return resolved_type, DEFAULT_SYMBOLS.get(resolved_type, "510300")

    spec = parse_asset_token(raw_symbol, asset_type=asset_type)
    return spec.asset_type, spec.symbol


def _parse_benchmark_spec(
    raw_benchmark: str | None,
    *,
    asset_type: str | None,
    adjust: str | None,
) -> AssetSpec | None:
    if raw_benchmark is None:
        return None
    cleaned = str(raw_benchmark).strip()
    if not cleaned:
        raise ValueError("--benchmark cannot be empty when specified.")
    return parse_asset_token(cleaned, asset_type=asset_type, adjust=adjust)


def _validate_output_targets(
    *,
    output: str | None,
    write_csv: bool,
    export_csv: str | None,
) -> tuple[Path | None, Path | None]:
    out_dir: Path | None = None
    if output is not None:
        raw_out = str(output).strip()
        if not raw_out:
            raise ValueError("--output directory path cannot be empty.")
        out_dir = Path(raw_out).expanduser()
        if out_dir.exists() and not out_dir.is_dir():
            raise ValueError(f"--output path exists and is not a directory: {out_dir}")
    elif write_csv:
        raise ValueError("--csv requires --output <dir> (or use --export-csv <path>).")

    csv_path: Path | None = None
    if export_csv is not None:
        raw_csv = str(export_csv).strip()
        if not raw_csv:
            raise ValueError("--export-csv path cannot be empty.")
        csv_path = Path(raw_csv).expanduser()
        if csv_path.exists() and csv_path.is_dir():
            raise ValueError(f"--export-csv path exists and is a directory: {csv_path}")

    return out_dir, csv_path


def _classify_cli_exception(exc: BaseException) -> tuple[str, int]:
    if isinstance(exc, _CliUsageError):
        return "caller", EXIT_USAGE_OR_CALLER
    if isinstance(exc, (ArtifactIOError, SnapshotError)):
        return "io_error", EXIT_UNEXPECTED
    if isinstance(exc, DataFetchError):
        category = classify_network_exception(exc)
        return category, _CATEGORY_TO_EXIT_CODE.get(category, EXIT_UNEXPECTED)
    if isinstance(exc, (ValueError, TypeError, ImportError)):
        return "caller", EXIT_USAGE_OR_CALLER
    mod = (type(exc).__module__ or "").lower()
    if mod.startswith(_TRANSPORT_MODULES) or isinstance(
        exc, (TimeoutError, ConnectionError, EOFError)
    ):
        category = classify_network_exception(exc)
        return category, _CATEGORY_TO_EXIT_CODE.get(category, EXIT_UPSTREAM)
    if type(exc) is OSError and any(
        tok in str(exc).lower()
        for tok in ("network", "socket", "connection", "offline", "timed out", "proxy")
    ):
        return "transport", EXIT_NETWORK
    return "unexpected", EXIT_UNEXPECTED


def _emit_error_json(
    *,
    command: str | None,
    exc: BaseException,
    category: str,
    exit_code: int,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "software_version": SOFTWARE_VERSION,
        "command": command,
        "status": "error",
        "error": {
            "category": category,
            "exit_code": exit_code,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "hint": format_diagnosis_user_message(exc),
        },
    }
    sys.stderr.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    sys.stderr.flush()


def _log_stderr(message: str, *, quiet: bool) -> None:
    if quiet:
        return
    sys.stderr.write(message.rstrip("\n") + "\n")
    sys.stderr.flush()


def _format_done_log(result: AnalysisResult | MultiAssetResult) -> str:
    if isinstance(result, MultiAssetResult):
        rows = result.data_summary.get("valuation_points", 0)
        return (
            f"[quant] done command={result.command} status={result.status} "
            f"assets={len(result.asset_order)} rows={rows}"
        )
    return (
        f"[quant] done status={result.status} rows={result.quality.total_rows} "
        f"source={result.asset.data_source or 'unknown'} "
        f"cache_hit={result.asset.cache_hit}"
    )


def run_cli_command(
    argv: Sequence[str] | None = None,
    *,
    loader: AssetLoaderFn | None = None,
    repository: QuoteRepository | None = None,
    fx_loader: Any | None = None,
    usdt_usd_loader: Any | None = None,
) -> tuple[int, AnalysisResult | MultiAssetResult | None]:
    """Execute ``quant fetch|analyze|compare|portfolio`` and return ``(exit_code, result)``."""
    command_hint: str | None = None
    if argv:
        first = str(argv[0]).strip()
        if first in {"fetch", "analyze", "compare", "portfolio"}:
            command_hint = first

    try:
        args = _parse_args(argv)
        command_hint = str(args.command)
        quiet = bool(getattr(args, "quiet", False))

        research = None
        if repository is not None and loader is not None:
            raise ValueError("Supply either a repository or a loader.")
        if args.data_mode or args.snapshot_dir or args.snapshot_id:
            mode = args.data_mode or ("offline" if args.snapshot_id else "prefer_local")
            if mode == "offline" and args.refresh:
                raise ValueError("--refresh is incompatible with offline replay.")
            directory = (
                args.snapshot_dir or Path.home() / ".local/share/quantitative_analysis/snapshots"
            )
            research = ResearchRepository(
                directory,
                mode=mode,
                repository=repository or DEFAULT_REPOSITORY,
                snapshot_id=args.snapshot_id,
                asset_loader=loader,
                fx_loader=fx_loader,
                usdt_usd_loader=usdt_usd_loader,
                refresh=bool(args.refresh),
            )
            loader, fx_loader, usdt_usd_loader = research.load, research.fx, research.usdt
        elif repository is not None:
            loader = repository.load
            if fx_loader is None:

                def fx_loader(symbol, start, end):
                    return repository.load("fx", symbol, start, end)

        if research is None or research.mode != "offline":
            apply_default_network_proxy_policy()

        start_ts, end_ts = resolve_analysis_date_window(args.start, args.end)
        out_dir, csv_export_path = _validate_output_targets(
            output=args.output,
            write_csv=bool(args.csv),
            export_csv=args.export_csv,
        )

        report_formats: tuple[str, ...] = ()
        chart_formats: tuple[str, ...] = ()
        chart_types: tuple[str, ...] | None = None
        if args.command in {"analyze", "compare", "portfolio"}:
            raw_report = getattr(args, "report", None)
            raw_charts = getattr(args, "charts", None)
            raw_chart_types = getattr(args, "chart_types", None)
            if raw_report is not None:
                from reporting.report_renderer import parse_report_formats

                report_formats = parse_report_formats(raw_report)
            if raw_charts is not None:
                from reporting.chart_renderer import parse_chart_formats

                chart_formats = parse_chart_formats(raw_charts)
            if raw_chart_types is not None:
                from reporting.chart_renderer import parse_chart_types

                chart_types = parse_chart_types(str(args.command), raw_chart_types)
            if (report_formats or chart_formats) and out_dir is None:
                raise ValueError("--report and --charts require --output <dir>.")

        start_str = start_ts.strftime("%Y-%m-%d")
        end_str = end_ts.strftime("%Y-%m-%d")

        if args.command in ("fetch", "analyze"):
            asset_type, symbol = _resolve_asset_and_symbol(
                asset_type=args.asset_type,
                symbol_pos=args.symbol_pos,
                symbol_opt=args.symbol_opt,
            )
            resolved_adjust = resolve_adjustment(asset_type, args.adjust)
            default_td = get_asset_trading_days(asset_type)
            if args.command == "analyze":
                request: AnalysisRequest | CompareRequest | PortfolioRequest = AnalysisRequest(
                    asset_type=asset_type,
                    symbol=symbol,
                    start_date=start_str,
                    end_date=end_str,
                    adjust=resolved_adjust,
                    base_currency=args.base_currency,
                    allow_usdt_peg=bool(args.allow_usdt_peg),
                    risk_free_rate=float(args.risk_free_rate),
                    trading_days=args.trading_days if args.trading_days is not None else default_td,
                    return_convention=args.return_convention,
                    sma_short=int(args.sma_short),
                    sma_long=int(args.sma_long),
                    bb_window=int(args.bb_window),
                    include_momentum=bool(args.include_momentum),
                    force_refresh=bool(args.refresh),
                    proxy_url=args.proxy,
                )
            else:
                request = AnalysisRequest(
                    asset_type=asset_type,
                    symbol=symbol,
                    start_date=start_str,
                    end_date=end_str,
                    adjust=resolved_adjust,
                    base_currency=args.base_currency,
                    allow_usdt_peg=bool(args.allow_usdt_peg),
                    trading_days=default_td,
                    force_refresh=bool(args.refresh),
                    proxy_url=args.proxy,
                )

            _log_stderr(
                f"[quant] {args.command} {request.asset_type}:{request.symbol} "
                f"({request.start_date}..{request.end_date}, base={request.base_currency})",
                quiet=quiet,
            )

            if args.command == "fetch":
                result: AnalysisResult | MultiAssetResult = fetch_single_asset(
                    request,
                    loader=loader,
                    fx_loader=fx_loader,
                    usdt_usd_loader=usdt_usd_loader,
                )
            else:
                result = analyze_single_asset(
                    request,
                    loader=loader,
                    fx_loader=fx_loader,
                    usdt_usd_loader=usdt_usd_loader,
                )
        elif args.command == "compare":
            specs = parse_asset_list(
                [*args.assets_pos, *args.assets_opts],
                asset_type=args.asset_type,
                adjust=args.adjust,
            )
            bench_spec = _parse_benchmark_spec(
                args.benchmark,
                asset_type=args.asset_type,
                adjust=args.adjust,
            )
            cmp_req = CompareRequest(
                assets=specs,
                start_date=start_str,
                end_date=end_str,
                benchmark=bench_spec,
                calendar_mode=args.calendar_mode,
                base_currency=args.base_currency,
                allow_usdt_peg=bool(args.allow_usdt_peg),
                risk_free_rate=float(args.risk_free_rate),
                trading_days=args.trading_days,
                return_convention=args.return_convention,
                rolling_window=int(args.rolling_window),
                cov_method=args.cov_method,
                force_refresh=bool(args.refresh),
                proxy_url=args.proxy,
            )
            _log_stderr(
                f"[quant] compare {','.join(s.leg_name for s in cmp_req.assets)} "
                f"({cmp_req.start_date}..{cmp_req.end_date}, cal={cmp_req.calendar_mode}, base={cmp_req.base_currency})",
                quiet=quiet,
            )
            result = compare_assets(
                cmp_req,
                loader=loader,
                fx_loader=fx_loader,
                usdt_usd_loader=usdt_usd_loader,
            )
        else:
            specs = parse_asset_list(
                [*args.assets_pos, *args.assets_opts],
                asset_type=args.asset_type,
                adjust=args.adjust,
            )
            bench_spec = _parse_benchmark_spec(
                args.benchmark,
                asset_type=args.asset_type,
                adjust=args.adjust,
            )
            custom_w = parse_weights(args.weights)
            port_req = PortfolioRequest(
                assets=specs,
                start_date=start_str,
                end_date=end_str,
                strategy=args.strategy,
                validation_mode=args.validation_mode,
                benchmark=bench_spec,
                calendar_mode=args.calendar_mode,
                base_currency=args.base_currency,
                allow_usdt_peg=bool(args.allow_usdt_peg),
                risk_free_rate=float(args.risk_free_rate),
                trading_days=args.trading_days,
                cov_method=args.cov_method,
                rebalance_freq=args.rebalance_freq,
                transaction_cost_bps=float(args.transaction_cost_bps),
                min_weight=float(args.min_weight),
                max_weight=float(args.max_weight),
                train_ratio=float(args.train_ratio),
                split_date=args.split_date,
                train_window=int(args.train_window),
                custom_weights=custom_w,
                frontier_points=int(args.frontier_points),
                num_random_portfolios=int(args.num_random_portfolios),
                random_seed=args.random_seed,
                include_sensitivity=bool(args.include_sensitivity),
                force_refresh=bool(args.refresh),
                proxy_url=args.proxy,
            )
            _log_stderr(
                f"[quant] portfolio {','.join(s.leg_name for s in port_req.assets)} "
                f"({port_req.start_date}..{port_req.end_date}, strategy={port_req.strategy}, "
                f"val={port_req.validation_mode}, base={port_req.base_currency})",
                quiet=quiet,
            )
            result = analyze_portfolio(
                port_req,
                loader=loader,
                fx_loader=fx_loader,
                usdt_usd_loader=usdt_usd_loader,
            )

        if args.include_data is None:
            include_data = args.command == "fetch" and out_dir is None
        else:
            include_data = bool(args.include_data)

        if research is not None:
            result = replace(
                result, data_summary={**result.data_summary, "snapshot_id": research.finish()}
            )

        final_result = write_outputs(
            result,
            out_dir=out_dir,
            write_csv=bool(args.csv),
            export_csv=csv_export_path,
            include_data=include_data,
            compact=bool(args.compact),
            report_formats=report_formats,
            chart_formats=chart_formats,
            chart_types=chart_types,
        )

        for warning in final_result.warnings:
            _log_stderr(f"[warn] {warning}", quiet=quiet)

        payload = final_result.to_dict(include_data=include_data)
        stdout_json = json.dumps(
            payload,
            ensure_ascii=False,
            indent=None if args.compact else 2,
            allow_nan=False,
        )
        sys.stdout.write(stdout_json + "\n")
        sys.stdout.flush()

        _log_stderr(_format_done_log(final_result), quiet=quiet)
        return EXIT_OK, final_result
    except Exception as exc:
        category, exit_code = _classify_cli_exception(exc)
        _emit_error_json(
            command=command_hint,
            exc=exc,
            category=category,
            exit_code=exit_code,
        )
        return exit_code, None


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint registered for the ``quant`` console script."""
    exit_code, _ = run_cli_command(argv)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
