"""Agent-friendly CLI (`quant`) for automated financial retrieval, comparison, and portfolio research."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from artifacts import (
    build_artifact_manifest,
    prepare_export_frame,
)
from asset_catalog import (
    DEFAULT_SYMBOLS,
    SUPPORTED_ASSET_TYPES,
    get_asset_trading_days,
    resolve_adjustment,
)
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
from data_sources.errors import DataFetchError
from data_sources.network import apply_default_network_proxy_policy, classify_network_exception
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

_MANAGED_FRAME_BASENAMES: tuple[str, ...] = (
    "prices",
    "returns",
    "correlation",
    "covariance",
    "rolling_metrics",
    "weights",
    "frontier",
    "backtest",
    "weight_history",
)

_MANAGED_OUTPUT_FILENAMES: frozenset[str] = frozenset(
    {
        "result.json",
        "data.parquet",
        "data.csv",
        "report.md",
        "report.html",
        *(f"{name}.parquet" for name in _MANAGED_FRAME_BASENAMES),
        *(f"{name}.csv" for name in _MANAGED_FRAME_BASENAMES),
    }
)

_MANAGED_CHART_TYPES: frozenset[str] = frozenset(
    {
        "price",
        "normalized",
        "drawdown",
        "rolling",
        "correlation",
        "backtest",
        "weights",
        "frontier",
        "weight_history",
    }
)

_MANAGED_CHART_SUFFIXES: frozenset[str] = frozenset({".html", ".png", ".svg"})


class _CliUsageError(ValueError):
    """Raised when CLI syntax or flag parsing fails so stderr can emit structured JSON."""


class _ArtifactIOError(OSError):
    """Raised when writing output files fails after fetch/analysis succeeds."""


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


def _parse_explicit_asset_token(raw_token: str, *, asset_type: str | None) -> tuple[str, str]:
    """Resolve an explicit asset token strictly without silent fallback guesses."""
    cleaned = str(raw_token or "").strip()
    if not cleaned:
        raise ValueError("Symbol must be a non-empty string.")

    for sep in (":", "|"):
        if sep in cleaned:
            prefix, remainder = cleaned.split(sep, 1)
            prefix_key = prefix.strip().lower()
            remainder_clean = remainder.strip()
            if not prefix_key or not remainder_clean:
                raise ValueError(
                    f"Malformed asset identifier {cleaned!r}: expected TYPE{sep}SYMBOL with non-empty parts."
                )
            if prefix_key not in SUPPORTED_ASSET_TYPES:
                raise ValueError(
                    f"Unknown asset type prefix {prefix_key!r} in {cleaned!r}; "
                    f"supported: {', '.join(SUPPORTED_ASSET_TYPES)}."
                )
            if asset_type is not None and prefix_key != asset_type:
                raise ValueError(
                    f"Conflicting asset_type: --asset-type={asset_type!r} vs prefix {prefix_key!r}."
                )
            return prefix_key, remainder_clean

    if asset_type is None:
        raise ValueError(
            f"Ambiguous bare symbol {cleaned!r}: specify -t/--asset-type or prefix as TYPE:SYMBOL "
            f"(e.g. etf:{cleaned}, us_stock:{cleaned})."
        )
    return asset_type, cleaned


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

    return _parse_explicit_asset_token(raw_symbol, asset_type=asset_type)


def _split_multi_asset_input(raw_item: str) -> list[str]:
    tokens: list[str] = []
    for line in str(raw_item).splitlines():
        for seg in line.split(";"):
            part = seg.strip()
            if not part:
                continue
            if "," not in part:
                tokens.append(part)
                continue
            comma_parts = [p.strip() for p in part.split(",")]
            if any(not p for p in comma_parts):
                raise ValueError(f"Empty token in comma-separated asset list: {part!r}")
            # Support single legacy `TYPE,SYMBOL` leg notation when unambiguous.
            if (
                len(comma_parts) == 2
                and ":" not in comma_parts[0]
                and "|" not in comma_parts[0]
                and ":" not in comma_parts[1]
                and "|" not in comma_parts[1]
                and comma_parts[0].lower() in SUPPORTED_ASSET_TYPES
                and comma_parts[1].lower() not in SUPPORTED_ASSET_TYPES
            ):
                tokens.append(f"{comma_parts[0].lower()}:{comma_parts[1]}")
            else:
                tokens.extend(comma_parts)
    return tokens


def _parse_multi_asset_specs(
    assets_pos: Sequence[str],
    assets_opts: Sequence[str],
    *,
    asset_type: str | None,
    adjust: str | None,
) -> tuple[AssetSpec, ...]:
    raw_items = [*assets_pos, *assets_opts]
    tokens: list[str] = []
    for item in raw_items:
        tokens.extend(_split_multi_asset_input(item))

    if not tokens:
        raise ValueError(
            "At least 2 asset specs are required (e.g. etf:510300 us_stock:SPY or --assets ...)."
        )

    specs: list[AssetSpec] = []
    for tok in tokens:
        atype, sym = _parse_explicit_asset_token(tok, asset_type=asset_type)
        leg_adjust = resolve_adjustment(atype, adjust) if adjust is not None else None
        specs.append(AssetSpec(asset_type=atype, symbol=sym, adjust=leg_adjust))
    return tuple(specs)


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
    atype, sym = _parse_explicit_asset_token(cleaned, asset_type=asset_type)
    leg_adjust = resolve_adjustment(atype, adjust) if adjust is not None else None
    return AssetSpec(asset_type=atype, symbol=sym, adjust=leg_adjust)


def _parse_custom_weights(raw_weights: str | None) -> tuple[float, ...] | None:
    if raw_weights is None:
        return None
    cleaned = str(raw_weights).strip()
    if not cleaned:
        raise ValueError("--weights cannot be empty when specified.")
    parts = [p for p in cleaned.replace(";", ",").replace(" ", ",").split(",") if p != ""]
    if not parts:
        raise ValueError("--weights must contain numeric weights matching --assets.")
    parsed: list[float] = []
    for p in parts:
        try:
            val = float(p)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric weight {p!r} in --weights.") from exc
        if not math.isfinite(val):
            raise ValueError(f"Non-finite weight {p!r} in --weights.")
        parsed.append(val)
    return tuple(parsed)


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


def _stage_temp_file(
    target_path: Path,
    writer: Callable[[Path], None],
    *,
    suffix: str = ".tmp",
) -> Path:
    parent = target_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target_path.name}.",
            suffix=suffix,
            dir=str(parent),
        )
    except OSError as exc:
        raise _ArtifactIOError(f"Failed staging output file {target_path}: {exc}") from exc

    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        writer(tmp_path)
        return tmp_path
    except OSError as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise _ArtifactIOError(f"Failed writing staged output file {target_path}: {exc}") from exc
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise _ArtifactIOError(f"Failed serializing output file {target_path}: {exc}") from exc


def _atomic_write(target_path: Path, writer: Callable[[Path], None]) -> Path:
    tmp_path = _stage_temp_file(target_path, writer, suffix=".tmp")
    try:
        os.replace(tmp_path, target_path)
        return target_path.resolve()
    except OSError as exc:
        raise _ArtifactIOError(f"Failed writing output file {target_path}: {exc}") from exc
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _resolve_non_symlink_managed_file(resolved_dir: Path, candidate: Path) -> Path | None:
    """Return ``candidate`` only when it is a regular non-symlink file inside ``resolved_dir``."""
    if candidate.is_symlink() or not candidate.is_file():
        return None
    resolved_candidate = candidate.resolve()
    if (
        resolved_candidate != candidate
        or resolved_candidate == resolved_dir
        or not resolved_candidate.is_relative_to(resolved_dir)
    ):
        return None
    return resolved_candidate


def _collect_stale_managed_artifacts(
    out_dir: Path,
    *,
    new_target_resolved: set[Path],
) -> list[Path]:
    """Identify previously generated quant artifacts in ``out_dir`` not produced by the current run."""
    if not out_dir.exists() or not out_dir.is_dir():
        return []
    resolved_dir = out_dir.resolve()
    charts_dir = resolved_dir / "charts"
    candidates: set[Path] = set()

    for name in _MANAGED_OUTPUT_FILENAMES:
        resolved_item = _resolve_non_symlink_managed_file(resolved_dir, resolved_dir / name)
        if resolved_item is not None:
            candidates.add(resolved_item)

    if not charts_dir.is_symlink() and charts_dir.is_dir() and charts_dir.resolve() == charts_dir:
        for p in charts_dir.iterdir():
            if p.stem in _MANAGED_CHART_TYPES and p.suffix.lower() in _MANAGED_CHART_SUFFIXES:
                resolved_chart = _resolve_non_symlink_managed_file(resolved_dir, p)
                if resolved_chart is not None and resolved_chart.parent == charts_dir:
                    candidates.add(resolved_chart)

    return sorted(
        p
        for p in candidates
        if p not in new_target_resolved
        and not p.is_symlink()
        and p != resolved_dir
        and p.is_relative_to(resolved_dir)
    )


def _write_outputs(
    result: AnalysisResult | MultiAssetResult,
    *,
    out_dir: Path | None,
    write_csv: bool,
    export_csv: Path | None,
    include_data: bool,
    compact: bool,
    report_formats: Sequence[str] = (),
    chart_formats: Sequence[str] = (),
    chart_types: Sequence[str] | None = None,
) -> AnalysisResult | MultiAssetResult:
    working_result: AnalysisResult | MultiAssetResult = result
    artifacts: dict[str, str] = {}
    write_tasks: list[tuple[Path, Callable[[Path], None]]] = []
    seen_targets: set[Path] = set()
    manifest_keys_by_target: dict[Path, tuple[str, str]] = {}

    def _enqueue(
        target: Path,
        writer: Callable[[Path], None],
        *,
        artifact_key: str | None = None,
        rel_path: str | None = None,
    ) -> Path:
        resolved = target.resolve()
        if resolved not in seen_targets:
            seen_targets.add(resolved)
            write_tasks.append((target, writer))
        if artifact_key is not None and rel_path is not None:
            manifest_keys_by_target[resolved] = (artifact_key, rel_path)
        return resolved

    def _enqueue_bytes(target: Path, content: bytes) -> Path:
        return _enqueue(target, lambda tmp, data=content: tmp.write_bytes(data))

    def _parquet_writer(df: pd.DataFrame) -> Callable[[Path], None]:
        write_idx = not isinstance(df.index, pd.RangeIndex)
        return lambda tmp: df.to_parquet(tmp, engine="pyarrow", index=write_idx)

    def _csv_writer(df: pd.DataFrame) -> Callable[[Path], None]:
        write_idx = not isinstance(df.index, pd.RangeIndex)
        return lambda tmp: df.to_csv(tmp, index=write_idx, encoding="utf-8-sig")

    out_dir_resolved = out_dir.resolve() if out_dir is not None else None

    if out_dir is not None and out_dir_resolved is not None:
        result_target = out_dir / "result.json"
        parquet_target = out_dir / "data.parquet"
        primary_df = prepare_export_frame(working_result.data_frame)

        artifacts["output_dir"] = "."
        artifacts["result_json"] = "result.json"
        artifacts["data_parquet"] = "data.parquet"
        _enqueue(
            parquet_target,
            _parquet_writer(primary_df),
            artifact_key="data_parquet",
            rel_path="data.parquet",
        )

        if write_csv:
            csv_target = out_dir / "data.csv"
            artifacts["data_csv"] = "data.csv"
            _enqueue(
                csv_target,
                _csv_writer(primary_df),
                artifact_key="data_csv",
                rel_path="data.csv",
            )

        if isinstance(working_result, MultiAssetResult):
            for frame_name, raw_frame in working_result.frames.items():
                prep = prepare_export_frame(raw_frame)
                pq_rel = f"{frame_name}.parquet"
                pq_path = out_dir / pq_rel
                artifacts[f"{frame_name}_parquet"] = pq_rel
                _enqueue(
                    pq_path,
                    _parquet_writer(prep),
                    artifact_key=f"{frame_name}_parquet",
                    rel_path=pq_rel,
                )
                if write_csv:
                    cs_rel = f"{frame_name}.csv"
                    cs_path = out_dir / cs_rel
                    artifacts[f"{frame_name}_csv"] = cs_rel
                    _enqueue(
                        cs_path,
                        _csv_writer(prep),
                        artifact_key=f"{frame_name}_csv",
                        rel_path=cs_rel,
                    )

        chart_artifacts: tuple[Any, ...] = ()
        skipped_charts: dict[str, str] = {}
        if chart_formats:
            from reporting.chart_renderer import render_charts

            chart_out = render_charts(
                working_result,
                out_dir,
                formats=chart_formats,
                chart_types=chart_types,
                file_writer=_enqueue_bytes,
            )
            chart_artifacts = chart_out.charts
            skipped_charts = dict(chart_out.skipped)
            for ch in chart_artifacts:
                ch_resolved = Path(ch.path).resolve()
                ch_rel = ch_resolved.relative_to(out_dir_resolved).as_posix()
                artifacts[ch.artifact_key] = ch_rel
                manifest_keys_by_target[ch_resolved] = (ch.artifact_key, ch_rel)
            if chart_out.warnings:
                merged_warns = list(working_result.warnings)
                for warn_msg in chart_out.warnings:
                    if warn_msg not in merged_warns:
                        merged_warns.append(warn_msg)
                working_result = replace(working_result, warnings=tuple(merged_warns))

        if report_formats:
            from reporting.report_renderer import render_report

            rep_out = render_report(
                working_result,
                out_dir,
                formats=report_formats,
                charts=chart_artifacts,
                skipped_charts=skipped_charts,
                file_writer=_enqueue_bytes,
            )
            for rep in rep_out.reports:
                rep_resolved = Path(rep.path).resolve()
                rep_rel = rep_resolved.relative_to(out_dir_resolved).as_posix()
                artifacts[rep.artifact_key] = rep_rel
                manifest_keys_by_target[rep_resolved] = (rep.artifact_key, rep_rel)

    if export_csv is not None:
        primary_csv_df = prepare_export_frame(working_result.data_frame)
        csv_resolved = _enqueue(export_csv, _csv_writer(primary_csv_df))
        if out_dir_resolved is None:
            artifacts["export_csv"] = str(csv_resolved)
        elif csv_resolved.is_relative_to(out_dir_resolved):
            csv_rel = csv_resolved.relative_to(out_dir_resolved).as_posix()
            artifacts["export_csv"] = csv_rel
            manifest_keys_by_target[csv_resolved] = ("export_csv", csv_rel)

    if out_dir is not None:
        seen_targets.add((out_dir / "result.json").resolve())
        final_result = working_result.with_artifacts(artifacts)
    else:
        final_result = working_result.with_artifacts(artifacts)
        if not write_tasks:
            return final_result

    stale_paths = (
        _collect_stale_managed_artifacts(out_dir, new_target_resolved=seen_targets)
        if out_dir is not None
        else []
    )

    staged_pairs: list[tuple[Path, Path]] = []
    staged_by_resolved: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    preexisting_targets: set[Path] = set()
    committed_targets: list[Path] = []
    removed_stale: list[Path] = []

    try:
        # Phase 1: stage data/chart/report files first, compute SHA-256 manifest, stage result.json last.
        for target, writer in write_tasks:
            tmp_path = _stage_temp_file(target, writer, suffix=".stage.tmp")
            staged_pairs.append((tmp_path, target))
            staged_by_resolved[target.resolve()] = tmp_path

        if out_dir is not None:
            manifest_inputs = {
                art_key: (rel_path, staged_by_resolved[res_path])
                for res_path, (art_key, rel_path) in manifest_keys_by_target.items()
                if res_path in staged_by_resolved
            }
            manifest = build_artifact_manifest(manifest_inputs)
            final_result = working_result.with_artifacts(artifacts, artifact_manifest=manifest)
            result_target = out_dir / "result.json"
            try:
                payload = final_result.to_dict(include_data=include_data)
                serialized = json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=None if compact else 2,
                    allow_nan=False,
                )
            except Exception as exc:
                raise _ArtifactIOError(f"Failed serializing result.json payload: {exc}") from exc
            result_tmp = _stage_temp_file(
                result_target,
                lambda tmp, text=serialized + "\n": tmp.write_text(text, encoding="utf-8"),
                suffix=".stage.tmp",
            )
            staged_pairs.append((result_tmp, result_target))

        # Phase 2: back up pre-existing targets and stale artifacts on same filesystem.
        staged_resolved = {t.resolve() for _, t in staged_pairs}
        all_paths = [target for _, target in staged_pairs] + stale_paths
        for path in all_paths:
            if path.is_file():
                resolved_p = path.resolve()
                if resolved_p in staged_resolved:
                    preexisting_targets.add(resolved_p)
                if resolved_p not in backups:
                    bak_path = _stage_temp_file(
                        path,
                        lambda tmp, src=path: shutil.copy2(src, tmp),
                        suffix=".bak.tmp",
                    )
                    backups[resolved_p] = bak_path

        # Phase 3: commit staged replacements and purge stale managed artifacts atomically.
        for idx, (tmp_path, target) in enumerate(staged_pairs):
            os.replace(tmp_path, target)
            staged_pairs[idx] = (Path(""), target)
            committed_targets.append(target)

        for stale in stale_paths:
            if stale.exists():
                stale.unlink()
                removed_stale.append(stale)

        if out_dir is not None:
            charts_dir = out_dir / "charts"
            if not charts_dir.is_symlink() and charts_dir.is_dir():
                try:
                    if not any(charts_dir.iterdir()):
                        charts_dir.rmdir()
                except OSError:
                    pass

        return final_result
    except _ArtifactIOError:
        # Roll back only files that were actually committed/unlinked.
        _rollback_multi_file_commit(
            committed_targets=committed_targets,
            removed_stale=removed_stale,
            preexisting_targets=preexisting_targets,
            backups=backups,
        )
        raise
    except Exception as exc:
        _rollback_multi_file_commit(
            committed_targets=committed_targets,
            removed_stale=removed_stale,
            preexisting_targets=preexisting_targets,
            backups=backups,
        )
        raise _ArtifactIOError(f"Failed committing output artifacts cleanly: {exc}") from exc
    finally:
        for tmp_path, _ in staged_pairs:
            if tmp_path != Path("") and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        for bak_path in backups.values():
            if bak_path.exists():
                try:
                    bak_path.unlink()
                except OSError:
                    pass


def _rollback_multi_file_commit(
    *,
    committed_targets: Sequence[Path],
    removed_stale: Sequence[Path],
    preexisting_targets: set[Path],
    backups: dict[Path, Path],
) -> None:
    """Restore any modified or deleted files from same-filesystem backups on commit failure."""
    for target in reversed(committed_targets):
        resolved_t = target.resolve()
        try:
            if resolved_t in preexisting_targets:
                bak = backups.get(resolved_t)
                if bak is not None and bak.exists():
                    os.replace(bak, target)
                    backups.pop(resolved_t, None)
            elif target.exists():
                target.unlink()
        except OSError:
            pass

    for stale in removed_stale:
        resolved_s = stale.resolve()
        bak = backups.get(resolved_s)
        if bak is not None and bak.exists():
            try:
                os.replace(bak, stale)
                backups.pop(resolved_s, None)
            except OSError:
                pass


def _classify_cli_exception(exc: BaseException) -> tuple[str, int]:
    if isinstance(exc, _CliUsageError):
        return "caller", EXIT_USAGE_OR_CALLER
    if isinstance(exc, _ArtifactIOError):
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
            apply_default_network_proxy_policy()

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
            specs = _parse_multi_asset_specs(
                args.assets_pos,
                args.assets_opts,
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
            apply_default_network_proxy_policy()
            result = compare_assets(
                cmp_req,
                loader=loader,
                fx_loader=fx_loader,
                usdt_usd_loader=usdt_usd_loader,
            )
        else:
            specs = _parse_multi_asset_specs(
                args.assets_pos,
                args.assets_opts,
                asset_type=args.asset_type,
                adjust=args.adjust,
            )
            bench_spec = _parse_benchmark_spec(
                args.benchmark,
                asset_type=args.asset_type,
                adjust=args.adjust,
            )
            custom_w = _parse_custom_weights(args.weights)
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
            apply_default_network_proxy_policy()
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

        final_result = _write_outputs(
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
