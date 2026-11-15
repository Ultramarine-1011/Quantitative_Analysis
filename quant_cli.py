"""Agent-friendly CLI (`quant`) for automated financial retrieval and quantitative analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

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
    SUPPORTED_RETURN_CONVENTIONS,
    AnalysisRequest,
    AnalysisResult,
)
from data_sources.errors import DataFetchError
from data_sources.network import apply_default_network_proxy_policy, classify_network_exception
from services.analysis_service import AssetLoaderFn, analyze_single_asset, fetch_single_asset

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


class _ArtifactIOError(OSError):
    """Raised when writing output files fails after fetch/analysis succeeds."""


class _QuantArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that raises ``_CliUsageError`` instead of emitting unstructured exits."""

    def error(self, message: str) -> Any:  # type: ignore[override]
        raise _CliUsageError(message)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "symbol_pos",
        nargs="?",
        default=None,
        metavar="SYMBOL",
        help="Optional symbol (e.g. 510300, AAPL) or TYPE:SYMBOL shorthand (e.g. ashare:600519)",
    )
    parser.add_argument(
        "-s",
        "--symbol",
        dest="symbol_opt",
        default=None,
        help="Target symbol code (defaults to example symbol for --asset-type)",
    )
    parser.add_argument(
        "-t",
        "--asset-type",
        choices=SUPPORTED_ASSET_TYPES,
        default=None,
        metavar="TYPE",
        help=f"Asset type (default: etf); supported: {', '.join(SUPPORTED_ASSET_TYPES)}",
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
        default="LOCAL",
        help="Reporting base currency (LOCAL, CNY, USD; default: LOCAL)",
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
        help="Output directory where atomic result.json and data.parquet will be written",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write data.csv inside --output directory",
    )
    parser.add_argument(
        "--export-csv",
        default=None,
        metavar="PATH",
        help="Optional standalone CSV export path",
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--include-data",
        dest="include_data",
        action="store_true",
        default=None,
        help="Embed row-level OHLCV/feature records in JSON output under 'data'",
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


def _build_parser() -> _QuantArgumentParser:
    parser = _QuantArgumentParser(
        prog="quant",
        description="Automated quantitative data retrieval and diagnostics CLI for AI Agents.",
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
    _add_common_arguments(fetch_parser)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Fetch, clean, compute indicators & risk/return metrics, and export structured results",
        description="Run single-asset quantitative analysis and emit result.json + data.parquet.",
    )
    _add_common_arguments(analyze_parser)
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

    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.command:
        raise _CliUsageError("Subcommand required: choose 'fetch' or 'analyze'.")
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
    explicit_asset_type = asset_type is not None
    resolved_type = asset_type or "etf"

    if raw_symbol is not None:
        cleaned = str(raw_symbol).strip()
        if not cleaned:
            raise ValueError("Symbol must be a non-empty string.")
        if ":" in cleaned:
            prefix, remainder = cleaned.split(":", 1)
            prefix_key = prefix.strip().lower()
            if prefix_key in SUPPORTED_ASSET_TYPES and remainder.strip():
                if explicit_asset_type and prefix_key != resolved_type:
                    raise ValueError(
                        f"Conflicting asset_type: --asset-type={resolved_type!r} "
                        f"vs symbol prefix {prefix_key!r}."
                    )
                resolved_type = prefix_key
                cleaned = remainder.strip()
        return resolved_type, cleaned

    return resolved_type, DEFAULT_SYMBOLS.get(resolved_type, "510300")


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


def _atomic_write(target_path: Path, writer: Callable[[Path], None]) -> Path:
    parent = target_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            dir=str(parent),
        )
    except OSError as exc:
        raise _ArtifactIOError(f"Failed preparing output file {target_path}: {exc}") from exc

    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        writer(tmp_path)
        os.replace(tmp_path, target_path)
        return target_path.resolve()
    except OSError as exc:
        raise _ArtifactIOError(f"Failed writing output file {target_path}: {exc}") from exc
    except Exception as exc:
        raise _ArtifactIOError(f"Failed serializing output file {target_path}: {exc}") from exc
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _write_outputs(
    result: AnalysisResult,
    *,
    out_dir: Path | None,
    write_csv: bool,
    export_csv: Path | None,
    include_data: bool,
    compact: bool,
) -> AnalysisResult:
    artifacts: dict[str, str] = {}

    if out_dir is not None:
        parquet_target = out_dir / "data.parquet"
        result_target = out_dir / "result.json"

        df_to_write = result.data_frame.copy() if result.data_frame is not None else pd.DataFrame()
        df_to_write.attrs = {}
        if isinstance(df_to_write.index, pd.DatetimeIndex) and not df_to_write.index.name:
            df_to_write.index.name = "Date"

        saved_parquet = _atomic_write(
            parquet_target,
            lambda tmp: df_to_write.to_parquet(tmp, engine="pyarrow", index=True),
        )
        artifacts["output_dir"] = str(out_dir.resolve())
        artifacts["result_json"] = str(result_target.resolve())
        artifacts["data_parquet"] = str(saved_parquet)

        if write_csv:
            csv_target = out_dir / "data.csv"
            saved_dir_csv = _atomic_write(
                csv_target,
                lambda tmp: df_to_write.to_csv(tmp, index=True, encoding="utf-8-sig"),
            )
            artifacts["data_csv"] = str(saved_dir_csv)

    if export_csv is not None:
        df_to_csv = result.data_frame.copy() if result.data_frame is not None else pd.DataFrame()
        df_to_csv.attrs = {}
        if isinstance(df_to_csv.index, pd.DatetimeIndex) and not df_to_csv.index.name:
            df_to_csv.index.name = "Date"
        saved_export_csv = _atomic_write(
            export_csv,
            lambda tmp: df_to_csv.to_csv(tmp, index=True, encoding="utf-8-sig"),
        )
        artifacts["export_csv"] = str(saved_export_csv)

    final_result = result.with_artifacts(artifacts)

    if out_dir is not None:
        result_target = out_dir / "result.json"
        payload = final_result.to_dict(include_data=include_data)
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            indent=None if compact else 2,
            allow_nan=False,
        )
        _atomic_write(
            result_target,
            lambda tmp: tmp.write_text(serialized + "\n", encoding="utf-8"),
        )

    return final_result


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


def run_cli_command(
    argv: Sequence[str] | None = None,
    *,
    loader: AssetLoaderFn | None = None,
    fx_loader: Any | None = None,
    usdt_usd_loader: Any | None = None,
) -> tuple[int, AnalysisResult | None]:
    """Execute ``quant fetch`` or ``quant analyze`` and return ``(exit_code, result)``."""
    command_hint: str | None = None
    if argv:
        first = str(argv[0]).strip()
        if first in {"fetch", "analyze"}:
            command_hint = first

    quiet = False
    try:
        args = _parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return (int(code) if isinstance(code, int) else EXIT_OK), None
    except _CliUsageError as exc:
        category, exit_code = _classify_cli_exception(exc)
        _emit_error_json(command=command_hint, exc=exc, category=category, exit_code=exit_code)
        return exit_code, None

    command_hint = args.command
    quiet = bool(getattr(args, "quiet", False))

    try:
        asset_type, symbol = _resolve_asset_and_symbol(
            asset_type=args.asset_type,
            symbol_pos=args.symbol_pos,
            symbol_opt=args.symbol_opt,
        )
        start_ts, end_ts = resolve_analysis_date_window(args.start, args.end)
        out_dir, csv_export_path = _validate_output_targets(
            output=args.output,
            write_csv=bool(args.csv),
            export_csv=args.export_csv,
        )

        resolved_adjust = resolve_adjustment(asset_type, args.adjust)

        default_td = get_asset_trading_days(asset_type)
        if args.command == "analyze":
            request = AnalysisRequest(
                asset_type=asset_type,
                symbol=symbol,
                start_date=start_ts.strftime("%Y-%m-%d"),
                end_date=end_ts.strftime("%Y-%m-%d"),
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
                start_date=start_ts.strftime("%Y-%m-%d"),
                end_date=end_ts.strftime("%Y-%m-%d"),
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
            result = fetch_single_asset(
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

        _log_stderr(
            f"[quant] done status={final_result.status} rows={final_result.quality.total_rows} "
            f"source={final_result.asset.data_source or 'unknown'} "
            f"cache_hit={final_result.asset.cache_hit}",
            quiet=quiet,
        )
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
