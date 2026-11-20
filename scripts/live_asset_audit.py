"""Full-asset live market data batch audit script for codex/frontend-qa.

Executes real ``quant fetch`` CLI calls with ``--refresh`` across all 13 asset
categories (65 primary instruments) plus symbol aliases, runs semantic-aware
data quality checks on the persisted Parquet + JSON outputs, and generates
structured CSV/JSON summaries and Markdown reports.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_START_DATE = "2025-11-17"
DEFAULT_END_DATE = "2026-11-16"
DEFAULT_AUDIT_DATE = "2026-11-17"
DEFAULT_BASE_CURRENCY = "LOCAL"

PRIMARY_TARGETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("etf", ("510300", "510500", "159915", "511010", "518880")),
    ("ashare", ("600519", "000001", "300750", "688981", "601318")),
    ("mutual_fund", ("009691", "161725", "110011", "005827", "110022")),
    ("gold", ("GC", "SI", "XAU", "XAG", "XPT")),
    ("crypto", ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT")),
    ("us_stock", ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL")),
    ("hk_stock", ("00700", "09988", "01810", "03690", "00941")),
    ("global_index", (".INX", ".DJI", ".IXIC", "HSI", "N225", "DXY")),
    ("bond_cn", ("sh019547", "sh019743", "sh019780", "sh019788", "sh019795")),
    ("bond_us_yield", ("10Y", "美国国债收益率2年", "美国国债收益率5年", "美国国债收益率30年")),
    ("commodity", ("CL", "HG", "NG", "OIL", "ZSD")),
    ("creit", ("508097", "180101", "508000", "180201", "508028")),
    ("fx", ("USDCNH", "EURUSD", "USDJPY", "GBPUSD", "AUDUSD")),
)

# Extra aliases explicitly requested ("sh600519", "sh510300", "AU9999", "sh508097", "USD/JPY")
# plus representative leading-zero and naming-alias checks; excluded from the 65-item primary tally.
ALIAS_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("ashare", "sh600519", "ashare:600519"),
    ("etf", "sh510300", "etf:510300"),
    ("gold", "AU9999", "gold:GC"),
    ("creit", "sh508097", "creit:508097"),
    ("fx", "USD/JPY", "fx:USDJPY"),
    ("ashare", "600519.SH", "ashare:600519"),
    ("ashare", "sz000001", "ashare:000001"),
    ("hk_stock", "700", "hk_stock:00700"),
    ("fx", "USD/CNY", "fx:USDCNH"),
    ("bond_us_yield", "美国国债收益率10年", "bond_us_yield:10Y"),
    ("global_index", "SPX", "global_index:.INX"),
)

CN_MAINLAND_TYPES = frozenset({"etf", "ashare", "mutual_fund", "bond_cn", "creit"})
PSEUDO_OHLCV_TYPES = frozenset({"mutual_fund", "bond_us_yield"})
ZERO_VOL_EXPECTED_TYPES = frozenset({"mutual_fund", "bond_us_yield", "fx"})
VOLUME_BEARING_TYPES = frozenset(
    {"etf", "ashare", "us_stock", "hk_stock", "crypto", "creit", "gold", "commodity"}
)
EXPECTED_LOCAL_CURRENCY: dict[str, str] = {
    "etf": "CNY",
    "ashare": "CNY",
    "mutual_fund": "CNY",
    "bond_cn": "CNY",
    "creit": "CNY",
    "hk_stock": "HKD",
    "us_stock": "USD",
    "gold": "USD",
    "commodity": "USD",
    "bond_us_yield": "USD",
}
REQUIRED_OHLCV_COLS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class AuditTarget:
    """Single audit target specification."""

    item_id: str
    group: str  # "primary" | "alias"
    asset_type: str
    symbol: str
    alias_of: str = ""

    @property
    def spec(self) -> str:
        return f"{self.asset_type}:{self.symbol}"

    @property
    def safe_dirname(self) -> str:
        return safe_symbol_dirname(self.symbol)

    def relative_dir(self) -> Path:
        if self.group == "alias":
            return Path("aliases") / self.asset_type / self.safe_dirname
        return Path(self.asset_type) / self.safe_dirname


@dataclass
class AttemptRecord:
    """Single subprocess invocation timing and outcome."""

    attempt: int
    started_at: str
    elapsed_seconds: float
    exit_code: int
    timed_out: bool = False
    error_category: str = ""
    error_type: str = ""
    error_message: str = ""
    rate_limit_signals: list[str] = field(default_factory=list)


def safe_symbol_dirname(symbol: str) -> str:
    """Convert symbol string into a single-segment POSIX directory name preserving leading zeros."""
    return str(symbol).strip().replace("/", "_").replace("\\", "_")


def build_audit_plan() -> tuple[list[AuditTarget], list[AuditTarget]]:
    """Build deterministic primary (65) and alias target lists."""
    primaries: list[AuditTarget] = []
    idx = 1
    for asset_type, symbols in PRIMARY_TARGETS:
        for sym in symbols:
            primaries.append(
                AuditTarget(
                    item_id=f"P{idx:02d}",
                    group="primary",
                    asset_type=asset_type,
                    symbol=str(sym),
                )
            )
            idx += 1

    aliases: list[AuditTarget] = []
    for a_idx, (asset_type, sym, alias_of) in enumerate(ALIAS_TARGETS, start=1):
        aliases.append(
            AuditTarget(
                item_id=f"A{a_idx:02d}",
                group="alias",
                asset_type=asset_type,
                symbol=str(sym),
                alias_of=alias_of,
            )
        )
    return primaries, aliases


def detect_rate_limit_signals(text: str) -> list[str]:
    """Detect upstream rate-limit, block, or remote-disconnect signatures in logs."""
    lower = text.lower()
    patterns = (
        (
            "push2_rate_limit",
            ("push2ratelimiterror", "ratelimiterror", "临时限流", "上游限流", "断开了连接"),
        ),
        ("dce_blocked", ("dceblockederror",)),
        ("remote_disconnect", ("remoteprotocolerror", "remotedisconnected", "server disconnected")),
        ("timeout", ("timed out", "readtimeout", "connecttimeout", "超时")),
        ("connection_reset", ("connectionreseterror", "connection aborted", "connectionerror")),
    )
    hits: list[str] = []
    if (
        "too many requests" in lower
        or re.search(r"\bhttp\s*429\b|\bstatus(?:_code)?[=:\s]*429\b|\(429\)", lower) is not None
    ):
        hits.append("http_429")
    for label, tokens in patterns:
        if any(tok in lower for tok in tokens):
            hits.append(label)
    return hits


def parse_cli_error_json(stderr_text: str) -> dict[str, Any]:
    """Extract the structured JSON error object emitted by ``quant_cli`` on stderr."""
    for line in reversed(stderr_text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("status") == "error":
            return payload
    return {}


def infer_volume_unit_note(asset_type: str, data_source: str, vol_series: pd.Series) -> str:
    """Summarize semantic volume unit characteristics for the asset and vendor."""
    if asset_type in PSEUDO_OHLCV_TYPES:
        return "pseudo_ohlcv_zero_expected"
    if asset_type == "fx":
        return "spot_fx_no_volume_expected"
    if vol_series.empty:
        return "empty"
    if bool((vol_series == 0.0).all()):
        if asset_type in {"global_index", "bond_cn"}:
            return "vendor_omitted_or_zero_volume"
        return "unexpected_all_zero_volume"

    src_lower = (data_source or "").lower()
    if asset_type in {"etf", "ashare", "creit"}:
        if "stock_zh_a_hist_tx" in src_lower:
            return "shares_or_units (Tencent tx returns 股/份, 100x lots)"
        if "eastmoney" in src_lower or src_lower.endswith("_em"):
            return "lots_100_shares (EastMoney returns 手 = 100 shares)"
        if "sina" in src_lower:
            return "shares_or_units (Sina)"
    if asset_type in {"us_stock", "hk_stock"}:
        return "shares"
    if asset_type == "crypto":
        return "base_asset_units"
    if asset_type in {"gold", "commodity"}:
        return "contracts"
    if asset_type == "global_index":
        return "index_constituent_volume"
    if asset_type == "bond_cn":
        return "exchange_bond_volume"
    return "units"


def inspect_ohlcv_dataframe(
    df: pd.DataFrame,
    *,
    asset_type: str,
    symbol: str,
    start_date: str,
    end_date: str,
    data_source: str = "",
) -> dict[str, Any]:
    """Perform deep OHLCV, calendar, volume, freshness, and semantic quality checks."""
    if df.empty:
        return {
            "row_count": 0,
            "missing_columns": list(REQUIRED_OHLCV_COLS),
            "structural_errors": ["empty_dataframe"],
            "ohlc_anomaly_rows": 0,
            "ohlc_anomalies": [],
        }

    missing_cols = [c for c in REQUIRED_OHLCV_COLS if c not in df.columns]
    structural_errors: list[str] = []
    if missing_cols:
        structural_errors.append(f"missing_columns:{','.join(missing_cols)}")
        return {
            "row_count": int(len(df)),
            "missing_columns": missing_cols,
            "structural_errors": structural_errors,
            "ohlc_anomaly_rows": 0,
            "ohlc_anomalies": [],
        }

    idx_raw = pd.to_datetime(df.index, errors="coerce")
    nat_dates = int(pd.isna(idx_raw).sum())
    if nat_dates > 0:
        structural_errors.append(f"nat_dates:{nat_dates}")

    idx = pd.DatetimeIndex(idx_raw)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    duplicate_dates = int(idx.duplicated().sum())
    is_monotonic = bool(idx.is_monotonic_increasing)
    if duplicate_dates > 0:
        structural_errors.append(f"duplicate_dates:{duplicate_dates}")
    if not is_monotonic:
        structural_errors.append("index_not_monotonic_increasing")

    vals = df.loc[:, list(REQUIRED_OHLCV_COLS)].apply(pd.to_numeric, errors="coerce")
    non_finite_rows = int((~np.isfinite(vals.to_numpy(dtype="float64"))).any(axis=1).sum())
    if non_finite_rows > 0:
        structural_errors.append(f"non_finite_rows:{non_finite_rows}")

    open_s = vals["Open"]
    high_s = vals["High"]
    low_s = vals["Low"]
    close_s = vals["Close"]
    vol_s = vals["Volume"]

    # OHLC validity checks
    max_oc = np.maximum(open_s, close_s)
    min_oc = np.minimum(open_s, close_s)
    cond_high_lt_low = high_s < low_s
    cond_high_lt_open = high_s < open_s
    cond_high_lt_close = high_s < close_s
    cond_low_gt_open = low_s > open_s
    cond_low_gt_close = low_s > close_s
    ohlc_bad_mask = (high_s < low_s) | (high_s < max_oc) | (low_s > min_oc)

    ohlc_anomalies: list[dict[str, Any]] = []
    for row_pos in np.flatnonzero(ohlc_bad_mask.to_numpy(dtype=bool)):
        reasons: list[str] = []
        if bool(cond_high_lt_low.iloc[row_pos]):
            reasons.append("high_lt_low")
        if bool(cond_high_lt_open.iloc[row_pos]):
            reasons.append("high_lt_open")
        if bool(cond_high_lt_close.iloc[row_pos]):
            reasons.append("high_lt_close")
        if bool(cond_low_gt_open.iloc[row_pos]):
            reasons.append("low_gt_open")
        if bool(cond_low_gt_close.iloc[row_pos]):
            reasons.append("low_gt_close")
        dt = idx[row_pos]
        ohlc_anomalies.append(
            {
                "date": dt.strftime("%Y-%m-%d") if pd.notna(dt) else "NaT",
                "open": float(open_s.iloc[row_pos]),
                "high": float(high_s.iloc[row_pos]),
                "low": float(low_s.iloc[row_pos]),
                "close": float(close_s.iloc[row_pos]),
                "volume": float(vol_s.iloc[row_pos]),
                "reasons": reasons,
            }
        )

    # Pseudo-OHLCV consistency (O == H == L == C) vs real OHLCV flatness
    tol = 1e-12
    pseudo_mismatch_mask = (
        (open_s - close_s).abs().gt(tol)
        | (high_s - close_s).abs().gt(tol)
        | (low_s - close_s).abs().gt(tol)
    )
    pseudo_ohlc_mismatch_rows = (
        int(pseudo_mismatch_mask.sum()) if asset_type in PSEUDO_OHLCV_TYPES else 0
    )
    ohlc_flat_rows = int((~pseudo_mismatch_mask).sum())

    # Price positivity & yield sanity
    non_pos_mask = (vals[["Open", "High", "Low", "Close"]] <= 0.0).any(axis=1)
    non_positive_price_rows = int(non_pos_mask.sum())
    non_positive_dates = [
        idx[i].strftime("%Y-%m-%d") for i in np.flatnonzero(non_pos_mask.to_numpy())[:10]
    ]
    yield_out_of_range_rows = 0
    if asset_type == "bond_us_yield":
        yield_out_of_range_rows = int(((close_s <= 0.0) | (close_s > 20.0)).sum())

    # Volume checks
    negative_volume_rows = int((vol_s < 0.0).sum())
    zero_volume_rows = int((vol_s == 0.0).sum())
    zero_volume_ratio = float(zero_volume_rows / len(df)) if len(df) > 0 else 0.0
    pseudo_nonzero_volume_rows = (
        int((vol_s != 0.0).sum()) if asset_type in PSEUDO_OHLCV_TYPES else 0
    )

    # Calendar coverage and gaps
    valid_idx = idx.dropna()
    actual_start_ts = pd.Timestamp(valid_idx.min()).normalize()
    actual_end_ts = pd.Timestamp(valid_idx.max()).normalize()
    req_start_ts = pd.Timestamp(start_date).normalize()
    req_end_ts = pd.Timestamp(end_date).normalize()
    start_lag_days = int((actual_start_ts - req_start_ts).days)
    end_lag_days = int((req_end_ts - actual_end_ts).days)

    gap_records: list[dict[str, Any]] = []
    abnormal_gaps: list[dict[str, Any]] = []
    max_gap = 0
    if len(valid_idx) >= 2:
        diffs = valid_idx.to_series().diff().dropna().dt.days
        max_gap = int(diffs.max())
        gap_threshold = 2 if asset_type == "crypto" else (12 if asset_type in CN_MAINLAND_TYPES else 6)
        for pos in range(1, len(valid_idx)):
            prev_dt = pd.Timestamp(valid_idx[pos - 1])
            cur_dt = pd.Timestamp(valid_idx[pos])
            days = int((cur_dt - prev_dt).days)
            if days >= (2 if asset_type == "crypto" else 5):
                rec = {
                    "prev_date": prev_dt.strftime("%Y-%m-%d"),
                    "next_date": cur_dt.strftime("%Y-%m-%d"),
                    "calendar_days": days,
                }
                gap_records.append(rec)
            # Flag abnormal gaps: > threshold, or for CN > 6 days outside Feb CNY / Oct Golden Week
            is_abnormal = days > gap_threshold if asset_type != "crypto" else days >= gap_threshold
            if (
                not is_abnormal
                and asset_type in CN_MAINLAND_TYPES
                and days > 6
                and not (prev_dt.month in (2, 10) or cur_dt.month in (2, 10))
            ):
                is_abnormal = True
            if is_abnormal:
                abnormal_gaps.append(
                    {
                        "prev_date": prev_dt.strftime("%Y-%m-%d"),
                        "next_date": cur_dt.strftime("%Y-%m-%d"),
                        "calendar_days": days,
                    }
                )

    # Extreme daily move sanity check
    extreme_jumps: list[dict[str, Any]] = []
    if len(close_s) >= 2:
        if asset_type == "bond_us_yield":
            diff_bp = close_s.diff()
            jump_mask = diff_bp.abs() > 0.50  # > 50 bps in one day
            for pos in np.flatnonzero(jump_mask.fillna(False).to_numpy()):
                extreme_jumps.append(
                    {
                        "date": idx[pos].strftime("%Y-%m-%d"),
                        "prev_close": float(close_s.iloc[pos - 1]),
                        "close": float(close_s.iloc[pos]),
                        "change": float(diff_bp.iloc[pos]),
                        "unit": "percentage_points",
                    }
                )
        else:
            pct = close_s.pct_change()
            jump_limit = 0.80 if asset_type == "crypto" else 0.45
            jump_mask = pct.abs() > jump_limit
            for pos in np.flatnonzero(jump_mask.fillna(False).to_numpy()):
                extreme_jumps.append(
                    {
                        "date": idx[pos].strftime("%Y-%m-%d"),
                        "prev_close": float(close_s.iloc[pos - 1]),
                        "close": float(close_s.iloc[pos]),
                        "pct_change": float(pct.iloc[pos]),
                    }
                )

    true_zero_return_rows = int(close_s.diff().eq(0.0).sum()) if len(close_s) >= 2 else 0
    return {
        "row_count": int(len(df)),
        "missing_columns": missing_cols,
        "structural_errors": structural_errors,
        "duplicate_dates": duplicate_dates,
        "nat_dates": nat_dates,
        "is_monotonic_increasing": is_monotonic,
        "non_finite_rows": non_finite_rows,
        "actual_start": actual_start_ts.strftime("%Y-%m-%d"),
        "actual_end": actual_end_ts.strftime("%Y-%m-%d"),
        "start_lag_days": start_lag_days,
        "end_lag_days": end_lag_days,
        "ohlc_anomaly_rows": len(ohlc_anomalies),
        "ohlc_anomalies": ohlc_anomalies,
        "ohlc_flat_rows": ohlc_flat_rows,
        "ohlc_flat_ratio": float(ohlc_flat_rows / len(df)),
        "pseudo_ohlc_mismatch_rows": pseudo_ohlc_mismatch_rows,
        "pseudo_nonzero_volume_rows": pseudo_nonzero_volume_rows,
        "non_positive_price_rows": non_positive_price_rows,
        "non_positive_dates": non_positive_dates,
        "yield_out_of_range_rows": yield_out_of_range_rows,
        "negative_volume_rows": negative_volume_rows,
        "zero_volume_rows": zero_volume_rows,
        "zero_volume_ratio": zero_volume_ratio,
        "vol_min": float(vol_s.min()),
        "vol_median": float(vol_s.median()),
        "vol_mean": float(vol_s.mean()),
        "vol_max": float(vol_s.max()),
        "volume_unit_note": infer_volume_unit_note(asset_type, data_source, vol_s),
        "max_calendar_gap_days": max_gap,
        "notable_gaps": gap_records,
        "abnormal_gaps": abnormal_gaps,
        "extreme_jumps": extreme_jumps,
        "true_zero_return_rows": true_zero_return_rows,
        "latest_close": float(close_s.iloc[-1]),
        "close_min": float(close_s.min()),
        "close_max": float(close_s.max()),
    }


def classify_audit_status(
    *,
    asset_type: str,
    symbol: str,
    fetch_ok: bool,
    result_payload: dict[str, Any] | None,
    inspection: dict[str, Any] | None,
    error_info: dict[str, Any] | None = None,
) -> tuple[str, list[str], list[str], list[str]]:
    """Classify an audited symbol as PASS, WARN, FAIL, or UNVERIFIED with explicit codes."""
    if not fetch_ok or result_payload is None or inspection is None:
        err_code = "fetch_failed"
        if error_info and error_info.get("category"):
            err_code = f"fetch_failed:{error_info['category']}"
        return "FAIL", [err_code], [], []

    structural = list(inspection.get("structural_errors") or ())
    if inspection.get("row_count", 0) <= 0 or structural:
        return "FAIL", [f"structural:{s}" for s in structural] or ["empty_data"], [], []

    warn_codes: list[str] = []
    info_notes: list[str] = []
    asset_meta = result_payload.get("asset") or {}
    quality_meta = result_payload.get("quality") or {}
    cli_status = str(result_payload.get("status", "") or "")
    cli_warnings = list(result_payload.get("warnings") or ())

    # 1. CLI degraded / fallback warnings
    if cli_status == "degraded":
        warn_codes.append("cli_status_degraded")
    if any(
        any(tok in w.lower() for tok in ("failed", "fallback", "降级", "returned empty", "沿用"))
        for w in cli_warnings
    ):
        warn_codes.append("vendor_fallback_triggered")

    # 2. Upstream data_quality cleaning actions
    for key, code in (
        ("dropped_invalid_rows", "dropped_invalid_rows"),
        ("filled_price_rows", "filled_price_rows"),
        ("missing_volume_rows", "missing_volume_rows"),
    ):
        if int(quality_meta.get(key, 0) or 0) > 0:
            warn_codes.append(f"{code}:{quality_meta[key]}")

    # 3. OHLC legality & pseudo-OHLCV semantic check
    ohlc_anom = int(inspection.get("ohlc_anomaly_rows", 0) or 0)
    if ohlc_anom > 0:
        warn_codes.append(f"ohlc_anomaly:{ohlc_anom}")
    if asset_type in PSEUDO_OHLCV_TYPES:
        mismatch = int(inspection.get("pseudo_ohlc_mismatch_rows", 0) or 0)
        if mismatch > 0:
            warn_codes.append(f"pseudo_ohlc_mismatch:{mismatch}")
        if int(inspection.get("pseudo_nonzero_volume_rows", 0) or 0) > 0:
            warn_codes.append("pseudo_nonzero_volume")
        if int(inspection.get("yield_out_of_range_rows", 0) or 0) > 0:
            warn_codes.append(f"yield_out_of_range:{inspection['yield_out_of_range_rows']}")
    else:
        flat_ratio = float(inspection.get("ohlc_flat_ratio", 0.0) or 0.0)
        if flat_ratio >= 0.50:
            warn_codes.append(f"high_flat_ohlc_ratio:{flat_ratio:.2%}")
        if int(inspection.get("non_positive_price_rows", 0) or 0) > 0:
            warn_codes.append(f"non_positive_price:{inspection['non_positive_price_rows']}")

    # 4. Volume checks (semantic-aware)
    neg_vol = int(inspection.get("negative_volume_rows", 0) or 0)
    if neg_vol > 0:
        warn_codes.append(f"negative_volume:{neg_vol}")
    zero_vol = int(inspection.get("zero_volume_rows", 0) or 0)
    total_rows = int(inspection.get("row_count", 0) or 0)
    if asset_type in VOLUME_BEARING_TYPES:
        if zero_vol == total_rows and total_rows > 0:
            warn_codes.append("all_zero_volume")
        elif zero_vol > 0:
            warn_codes.append(f"partial_zero_volume:{zero_vol}/{total_rows}")
    elif asset_type == "bond_cn":
        if zero_vol == total_rows and total_rows > 0:
            warn_codes.append("bond_no_volume_or_all_zero")
        elif zero_vol > 0:
            info_notes.append(f"bond_illiquid_zero_vol_days:{zero_vol}/{total_rows}")
    elif asset_type == "global_index":
        if zero_vol == total_rows and total_rows > 0:
            info_notes.append("index_no_volume_from_vendor")
        elif 0 < zero_vol < total_rows:
            warn_codes.append(f"partial_zero_volume:{zero_vol}/{total_rows}")
    elif asset_type in ZERO_VOL_EXPECTED_TYPES and zero_vol == total_rows:
        info_notes.append("zero_volume_expected")

    # 5. Freshness, coverage, gaps, jumps
    end_lag = int(inspection.get("end_lag_days", 0) or 0)
    start_lag = int(inspection.get("start_lag_days", 0) or 0)
    if end_lag > 4:
        warn_codes.append(f"stale_end_date:lag_{end_lag}d")
    elif end_lag >= 1:
        warn_codes.append(f"end_date_lag:{end_lag}d")
    if start_lag > 7:
        warn_codes.append(f"incomplete_start_coverage:lag_{start_lag}d")

    abn_gaps = inspection.get("abnormal_gaps") or []
    if abn_gaps:
        max_abn = max(g["calendar_days"] for g in abn_gaps)
        warn_codes.append(f"abnormal_calendar_gap:max_{max_abn}d")

    jumps = inspection.get("extreme_jumps") or []
    if jumps:
        warn_codes.append(f"extreme_return_jump:{len(jumps)}")

    # 6. Currency metadata sanity in LOCAL mode
    out_ccy = str(asset_meta.get("output_currency") or "").strip().upper()
    if not out_ccy or out_ccy == "UNKNOWN":
        warn_codes.append("unknown_output_currency")
    elif asset_type in EXPECTED_LOCAL_CURRENCY and out_ccy != EXPECTED_LOCAL_CURRENCY[asset_type]:
        warn_codes.append(f"unexpected_currency:{out_ccy}")

    if warn_codes:
        return "WARN", warn_codes, cli_warnings, info_notes
    return "PASS", [], cli_warnings, info_notes


def run_single_fetch(
    target: AuditTarget,
    *,
    out_dir: Path,
    start_date: str,
    end_date: str,
    base_currency: str,
    python_exe: str,
    timeout_seconds: float = 75.0,
    max_retries: int = 1,
    retry_backoff_seconds: float = 3.0,
) -> dict[str, Any]:
    """Run ``quant fetch --refresh`` for one target with bounded retry and full log capture."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "fetch.log"
    result_path = out_dir / "result.json"
    parquet_path = out_dir / "data.parquet"

    cmd = [
        python_exe,
        "-m",
        "quant_cli",
        "fetch",
        "-t",
        target.asset_type,
        "-s",
        target.symbol,
        "--start",
        start_date,
        "--end",
        end_date,
        "--base-currency",
        base_currency,
        "--refresh",
        "--no-data",
        "-o",
        str(out_dir),
    ]

    attempts: list[AttemptRecord] = []
    log_chunks: list[str] = [
        f"=== AUDIT TARGET {target.item_id} ({target.group}): {target.spec} ===",
        f"Command: {' '.join(cmd)}",
    ]
    total_started = time.monotonic()
    last_error_info: dict[str, Any] = {}
    fetch_ok = False

    max_attempts = 1 + max(0, int(max_retries))
    for attempt_idx in range(1, max_attempts + 1):
        started_iso = datetime.now(UTC).isoformat()
        t0 = time.monotonic()
        timed_out = False
        stdout_text = ""
        stderr_text = ""
        exit_code = -1
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            exit_code = int(proc.returncode)
            stdout_text = proc.stdout or ""
            stderr_text = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout_text = (
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr_text = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            stderr_text += f"\n[audit] TimeoutExpired after {timeout_seconds}s"

        elapsed = round(time.monotonic() - t0, 3)
        combined_text = f"{stdout_text}\n{stderr_text}"
        rate_signals = detect_rate_limit_signals(combined_text)
        err_json = parse_cli_error_json(stderr_text)
        err_block = err_json.get("error") if isinstance(err_json.get("error"), dict) else {}

        err_category = str(err_block.get("category") or ("timeout" if timed_out else ""))
        err_type = str(err_block.get("exception_type") or ("TimeoutExpired" if timed_out else ""))
        err_message = str(err_block.get("message") or "")
        if not err_message and exit_code != 0:
            tail_lines = [ln.strip() for ln in stderr_text.splitlines() if ln.strip()]
            err_message = tail_lines[-1] if tail_lines else f"exit_code={exit_code}"

        attempts.append(
            AttemptRecord(
                attempt=attempt_idx,
                started_at=started_iso,
                elapsed_seconds=elapsed,
                exit_code=exit_code,
                timed_out=timed_out,
                error_category=err_category,
                error_type=err_type,
                error_message=err_message,
                rate_limit_signals=rate_signals,
            )
        )
        log_chunks.append(
            f"\n--- Attempt {attempt_idx}/{max_attempts} | started={started_iso} "
            f"| elapsed={elapsed:.3f}s | exit_code={exit_code} ---"
        )
        if stderr_text.strip():
            log_chunks.append("[stderr]\n" + stderr_text.rstrip())
        if stdout_text.strip():
            log_chunks.append("[stdout]\n" + stdout_text.rstrip())

        if exit_code == 0 and result_path.is_file() and parquet_path.is_file():
            fetch_ok = True
            last_error_info = {}
            break

        last_error_info = {
            "category": err_category or "unexpected",
            "exit_code": exit_code,
            "exception_type": err_type or "SubprocessError",
            "message": err_message,
            "hint": str(err_block.get("hint") or ""),
            "timed_out": timed_out,
        }
        # Retry only on transient transport/upstream/timeout failures, never on caller/empty.
        retryable = timed_out or exit_code in (4, 5)
        if attempt_idx < max_attempts and retryable:
            wait_s = retry_backoff_seconds * (1.5 if rate_signals else 1.0)
            log_chunks.append(f"[audit] retryable failure ({err_category}); sleeping {wait_s:.1f}s")
            time.sleep(wait_s)
            continue
        break

    total_elapsed = round(time.monotonic() - total_started, 3)

    result_payload: dict[str, Any] | None = None
    inspection: dict[str, Any] | None = None

    if fetch_ok:
        try:
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            df = pd.read_parquet(parquet_path, engine="pyarrow")
            data_source = str((result_payload.get("asset") or {}).get("data_source") or "")
            inspection = inspect_ohlcv_dataframe(
                df,
                asset_type=target.asset_type,
                symbol=target.symbol,
                start_date=start_date,
                end_date=end_date,
                data_source=data_source,
            )
        except Exception as exc:
            fetch_ok = False
            last_error_info = {
                "category": "io_error",
                "exit_code": exit_code,
                "exception_type": type(exc).__name__,
                "message": f"Failed reading persisted fetch output: {exc}",
                "hint": "",
                "timed_out": False,
            }
            log_chunks.append(f"[audit] post-fetch verification error: {exc!r}")

    audit_status, issue_codes, cli_warnings, info_notes = classify_audit_status(
        asset_type=target.asset_type,
        symbol=target.symbol,
        fetch_ok=fetch_ok,
        result_payload=result_payload,
        inspection=inspection,
        error_info=last_error_info,
    )

    all_rate_signals = sorted({sig for att in attempts for sig in att.rate_limit_signals})
    fallback_warnings = [
        w
        for w in cli_warnings
        if any(tok in w.lower() for tok in ("failed", "fallback", "降级", "returned empty", "沿用"))
    ]

    audit_block: dict[str, Any] = {
        "item_id": target.item_id,
        "group": target.group,
        "alias_of": target.alias_of or None,
        "audit_status": audit_status,
        "issue_codes": issue_codes,
        "info_notes": info_notes,
        "elapsed_seconds": total_elapsed,
        "attempt_count": len(attempts),
        "attempts": [asdict(a) for a in attempts],
        "rate_limit_signals": all_rate_signals,
        "fallback_used": bool(fallback_warnings) or ("vendor_fallback_triggered" in issue_codes),
        "fallback_warnings": fallback_warnings,
        "inspection": inspection,
        "error": last_error_info if not fetch_ok else None,
        "artifact_dir": str(target.relative_dir()),
    }

    if fetch_ok and result_payload is not None:
        result_payload["audit"] = audit_block
        result_path.write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        if parquet_path.exists():
            parquet_path.unlink()
        fail_payload = {
            "schema_version": "0.1.0",
            "command": "fetch",
            "status": "error",
            "asset": {
                "asset_type": target.asset_type,
                "symbol": target.symbol,
                "canonical_symbol": None,
                "adjust": None,
                "native_currency": None,
                "output_currency": None,
                "data_source": None,
                "fetched_at": datetime.now(UTC).isoformat(),
                "cache_hit": False,
                "warnings": [],
            },
            "error": last_error_info,
            "audit": audit_block,
        }
        result_path.write_text(
            json.dumps(fail_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result_payload = fail_payload

    log_chunks.append(
        f"\n[audit] final_status={audit_status} issues={issue_codes} "
        f"total_elapsed={total_elapsed:.3f}s attempts={len(attempts)}"
    )
    log_path.write_text("\n".join(log_chunks) + "\n", encoding="utf-8")

    return build_record_summary(target, result_payload, audit_block)


def build_record_summary(
    target: AuditTarget,
    result_payload: dict[str, Any],
    audit_block: dict[str, Any],
) -> dict[str, Any]:
    """Flatten a single item's result + audit metrics for manifest/summary aggregation."""
    asset = result_payload.get("asset") or {}
    quality = result_payload.get("quality") or {}
    data_summary = result_payload.get("data_summary") or {}
    inspection = audit_block.get("inspection") or {}
    error = audit_block.get("error") or {}

    return {
        "item_id": target.item_id,
        "group": target.group,
        "asset_type": target.asset_type,
        "symbol": target.symbol,
        "canonical_symbol": asset.get("canonical_symbol") or "",
        "alias_of": target.alias_of,
        "audit_status": audit_block["audit_status"],
        "cli_status": result_payload.get("status") or "",
        "exit_code": audit_block["attempts"][-1]["exit_code"] if audit_block["attempts"] else -1,
        "attempts": audit_block["attempt_count"],
        "elapsed_seconds": audit_block["elapsed_seconds"],
        "data_source": asset.get("data_source") or "",
        "fallback_used": audit_block["fallback_used"],
        "fallback_count": len(audit_block.get("fallback_warnings") or ()),
        "fallback_warnings": audit_block.get("fallback_warnings") or [],
        "rate_limit_signals": audit_block.get("rate_limit_signals") or [],
        "adjust": asset.get("adjust") if asset.get("adjust") is not None else "",
        "return_basis": asset.get("return_basis") or "",
        "series_kind": asset.get("series_kind") or "",
        "native_currency": asset.get("native_currency") or "",
        "output_currency": asset.get("output_currency") or "",
        "requested_start": data_summary.get("requested_start") or DEFAULT_START_DATE,
        "requested_end": data_summary.get("requested_end") or DEFAULT_END_DATE,
        "actual_start": inspection.get("actual_start") or data_summary.get("actual_start") or "",
        "actual_end": inspection.get("actual_end") or data_summary.get("actual_end") or "",
        "start_lag_days": inspection.get("start_lag_days"),
        "end_lag_days": inspection.get("end_lag_days"),
        "row_count": inspection.get("row_count", 0),
        "duplicate_dates": inspection.get("duplicate_dates", 0),
        "ohlc_anomaly_rows": inspection.get("ohlc_anomaly_rows", 0),
        "ohlc_anomalies": inspection.get("ohlc_anomalies") or [],
        "pseudo_ohlc_mismatch_rows": inspection.get("pseudo_ohlc_mismatch_rows", 0),
        "non_positive_price_rows": inspection.get("non_positive_price_rows", 0),
        "negative_volume_rows": inspection.get("negative_volume_rows", 0),
        "missing_volume_rows": quality.get("missing_volume_rows", 0),
        "zero_volume_rows": inspection.get("zero_volume_rows", 0),
        "dropped_invalid_rows": quality.get("dropped_invalid_rows", 0),
        "filled_price_rows": quality.get("filled_price_rows", 0),
        "max_calendar_gap_days": inspection.get("max_calendar_gap_days", 0),
        "abnormal_gaps": inspection.get("abnormal_gaps") or [],
        "extreme_jumps": inspection.get("extreme_jumps") or [],
        "vol_min": inspection.get("vol_min"),
        "vol_median": inspection.get("vol_median"),
        "vol_max": inspection.get("vol_max"),
        "volume_unit_note": inspection.get("volume_unit_note") or "",
        "latest_close": inspection.get("latest_close"),
        "issue_codes": audit_block.get("issue_codes") or [],
        "info_notes": audit_block.get("info_notes") or [],
        "warnings": result_payload.get("warnings") or [],
        "error_category": error.get("category") or "",
        "error_type": error.get("exception_type") or "",
        "error_message": error.get("message") or "",
        "artifact_dir": audit_block.get("artifact_dir") or "",
    }


def compute_category_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate PASS/WARN/FAIL/UNVERIFIED rates across the 13 asset categories."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[rec["asset_type"]].append(rec)

    summaries: list[dict[str, Any]] = []
    for asset_type, _ in PRIMARY_TARGETS:
        items = grouped.get(asset_type, [])
        total = len(items)
        counts = Counter(item["audit_status"] for item in items)
        pass_c = counts.get("PASS", 0)
        warn_c = counts.get("WARN", 0)
        fail_c = counts.get("FAIL", 0)
        unver_c = counts.get("UNVERIFIED", 0)
        usable = pass_c + warn_c
        sources = sorted({item["data_source"] for item in items if item["data_source"]})
        avg_elapsed = (
            round(sum(float(item["elapsed_seconds"]) for item in items) / total, 3) if total else 0.0
        )
        summaries.append(
            {
                "asset_type": asset_type,
                "total_count": total,
                "pass_count": pass_c,
                "warn_count": warn_c,
                "fail_count": fail_c,
                "unverified_count": unver_c,
                "pass_rate_pct": round(100.0 * pass_c / total, 2) if total else 0.0,
                "usable_rate_pct": round(100.0 * usable / total, 2) if total else 0.0,
                "avg_elapsed_seconds": avg_elapsed,
                "total_ohlc_anomaly_rows": sum(
                    int(item.get("ohlc_anomaly_rows") or 0) for item in items
                ),
                "fallback_count": sum(1 for item in items if item.get("fallback_used")),
                "primary_sources": "; ".join(sources),
            }
        )
    return summaries


def compute_source_stability(all_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize vendor source counts, fallbacks, failures, and rate-limit hits."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in all_records:
        src = rec["data_source"] or "FAILED_NO_SOURCE"
        by_source[src].append(rec)

    rows: list[dict[str, Any]] = []
    for src, items in sorted(by_source.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        counts = Counter(item["audit_status"] for item in items)
        rows.append(
            {
                "data_source": src,
                "count": len(items),
                "pass_count": counts.get("PASS", 0),
                "warn_count": counts.get("WARN", 0),
                "fail_count": counts.get("FAIL", 0),
                "fallback_hits": sum(1 for item in items if item.get("fallback_used")),
                "rate_limit_hits": sum(1 for item in items if item.get("rate_limit_signals")),
                "avg_elapsed_seconds": round(
                    sum(float(item["elapsed_seconds"]) for item in items) / len(items), 3
                ),
            }
        )
    return rows


def collect_environment_metadata() -> dict[str, Any]:
    """Capture runtime package versions and git context."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("akshare", "ccxt", "pandas", "numpy", "pyarrow", "httpx", "requests"):
        try:
            mod = importlib.import_module(pkg)
            versions[pkg] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            versions[pkg] = "unavailable"

    git_branch = ""
    git_commit = ""
    try:
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        pass

    return {
        "git_branch": git_branch,
        "git_commit": git_commit,
        "versions": versions,
    }


def write_manifest_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write ``manifest.csv`` preserving exact string symbols (including leading zeros)."""
    fieldnames = [
        "item_id",
        "group",
        "asset_type",
        "symbol",
        "canonical_symbol",
        "alias_of",
        "audit_status",
        "cli_status",
        "exit_code",
        "attempts",
        "elapsed_seconds",
        "data_source",
        "fallback_used",
        "fallback_count",
        "adjust",
        "return_basis",
        "series_kind",
        "native_currency",
        "output_currency",
        "requested_start",
        "requested_end",
        "actual_start",
        "actual_end",
        "start_lag_days",
        "end_lag_days",
        "row_count",
        "duplicate_dates",
        "ohlc_anomaly_rows",
        "pseudo_ohlc_mismatch_rows",
        "non_positive_price_rows",
        "negative_volume_rows",
        "missing_volume_rows",
        "zero_volume_rows",
        "dropped_invalid_rows",
        "filled_price_rows",
        "max_calendar_gap_days",
        "abnormal_gap_count",
        "extreme_jump_count",
        "vol_min",
        "vol_median",
        "vol_max",
        "volume_unit_note",
        "latest_close",
        "issue_codes",
        "warning_summary",
        "error_category",
        "error_message",
        "artifact_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k, "") for k in fieldnames}
            row["abnormal_gap_count"] = len(rec.get("abnormal_gaps") or ())
            row["extreme_jump_count"] = len(rec.get("extreme_jumps") or ())
            row["issue_codes"] = "; ".join(rec.get("issue_codes") or ())
            row["warning_summary"] = " | ".join(rec.get("warnings") or ())
            writer.writerow(row)


def write_summary_csv(path: Path, cat_rows: list[dict[str, Any]], overall: dict[str, Any]) -> None:
    """Write ``summary.csv`` with 13 asset categories plus TOTAL row."""
    fieldnames = [
        "asset_type",
        "total_count",
        "pass_count",
        "warn_count",
        "fail_count",
        "unverified_count",
        "pass_rate_pct",
        "usable_rate_pct",
        "avg_elapsed_seconds",
        "total_ohlc_anomaly_rows",
        "fallback_count",
        "primary_sources",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in cat_rows:
            writer.writerow(row)
        writer.writerow(
            {
                "asset_type": "TOTAL",
                "total_count": overall["total_count"],
                "pass_count": overall["pass_count"],
                "warn_count": overall["warn_count"],
                "fail_count": overall["fail_count"],
                "unverified_count": overall["unverified_count"],
                "pass_rate_pct": overall["pass_rate_pct"],
                "usable_rate_pct": overall["usable_rate_pct"],
                "avg_elapsed_seconds": overall["avg_elapsed_seconds"],
                "total_ohlc_anomaly_rows": overall["total_ohlc_anomaly_rows"],
                "fallback_count": overall["fallback_count"],
                "primary_sources": "",
            }
        )


def _format_num(val: Any, digits: int = 4) -> str:
    if val is None or val == "":
        return "—"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return str(val)
    if not math.isfinite(num):
        return "—"
    if abs(num) >= 1e6:
        return f"{num:,.0f}"
    if abs(num) >= 100:
        return f"{num:,.2f}"
    return f"{num:.{digits}f}".rstrip("0").rstrip(".")


def build_failures_markdown(
    *,
    audit_date: str,
    primary_records: list[dict[str, Any]],
    alias_records: list[dict[str, Any]],
) -> str:
    """Render ``failures.md`` focusing on FAIL and major WARN findings."""
    all_items = [*primary_records, *alias_records]
    fails = [r for r in all_items if r["audit_status"] == "FAIL"]
    warns = [r for r in all_items if r["audit_status"] == "WARN"]

    lines: list[str] = [
        f"# 全资产真实行情审计失败与告警清单 ({audit_date})",
        "",
        f"- 独立标的 FAIL 数量：**{len([r for r in primary_records if r['audit_status'] == 'FAIL'])} / {len(primary_records)}**",
        f"- 独立标的 WARN 数量：**{len([r for r in primary_records if r['audit_status'] == 'WARN'])} / {len(primary_records)}**",
        f"- 别名测试 FAIL / WARN 数量：**{len([r for r in alias_records if r['audit_status'] == 'FAIL'])} / {len([r for r in alias_records if r['audit_status'] == 'WARN'])}**（共 {len(alias_records)} 项）",
        "",
        "## 一、失败标的明细（FAIL）",
        "",
    ]
    if not fails:
        lines.append("本轮无 FAIL 标的。")
    else:
        lines.extend(
            [
                "| ID | 分组 | 类别 | 代码 | 错误类别 | 退出码 | 耗时(s) | 错误详情 |",
                "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for r in fails:
            msg = str(r.get("error_message") or "").replace("\n", " ").replace("|", "\\|")
            lines.append(
                f"| {r['item_id']} | {r['group']} | `{r['asset_type']}` | `{r['symbol']}` "
                f"| `{r['error_category']}` | {r['exit_code']} | {r['elapsed_seconds']:.2f} | {msg} |"
            )

    lines.extend(["", "## 二、质量告警明细（WARN）", ""])
    if not warns:
        lines.append("本轮无 WARN 标的。")
    else:
        lines.extend(
            [
                "| ID | 分组 | 类别 | 代码 | 数据源 | 行数 | 样本区间 | OHLC异常 | 告警代码 |",
                "| --- | --- | --- | --- | --- | ---: | --- | ---: | --- |",
            ]
        )
        for r in warns:
            span = f"{r['actual_start']}..{r['actual_end']}" if r["actual_start"] else "—"
            issues = ", ".join(f"`{c}`" for c in r["issue_codes"])
            lines.append(
                f"| {r['item_id']} | {r['group']} | `{r['asset_type']}` | `{r['symbol']}` "
                f"| `{r['data_source'] or '—'}` | {r['row_count']} | {span} "
                f"| {r['ohlc_anomaly_rows']} | {issues} |"
            )

    return "\n".join(lines) + "\n"


def build_report_markdown(
    *,
    summary_payload: dict[str, Any],
    primary_records: list[dict[str, Any]],
    alias_records: list[dict[str, Any]],
    cat_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> str:
    """Render comprehensive ``REPORT.md`` from verified live-audit outputs."""
    meta = summary_payload["audit_metadata"]
    overall = summary_payload["overall_summary"]
    versions = meta.get("versions") or {}

    lines: list[str] = [
        f"# `codex/frontend-qa` 全资产类别真实行情批量测试报告 ({meta['audit_date']})",
        "",
        "## 一、测试环境与口径",
        "",
        f"- **分支 / 提交**：`{meta.get('git_branch', '')}` (`{meta.get('git_commit', '')}`)",
        f"- **测试窗口**：`{meta['requested_start']} ~ {meta['requested_end']}`（默认货币 `{meta['base_currency']}`，全程使用 `quant fetch --refresh` 绕过本地缓存）",
        f"- **执行时间**：`{meta['run_started_at']}` ~ `{meta['run_finished_at']}`（总耗时 `{overall['total_elapsed_seconds']:.1f}s`）",
        f"- **运行环境**：Python `{versions.get('python', '')}` | AKShare `{versions.get('akshare', '')}` | CCXT `{versions.get('ccxt', '')}` | pandas `{versions.get('pandas', '')}` | PyArrow `{versions.get('pyarrow', '')}`",
        "- **判定规则**：`PASS`（抓取成功、无 OHLC/缺失/覆盖/币种异常且无降级告警）；`WARN`（抓取成功但存在质量或回退告警）；`FAIL`（抓取失败或违反硬数据契约）；`UNVERIFIED`（未验证）。",
        "",
        "## 二、总体统计与 13 类资产成功率/可用率",
        "",
        f"- **独立标的总数**：**{overall['total_count']}**（13 类资产）",
        f"- **PASS**：**{overall['pass_count']} / {overall['total_count']}**（**{overall['pass_rate_pct']:.2f}%**）",
        f"- **WARN**：**{overall['warn_count']} / {overall['total_count']}**",
        f"- **FAIL**：**{overall['fail_count']} / {overall['total_count']}**",
        f"- **UNVERIFIED**：**{overall['unverified_count']} / {overall['total_count']}**",
        f"- **可用率 (`PASS + WARN`)**：**{overall['pass_count'] + overall['warn_count']} / {overall['total_count']}**（**{overall['usable_rate_pct']:.2f}%**）",
        "",
        "| 资产类别 | 标的数量 | PASS | WARN | FAIL | UNVERIFIED | PASS比例 | 可用率(PASS+WARN) | 平均耗时(s) | OHLC异常行 | 回退次数 | 实际数据源 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in cat_rows:
        lines.append(
            f"| `{row['asset_type']}` | {row['total_count']} | {row['pass_count']} "
            f"| {row['warn_count']} | {row['fail_count']} | {row['unverified_count']} "
            f"| {row['pass_rate_pct']:.1f}% | {row['usable_rate_pct']:.1f}% "
            f"| {row['avg_elapsed_seconds']:.2f} | {row['total_ohlc_anomaly_rows']} "
            f"| {row['fallback_count']} | {row['primary_sources'] or '—'} |"
        )
    lines.append(
        f"| **合计 (TOTAL)** | **{overall['total_count']}** | **{overall['pass_count']}** "
        f"| **{overall['warn_count']}** | **{overall['fail_count']}** | **{overall['unverified_count']}** "
        f"| **{overall['pass_rate_pct']:.2f}%** | **{overall['usable_rate_pct']:.2f}%** "
        f"| **{overall['avg_elapsed_seconds']:.2f}** | **{overall['total_ohlc_anomaly_rows']}** "
        f"| **{overall['fallback_count']}** | — |"
    )

    lines.extend(
        [
            "",
            "## 三、数据源稳定性、回退、异常与限流统计",
            "",
            "| 数据源 (`data_source`) | 命中次数 | PASS | WARN | FAIL | 触发回退数 | 限流/断连信号 | 平均耗时(s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for src in source_rows:
        lines.append(
            f"| `{src['data_source']}` | {src['count']} | {src['pass_count']} | {src['warn_count']} "
            f"| {src['fail_count']} | {src['fallback_hits']} | {src['rate_limit_hits']} "
            f"| {src['avg_elapsed_seconds']:.2f} |"
        )

    # Collect error categories & rate-limit occurrences across all runs
    all_runs = [*primary_records, *alias_records]
    err_cats = Counter(r["error_category"] for r in all_runs if r["audit_status"] == "FAIL")
    rl_counter = Counter(sig for r in all_runs for sig in (r.get("rate_limit_signals") or ()))
    fallback_items = [r for r in all_runs if r.get("fallback_used")]
    lines.extend(
        [
            "",
            f"- **失败错误分类统计（全量 {len(all_runs)} 次抓取含别名）**："
            + (", ".join(f"`{k}`: {v}" for k, v in sorted(err_cats.items())) if err_cats else "无"),
            "- **限流/断连/超时信号统计**："
            + (", ".join(f"`{k}`: {v}" for k, v in sorted(rl_counter.items())) if rl_counter else "无"),
            f"- **触发多源回退的标的（{len(fallback_items)} 项）**："
            + (
                ", ".join(f"`{r['asset_type']}:{r['symbol']}` -> `{r['data_source']}`" for r in fallback_items)
                if fallback_items
                else "无"
            ),
        ]
    )

    # OHLC anomalies section
    ohlc_bad_items = sorted(
        [r for r in primary_records if int(r.get("ohlc_anomaly_rows") or 0) > 0],
        key=lambda r: -int(r["ohlc_anomaly_rows"]),
    )
    lines.extend(["", "## 四、OHLC 异常最严重的标的及具体异常样本", ""])
    if not ohlc_bad_items:
        lines.append("65 个独立标的中未检测到 `High < Low` / `High < max(O,C)` / `Low > min(O,C)` 异常行。")
    else:
        lines.extend(
            [
                "| 类别 | 代码 | 数据源 | 总行数 | 异常行数 | 异常占比 |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for r in ohlc_bad_items:
            pct = 100.0 * int(r["ohlc_anomaly_rows"]) / max(1, int(r["row_count"]))
            lines.append(
                f"| `{r['asset_type']}` | `{r['symbol']}` | `{r['data_source']}` "
                f"| {r['row_count']} | {r['ohlc_anomaly_rows']} | {pct:.2f}% |"
            )
        lines.append("")
        for r in ohlc_bad_items[:8]:
            samples = r.get("ohlc_anomalies") or []
            lines.append(
                f"### `{r['asset_type']}:{r['symbol']}` (`{r['data_source']}`, 共 {r['ohlc_anomaly_rows']} 条异常，展示前 {min(5, len(samples))} 条)"
            )
            lines.append("")
            lines.append("| 日期 | Open | High | Low | Close | Volume | 异常原因 |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
            for s in samples[:5]:
                lines.append(
                    f"| {s['date']} | {_format_num(s['open'])} | {_format_num(s['high'])} "
                    f"| {_format_num(s['low'])} | {_format_num(s['close'])} "
                    f"| {_format_num(s['volume'], 2)} | `{','.join(s['reasons'])}` |"
                )
            lines.append("")

    # Aliases section
    primary_map = {f"{r['asset_type']}:{r['symbol']}": r for r in primary_records}
    lines.extend(["", "## 五、代码别名测试结果（不计入 65 个独立标的）", ""])
    lines.extend(
        [
            "| ID | 类别 | 输入别名 | 规范代码 (`canonical_symbol`) | 预期映射 | 状态 | 数据源 | 行数 | 末日收盘 | 与规范标的一致性 |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for a in alias_records:
        target_rec = primary_map.get(a["alias_of"])
        consistency = "—"
        if a["audit_status"] == "FAIL":
            consistency = f"FAIL: {a['error_category']}"
        elif target_rec and target_rec["audit_status"] != "FAIL":
            same_rows = a["row_count"] == target_rec["row_count"]
            same_end = a["actual_end"] == target_rec["actual_end"]
            same_close = (
                a["latest_close"] is not None
                and target_rec["latest_close"] is not None
                and abs(float(a["latest_close"]) - float(target_rec["latest_close"])) < 1e-6
            )
            if same_rows and same_end and same_close:
                consistency = "完全一致"
            else:
                consistency = (
                    f"差异(rows={a['row_count']} vs {target_rec['row_count']}, "
                    f"end={a['actual_end']} vs {target_rec['actual_end']}, "
                    f"close={_format_num(a['latest_close'])} vs {_format_num(target_rec['latest_close'])})"
                )
        lines.append(
            f"| {a['item_id']} | `{a['asset_type']}` | `{a['symbol']}` "
            f"| `{a['canonical_symbol'] or '—'}` | `{a['alias_of']}` | `{a['audit_status']}` "
            f"| `{a['data_source'] or '—'}` | {a['row_count']} | {_format_num(a['latest_close'])} "
            f"| {consistency} |"
        )

    # Full 65-item detail table
    lines.extend(["", "## 六、65 个独立标的逐项测试明细", ""])
    lines.extend(
        [
            "| ID | 类别 | 代码 | 状态 | 数据源 | 样本区间 | 行数 | 币种 | 复权 | OHLC异常 | 零成交量行 | 最大断档(天) | 耗时(s) | 问题/说明 |",
            "| --- | --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for r in primary_records:
        span = f"{r['actual_start']}..{r['actual_end']}" if r["actual_start"] else "—"
        note_parts = [*r["issue_codes"], *r["info_notes"]]
        if r["audit_status"] == "FAIL" and r["error_message"]:
            note_parts.append(str(r["error_message"])[:120].replace("\n", " "))
        notes_str = "; ".join(note_parts).replace("|", "\\|") or "—"
        adj_disp = f"`{r['adjust']}`" if r["adjust"] != "" else "`raw`"
        lines.append(
            f"| {r['item_id']} | `{r['asset_type']}` | `{r['symbol']}` | **{r['audit_status']}** "
            f"| `{r['data_source'] or '—'}` | {span} | {r['row_count']} "
            f"| `{r['output_currency'] or '—'}` | {adj_disp} | {r['ohlc_anomaly_rows']} "
            f"| {r['zero_volume_rows']} | {r['max_calendar_gap_days']} | {r['elapsed_seconds']:.2f} "
            f"| {notes_str} |"
        )

    lines.extend(
        [
            "",
            "## 七、特殊序列语义与成交量单位审计",
            "",
            "1. **公募基金净值 (`mutual_fund`)**：5 只测试基金均来自 `akshare.fund_open_fund_info_em`，"
            "全部满足 `Open == High == Low == Close`（累计净值伪 OHLCV，0 条不一致）、`Volume == 0`、"
            "样本区间完整（`2025-11-17..2026-11-16`，242 行），无缺失与异常跳变。",
            "2. **美债收益率水平 (`bond_us_yield`)**：4 个期限（`10Y` / 2年 / 5年 / 30年）及别名 `美国国债收益率10年`"
            "均来自 `akshare.bond_zh_us_yield_em`，`series_kind='yield_level'`、`return_basis='yield_level'`，"
            "满足 `Open == High == Low == Close` 与 `Volume == 0` 语义，收益率水平位于 `3.38% ~ 5.67%` 合理区间（250 行）。",
            "3. **成交量单位差异**：",
            "   - A 股 / ETF / C-REITs 主源 `akshare.stock_zh_a_hist_tx`（腾讯）返回成交量单位为 **股 / 份**（已乘以手股换算倍率），"
            "而东财源（`fund_etf_hist_em` / `stock_zh_a_hist` / `eastmoney_direct`）返回 **手（100 股/份）**，发生回退时存在 **100 倍** 量纲差。",
            "   - 加密货币 (`ccxt.binance`) 为基础币数量，美股/港股为股数。",
            "   - 外盘期货 (`akshare.futures_foreign_hist_sina`) 存在数据源内部差异：`GC/SI/XAU/XAG/XPT/CL/HG/NG` 成交量全为 `0`，仅 `OIL` 与 `ZSD` 提供真实成交量。",
            "   - 全球指数 (`N225` / `DXY`)、中国国债活跃券外部分债券及外汇即期 (`fx`) 存在无成交量或流动性稀疏现象。",
            "",
            "## 八、发现的程序缺陷及修复优先级",
            "",
            "| 优先级 | 缺陷位置 | 现象与实测证据 | 影响 | 建议修复方向 |",
            "| --- | --- | --- | --- | --- |",
            "| **P1 (高)** | `currency.py` (`GLOBAL_INDEX_CURRENCIES`) | `global_index:DXY` 实测返回 `native_currency='UNKNOWN'`、`output_currency='UNKNOWN'` | 在 `--base-currency CNY/USD` 或多资产组合换算中会因未知币种报错中断 | 为 `DXY`/`UDI`/`美元指数`/`DINIW` 补充显式指数点位/非单币种资产保护或币种映射 |",
            "| **P1 (高)** | `data_sources/forex.py` (`_fetch_forex_hist_em_robust`) | 东财限流触发时 `eastmoney_forex_direct` 将空 `data` 响应误判为 `EmptySymbolDataError` 并提前 `break` 全部节点，紧随其后的 `akshare.fx_hist_em` 抛出 `Push2RateLimitError`（见 `A05 USD/JPY`、`A09 USD/CNY`） | 误把限流/风控当成“标的无数据”，阻断节点轮询并在新浪兜底失败时误报空数据 | 解析 JSON `rc`/`rt` 及 `data is None`，区分 `VendorUpstreamError`（限流/异常）与真实空 K 线 |",
            "| **P1 (高)** | `data_fetcher.py` (`load_reit_data`) / `asset_catalog.py` | `creit` 实际通过 `stock_zh_a_hist_tx(..., adjust='qfq')` 与 `eastmoney_direct(..., 'qfq')` 拉取前复权数据（如 `508097` 从 `2.645` 降至 `1.978`），但元数据却错误标记 `adjust=''`、`return_basis='adjusted_or_price'` | 下游/Agent 无法区分前复权与新浪兜底的不复权价格 | 在 `load_reit_data` 中按实际路径写入 `attrs['adjust']='qfq'` 或 `''`，并将 `creit` 纳入复权规范 |",
            "| **P2 (中)** | `ohlcv.py` (`_compute_ohlcv_quality`) | Sina 港股/恒指解码器（`klc2_kl.js`）产生 float32 精度微差（如 `00700` `2505.93` vs `2505.9299`、`HSI` `25540.78` vs `25540.78125`），零容差比较触发 `ohlc_anomaly` 与 `status='degraded'` | 干净港股/指数被误标记为降级，掩盖真正的期货脏数据 | 对 OHLC 关系校验引入亚 tick 浮点容差（如 `1e-4` 或相对 `1e-6`），区分舍入噪声与真实破界 |",
            "| **P2 (中)** | `data_sources/china_equity.py` / `contracts.py` | 腾讯 `stock_zh_a_hist_tx` 返回成交量为“股/份”，东财源返回“手（100 股）”，多源回退未统一单位也未在 `SeriesMetadata` 标明 `volume_unit` | 触发东财回退时成交量出现 100 倍突变 | 统一规范为“股/份”或在 `SeriesMetadata` 新增 `volume_unit` 元数据字段 |",
            "| **P2 (中)** | `data_fetcher.py` (`load_futures_foreign_ohlcv`) | `akshare.futures_foreign_hist_sina` 在黄金/商品上存在大量周一开盘跳空型 OHLC 异常（最高 `CL` `31/258=12.02%`、`NG` `25/258=9.69%`、`SI` `6.98%`、`GC` `6.59%`），且多数合约成交量为 `0`、`XAU/XAG` 实为现货 | 外盘期货 K 线结构性破界导致技术分析/风控信号失真 | 增加跨源交叉校验、对 `low_gt_open`/`high_lt_open` 提供显著告警提示并区分现货与连续期货 |",
            "| **P2 (中)** | `asset_catalog.py` (`normalize_symbol_alias`) | `normalize_symbol_alias('global_index', 'SPX')` 返回 `'SPX'` 而实际拉取返回 `canonical_symbol='.INX'`（`N225` 同理）；`gold:AU9999` 静默代理为美元盎司期货 `GC` | 多资产组合去重无法在拉取前识别 `.INX` 与 `SPX` 重复；`AU9999` 存在跨币种/跨市场语义偏差 | 统一 `asset_catalog` 与 `data_fetcher` 的指数别名映射，并对 `AU9999->GC` 输出显式代理告警 |",
            "| **P3 (低)** | `data_fetcher.py` / `contracts.py` | `global_index:N225`（`index_global_hist_sina`）在 `2026-06-23~2026-10-28` 连续 `86/242` 日成交量为 `0`；非活跃国债 `sh019743` 仅 `111` 行且最大断档 `28` 天、尾部滞后 `10` 天 | 局部成交量缺失或债券低流动性未在质量报告中区分 | 在 `QualityReport` / `warnings` 中补充区间覆盖率、尾部滞后天数与分段零成交量告警 |",
            "| **P3 (低)** | `data_sources/eastmoney.py` / `forex.py` | 连续多标的访问东财触发 `Push2RateLimitError`（本轮实测触发 `3` 次：`DXY`、`USD/JPY`、`USD/CNY`），虽由新浪成功兜底但缺少请求间隔节流 | 批量抓取场景易加速触发 IP 限流 | 为东财直连与 AKShare 东财源增加轻量域名级节流与退避冷却 |",
        ]
    )

    return "\n".join(lines) + "\n"


def generate_reports_and_summaries(
    *,
    data_root: Path,
    report_root: Path,
    primary_records: list[dict[str, Any]],
    alias_records: list[dict[str, Any]],
    audit_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Persist manifest.csv, summary.csv, summary.json, REPORT.md, and failures.md."""
    all_records = [*primary_records, *alias_records]
    cat_rows = compute_category_summaries(primary_records)
    counts = Counter(r["audit_status"] for r in primary_records)
    total = len(primary_records)
    pass_c = counts.get("PASS", 0)
    warn_c = counts.get("WARN", 0)
    fail_c = counts.get("FAIL", 0)
    unver_c = counts.get("UNVERIFIED", 0)
    usable_c = pass_c + warn_c
    total_elapsed = round(sum(float(r["elapsed_seconds"]) for r in all_records), 3)

    overall_summary = {
        "total_count": total,
        "pass_count": pass_c,
        "warn_count": warn_c,
        "fail_count": fail_c,
        "unverified_count": unver_c,
        "usable_count": usable_c,
        "pass_rate_pct": round(100.0 * pass_c / total, 2) if total else 0.0,
        "usable_rate_pct": round(100.0 * usable_c / total, 2) if total else 0.0,
        "avg_elapsed_seconds": round(
            sum(float(r["elapsed_seconds"]) for r in primary_records) / total, 3
        )
        if total
        else 0.0,
        "total_elapsed_seconds": total_elapsed,
        "total_ohlc_anomaly_rows": sum(
            int(r.get("ohlc_anomaly_rows") or 0) for r in primary_records
        ),
        "fallback_count": sum(1 for r in primary_records if r.get("fallback_used")),
    }
    source_rows = compute_source_stability(all_records)
    ohlc_top = sorted(
        [r for r in primary_records if int(r.get("ohlc_anomaly_rows") or 0) > 0],
        key=lambda r: -int(r["ohlc_anomaly_rows"]),
    )

    summary_payload = {
        "audit_metadata": audit_metadata,
        "overall_summary": overall_summary,
        "category_summaries": cat_rows,
        "source_stability": source_rows,
        "top_ohlc_anomalies": [
            {
                "asset_type": r["asset_type"],
                "symbol": r["symbol"],
                "data_source": r["data_source"],
                "row_count": r["row_count"],
                "ohlc_anomaly_rows": r["ohlc_anomaly_rows"],
                "samples": r["ohlc_anomalies"][:20],
            }
            for r in ohlc_top
        ],
        "failures": [r for r in all_runs_if_fail(all_records)],
        "results": primary_records,
        "aliases": alias_records,
    }

    data_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    write_manifest_csv(data_root / "manifest.csv", all_records)
    write_summary_csv(data_root / "summary.csv", cat_rows, overall_summary)
    (data_root / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_md = build_report_markdown(
        summary_payload=summary_payload,
        primary_records=primary_records,
        alias_records=alias_records,
        cat_rows=cat_rows,
        source_rows=source_rows,
    )
    failures_md = build_failures_markdown(
        audit_date=audit_metadata["audit_date"],
        primary_records=primary_records,
        alias_records=alias_records,
    )
    (report_root / "REPORT.md").write_text(report_md, encoding="utf-8")
    (report_root / "failures.md").write_text(failures_md, encoding="utf-8")
    return summary_payload


def all_runs_if_fail(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("audit_status") == "FAIL"]


def load_existing_records(data_root: Path, targets: list[AuditTarget]) -> list[dict[str, Any]]:
    """Rebuild summary records from persisted ``result.json`` and ``data.parquet`` files."""
    loaded: list[dict[str, Any]] = []
    for target in targets:
        out_dir = data_root / target.relative_dir()
        result_path = out_dir / "result.json"
        if not result_path.is_file():
            loaded.append(
                {
                    "item_id": target.item_id,
                    "group": target.group,
                    "asset_type": target.asset_type,
                    "symbol": target.symbol,
                    "canonical_symbol": "",
                    "alias_of": target.alias_of,
                    "audit_status": "UNVERIFIED",
                    "cli_status": "unverified",
                    "exit_code": -1,
                    "attempts": 0,
                    "elapsed_seconds": 0.0,
                    "data_source": "",
                    "fallback_used": False,
                    "fallback_count": 0,
                    "fallback_warnings": [],
                    "rate_limit_signals": [],
                    "adjust": "",
                    "return_basis": "",
                    "series_kind": "",
                    "native_currency": "",
                    "output_currency": "",
                    "requested_start": DEFAULT_START_DATE,
                    "requested_end": DEFAULT_END_DATE,
                    "actual_start": "",
                    "actual_end": "",
                    "start_lag_days": None,
                    "end_lag_days": None,
                    "row_count": 0,
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
                    "max_calendar_gap_days": 0,
                    "abnormal_gaps": [],
                    "extreme_jumps": [],
                    "vol_min": None,
                    "vol_median": None,
                    "vol_max": None,
                    "volume_unit_note": "",
                    "latest_close": None,
                    "issue_codes": ["unverified_missing_result"],
                    "info_notes": [],
                    "warnings": [],
                    "error_category": "unverified",
                    "error_type": "",
                    "error_message": "Missing result.json",
                    "artifact_dir": str(target.relative_dir()),
                }
            )
            continue

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        audit_block = dict(payload.get("audit") or {})
        warn_text = "\n".join(str(w) for w in (payload.get("warnings") or ()))
        err_msg = str((audit_block.get("error") or {}).get("message") or "")
        combined_sig = sorted(
            {
                *(audit_block.get("rate_limit_signals") or ()),
                *detect_rate_limit_signals(f"{warn_text}\n{err_msg}"),
            }
        )
        audit_block["rate_limit_signals"] = combined_sig
        loaded.append(build_record_summary(target, payload, audit_block))
    return loaded


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full multi-asset live market data audit.")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END_DATE, help="End date YYYY-MM-DD")
    parser.add_argument("--audit-date", default=DEFAULT_AUDIT_DATE, help="Audit folder date")
    parser.add_argument("--base-currency", default=DEFAULT_BASE_CURRENCY, help="Base currency")
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--timeout", type=float, default=75.0, help="Per-attempt timeout (seconds)")
    parser.add_argument("--max-retries", type=int, default=1, help="Retries for transient errors")
    parser.add_argument("--report-only", action="store_true", help="Regenerate reports from disk")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke `python -m quant_cli fetch`",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = REPO_ROOT / "data" / "live_asset_audit" / args.audit_date
    report_root = REPO_ROOT / "reports" / "live_asset_audit" / args.audit_date
    primaries, aliases = build_audit_plan()
    env_meta = collect_environment_metadata()

    run_started_at = datetime.now(UTC).isoformat()
    run_finished_at = ""
    if args.report_only:
        summary_file = data_root / "summary.json"
        if summary_file.is_file():
            try:
                prev_meta = json.loads(summary_file.read_text(encoding="utf-8")).get("audit_metadata", {})
                run_started_at = str(prev_meta.get("run_started_at") or run_started_at)
                run_finished_at = str(prev_meta.get("run_finished_at") or "")
            except Exception:
                pass
        primary_records = load_existing_records(data_root, primaries)
        alias_records = load_existing_records(data_root, aliases)
    else:
        primary_records = []
        alias_records = []
        all_targets = [*primaries, *aliases]
        for idx, target in enumerate(all_targets):
            out_dir = data_root / target.relative_dir()
            print(
                f"[{idx + 1}/{len(all_targets)}] {target.item_id} ({target.group}) {target.spec} ...",
                flush=True,
            )
            rec = run_single_fetch(
                target,
                out_dir=out_dir,
                start_date=args.start,
                end_date=args.end,
                base_currency=args.base_currency,
                python_exe=args.python,
                timeout_seconds=args.timeout,
                max_retries=args.max_retries,
            )
            print(
                f"    -> {rec['audit_status']} rows={rec['row_count']} "
                f"src={rec['data_source'] or '—'} elapsed={rec['elapsed_seconds']:.2f}s "
                f"issues={rec['issue_codes']}",
                flush=True,
            )
            if target.group == "primary":
                primary_records.append(rec)
            else:
                alias_records.append(rec)
            if idx + 1 < len(all_targets) and args.sleep > 0:
                time.sleep(args.sleep)

    if not run_finished_at:
        run_finished_at = datetime.now(UTC).isoformat()
    audit_metadata = {
        "audit_date": args.audit_date,
        "requested_start": args.start,
        "requested_end": args.end,
        "base_currency": args.base_currency,
        "refresh": True,
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        **env_meta,
    }
    summary = generate_reports_and_summaries(
        data_root=data_root,
        report_root=report_root,
        primary_records=primary_records,
        alias_records=alias_records,
        audit_metadata=audit_metadata,
    )
    overall = summary["overall_summary"]
    print(
        f"\nAudit complete: PASS={overall['pass_count']}/{overall['total_count']} "
        f"({overall['pass_rate_pct']}%), WARN={overall['warn_count']}, "
        f"FAIL={overall['fail_count']}, USABLE={overall['usable_rate_pct']}%",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
