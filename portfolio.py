"""家庭资产台账的导入、清洗、估值与汇总。"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd


PORTFOLIO_COLUMNS: list[str] = [
    "account",
    "item_type",
    "asset_class",
    "asset_type",
    "symbol",
    "name",
    "quantity",
    "current_price",
    "currency",
    "is_risk_asset",
    "target_weight",
    "notes",
]
REQUIRED_PORTFOLIO_COLUMNS: tuple[str, ...] = (
    "item_type",
    "asset_class",
    "quantity",
    "current_price",
)


def _parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "是", "风险", "risk"}:
        return True
    if text in {"0", "false", "f", "no", "n", "否", "", "nan", "none"}:
        return False
    return False


def _coerce_target_weight(series: pd.Series) -> pd.Series:
    weights = pd.to_numeric(series, errors="coerce").fillna(0.0).astype("float64")
    weights = weights.where(weights <= 1.0, weights / 100.0)
    return weights.clip(lower=0.0)


def empty_holdings_frame() -> pd.DataFrame:
    """返回 Streamlit 可编辑表的空台账模板。"""
    return pd.DataFrame(columns=PORTFOLIO_COLUMNS)


def normalize_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """统一家庭资产台账字段、类型和空值。

    必填列为 ``item_type``、``asset_class``、``quantity``、``current_price``；
    其他列缺失时使用安全默认值补齐，便于手动录入表逐步填写。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    missing = [col for col in REQUIRED_PORTFOLIO_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("缺少必填列: %s" % ", ".join(missing))

    out = df.copy()
    for col in PORTFOLIO_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out.loc[:, PORTFOLIO_COLUMNS].copy()
    out = out.dropna(how="all")
    if out.empty:
        return empty_holdings_frame()

    text_cols = [
        "account",
        "item_type",
        "asset_class",
        "asset_type",
        "symbol",
        "name",
        "currency",
        "notes",
    ]
    for col in text_cols:
        out[col] = out[col].fillna("").astype(str).str.strip()

    out["item_type"] = out["item_type"].str.lower()
    out["item_type"] = out["item_type"].replace(
        {
            "资产": "asset",
            "负债": "liability",
        }
    )
    invalid_types = sorted(set(out["item_type"]) - {"asset", "liability"})
    if invalid_types:
        raise ValueError("item_type 只能为 asset 或 liability: %s" % ", ".join(invalid_types))

    out["asset_class"] = out["asset_class"].replace("", "未分类")
    out["currency"] = out["currency"].replace("", "CNY").str.upper()
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce").fillna(0.0).astype("float64")
    out["current_price"] = (
        pd.to_numeric(out["current_price"], errors="coerce").fillna(0.0).astype("float64")
    )
    out["is_risk_asset"] = out["is_risk_asset"].map(_parse_bool).astype(bool)
    out["target_weight"] = _coerce_target_weight(out["target_weight"])
    return out


def compute_holdings_values(df: pd.DataFrame) -> pd.DataFrame:
    """添加 ``market_value`` 列；负债保持正数，汇总时再单独扣减。"""
    out = normalize_holdings(df)
    if out.empty:
        valued = out.copy()
        valued["market_value"] = pd.Series(dtype="float64")
        return valued
    out["market_value"] = (out["quantity"].abs() * out["current_price"].abs()).astype("float64")
    return out


def summarize_assets(df: pd.DataFrame) -> dict[str, Any]:
    """按家庭资产口径汇总净资产、占比与资产类别。"""
    valued = compute_holdings_values(df)
    assets = valued[valued["item_type"] == "asset"].copy()
    liabilities = valued[valued["item_type"] == "liability"].copy()

    gross_asset_value = float(assets["market_value"].sum()) if not assets.empty else 0.0
    total_liability_value = float(liabilities["market_value"].sum()) if not liabilities.empty else 0.0
    net_asset_value = gross_asset_value - total_liability_value
    risk_asset_value = float(assets.loc[assets["is_risk_asset"], "market_value"].sum()) if not assets.empty else 0.0
    risk_asset_ratio = risk_asset_value / gross_asset_value if gross_asset_value > 0 else 0.0

    if assets.empty:
        by_class = pd.DataFrame(columns=["market_value", "asset_ratio", "target_weight"])
        by_class.index.name = "asset_class"
    else:
        by_class = assets.groupby("asset_class", dropna=False).agg(
            market_value=("market_value", "sum"),
            target_weight=("target_weight", "sum"),
        )
        by_class["asset_ratio"] = by_class["market_value"] / gross_asset_value
        by_class = by_class.loc[:, ["market_value", "asset_ratio", "target_weight"]].sort_values(
            "market_value", ascending=False
        )
    return {
        "gross_asset_value": gross_asset_value,
        "total_liability_value": total_liability_value,
        "net_asset_value": net_asset_value,
        "risk_asset_value": risk_asset_value,
        "risk_asset_ratio": risk_asset_ratio,
        "asset_class_summary": by_class,
        "valued_holdings": valued,
    }


def holdings_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """导出 UTF-8-SIG CSV，便于 Excel 打开。"""
    normalized = normalize_holdings(df)
    buffer = io.StringIO()
    normalized.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


def read_holdings_csv(file: Any) -> pd.DataFrame:
    """读取上传文件或二进制流中的家庭资产 CSV。"""
    raw = pd.read_csv(file, dtype=str, keep_default_na=False)
    return normalize_holdings(raw)


__all__ = [
    "PORTFOLIO_COLUMNS",
    "REQUIRED_PORTFOLIO_COLUMNS",
    "compute_holdings_values",
    "empty_holdings_frame",
    "holdings_to_csv_bytes",
    "normalize_holdings",
    "read_holdings_csv",
    "summarize_assets",
]
