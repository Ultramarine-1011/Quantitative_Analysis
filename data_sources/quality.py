"""Source-specific quote semantics; prices and volume values are never rescaled."""

import pandas as pd

from asset_catalog import SUPPORTED_ADJUSTMENTS, resolve_adjustment

_LOT_SOURCES = {
    "eastmoney_direct",
    "akshare.fund_etf_hist_em",
    "akshare.stock_zh_a_hist",
}
_SHARE_SOURCES = {
    "akshare.stock_zh_a_hist_tx",
    "akshare.stock_zh_a_daily_sina",
    "akshare.fund_etf_hist_sina",
}


def source_volume_unit(asset_type: str, source: str) -> str | None:
    if asset_type in {"ashare", "etf", "creit"}:
        if source in _LOT_SOURCES:
            return "lots_100"
        if source in _SHARE_SOURCES:
            return "shares_or_units"
    return {
        "fx": "not_applicable",
        "bond_us_yield": "not_applicable",
        "mutual_fund": "not_applicable",
        "us_stock": "shares",
        "hk_stock": "shares",
        "crypto": "base_asset_units",
    }.get(asset_type)


def annotate_quote_quality(
    frame: pd.DataFrame, asset_type: str, requested_adjust: str | None
) -> None:
    """Keep the actual provider conventions and flag conflicts for downstream research."""
    attrs = frame.attrs
    source = str(attrs.get("data_source", ""))
    warnings = list(attrs.get("quality_warnings") or ())
    expected_unit = source_volume_unit(asset_type, source)
    if expected_unit:
        if attrs.setdefault("volume_unit", expected_unit) != expected_unit:
            warnings.append("volume_unit_mismatch")
    if asset_type in SUPPORTED_ADJUSTMENTS and attrs.get("adjust") is not None:
        if attrs["adjust"] != resolve_adjustment(asset_type, requested_adjust):
            warnings.append("adjustment_mismatch")
    if asset_type in {"gold", "commodity"} and "Volume" in frame and frame["Volume"].eq(0).all():
        warnings.append("all_zero_volume")
    if asset_type == "creit":
        if source in {"eastmoney_direct", "akshare.stock_zh_a_hist_tx"}:
            attrs.update(adjust="qfq", return_basis="adjusted_price")
        elif source == "akshare.fund_etf_hist_sina":
            attrs.update(adjust="", return_basis="unadjusted_price")
        elif source in {"akshare.fund_reits_hist_em", "akshare.reits_hist_em"}:
            warnings.append("adjustment_unspecified")
    if warnings:
        attrs["quality_warnings"] = list(dict.fromkeys(warnings))
