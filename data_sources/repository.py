"""Quote requests, replaceable providers and the shared short-lived market cache."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import pandas as pd

from asset_catalog import (
    SUPPORTED_ADJUSTMENTS,
    SUPPORTED_ASSET_TYPES,
    normalize_symbol_alias,
    resolve_adjustment,
    resolve_return_basis,
)
from currency import infer_asset_currency
from data_sources.cache import load_cached_ohlcv, save_cached_ohlcv
from data_sources.crypto import normalize_proxy_url
from data_sources.quality import annotate_quote_quality
from ohlcv import validate_analysis_date_range


@dataclass(frozen=True)
class QuoteRequest:
    asset_type: str
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    adjust: str | None

    @classmethod
    def create(
        cls, asset_type: str, symbol: str, start: object, end: object, adjust: str | None = None
    ):
        key = asset_type.strip().lower()
        if key not in SUPPORTED_ASSET_TYPES:
            raise ValueError(f"不支持的资产类型: {asset_type}")
        if adjust is not None:
            resolve_adjustment(key, adjust)
        first, last = validate_analysis_date_range(start, end)
        return cls(key, symbol, first, last, adjust)

    def identity(self) -> dict[str, str]:
        return dict(
            asset_type=self.asset_type,
            symbol=normalize_symbol_alias(self.asset_type, self.symbol),
            adjust=resolve_adjustment(self.asset_type, self.adjust),
            start=str(self.start.date()),
            end=str(self.end.date()),
        )


class QuoteProvider(Protocol):
    def __call__(self, request: QuoteRequest, *, proxy_url: str | None) -> pd.DataFrame: ...


class QuoteRepository(Protocol):
    def load(
        self,
        asset_type: str,
        symbol: str,
        start_date: object,
        end_date: object,
        proxy_url: str | None = None,
        *,
        adjust: str | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame: ...


def attach_quote_metadata(frame: pd.DataFrame, request: QuoteRequest) -> None:
    key = request.asset_type
    annotate_quote_quality(frame, key, request.adjust)
    attrs = frame.attrs
    symbol = str(attrs.get("symbol") or request.symbol).strip()
    attrs.setdefault("asset_type", key)
    if key in SUPPORTED_ADJUSTMENTS:
        attrs.setdefault("adjust", resolve_adjustment(key, request.adjust))
    attrs["return_basis"] = resolve_return_basis(
        key, adjust=attrs.get("adjust", request.adjust), attrs=attrs
    )
    attrs.setdefault("canonical_symbol", normalize_symbol_alias(key, symbol))
    attrs.setdefault("currency", infer_asset_currency(key, symbol, attrs))
    attrs.setdefault("fetched_at", datetime.now(UTC).isoformat())
    attrs.setdefault("cache_hit", False)


class CachedQuoteRepository:
    """Read through the TTL cache; refresh bypasses reads but updates successful quotes."""

    def __init__(self, provider: QuoteProvider, *, namespace: str):
        self.namespace = namespace
        self.provider = provider

    def load(
        self,
        asset_type: str,
        symbol: str,
        start_date: object,
        end_date: object,
        proxy_url: str | None = None,
        *,
        adjust: str | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        req = QuoteRequest.create(asset_type, symbol, start_date, end_date, adjust)
        if req.asset_type == "crypto" and proxy_url is not None:
            normalize_proxy_url(proxy_url)
        args = (req.asset_type, symbol, req.start, req.end)
        frame = load_cached_ohlcv(
            *args, adjust=adjust, force_refresh=force_refresh, namespace=self.namespace
        )
        if frame is not None:
            attach_quote_metadata(frame, req)
            return frame
        frame = self.provider(req, proxy_url=proxy_url)
        attach_quote_metadata(frame, req)
        if not frame.empty:
            path = save_cached_ohlcv(frame, *args, adjust=adjust, namespace=self.namespace)
            if path is not None:
                frame.attrs["cache_path"] = str(path)
        return frame
