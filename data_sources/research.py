"""Per-run repository that records or replays all market and currency inputs."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from data_sources.errors import EmptySymbolDataError, SnapshotError, VendorUpstreamError
from data_sources.network import classify_network_exception
from data_sources.repository import QuoteRepository, QuoteRequest, attach_quote_metadata
from data_sources.snapshots import SnapshotStore, request_key

DataMode = Literal["online", "prefer_local", "offline"]


class ResearchRepository:
    """Create one instance per analysis. Publish after all requested legs succeed."""

    def __init__(
        self,
        store: Path,
        *,
        mode: DataMode,
        repository: QuoteRepository,
        snapshot_id: str | None = None,
        asset_loader: Callable | None = None,
        refresh: bool = False,
        fx_loader: Callable | None = None,
        usdt_usd_loader: Callable | None = None,
    ):
        if mode not in ("online", "prefer_local", "offline"):
            raise ValueError(f"Unknown data mode: {mode}")
        if snapshot_id and mode != "offline":
            raise ValueError("A pinned snapshot requires offline mode.")
        self.store = SnapshotStore(store)
        self.repository = repository
        self.mode = mode
        self.snapshot_id, self.available = (
            (None, {}) if mode == "online" else self.store.select(snapshot_id)
        )
        if mode == "offline" and self.snapshot_id is None:
            raise SnapshotError(f"No published research snapshot in {store}")
        self.inputs: dict[str, dict[str, Any]] = {}
        self.asset_loader = asset_loader or repository.load
        self.refresh = refresh
        self.fx_loader = fx_loader
        self.usdt_usd_loader = usdt_usd_loader

    def _read(
        self,
        role: str,
        req: QuoteRequest,
        fetch: Callable[[], pd.DataFrame],
        *,
        refresh: bool = False,
    ):
        refresh = refresh or self.refresh
        if refresh and self.mode == "offline":
            raise ValueError("Offline data cannot be refreshed.")
        query = {"role": role, **req.identity()}
        key = request_key(query)
        entry = self.inputs.get(key)
        if entry is None and not refresh:
            entry = self.available.get(key)
            # Retry previously unavailable auxiliary rates on the next online-capable run.
            if self.mode == "prefer_local" and entry and "unavailable" in entry:
                entry = None
        if entry is not None:
            if entry["query"] != query:
                raise SnapshotError("Snapshot input does not match the requested quote")
            self.inputs[key] = entry
            if "unavailable" in entry:
                raise VendorUpstreamError(f"Recorded unavailable rate: {entry['unavailable']}")
            return self.store.load_frame(entry)
        if self.mode == "offline":
            raise SnapshotError(f"Missing offline input: {query}")
        try:
            frame = fetch()
            if frame is None or frame.empty:
                raise EmptySymbolDataError(f"Empty market input: {query}")
        except Exception as exc:
            if role != "asset":
                self.inputs[key] = dict(query=query, unavailable=classify_network_exception(exc))
            raise
        frame = frame.copy(deep=False)
        attach_quote_metadata(frame, req)
        frame.attrs.setdefault("data_source", "unspecified")
        frame.attrs.setdefault("volume_unit", "unknown")
        self.inputs[key] = self.store.save_frame(query, frame)
        return frame

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
    ):
        req = QuoteRequest.create(asset_type, symbol, start_date, end_date, adjust)
        return self._read(
            "asset",
            req,
            lambda: self.asset_loader(
                asset_type=asset_type,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                proxy_url=proxy_url,
                adjust=adjust,
                force_refresh=force_refresh or self.refresh or self.mode == "online",
            ),
            refresh=force_refresh,
        )

    def fx(self, symbol: str, start: object, end: object):
        req = QuoteRequest.create("fx", symbol, start, end)

        def fetch():
            if self.fx_loader is not None:
                return self.fx_loader(symbol, start, end)
            return self.repository.load(
                "fx", symbol, start, end, force_refresh=self.refresh or self.mode == "online"
            )

        return self._read("fx", req, fetch)

    def usdt(self, start: object, end: object, proxy_url: str | None):
        req = QuoteRequest.create("crypto", "USDT/USD", start, end)

        def fetch():
            if self.usdt_usd_loader is not None:
                return self.usdt_usd_loader(start, end, proxy_url)
            from data_sources.crypto import fetch_usdt_usd_ohlcv

            return fetch_usdt_usd_ohlcv(start, end, proxy_url=proxy_url)

        return self._read("usdt_usd", req, fetch)

    def finish(self) -> str:
        """Return the immutable manifest ID to include in the result's data_summary."""
        if self.mode == "offline":
            assert self.snapshot_id is not None
            return self.snapshot_id
        return self.store.publish_run(self.inputs)
