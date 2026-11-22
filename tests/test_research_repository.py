"""Exercise publication, replacement and offline replay through the public CLI."""

import json

import pandas as pd
import pytest

from artifacts import load_result_from_artifacts
from data_sources.errors import SnapshotError
from data_sources.repository import CachedQuoteRepository
from data_sources.research import ResearchRepository
from data_sources.snapshots import SnapshotStore
from quant_cli import run_cli_command
from tests.test_m6_1_artifacts_and_migration import _multi_asset_loaders


def forbidden(*args, **kwargs):
    pytest.fail("offline replay called a quote provider")


@pytest.mark.parametrize(
    ("cmd", "args"),
    [
        ("fetch", ["us_stock:SPY"]),
        ("analyze", ["us_stock:SPY"]),
        ("compare", ["etf:510300", "us_stock:SPY", "--benchmark", "global_index:000300"]),
        (
            "portfolio",
            [
                "etf:510300",
                "us_stock:SPY",
                "--benchmark",
                "global_index:000300",
                "--strategy",
                "equal_weight",
                "--no-sensitivity",
            ],
        ),
    ],
)
def test_record_relocate_recompute_and_render(tmp_path, cmd, args, capsys):
    store = tmp_path / "research"
    loader, fx_loader = _multi_asset_loaders()
    common = [
        cmd,
        *args,
        "--start",
        "2024-01-02",
        "--end",
        "2024-05-10",
        "--base-currency",
        "CNY",
        "-q",
    ]
    code, live = run_cli_command(
        [*common, "--snapshot-dir", str(store), "--data-mode", "online"],
        loader=loader,
        fx_loader=fx_loader,
    )
    assert code == 0, capsys.readouterr().err
    pinned = live.data_summary["snapshot_id"]
    manifest = json.loads((store / "manifests" / f"{pinned}.json").read_bytes())
    assert {"fx", "asset"} == {item["query"]["role"] for item in manifest["inputs"].values()}
    store = store.rename(tmp_path / "copy")
    out = tmp_path / "out"
    code, replay = run_cli_command(
        [
            *common,
            "--snapshot-dir",
            str(store),
            "--snapshot-id",
            pinned,
            "-o",
            str(out),
            *(["--report", "all"] if cmd != "fetch" else []),
        ],
        loader=forbidden,
        fx_loader=forbidden,
        usdt_usd_loader=forbidden,
    )
    assert code == 0, capsys.readouterr().err
    assert live.to_dict()["metrics"] == replay.to_dict()["metrics"]
    assert live.warnings == replay.warnings
    pd.testing.assert_frame_equal(live.data_frame, replay.data_frame)
    assert load_result_from_artifacts(out).data_summary["snapshot_id"] == pinned
    if cmd != "fetch":
        assert (out / "report.html").is_file()


def test_refresh_preserves_pinned_run_and_corrupt_local_data_fails_closed(tmp_path, sample_ohlcv):
    calls = []

    def provider(req, *, proxy_url):
        calls.append(req)
        frame = sample_ohlcv.copy() * len(calls)
        frame.attrs.update(data_source="test_provider", currency="CNY", volume_unit="units")
        return frame

    base = CachedQuoteRepository(provider, namespace="test")
    args = ("etf", "510300", "2024-01-02", "2024-06-01")
    first = ResearchRepository(tmp_path, mode="prefer_local", repository=base)
    original = first.load(*args)
    old_id = first.finish()
    cached = ResearchRepository(tmp_path, mode="prefer_local", repository=base)
    pd.testing.assert_frame_equal(cached.load(*args), original)
    assert cached.finish() == old_id and len(calls) == 1
    fresh = ResearchRepository(tmp_path, mode="online", repository=base)
    assert not fresh.load(*args).equals(original)
    assert fresh.finish() != old_id
    replay = ResearchRepository(tmp_path, mode="offline", repository=base, snapshot_id=old_id)
    pd.testing.assert_frame_equal(replay.load(*args), original)
    with pytest.raises(SnapshotError, match="Missing offline input"):
        replay.load("etf", "other", args[2], args[3])
    entry = next(iter(replay.inputs.values()))
    (tmp_path / "objects" / (entry["sha256"] + ".parquet")).write_bytes(b"partial")
    with pytest.raises(SnapshotError, match="SHA-256"):
        replay.load(*args)
    assert len(calls) == 2


def test_failed_snapshot_publication_preserves_prior_selection(tmp_path, monkeypatch):
    store = SnapshotStore(tmp_path)
    prior = store.publish_run({})

    def failed(*args):
        raise OSError("injected pointer replacement failure")

    monkeypatch.setattr("os.replace", failed)
    with pytest.raises(OSError, match="injected"):
        store.publish_run({"other": {}})
    assert store.select(None) == (prior, {})
    assert not any(p.is_file() and p.name != "latest" for p in tmp_path.iterdir())


@pytest.mark.parametrize("available", [True, False])
def test_stablecoin_rate_and_explicit_peg_replay(tmp_path, sample_ohlcv, available, capsys):
    calls = []

    def asset(**kwargs):
        frame = sample_ohlcv.copy()
        frame.attrs.update(
            asset_type="crypto", symbol="BTC/USDT", currency="USDT", data_source="fixture"
        )
        return frame

    def stablecoin(*args):
        calls.append(1)
        if not available:
            raise ConnectionError("unavailable fixture")
        return sample_ohlcv.assign(Close=0.99)

    common = [
        "fetch",
        "crypto:BTC/USDT",
        "--base-currency",
        "USD",
        "--allow-usdt-peg",
        "--no-data",
        "--start",
        "2024-01-02",
        "--end",
        "2024-05-31",
        "--snapshot-dir",
        str(tmp_path),
    ]
    code, first = run_cli_command(
        [*common, "--data-mode", "online"], loader=asset, usdt_usd_loader=stablecoin
    )
    assert code == 0, capsys.readouterr().err
    code, replay = run_cli_command(
        [*common, "--data-mode", "offline"], loader=forbidden, usdt_usd_loader=forbidden
    )
    assert code == 0, capsys.readouterr().err
    assert calls == [1]
    assert first.warnings == replay.warnings
    pd.testing.assert_frame_equal(first.data_frame, replay.data_frame)
    if available:
        entries = SnapshotStore(tmp_path).select(first.data_summary["snapshot_id"])[1].values()
        fx = next(entry for entry in entries if entry["query"]["role"] == "usdt_usd")
        (tmp_path / "objects" / (fx["sha256"] + ".parquet")).unlink()
        code, _ = run_cli_command([*common, "--data-mode", "offline"], loader=forbidden)
        assert code == 1 and "Missing snapshot file" in capsys.readouterr().err
