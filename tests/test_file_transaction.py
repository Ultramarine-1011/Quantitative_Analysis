from pathlib import Path

import pytest

from artifacts.transaction import ArtifactIOError, FileTransaction


def test_failure_removing_stale_file_restores_all_prior_files(tmp_path, monkeypatch):
    first, second, new, stale, fails = (tmp_path / name for name in ("a", "b", "c", "d", "e"))
    for f in (first, second, stale, fails):
        f.write_bytes(f.name.encode())
    unlink = Path.unlink

    def fail_once(path, *args, **kwargs):
        if path == fails:
            raise OSError("injected cleanup failure")
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    with pytest.raises(ArtifactIOError, match="injected"):
        with FileTransaction() as txn:
            for target in (first, second, new):
                txn.stage(target, lambda tmp: tmp.write_bytes(b"replacement"))
            txn.commit([stale, fails])
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == {
        "a": b"a",
        "b": b"b",
        "d": b"d",
        "e": b"e",
    }


def test_failed_restore_keeps_recovery_copy_and_restores_other_targets(tmp_path, monkeypatch):
    import os

    first, second, last = (tmp_path / name for name in ("a", "b", "c"))
    for target in (first, second, last):
        target.write_bytes(b"old")
    replace = os.replace

    def interrupt(src, dest):
        if dest == last or (dest == second and str(src).endswith(".bak.tmp")):
            raise OSError("disk fault")
        replace(src, dest)

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(ArtifactIOError, match="incomplete rollback"):
        with FileTransaction() as txn:
            for target in (first, second, last):
                txn.stage(target, lambda tmp: tmp.write_bytes(b"new"))
            txn.commit()
    assert first.read_bytes() == last.read_bytes() == b"old"
    (backup,) = tmp_path.glob(".*.bak.tmp")
    assert backup.read_bytes() == b"old" and second.read_bytes() == b"new"


def test_concurrent_bundle_publishers_leave_one_complete_version(tmp_path, sample_ohlcv):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from artifacts import load_result_from_artifacts
    from artifacts.store import write_result_outputs
    from contracts import AnalysisRequest
    from services import fetch_single_asset

    barrier = Barrier(4)

    def publish(i):
        request = AnalysisRequest("etf", str(i), "2024-01-02", "2024-05-01")
        frame = sample_ohlcv.copy() * i
        frame.attrs.update(symbol=str(i))
        result = fetch_single_asset(request, raw_df=frame)
        barrier.wait(timeout=5)
        write_result_outputs(result, out_dir=tmp_path, write_csv=i % 2 == 0)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(publish, range(1, 5)))
    result = load_result_from_artifacts(tmp_path)
    i = int(result.asset.symbol)
    assert result.data_frame["Close"].iloc[0] == sample_ohlcv["Close"].iloc[0] * i
    assert (tmp_path / "data.csv").exists() == (i % 2 == 0)
