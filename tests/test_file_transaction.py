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
