"""Cooperative process and thread locks for cache entries and bundle commits."""

import errno
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX platforms
    msvcrt = None  # type: ignore[assignment]

_KEY_THREAD_LOCKS_GUARD = threading.Lock()
_KEY_THREAD_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_FILE_LOCK_TIMEOUT_SECONDS = 30.0


@contextmanager
def _msvcrt_file_lock(
    fh: Any, *, timeout_seconds: float = _FILE_LOCK_TIMEOUT_SECONDS
) -> Iterator[None]:
    """Cross-process byte-range lock on Windows via ``msvcrt.locking``."""
    fh.seek(0, os.SEEK_END)
    if fh.tell() == 0:
        fh.write(b"\0")
        fh.flush()
    fh.seek(0)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[union-attr]
            break
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EDEADLK, errno.EAGAIN}:
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for cache entry Windows file lock.") from exc
            time.sleep(0.005)
    try:
        yield
    finally:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[union-attr]


@contextmanager
def entry_lock(directory: Path, key: str) -> Iterator[None]:
    """Serialize read/write transitions on a single cache key across threads and processes."""
    lock_scope = (os.path.abspath(os.fspath(directory)), key)
    with _KEY_THREAD_LOCKS_GUARD:
        tlock = _KEY_THREAD_LOCKS.setdefault(lock_scope, threading.Lock())
    with tlock:
        if fcntl is None and msvcrt is None:
            yield
            return
        lock_dir = directory / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{key}.lock"
        with lock_path.open("a+b") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            else:
                with _msvcrt_file_lock(fh):
                    yield
