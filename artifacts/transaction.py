"""Same-filesystem staging and rollback for related output files."""

import os
import shutil
import tempfile
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from pathlib import Path


class ArtifactIOError(OSError):
    """The bundle could not be serialized or committed."""


class FileTransaction(ExitStack):
    """Stage files beside their destinations, then publish the complete write set."""

    def __init__(self):
        super().__init__()
        self.staged: dict[Path, Path] = {}

    def _temp(self, target: Path, suffix: str) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=suffix, dir=target.parent)
        os.close(fd)
        path = Path(name)
        self.callback(path.unlink, missing_ok=True)
        return path

    def stage(self, target: Path, writer: Callable[[Path], object]) -> Path:
        try:
            if target in self.staged:
                raise ValueError(f"Duplicate output target: {target}")
            tmp = self._temp(target, ".stage.tmp")
            writer(tmp)
            self.staged[target] = tmp
            return tmp
        except Exception as exc:
            raise ArtifactIOError(f"Failed staging output file {target}: {exc}") from exc

    def commit(self, stale: Iterable[Path] = ()) -> None:
        backups: dict[Path, Path] = {}
        changed: list[Path] = []
        stale = tuple(stale)
        try:
            for target in (*self.staged, *stale):
                if target.is_symlink() or target.is_file():
                    backup = self._temp(target, ".bak.tmp")
                    if target.is_symlink():
                        backup.unlink()
                        backup.symlink_to(os.readlink(target))
                    else:
                        shutil.copy2(target, backup)
                    backups[target] = backup
            for target, tmp in self.staged.items():
                os.replace(tmp, target)
                changed.append(target)
            for target in stale:
                target.unlink()
                changed.append(target)
        except Exception as exc:
            for target in reversed(changed):
                if target in backups:
                    os.replace(backups[target], target)
                else:
                    target.unlink(missing_ok=True)
            raise ArtifactIOError(f"Failed committing output artifacts cleanly: {exc}") from exc
