"""Access to the files of one scene.

All reads of scene data go through a :class:`Storage`. Decoders take bytes,
so a scene can live in a folder (:class:`DirStorage`) or in the release
archives (:class:`ZipStorage`) without changing anything above this layer.

Storages are plain data (paths), so they pickle into DataLoader workers;
open file handles are per process.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    """Read-only access to the members of one scene (names relative to the scene, ``/``-separated)."""

    def exists(self, name: str) -> bool:
        """Whether the member (file) exists."""

    def is_dir(self, name: str) -> bool:
        """Whether *name* is a folder that holds members."""

    def list(self, prefix: str) -> list[str]:
        """Names of the files directly in folder *prefix*, sorted; empty if there is none."""

    def read_bytes(self, name: str) -> bytes:
        """The member's content.

        Raises:
            FileNotFoundError: If it does not exist.
        """

    def describe(self, name: str = "") -> str:
        """Where a member lives, for messages."""


class DirStorage:
    """A scene stored as an extracted folder."""

    __slots__ = ("root",)

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"DirStorage({str(self.root)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DirStorage) and other.root == self.root

    def __hash__(self) -> int:
        return hash(self.root)

    def __getstate__(self) -> dict:
        return {"root": self.root}

    def __setstate__(self, state: dict) -> None:
        self.root = state["root"]

    def path(self, name: str) -> Path:
        """Filesystem path of a member."""
        return self.root / name

    def locate(self, name: str) -> tuple[Path, int, int, Optional[str]]:
        """Where a member's bytes are: ``(file, offset, size, member name in the zip or None)``.

        Raises:
            FileNotFoundError: If it does not exist.
        """
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        return path, 0, path.stat().st_size, None

    def exists(self, name: str) -> bool:
        return (self.root / name).is_file()

    def is_dir(self, name: str) -> bool:
        return (self.root / name).is_dir()

    def list(self, prefix: str) -> list[str]:
        folder = self.root / prefix
        try:
            with os.scandir(folder) as entries:
                names = [e.name for e in entries if e.is_file()]
        except (FileNotFoundError, NotADirectoryError):
            return []
        prefix = prefix.rstrip("/")
        return sorted(f"{prefix}/{n}" if prefix else n for n in names)

    def read_bytes(self, name: str) -> bytes:
        try:
            return (self.root / name).read_bytes()
        except IsADirectoryError:
            raise FileNotFoundError(f"Not a file: {self.root / name}") from None

    def describe(self, name: str = "") -> str:
        return str(self.root / name) if name else str(self.root)


class ZipStorage(DirStorage):
    """A scene in the release archive format (:mod:`kitscenes._archive`), read without extracting.

    Loose files come from the folder; archive members are read with one exact
    ``pread`` at the offset in ``index.parquet``. A component whose archive is
    missing but whose folder exists (extracted) is read from the folder, one
    with neither is simply absent (not downloaded).

    Args:
        root: The scene folder (with ``index.parquet``).
        verify: Check each member's CRC32 (costs ~25 % on one core; archives are
            already SHA-256-checked on download). A mismatch, and a member the zip ends
            before (a short read), raise :class:`kitscenes.errors.ChecksumError`.
    """

    __slots__ = ("verify", "_index", "_handles", "_pid")

    def __init__(self, root: Path | str, *, verify: bool = False) -> None:
        super().__init__(root)
        self.verify = verify
        self._index = None
        self._handles: dict[str, int] = {}
        self._pid = os.getpid()

    def __repr__(self) -> str:
        return f"ZipStorage({str(self.root)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ZipStorage) and other.root == self.root

    def __hash__(self) -> int:
        return hash(("zip", self.root))

    def __getstate__(self) -> dict:
        return {"root": self.root, "verify": self.verify}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["root"], verify=state["verify"])

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Close the archive file handles (they reopen on the next read)."""
        if getattr(self, "_pid", None) == os.getpid():
            for fd in getattr(self, "_handles", {}).values():
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._handles = {}

    @property
    def index(self):
        """The parsed ``index.parquet`` (:class:`kitscenes._archive.ArchiveIndex`)."""
        if self._index is None:
            from kitscenes._archive import INDEX_FILE, ArchiveIndex

            self._index = ArchiveIndex.from_bytes((self.root / INDEX_FILE).read_bytes(), index_overlays(self.root))
        return self._index

    def _archive(self, component: str) -> Path:
        from kitscenes._archive import ARCHIVE_SUFFIX

        return self.root / f"{component}{ARCHIVE_SUFFIX}"

    def packed(self, name: str):
        """The ``index.parquet`` entry ``(component, offset, size, crc32)`` of member *name* if it is
        read from its zip (the zip is in the scene folder), else None (extracted, not downloaded, or no member)."""
        entry = self.index.entry(name)
        if entry is None or not self._archive(entry[0]).is_file():
            return None
        return entry

    def exists(self, name: str) -> bool:
        return self.packed(name) is not None or super().exists(name)

    def locate(self, name: str) -> tuple[Path, int, int, Optional[str]]:
        entry = self.packed(name)
        if entry is None:  # extracted (or missing: FileNotFoundError)
            return super().locate(name)
        component, offset, size, _ = entry
        return self._archive(component), offset, size, f"{self.root.name}/{name}"  # members are <scene_id>/<path>

    def is_dir(self, name: str) -> bool:
        if super().is_dir(name):
            return True
        return any(self._archive(c).is_file() for c in self.index.folder_components(name))

    def list(self, prefix: str) -> list[str]:
        from kitscenes._archive import component_of

        prefix = prefix.rstrip("/")
        names = set(super().list(prefix))
        present = {c for c in self.index.folder_components(prefix) if self._archive(c).is_file()}
        if present:
            names.update(m for m in self.index.children(prefix) if component_of(m) in present)
        return sorted(names)

    def read_bytes(self, name: str) -> bytes:
        entry = self.packed(name)
        if entry is None:
            return super().read_bytes(name)
        component, offset, size, crc = entry
        data = os.pread(self._handle(component), size, offset)
        if len(data) != size:
            from kitscenes.errors import ChecksumError

            raise ChecksumError(f"Short read of {name} from {self._archive(component)} ({len(data)} of {size} bytes): "
                                "the zip is truncated; download the scene again")
        if self.verify:
            import zlib

            if zlib.crc32(data) != crc:
                from kitscenes.errors import ChecksumError

                raise ChecksumError(f"CRC mismatch for {name} in {self._archive(component)}: the member is corrupt; "
                                    "download the scene again")
        return data

    def _handle(self, component: str) -> int:
        if self._pid != os.getpid():  # forked: never share file offsets/handles with the parent
            self._handles = {}
            self._pid = os.getpid()
        fd = self._handles.get(component)
        if fd is None:
            fd = os.open(self._archive(component), os.O_RDONLY)
            kept = self._handles.setdefault(component, fd)  # atomic: threads racing here keep one handle
            if kept != fd:
                os.close(fd)
            fd = kept
        return fd

    def describe(self, name: str = "") -> str:
        entry = self.index.entry(name) if name else None
        if entry is not None:
            return f"{self._archive(entry[0])}:{name}"
        return super().describe(name)


def index_overlays(scene_dir: Path | str) -> list[bytes]:
    """The index rows of re-written components of a scene folder (``.transcoded/<camera>.parquet`` of the record
    that matches the zip on disk, :mod:`kitscenes.transcode`), to read on top of its ``index.parquet``."""
    from kitscenes.transcode import read_overlays

    return read_overlays(scene_dir)


def open_storage(scene_dir: Path | str, *, verify: bool = False) -> DirStorage:
    """:class:`ZipStorage` for a scene folder with ``index.parquet`` (*verify*: CRC-32 per member), else :class:`DirStorage`."""
    from kitscenes._archive import INDEX_FILE

    scene_dir = Path(scene_dir)
    return ZipStorage(scene_dir, verify=verify) if (scene_dir / INDEX_FILE).is_file() else DirStorage(scene_dir)
