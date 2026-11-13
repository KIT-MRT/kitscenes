"""Reading raw aerial tiles: files, members of zips (in place), tile-store ``.npy``.

A tile is a JPEG (the release), a PNG or an
``.npy`` of a tile store (uncompressed, memory-mapped). JPEG and PNG tiles
are read as files or as members of a zip, by offset without extracting
(the release zips store their members uncompressed); decoded tiles are kept
in a bounded LRU. All handles and caches are per process
(:class:`TileReader`): a forked or unpickled copy starts with none, so the
objects that hold a reader go into DataLoader workers as they are.
"""

from __future__ import annotations

import io
import os
import struct
import threading
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Hashable, Optional

import numpy as np

#: Tile formats by file suffix.
SUFFIX_FORMAT = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".npy": "npy"}

#: Default budget of decoded JPEG/PNG tiles per process and tile set (a 2048 px RGB tile is 12 MiB).
DEFAULT_TILE_CACHE_MB = 256
#: Open tile-store memory maps per process (address space only; the pages are the OS page cache).
MAX_OPEN_NPY = 4096

_LOCAL_HEADER = struct.Struct("<4s2B4HL2L2H")  # zip local file header, 30 bytes
_LOCAL_SIGNATURE = b"PK\x03\x04"


class LRU:
    """A thread-safe LRU of arrays, bounded by their total size in bytes and by their count."""

    def __init__(self, max_bytes: int, max_items: int = 1 << 30) -> None:
        self.max_bytes = int(max_bytes)
        self.max_items = int(max_items)
        self._items: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def get(self, key: Hashable, load: Callable[[], np.ndarray], nbytes: Callable[[np.ndarray], int]) -> np.ndarray:
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
                self.hits += 1
                return value[0]
            self.misses += 1
        array = load()  # outside the lock: other threads may read meanwhile
        size = nbytes(array)
        if size > self.max_bytes:
            return array
        with self._lock:
            if key not in self._items:
                self._items[key] = (array, size)
                self._bytes += size
                while self._items and (self._bytes > self.max_bytes or len(self._items) > self.max_items):
                    _, (_, dropped) = self._items.popitem(last=False)
                    self._bytes -= dropped
        return array


def decode_image(data: bytes) -> np.ndarray:
    """A JPEG or PNG as an (H, W, 3) uint8 RGB array."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"))


def load_npy(path: str) -> np.ndarray:
    """A tile-store tile, memory-mapped read-only."""
    return np.load(path, mmap_mode="r")


def npy_is_valid(path: Path | str, shape: tuple[int, ...]) -> bool:
    """Whether *path* is a complete uint8 ``.npy`` of *shape* (header and file size checked, no pixels read)."""
    try:
        with open(path, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                header = np.lib.format.read_array_header_1_0(fh)
            elif version == (2, 0):
                header = np.lib.format.read_array_header_2_0(fh)
            else:
                return False
            offset = fh.tell()
        size = os.path.getsize(path)
    except (OSError, ValueError):
        return False
    got_shape, fortran, dtype = header
    return (tuple(got_shape) == tuple(shape) and dtype == np.uint8 and not fortran
            and size == offset + int(np.prod(shape)))


class ZipMembers:
    """The members of one zip, read in place: by offset when stored (no extraction), else via :mod:`zipfile`.

    Opened in one process; :class:`TileReader` makes one per process.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        with zipfile.ZipFile(self.path) as zf:
            self.infos = {i.filename: i for i in zf.infolist() if not i.is_dir()}
        self._fd = os.open(self.path, os.O_RDONLY)
        self._offsets: dict[str, int] = {}
        self._zipfile: Optional[zipfile.ZipFile] = None
        self._lock = threading.Lock()

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        if self._zipfile is not None:
            self._zipfile.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass

    def __contains__(self, name: str) -> bool:
        return name in self.infos

    def _data_offset(self, info: zipfile.ZipInfo) -> int:
        offset = self._offsets.get(info.filename)
        if offset is None:
            header = os.pread(self._fd, _LOCAL_HEADER.size, info.header_offset)
            fields = _LOCAL_HEADER.unpack(header)
            if fields[0] != _LOCAL_SIGNATURE:
                raise zipfile.BadZipFile(f"{self.path}: bad local header of {info.filename}")
            name_len, extra_len = fields[-2], fields[-1]
            offset = self._offsets[info.filename] = info.header_offset + _LOCAL_HEADER.size + name_len + extra_len
        return offset

    def read(self, name: str) -> bytes:
        info = self.infos[name]
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:  # compressed or encrypted
            with self._lock:
                if self._zipfile is None:
                    self._zipfile = zipfile.ZipFile(self.path)
                return self._zipfile.read(name)  # zipfile checks the CRC
        data = os.pread(self._fd, info.file_size, self._data_offset(info))
        if len(data) != info.file_size:
            raise zipfile.BadZipFile(f"{self.path}: {name} is truncated")
        return data


class TileReader:
    """Per-process handles and caches for reading tiles: open zips, memory maps, decoded tiles.

    Every access checks the process ID: after a fork the child drops what it
    inherited (locks, file descriptors shared with the parent) and opens its
    own. Not pickled (the owner re-creates it).
    """

    def __init__(self, cache_mb: float = DEFAULT_TILE_CACHE_MB) -> None:
        self.cache_mb = float(cache_mb)
        self._pid = -1
        self._reset()

    def _reset(self) -> None:
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._zips: dict[str, ZipMembers] = {}
        self._npy = LRU(1 << 62, MAX_OPEN_NPY)
        self.decoded = LRU(int(self.cache_mb * (1 << 20)))

    def _check_pid(self) -> None:
        if self._pid != os.getpid():
            self._reset()

    def zip(self, path: str) -> ZipMembers:
        self._check_pid()
        members = self._zips.get(path)
        if members is None:
            with self._lock:
                members = self._zips.get(path)
                if members is None:
                    members = self._zips[path] = ZipMembers(path)
        return members

    def read_bytes(self, path: str, member: Optional[str]) -> bytes:
        """The encoded bytes of a tile file, or of member *member* of the zip at *path*."""
        if member is not None:
            return self.zip(path).read(member)
        with open(path, "rb") as fh:
            return fh.read()

    def pixels(self, path: str, member: Optional[str], fmt: str) -> np.ndarray:
        """The (H, W, 3) uint8 pixels of a tile: memory-mapped (npy) or decoded once and cached (JPEG/PNG)."""
        self._check_pid()
        key = (path, member)
        if fmt == "npy":
            return self._npy.get(key, lambda: load_npy(path), lambda a: 0)
        if self.decoded.max_bytes <= 0:
            return decode_image(self.read_bytes(path, member))
        return self.decoded.get(key, lambda: decode_image(self.read_bytes(path, member)), lambda a: a.nbytes)
