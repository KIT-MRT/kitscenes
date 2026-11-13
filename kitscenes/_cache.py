"""Opt-in cache for decoded payloads (images, point clouds).

Off by default: every accessor decodes again and returns arrays the caller
owns. A cache only pays off when the same frame is read several times (in a
notebook); for sequential training reads it costs memory per worker and
hits nothing. Cached payloads are made read-only, so a caller that modifies
one fails loudly instead of corrupting the next reader.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Callable, Hashable, TypeVar

import numpy as np

from kitscenes.points import PointCloud, readonly

T = TypeVar("T")


def _size(payload: object) -> int:
    if isinstance(payload, np.ndarray):
        return int(payload.nbytes)
    if isinstance(payload, PointCloud):
        arrays = [payload.xyz, payload.measured, *payload.attrs.values()]
        if payload.source_index is not None:
            arrays.append(payload.source_index)
        return sum(int(a.nbytes) for a in arrays)
    if isinstance(payload, bytes):
        return len(payload)
    return 0


def _freeze(payload: object) -> None:
    if isinstance(payload, np.ndarray):
        payload.flags.writeable = False
    elif isinstance(payload, PointCloud):
        readonly(payload)


class PayloadCache:
    """Thread-safe LRU with a byte budget; payloads it returns are read-only.

    Args:
        max_bytes: Budget. A payload larger than the budget is not kept.
    """

    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self._max_bytes = max_bytes
        self._current_bytes = 0
        self._entries: OrderedDict = OrderedDict()
        self._lock = RLock()

    def __repr__(self) -> str:
        return f"PayloadCache({self._current_bytes}/{self._max_bytes} bytes, {len(self._entries)} entries)"

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._current_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._current_bytes = 0

    def get_or_load(self, key: Hashable, load: Callable[[], T]) -> T:
        """The cached payload for *key*, or ``load()`` stored read-only."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry[0]
        payload = load()
        size = _size(payload)
        if 0 < size <= self._max_bytes:
            _freeze(payload)
            with self._lock:
                old = self._entries.pop(key, None)
                if old is not None:
                    self._current_bytes -= old[1]
                while self._entries and self._current_bytes + size > self._max_bytes:
                    _, (_, evicted) = self._entries.popitem(last=False)
                    self._current_bytes -= evicted
                self._entries[key] = (payload, size)
                self._current_bytes += size
        return payload

    def __getstate__(self) -> dict:
        """Only the budget crosses a process boundary; a worker starts empty."""
        return {"max_bytes": self._max_bytes}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["max_bytes"])
