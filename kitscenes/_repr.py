"""Short reprs for the dataclasses that hold arrays, so ``print(obj)`` in a notebook stays one line."""

from __future__ import annotations

import dataclasses
from enum import Enum

import numpy as np

_SHOWN = 8  # arrays up to this many elements (and 1D) are printed with their values


def _short(value, depth: int = 0) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim <= 1 and value.size <= _SHOWN:
            return np.array2string(value, precision=4, separator=", ", suppress_small=True)
        return f"<{value.dtype} {tuple(value.shape)}>"
    if isinstance(value, np.generic):
        return repr(value.item())
    if isinstance(value, Enum):
        return repr(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return compact_repr(value) if depth == 0 else type(value).__name__
    if isinstance(value, dict):
        kinds = {type(v).__name__ for v in value.values()}
        if value and len(kinds) == 1 and all(dataclasses.is_dataclass(v) for v in value.values()):
            return f"{{{len(value)} {kinds.pop()}}}"
        if depth >= 1 or len(value) > 12:
            return f"{{{len(value)} items}}"
        return "{" + ", ".join(f"{k!r}: {_short(v, depth + 1)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        simple = all(isinstance(v, (int, float, str, bool, type(None), np.generic)) for v in value)
        if simple and len(value) <= _SHOWN:
            return repr(tuple(value) if isinstance(value, tuple) else list(value))
        kinds = {type(v).__name__ for v in value}
        kind = kinds.pop() if len(kinds) == 1 else "items"
        return f"[{len(value)} {kind}]"
    return repr(value)


def compact_repr(obj) -> str:
    """``Name(field=value, ...)`` with arrays as ``<dtype (shape)>`` and nested containers summarised."""
    parts = [f"{f.name}={_short(getattr(obj, f.name))}" for f in dataclasses.fields(obj) if f.repr]
    return f"{type(obj).__name__}({', '.join(parts)})"
