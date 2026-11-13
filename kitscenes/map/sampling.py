"""How many points a map label line gets, and where.

The ML converter resamples every line to a fixed number of points, equally
spaced along its 3D length (:class:`FixedPoints`, its default). The other
rules here work on the converter's unresampled output (the map's own
vertices, cut to the region and in the ego frame):

- :class:`Spacing`: equally spaced points at most *metres* apart, so the
  count grows with the length of the line.
- :class:`Adaptive`: curvature-aware. The map's vertices are thinned where
  they add less than *tolerance* (Ramer-Douglas-Peucker: straight runs lose
  their inner vertices, curves keep theirs), then gaps longer than
  *max_spacing* are filled with equally spaced points.
- ``None``: the map's vertices as they are.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Union

import numpy as np


@dataclasses.dataclass(frozen=True)
class FixedPoints:
    """*n* points per line, equally spaced along it (the converter's own resampling)."""

    n: int = 20

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("FixedPoints needs n >= 2")


@dataclasses.dataclass(frozen=True)
class Spacing:
    """Equally spaced points at most *metres* apart (at least the two ends)."""

    metres: float = 1.0

    def __post_init__(self) -> None:
        if not self.metres > 0:
            raise ValueError("Spacing needs metres > 0")


@dataclasses.dataclass(frozen=True)
class Adaptive:
    """The map's vertices, thinned to *tolerance* (metres) and filled to at most *max_spacing* apart."""

    tolerance: float = 0.05
    max_spacing: float = 5.0

    def __post_init__(self) -> None:
        if self.tolerance < 0 or not self.max_spacing > 0:
            raise ValueError("Adaptive needs tolerance >= 0 and max_spacing > 0")


Sampling = Optional[Union[FixedPoints, Spacing, Adaptive]]


def resample(line: np.ndarray, sampling: Sampling) -> np.ndarray:
    """*line* (N, D) resampled by *sampling* (``None``: unchanged)."""
    line = np.asarray(line, dtype=np.float64)
    if sampling is None or len(line) < 2:
        return line
    if isinstance(sampling, FixedPoints):
        return _equidistant(line, sampling.n)
    if isinstance(sampling, Spacing):
        length = _cumulative(line)[-1]
        return _equidistant(line, max(2, int(np.ceil(length / sampling.metres - 1e-9)) + 1))
    if isinstance(sampling, Adaptive):
        return _fill(simplify(line, sampling.tolerance), sampling.max_spacing)
    raise TypeError(f"unknown sampling {sampling!r}")


def _cumulative(line: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])


def _equidistant(line: np.ndarray, n: int) -> np.ndarray:
    s = _cumulative(line)
    if s[-1] <= 0:
        return np.repeat(line[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, line[:, k]) for k in range(line.shape[1])], axis=1)


def simplify(line: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker: the vertices of *line* that matter at *tolerance* (metres, 3D distance)."""
    line = np.asarray(line, dtype=np.float64)
    if len(line) < 3:
        return line
    keep = np.zeros(len(line), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(line) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        inner = line[a + 1:b]
        d = _distance_to_segment(inner, line[a], line[b])
        k = int(np.argmax(d))
        if d[k] > tolerance:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return line[keep]


def _distance_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab)
    t = np.zeros(len(points)) if denom == 0 else np.clip((points - a) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _fill(line: np.ndarray, max_spacing: float) -> np.ndarray:
    pieces = [line[:1]]
    for a, b in zip(line[:-1], line[1:]):
        n = int(np.ceil(np.linalg.norm(b - a) / max_spacing - 1e-9))
        t = np.arange(1, max(n, 1) + 1) / max(n, 1)
        pieces.append(a + t[:, None] * (b - a))
    return np.concatenate(pieces)
