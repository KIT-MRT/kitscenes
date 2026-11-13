"""Ego poses from ``poses.txt`` (TUM format: ``t tx ty tz qx qy qz qw`` per line)."""

from __future__ import annotations

import logging

import numpy as np

from kitscenes.errors import DataFormatError
from kitscenes._layout import POSES
from kitscenes.schema import EgoPose
from kitscenes._timestamps import seconds_to_ns

logger = logging.getLogger(__name__)


def parse_poses(text: str, source: str = POSES) -> list[EgoPose]:
    """Parse ``poses.txt`` content.

    The timestamps (seconds relative to the scene start) are parsed exactly,
    see :func:`kitscenes._timestamps.seconds_to_ns`. Malformed lines are skipped
    with a warning.
    """
    poses: list[EgoPose] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            if len(parts) != 8:
                raise ValueError("expected 8 values")
            values = np.array([float(p) for p in parts[1:]], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError("non-finite value")
            timestamp_ns = seconds_to_ns(parts[0])
        except (ValueError, DataFormatError) as err:
            logger.warning("Skipping malformed line in %s (%s): %r", source, err, line)
            continue
        poses.append(EgoPose(t_ns=timestamp_ns, xyz=values[:3], quat=values[3:]))
    return poses
