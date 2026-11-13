"""Fusing several lidar sweeps into one point cloud in the base frame.

:func:`fuse_point_clouds` is the pure merging logic on loaded
:class:`~kitscenes.points.PointCloud` objects; :meth:`kitscenes.frame.Frame.fused_lidar`
loads the sweeps of a frame and calls it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from kitscenes.points import PointCloud
from kitscenes.schema import Provenance


def fuse_point_clouds(clouds: Sequence[PointCloud], *, t_ns: Optional[int] = None) -> PointCloud:
    """Concatenate base-frame clouds, keeping only measured rows.

    Rows without a measured point (at the sensor origin: no-returns, slots
    without a point) are dropped, so none of them becomes a point at the
    origin of the fused cloud.

    The attributes all clouds share are kept (e.g. ``reflectivity``,
    ``timestamp``); ``source_index`` maps every point to its cloud in
    ``sensor``. ``motion_compensated`` is True only if every input was.

    Raises:
        ValueError: If a cloud is not in the base frame, or the clouds have
            different times.
    """
    for cloud in clouds:
        if cloud.frame != "base":
            raise ValueError(f"Cannot fuse {cloud.sensor!r}: it is in the {cloud.frame} frame, not the base frame.")
    times = {cloud.t_ns for cloud in clouds}
    if len(times) > 1:
        raise ValueError(f"Cannot fuse clouds of different times: {sorted(times)}")
    if t_ns is None:
        if not times:
            raise ValueError("Nothing to fuse and no t_ns given.")
        t_ns = times.pop()

    parts = [cloud.compact() for cloud in clouds]
    shared = [k for k in (parts[0].attrs if parts else {}) if all(k in p.attrs for p in parts)]
    reasons = [f"{p.sensor}: {p.reason}" for p in parts if not p.motion_compensated]
    return PointCloud(
        xyz=np.concatenate([p.xyz for p in parts]) if parts else np.zeros((0, 3), np.float32),
        attrs={k: np.concatenate([p.attrs[k] for p in parts]) for k in shared},
        measured=np.ones(sum(len(p) for p in parts), dtype=bool),
        frame="base",
        t_ns=t_ns,
        sensor=tuple(str(p.sensor) for p in parts),
        motion_compensated=not reasons,
        reason="; ".join(reasons) or None,
        source_index=np.concatenate(
            [np.full(len(p), i, dtype=np.uint8) for i, p in enumerate(parts)]
        ) if parts else np.zeros(0, np.uint8),
        provenance=Provenance.GEOMETRY_DERIVED,
    )
