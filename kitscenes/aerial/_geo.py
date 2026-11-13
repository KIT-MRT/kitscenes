"""Scene frame -> EPSG:25832, the CRS of the DOP aerial imagery.

A scene's world frame is a local UTM frame around ``maps/origin.json``
(:class:`kitscenes.map.MapOrigin`): x, y are metres east and north of the
origin. The origin is placed in EPSG:25832 and the local offsets are added
to it, so scene yaw equals world yaw. EPSG:25832 (ETRS89) and the origin's
UTM zone 32N (WGS 84) differ by about 0.1 mm here (pyproj applies no datum
shift between them).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kitscenes.map import MapOrigin


@lru_cache(maxsize=1)
def _latlon_to_epsg25832():
    import pyproj

    return pyproj.Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)


def latlon_to_epsg25832(lat: float, lon: float) -> tuple[float, float]:
    """WGS84 (lat, lon) in degrees -> EPSG:25832 (easting, northing) in metres.

    The arguments are (lat, lon); the result is (x, y).
    """
    return _latlon_to_epsg25832().transform(lon, lat)


def scene_xy_to_epsg25832(origin: "MapOrigin", xy: np.ndarray) -> np.ndarray:
    """(N, 2+) scene-frame points -> (N, 2) EPSG:25832 easting, northing (a pure translation)."""
    xy = np.asarray(xy, dtype=np.float64)[..., :2]
    return xy + np.array(latlon_to_epsg25832(origin.lat, origin.lon))


def heading_from_rotation(rotation) -> np.ndarray:
    """Heading (radians about +z) of the vehicle's forward axis, from a scipy ``Rotation``.

    ``atan2`` of the rotation matrix's first column: the intrinsic yaw of a
    yaw-pitch-roll (``Rz @ Ry @ Rx``) factorization, so a crop follows where
    the vehicle points even when the pose has pitch and roll.
    """
    matrix = rotation.as_matrix()
    return np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])
