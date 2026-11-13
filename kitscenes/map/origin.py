"""The map frame: a local UTM frame around ``maps/origin.json``.

Scene maps and ``poses.txt`` share one frame: x = UTM easting minus the
easting of the origin, y = UTM northing minus its northing, z = height above
the ellipsoid as stored. This equals lanelet2's ``UtmProjector`` frame
exactly, but needs only pyproj.
"""

from __future__ import annotations

import dataclasses
import json
from functools import cached_property

import numpy as np


@dataclasses.dataclass(frozen=True)
class MapOrigin:
    """WGS-84 origin of a map frame.

    Attributes:
        lat, lon: Origin in degrees.
    """

    lat: float
    lon: float

    @classmethod
    def from_json(cls, data: bytes | str) -> "MapOrigin":
        """From ``maps/origin.json`` (``{"latitude": ..., "longitude": ...}``)."""
        d = json.loads(data)
        return cls(float(d["latitude"]), float(d["longitude"]))

    @property
    def utm_zone(self) -> int:
        return int((self.lon + 180.0) // 6.0) % 60 + 1

    @property
    def epsg(self) -> int:
        """EPSG code of the UTM zone of the origin (326zz north, 327zz south)."""
        return (32600 if self.lat >= 0 else 32700) + self.utm_zone

    @cached_property
    def _to_utm(self):
        import pyproj

        return pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)

    @cached_property
    def _from_utm(self):
        import pyproj

        return pyproj.Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)

    @cached_property
    def utm_offset(self) -> np.ndarray:
        """(2,) absolute UTM easting, northing of the origin."""
        e, n = self._to_utm.transform(self.lon, self.lat)
        return np.array([e, n])

    def to_utm(self, xyz: np.ndarray) -> np.ndarray:
        """Map-frame points (N, 2 or 3) -> absolute UTM in :attr:`epsg`."""
        out = np.array(xyz, dtype=np.float64, copy=True)
        out[..., :2] += self.utm_offset
        return out

    def from_utm(self, xyz: np.ndarray) -> np.ndarray:
        out = np.array(xyz, dtype=np.float64, copy=True)
        out[..., :2] -= self.utm_offset
        return out

    def to_latlon(self, xyz: np.ndarray) -> np.ndarray:
        """Map-frame points (N, 2 or 3) -> (N, 2) latitude, longitude in degrees."""
        utm = self.to_utm(np.atleast_2d(xyz))
        lon, lat = self._from_utm.transform(utm[:, 0], utm[:, 1])
        return np.column_stack([lat, lon])

    def from_latlon(self, lat, lon) -> np.ndarray:
        """Latitude, longitude (scalars or arrays) -> (N, 2) map-frame x, y."""
        e, n = self._to_utm.transform(np.atleast_1d(lon), np.atleast_1d(lat))
        return np.column_stack([e, n]) - self.utm_offset

    def __getstate__(self) -> dict:
        return {"lat": self.lat, "lon": self.lon}

    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, "lat", state["lat"])
        object.__setattr__(self, "lon", state["lon"])


#: The map origin of each city. Every scene of a city has this ``maps/origin.json``, and the splits that ship no
#: maps but poses in the map frame (``pose_frame == "local_utm"``: test, map_change_test) use the same frame. Lets
#: those scenes be placed on the aerial imagery (:func:`kitscenes.aerial.scene_poses`).
CITY_ORIGINS: dict[str, MapOrigin] = {
    "Karlsruhe": MapOrigin(49.01439, 8.41722),
    "Frankfurt": MapOrigin(50.110423, 8.682138),
    "Sindelfingen": MapOrigin(48.70782, 8.98137),
}
