"""Coordinates: the scene's metric frame -> the viewer's ENU.

The viewer places everything in a local ENU: an equirectangular tangent plane
about an anchor lat/lon, z = ele (`enu_to_lat_lon` is its inverse).

A KITScenes scene lives in lanelet2's `UtmProjector` frame (a city-local
transverse Mercator anchored at `maps/origin.json`), and so does its map. The
two frames differ by the meridian convergence (about 0.25 deg, 4.3 m per
kilometre) and by an anisotropic scale (about 0.3 %; the ENU uses the
equatorial radius for both axes). So the map and the point clouds both go
through `SceneFrame`, which runs the exact chain through pyproj and matches
lanelet2's own projector to under a micrometre.
"""

import math

EARTH_R = 6378137.0  # the viewer's ENU uses the WGS84 equatorial radius; the JOSM plugin must use the same


def enu_to_lat_lon(x, y, lat0, lon0):
    """(lat, lon) of viewer ENU metres about the anchor (lat0, lon0)."""
    lat = lat0 + math.degrees(y / EARTH_R)
    cos0 = math.cos(math.radians(lat0))
    if abs(cos0) < 1e-12:
        cos0 = 1e-12
    return lat, lon0 + math.degrees(x / (EARTH_R * cos0))


def round_coord(value, decimals=3):
    """Round half away from zero, the rounding of every coordinate sent to the page."""
    factor = 10.0 ** decimals
    scaled = value * factor
    return math.floor(scaled + 0.5) / factor if scaled >= 0 else math.ceil(scaled - 0.5) / factor


def utm_epsg(lat, lon):
    """EPSG code of the UTM zone lanelet2's `UtmProjector` picks for an origin."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    return "EPSG:%d" % ((32600 if lat >= 0 else 32700) + zone)


class SceneFrame:
    """Exact transform from a scene's local UTM metres to the viewer's ENU.

    `origin_lat/lon` is the scene's projector origin (`maps/origin.json`),
    which is the point its local metres are measured from; `anchor_lat/lon` is
    the viewer's ENU anchor. Both single points and whole clouds take the same
    path, so a point cloud cannot drift against the map.
    """

    def __init__(self, origin_lat, origin_lon, anchor_lat, anchor_lon, epsg=None):
        import pyproj
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.anchor_lat = anchor_lat
        self.anchor_lon = anchor_lon
        self.epsg = epsg or utm_epsg(origin_lat, origin_lon)
        self._fwd = pyproj.Transformer.from_crs("EPSG:4326", self.epsg, always_xy=True)
        self._inv = pyproj.Transformer.from_crs(self.epsg, "EPSG:4326", always_xy=True)
        # Where the scene's (0, 0) sits in absolute UTM. lanelet2's projector
        # offsets by exactly this (useOffset=True).
        self.e0, self.n0 = self._fwd.transform(origin_lon, origin_lat)
        self._kx = math.radians(1.0) * math.cos(math.radians(anchor_lat)) * EARTH_R
        self._ky = math.radians(1.0) * EARTH_R

    def to_lat_lon(self, x, y):
        lon, lat = self._inv.transform(x + self.e0, y + self.n0)
        return lat, lon

    def apply(self, x, y):
        """One point, scene metres -> viewer ENU metres."""
        lat, lon = self.to_lat_lon(x, y)
        return ((lon - self.anchor_lon) * self._kx, (lat - self.anchor_lat) * self._ky)

    def apply_array(self, xy):
        """(N, 2) scene metres -> (N, 2) viewer ENU metres."""
        import numpy as np
        xy = np.asarray(xy, dtype=np.float64)
        lon, lat = self._inv.transform(xy[:, 0] + self.e0, xy[:, 1] + self.n0)
        return np.column_stack([(np.asarray(lon) - self.anchor_lon) * self._kx,
                                (np.asarray(lat) - self.anchor_lat) * self._ky])

    def rotation_deg_array(self, xy, step=10.0):
        """How far the scene's x axis is turned from ENU east at (N, 2) scene points, in degrees.

        Numeric and vectorised: a city's trajectories are tens of thousands
        of poses.
        """
        import numpy as np
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        here = self.apply_array(xy)
        east = self.apply_array(xy + np.array([step, 0.0]))
        return np.degrees(np.arctan2(east[:, 1] - here[:, 1], east[:, 0] - here[:, 0]))


class LocalFrame:
    """The scene's own metres as viewer ENU, unchanged: for a scene that cannot be placed.

    The test-e2e splits' poses are relative to the scene's first pose, so
    there is no lat/lon to turn or scale them by and no anchor; the scene is
    shown as it is, through `SceneFrame`'s interface.
    """

    def apply(self, x, y):
        return float(x), float(y)

    def apply_array(self, xy):
        import numpy as np
        return np.array(xy, dtype=np.float64).reshape(-1, 2)

    def rotation_deg_array(self, xy, step=10.0):
        import numpy as np
        return np.zeros(len(np.asarray(xy).reshape(-1, 2)))
