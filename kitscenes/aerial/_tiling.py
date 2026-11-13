"""The tile grid of an area."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TileGrid:
    """A frozen tile grid for one (area_id, raw_cache_id) pair.

    origin_e/origin_n is the EPSG:25832 (easting, northing) of tile
    (0, 0)'s lower-left corner.
    """

    origin_e: float
    origin_n: float
    tile_extent_m: float

    def tile_index(self, easting: float, northing: float) -> tuple[int, int]:
        """Return the (i, j) tile index containing an EPSG:25832 (easting,
        northing) point.
        """
        i = math.floor((easting - self.origin_e) / self.tile_extent_m)
        j = math.floor((northing - self.origin_n) / self.tile_extent_m)
        return i, j
