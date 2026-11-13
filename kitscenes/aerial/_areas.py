"""Areas: the 10 km EPSG:25832 buckets the tiles are grouped by."""

from __future__ import annotations

import math

from kitscenes.aerial._geo import latlon_to_epsg25832

DEFAULT_AREA_BUCKET_SIZE_M = 10_000.0


def bucket_area_id(lat: float, lon: float, bucket_size_m: float = DEFAULT_AREA_BUCKET_SIZE_M) -> str:
    """``area_<min_e>_<min_n>``: the EPSG:25832 bucket of a map origin."""
    easting, northing = latlon_to_epsg25832(lat, lon)
    min_e = math.floor(easting / bucket_size_m) * bucket_size_m
    min_n = math.floor(northing / bucket_size_m) * bucket_size_m
    return f"area_{int(min_e)}_{int(min_n)}"
