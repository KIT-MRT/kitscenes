"""AerialCacheConfig: the parameters a crop cache was prepared with."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class AerialCacheConfig:
    """Parameters of a prepared crop cache.

    Attributes:
        provider: Imagery provider (``"dop20"``).
        resolution_m: Metres per pixel.
        roi_m: (longitudinal_m, lateral_m) of a crop.
        tile_size_px: Requested raw-tile size (the provider may cap it).
        orientation: ``"ego"`` (forward is +column).
        version: Free version tag, part of :attr:`cache_id`.
        cache_id: Name of the crop cache, e.g. ``dop20_res020_roi60x30_ego_v1``.
        raw_cache_id: Name of the raw-tile cache, e.g. ``dop20_res020_tile2048``.
    """

    provider: str
    resolution_m: float
    roi_m: tuple[float, float]
    tile_size_px: int
    orientation: str
    version: str
    cache_id: str
    raw_cache_id: str
