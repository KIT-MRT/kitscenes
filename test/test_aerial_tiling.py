"""kitscenes.aerial._tiling: the tile grid."""

from __future__ import annotations

from kitscenes.aerial._tiling import TileGrid


def test_tile_index() -> None:
    grid = TileGrid(origin_e=470000.0, origin_n=5550000.0, tile_extent_m=100.0)
    assert grid.tile_index(470000.0, 5550000.0) == (0, 0)
    assert grid.tile_index(470150.0, 5550250.0) == (1, 2)
    assert grid.tile_index(469950.0, 5549950.0) == (-1, -1)  # below the origin
