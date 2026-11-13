"""The geometry of an aerial crop (AerialCrops.crop on synthetic PNG tiles).

roi = (longitudinal_m, lateral_m); a crop is (round(lateral_m / res), round(longitudinal_m / res), 3); the ego is
the centre pixel, forward points to +column and left to -row.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from kitscenes.aerial import AerialCrops, AerialTiles, TileCoverageError

AREA = "area_x"


def write_tiles(aerial: Path, tiles: dict, size_px: int, resolution_m: float, origin=(0.0, 0.0)) -> Path:
    """An aerial folder with the PNG tiles ``{(i, j): pixels or a grey level}`` of one area."""
    raw_id = f"dop20_res{round(resolution_m * 100):03d}_tile{size_px}"
    entries = []
    for (i, j), content in tiles.items():
        rel = f"raw_tiles/{raw_id}/{AREA}/tile_{i}_{j}.png"
        path = aerial / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = content if isinstance(content, np.ndarray) else np.full((size_px, size_px, 3), content, np.uint8)
        Image.fromarray(pixels).save(path)
        entries.append({"i": i, "j": j, "path": rel})
    manifest = {"raw_cache_id": raw_id, "provider": "dop20", "resolution_m_per_px": resolution_m,
                "tile_size_px": size_px, "effective_tile_size_px": size_px, "tile_extent_m": resolution_m * size_px,
                "areas": {AREA: {"origin_xy": list(origin), "crs": "EPSG:25832", "tiles": entries}}}
    (aerial / "metadata").mkdir(parents=True, exist_ok=True)
    (aerial / "metadata" / f"raw_tile_manifest_{raw_id}.json").write_text(json.dumps(manifest))
    return aerial


def marker_tiles(aerial: Path, marker_xy: tuple[float, float], resolution_m: float) -> Path:
    """One black 64 m tile with a red 3 x 3 px block around *marker_xy* (so every bilinear sample at the marker
    reads red, whatever the heading)."""
    size = round(64.0 / resolution_m)
    pixels = np.zeros((size, size, 3), np.uint8)
    col = round(marker_xy[0] / resolution_m - 0.5)
    row = round((64.0 - marker_xy[1]) / resolution_m - 0.5)
    pixels[row - 1:row + 2, col - 1:col + 2] = [255, 0, 0]
    return write_tiles(aerial, {(0, 0): pixels}, size, resolution_m)


def test_shape_follows_roi_and_resolution(tmp_path: Path) -> None:
    crops = AerialCrops(write_tiles(tmp_path, {(0, 0): 0}, 512, 0.2), (60.0, 30.0))
    assert crops.shape == (150, 300, 3) and crops.crop((51.2, 51.2), 0.0).shape == (150, 300, 3)


def test_marker_lands_at_its_pixel_with_a_heading_off_the_axes(tmp_path: Path) -> None:
    res, roi, ego, yaw = 0.5, (20.5, 20.5), (32.0, 32.0), math.radians(37.0)
    forward_m, left_m = 6.0, 3.0  # ahead of the ego and to its left
    marker = (ego[0] + math.cos(yaw) * forward_m - math.sin(yaw) * left_m,
              ego[1] + math.sin(yaw) * forward_m + math.cos(yaw) * left_m)
    crop = AerialCrops(marker_tiles(tmp_path, marker, res), roi, backend="scipy").crop(ego, yaw)
    h, w = crop.shape[:2]
    col = round((w - 1) / 2 + forward_m / res)
    row = round((h - 1) / 2 - left_m / res)
    np.testing.assert_array_equal(crop[row, col], [255, 0, 0])
    assert not np.array_equal(crop[col, row], [255, 0, 0])  # rows and columns are not swapped


def test_ego_is_the_centre_pixel(tmp_path: Path) -> None:
    crop = AerialCrops(marker_tiles(tmp_path, (32.5, 32.5), 1.0), (11.0, 11.0), backend="scipy").crop((32.5, 32.5), 0.0)
    np.testing.assert_array_equal(crop[5, 5], [255, 0, 0])


def test_window_from_several_tiles(tmp_path: Path) -> None:
    """East is +column, and the tile with the larger j (further north) is on top."""
    tiles = AerialTiles(write_tiles(tmp_path, {(0, 0): 100, (1, 0): 200, (0, 1): 50}, 10, 0.1))
    sampler = tiles.sampler(AREA)
    east = sampler.sample_bbox((0.0, 0.0, 2.0, 1.0), 0.1)
    assert east.shape == (10, 20, 3)
    np.testing.assert_array_equal(east[:, :10], 100)
    np.testing.assert_array_equal(east[:, 10:], 200)
    north = sampler.sample_bbox((0.0, 0.0, 1.0, 2.0), 0.1)
    assert north.shape == (20, 10, 3)
    np.testing.assert_array_equal(north[:10], 50)
    np.testing.assert_array_equal(north[10:], 100)
    with pytest.raises(TileCoverageError):
        sampler.sample_bbox((1.0, 1.0, 2.0, 2.0), 0.1)  # tile (1, 1) is not there


def test_window_off_the_pixel_grid_is_refused(tmp_path: Path) -> None:
    """Start and width are rounded on their own: a window that would come out a column short raises."""
    sampler = AerialTiles(write_tiles(tmp_path, {(0, 0): 100}, 3, 0.5)).sampler(AREA)
    # tile extent 1.5 m; start round(0.75 / 0.5) = 2 and width 2: columns [2:4] of a 3 px tile
    with pytest.raises(ValueError, match="truncat"):
        sampler.sample_bbox((0.75, 0.0, 1.5, 0.5), 0.5)


def ramp_tiles(aerial: Path, scale: float = 3.0) -> Path:
    """1 m pixels, 20 m tiles, 80 x 80 m from the origin; red / green are *scale* times the easting / northing
    of the pixel centre: a linear ramp, which bilinear sampling reproduces exactly."""
    tiles = {}
    centres = np.arange(20) + 0.5
    for i in range(4):
        for j in range(4):
            pixels = np.zeros((20, 20, 3), np.uint8)
            pixels[..., 0] = np.round(scale * (20 * i + centres))[None, :]
            pixels[..., 1] = np.round(scale * (20 * (j + 1) - centres))[:, None]
            tiles[(i, j)] = pixels
    return write_tiles(aerial, tiles, 20, 1.0)


@pytest.mark.parametrize("ego_xy, yaw", [
    ((40.0, 40.0), 0.0),
    ((40.45, 37.8), 0.0),     # the sample window starts 0.47 px off the pixel grid
    ((38.62, 41.13), 0.7),
    ((41.9, 39.55), -2.3),
])
def test_every_crop_pixel_is_at_its_world_position(tmp_path: Path, ego_xy, yaw) -> None:
    """Georegistration to a fraction of a pixel."""
    crop = AerialCrops(ramp_tiles(tmp_path), (9.0, 5.0), backend="scipy").crop(ego_xy, yaw).astype(float)
    h, w = crop.shape[:2]
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = cols - (w - 1) / 2
    y = -(rows - (h - 1) / 2)
    e = ego_xy[0] + np.cos(yaw) * x - np.sin(yaw) * y
    n = ego_xy[1] + np.sin(yaw) * x + np.cos(yaw) * y
    # 3 levels per metre; tile values and the crop are rounded to integers: within 1 level (0.11 px)
    assert np.abs(crop[..., 0] - 3 * e).max() <= 1.0
    assert np.abs(crop[..., 1] - 3 * n).max() <= 1.0
