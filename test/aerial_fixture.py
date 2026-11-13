"""Synthetic scenes and raw aerial tiles for the aerial tests: small PNG tiles in the release layout.

The tiles are painted with their own position: red is the EPSG:25832 easting of a pixel, green its northing
(both floored, modulo 256), so a crop shows where it was cut.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import kitscenes
from kitscenes.aerial import scene_poses
from kitscenes.aerial._areas import bucket_area_id
from release_fixture import write_poses, write_timestamps

# Two places ~35 km apart: different 10 km areas.
FRANKFURT = (50.110423, 8.682138)
KARLSRUHE = (49.006889, 8.403653)

#: 1 m/px and 20 m tiles, so the tests stay fast.
RESOLUTION_M = 1.0
TILE_SIZE_PX = 20
RAW_CACHE_ID = "dop20_res100_tile20"


def write_scene(path: Path, latlon=None, n: int = 3, **pose_kwargs) -> Path:
    """Timestamps, poses and (with *latlon*) maps/origin.json: all the aerial code reads."""
    path.mkdir(parents=True)
    write_timestamps(path, n)
    write_poses(path, n, **pose_kwargs)
    if latlon is not None:
        (path / "maps").mkdir()
        (path / "maps" / "origin.json").write_text(json.dumps({"latitude": latlon[0], "longitude": latlon[1]}))
    return path


def paint(min_e: float, max_n: float, size_px: int = TILE_SIZE_PX, resolution_m: float = RESOLUTION_M) -> np.ndarray:
    """The pixels of a tile whose north-west corner is (*min_e*, *max_n*)."""
    cols = min_e + (np.arange(size_px) + 0.5) * resolution_m
    rows = max_n - (np.arange(size_px) + 0.5) * resolution_m  # row 0 is the north edge
    image = np.zeros((size_px, size_px, 3), dtype=np.uint8)
    image[..., 0] = (np.floor(cols)[None, :] % 256).astype(np.uint8)
    image[..., 1] = (np.floor(rows)[:, None] % 256).astype(np.uint8)
    image[..., 2] = 7
    return image


def write_tiles(root: Path, aerial: Optional[Path] = None, *, scenes: Optional[list[str]] = None,
                margin_m: float = 12.0) -> Path:
    """PNG tiles within *margin_m* of every pose of the scenes of *root* that can be placed, into *aerial*
    (default ``<root>/aerial``): ``raw_tiles/``, the tile manifest and the area index. Returns the folder."""
    aerial = Path(aerial) if aerial is not None else Path(root) / "aerial"
    extent = RESOLUTION_M * TILE_SIZE_PX
    areas: dict[str, dict] = {}
    scene_to_area: dict[str, str] = {}
    for scene in kitscenes.open(root):
        if scenes is not None and scene.id not in scenes:
            continue
        try:
            poses = scene_poses(scene)
        except LookupError:  # no map origin
            continue
        area_id = bucket_area_id(poses.lat, poses.lon)
        scene_to_area[scene.id] = area_id
        area = areas.setdefault(area_id, {"points": [], "scene_ids": []})
        area["points"] += [(float(e), float(n)) for e, n in poses.xy]
        area["scene_ids"].append(scene.id)
    manifest = {"raw_cache_id": RAW_CACHE_ID, "provider": "dop20", "resolution_m_per_px": RESOLUTION_M,
                "tile_size_px": TILE_SIZE_PX, "effective_tile_size_px": TILE_SIZE_PX, "tile_extent_m": extent,
                "areas": {}}
    for area_id, area in areas.items():
        origin_e = math.floor(min(e for e, _ in area["points"]) / extent) * extent
        origin_n = math.floor(min(n for _, n in area["points"]) / extent) * extent
        tiles = set()
        for e, n in area["points"]:
            for i in range(math.floor((e - margin_m - origin_e) / extent), math.floor((e + margin_m - origin_e) / extent) + 1):
                for j in range(math.floor((n - margin_m - origin_n) / extent), math.floor((n + margin_m - origin_n) / extent) + 1):
                    tiles.add((i, j))
        entries = []
        for i, j in sorted(tiles):
            rel = f"raw_tiles/{RAW_CACHE_ID}/{area_id}/tile_{i}_{j}.png"
            path = aerial / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(paint(origin_e + i * extent, origin_n + (j + 1) * extent)).save(path)
            entries.append({"i": i, "j": j, "path": rel})
        manifest["areas"][area_id] = {"origin_xy": [origin_e, origin_n], "crs": "EPSG:25832", "tiles": entries}
    meta = aerial / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / f"raw_tile_manifest_{RAW_CACHE_ID}.json").write_text(json.dumps(manifest))
    (meta / "area_index.json").write_text(json.dumps({
        "strategy": "epsg25832_bucket", "bucket_size_m": 10000.0, "scene_to_area": scene_to_area,
        "areas": {a: {"scene_ids": sorted(v["scene_ids"])} for a, v in areas.items()}}))
    return aerial
