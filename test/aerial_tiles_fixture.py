"""A raw aerial tile set in the release layout (extracted or zipped), painted with a smooth pattern."""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

RAW_ID = "dop20_res100_tile32"
SIZE = 32  # px per tile, 1 m/px: 32 m tiles
AREA = "area_0_0"
ORIGIN = (1000.0, 2000.0)
TILES = [(i, j) for i in range(3) for j in range(3)]


def pixels(i: int, j: int, origin=ORIGIN) -> np.ndarray:
    """A smooth RGB pattern of the world position (JPEG keeps it within a few levels)."""
    e = origin[0] + i * SIZE + np.arange(SIZE) + 0.5
    n = origin[1] + (j + 1) * SIZE - np.arange(SIZE) - 0.5  # row 0 is the north edge
    ee, nn = np.meshgrid(e, n)
    image = np.stack([128 + 100 * np.sin(ee / 7), 128 + 100 * np.cos(nn / 5), 128 + 60 * np.sin((ee + nn) / 11)], -1)
    return np.round(image).astype(np.uint8)


def encode(image: np.ndarray, fmt: str) -> bytes:
    import io

    buf = io.BytesIO()
    if fmt == "jpeg":
        Image.fromarray(image).save(buf, format="JPEG", quality=95, subsampling=0)
    else:
        Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def write_aligned_zip(path: Path, members: dict[str, bytes], align: int = 4096, deflate: bool = False) -> None:
    """A stored zip whose members start at multiples of *align* (like the release packer), or a deflated one."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED if deflate else zipfile.ZIP_STORED) as zf:
        for name, data in sorted(members.items()):
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            if deflate:
                zi.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(zi, data)
                continue
            zi.file_size, zi.CRC, zi.compress_size = len(data), 0, 0  # placeholders zipfile fills in
            base = len(zi.FileHeader(False))
            pad = (-(zf.fp.tell() + base)) % align
            while 0 < pad < 4:
                pad += align
            zi.extra = struct.pack("<HH", 0xD935, pad - 4) + b"\0" * (pad - 4) if pad else b""
            with zf.open(zi, "w") as dst:
                dst.write(data)
            zi.extra = b""


def write_tiles(
    aerial: Path, *, fmt: str = "png", layout: str = "dir", tiles=TILES, area: str = AREA, origin=ORIGIN,
    scenes: dict | None = None, size: int = SIZE, content=pixels, deflate: bool = False,
) -> Path:
    """An aerial folder with raw tiles: ``layout`` "dir" (extracted) or "zip" (one zip per area)."""
    suffix = {"jpeg": ".jpg", "png": ".png"}[fmt]
    tiles_dir = aerial / "raw_tiles" / RAW_ID
    entries, members = [], {}
    for i, j in tiles:
        name = f"tile_{i}_{j}{suffix}"
        data = encode(content(i, j, origin), fmt)
        sidecar = json.dumps({"i": i, "j": j}).encode()
        rel = f"raw_tiles/{RAW_ID}/{area}/{name}"
        if layout == "zip":
            members[f"{area}/{name}"] = data
            members[f"{area}/tile_{i}_{j}.json"] = sidecar
        else:
            path = aerial / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.with_suffix(".json").write_bytes(sidecar)
        entries.append({"i": i, "j": j, "path": rel})
    if layout == "zip":
        tiles_dir.mkdir(parents=True, exist_ok=True)
        write_aligned_zip(tiles_dir / f"{area}.zip", members, deflate=deflate)
    meta = aerial / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    manifest_path = meta / f"raw_tile_manifest_{RAW_ID}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {
        "raw_cache_id": RAW_ID, "provider": "dop20", "resolution_m_per_px": 1.0, "tile_size_px": size,
        "effective_tile_size_px": size, "tile_extent_m": float(size), "areas": {}}
    if fmt == "jpeg":
        manifest["tile_format"] = {"format": "jpeg", "quality": 95, "chroma_subsampling": "4:4:4"}
    manifest["areas"][area] = {"origin_xy": list(origin), "crs": "EPSG:25832", "tiles": entries}
    manifest_path.write_text(json.dumps(manifest))
    index = {"strategy": "epsg25832_bucket", "bucket_size_m": 10000.0, "scene_to_area": dict(scenes or {}),
             "areas": {area: {"scene_ids": sorted(scenes or {})}}}
    (meta / "area_index.json").write_text(json.dumps(index))
    (meta / "provider_metadata.json").write_text(json.dumps({"state": {"licence": "test"}}))
    (aerial / "ATTRIBUTION.md").write_text("test imagery\n")
    return aerial
