"""A feature tile set of the aerial imagery in the release layout (extracted or zipped)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from aerial_tiles_fixture import write_aligned_zip

SET = "dinov3_test_tile8"
AREA = "area_0_0"
ORIGIN = (1000.0, 2000.0)
TOKENS = 8  # tokens per tile edge
TOKEN_M = 2.0  # 16 m tiles
LAYERS = (4, 11)
CHANNELS = 3
TILES = [(i, j) for i in range(3) for j in range(3)]


def float32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Round to nearest even bfloat16, as uint16 bit patterns."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def safetensors_bytes(tensors: dict[str, np.ndarray], metadata: dict | None = None) -> bytes:
    """A safetensors file of bfloat16 tensors (given as float32)."""
    header, chunks, offset = {}, [], 0
    for name, value in tensors.items():
        data = float32_to_bf16(value).tobytes()
        header[name] = {"dtype": "BF16", "shape": list(value.shape), "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    if metadata:
        header["__metadata__"] = {k: json.dumps(v) for k, v in metadata.items()}
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw + b"".join(chunks)


def token_field(i: int, j: int, origin=ORIGIN) -> np.ndarray:
    """(L, C, T, T): channel 0 = easting, 1 = northing of the token centre (relative to the origin),
    channel 2 = layer index; bf16-exact (multiples of 0.5 below 256)."""
    k = np.arange(TOKENS) + 0.5
    e = i * TOKENS * TOKEN_M + k * TOKEN_M
    n = (j + 1) * TOKENS * TOKEN_M - k * TOKEN_M  # row 0 is the north edge
    ee, nn = np.meshgrid(e, n)
    out = np.empty((len(LAYERS), CHANNELS, TOKENS, TOKENS), dtype=np.float32)
    for li in range(len(LAYERS)):
        out[li, 0], out[li, 1], out[li, 2] = ee, nn, li
    return out


def write_features(root: Path, *, layout: str = "dir", tiles=TILES, field=token_field, area: str = AREA,
                   origin=ORIGIN) -> Path:
    """A dataset root with ``aerial/features/<SET>/``: extracted files or one stored zip per area."""
    folder = root / "aerial" / "features" / SET
    folder.mkdir(parents=True, exist_ok=True)
    members = {}
    for i, j in tiles:
        maps = field(i, j)
        data = safetensors_bytes({"maps": maps, "cls": maps.mean(axis=(2, 3))}, {"i": i, "j": j})
        members[f"{area}/tile_{i}_{j}.safetensors"] = data
    if layout == "zip":
        write_aligned_zip(folder / f"{area}.zip", members)
    else:
        for name, data in members.items():
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(data)
    manifest = {
        "feature_set": SET, "layers": list(LAYERS), "channels": CHANNELS, "tokens_per_tile": TOKENS,
        "tile_extent_m": TOKENS * TOKEN_M,
        "areas": {area: {"origin_xy": list(origin), "tiles": [{"i": i, "j": j, "missing_neighbors": []}
                                                                for i, j in tiles]}},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "TERMS.md").write_text("terms\n")
    return root
