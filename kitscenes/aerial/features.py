"""DINOv3 feature tiles of the aerial imagery: :class:`AerialFeatures`, and ego-aligned crops :class:`AerialFeatureCrops`.

The release ships features of high-resolution orthophotos (8 cm/px; the
imagery itself may not be redistributed) computed with the frozen DINOv3
ViT-L/16 satellite backbone (``dinov3_vitl16_pretrain_sat493m``), one file
per 2048 px tile (163.84 m), in ``<dataset root>/aerial/features/<set>/``::

    aerial/features/<set>/manifest.json                     tile grid per area, layers, token grid, normalisation
    aerial/features/<set>/TERMS.md                          terms of use (no reconstruction of the imagery)
    aerial/features/<set>/<area>.zip                        as downloaded: one zip per 10 km area, members
                                                            <area>/tile_<i>_<j>.safetensors, stored
    aerial/features/<set>/<area>/tile_<i>_<j>.safetensors   or extracted

``<set>`` is e.g. ``dinov3_vitl16_sat493m_dop10_res008_tile2048``. Each tile
holds ``maps``, (L, C, 128, 128) bfloat16: the patch tokens of L blocks
(``manifest["layers"]``, e.g. 4, 11, 17, 23), one token per 16 px = 1.28 m,
row 0 the north edge; and ``cls``, (L, C), the class tokens of the same
forward. Each tile was encoded inside a 2560 px window of its neighbours, so
tokens at a tile edge saw the real surroundings (where there are
neighbours: ``manifest`` lists the tiles without). The channels are in a
fixed basis that differs from the public backbone's by an unpublished
orthogonal rotation (the same for every layer and the class token):
distances, inner products and anything a linear layer learns are unchanged,
the public DINOv3 heads do not apply. Reconstructing the imagery from the
features is not permitted (``TERMS.md``).

Reading needs no extra packages: the safetensors files are read by byte
range (only the layers asked for), from the zip in place or the extracted
files, and the bfloat16 values converted to float32 here.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, NamedTuple, Optional, Sequence, Union

import numpy as np

from kitscenes.aerial._areas import DEFAULT_AREA_BUCKET_SIZE_M
from kitscenes.aerial._crop import TileCoverageError, TileSampler, crop_sample_bbox
from kitscenes.aerial._frame_crops import PoseCrops
from kitscenes.aerial._tile_io import LRU, ZipMembers
from kitscenes.aerial._tiling import TileGrid
from kitscenes.aerial.tiles import AERIAL_DIR, PathLike

if TYPE_CHECKING:
    from kitscenes.aerial._poses import ScenePoses
    from kitscenes.scene import Scene

FEATURES_DIR = "features"
FEATURE_MANIFEST = "manifest.json"
TERMS = "TERMS.md"
#: Open memory maps of feature files per process (address space only; the pages are the OS page cache).
MAX_OPEN_MAPS = 1024

_DTYPES = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}


def _set_parents(path: Path) -> list[Path]:
    return [path, path / FEATURES_DIR, path / AERIAL_DIR / FEATURES_DIR]


def _sets_in(parent: Path) -> list[str]:
    if not parent.is_dir():
        return []
    return sorted(p.name for p in parent.iterdir() if (p / FEATURE_MANIFEST).is_file())


def feature_set_ids(path: PathLike) -> list[str]:
    """The feature sets in *path* (a dataset root, its ``aerial/`` or ``aerial/features/``), sorted."""
    path = Path(path)
    if (path / FEATURE_MANIFEST).is_file():
        return [path.name]
    for parent in _set_parents(path):
        ids = _sets_in(parent)
        if ids:
            return ids
    return []


def find_feature_dir(path: PathLike, feature_set: Optional[str] = None) -> Path:
    """The folder of one feature set: *path* itself (it holds ``manifest.json``) or found below it.

    Raises:
        FileNotFoundError: If there is none.
        ValueError: If there are several and *feature_set* is None.
    """
    path = Path(path)
    if (path / FEATURE_MANIFEST).is_file() and feature_set in (None, path.name):
        return path
    for parent in _set_parents(path):
        ids = _sets_in(parent)
        if feature_set is not None and feature_set in ids:
            return parent / feature_set
        if feature_set is None and len(ids) == 1:
            return parent / ids[0]
        if feature_set is None and len(ids) > 1:
            raise ValueError(f"Pick a feature_set: {parent} holds {ids}")
    raise FileNotFoundError(
        f"No aerial feature tiles in {path}"
        + (f" for feature set {feature_set!r}" if feature_set else "")
        + f": expected {AERIAL_DIR}/{FEATURES_DIR}/<set>/{FEATURE_MANIFEST}. They come with a full 'kitscenes "
        "download ROOT' (unless --no-aerial-features) and with a filtered one with --aerial-features (the areas of "
        "the selected scenes, e.g. '--splits val --components none --aerial-features'), or pass the folder that "
        "holds them."
    )


def bf16_to_float32(bits: np.ndarray) -> np.ndarray:
    """bfloat16 values (as their uint16 bit patterns) -> float32 (exact)."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


class FeatureLocation(NamedTuple):
    """Where one feature tile is: a file (*member* None) or member *member* of the zip *path*."""

    path: str
    member: Optional[str]

    def describe(self) -> str:
        return f"{self.path}:{self.member}" if self.member else self.path


class _Tensor(NamedTuple):
    dtype: type
    shape: tuple[int, ...]
    offset: int  # absolute file offset of the first byte


class _FeatureReader:
    """Per-process access to safetensors feature files: open zips and memory maps of their tensors."""

    def __init__(self) -> None:
        self._pid = -1
        self._reset()

    def _reset(self) -> None:
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._zips: dict[str, ZipMembers] = {}
        self._maps = LRU(1 << 62, MAX_OPEN_MAPS)
        self._headers: dict[tuple[str, Optional[str]], tuple[dict[str, _Tensor], dict]] = {}

    def _span(self, loc: FeatureLocation) -> tuple[int, int]:
        """(offset, size) of the file's bytes: the whole file, or the stored zip member."""
        if loc.member is None:
            return 0, os.path.getsize(loc.path)
        members = self._zips.get(loc.path)
        if members is None:
            with self._lock:
                members = self._zips.setdefault(loc.path, ZipMembers(loc.path))
        info = members.infos[loc.member]
        if info.compress_type != 0 or info.flag_bits & 0x1:
            raise ValueError(f"{loc.describe()} is compressed or encrypted; feature zips must store their members")
        return members._data_offset(info), info.file_size

    def header(self, loc: FeatureLocation) -> tuple[dict[str, _Tensor], dict]:
        """The tensors (dtype, shape, absolute offset) and the ``__metadata__`` of a safetensors file."""
        if self._pid != os.getpid():
            self._reset()
        key = (loc.path, loc.member)
        hit = self._headers.get(key)
        if hit is not None:
            return hit
        start, size = self._span(loc)
        with open(loc.path, "rb") as fh:
            fh.seek(start)
            (n,) = np.frombuffer(fh.read(8), dtype="<u8")
            if n > size - 8:
                raise ValueError(f"{loc.describe()} is no safetensors file (header of {n} bytes)")
            header = json.loads(fh.read(int(n)))
        data_start = start + 8 + int(n)
        tensors = {}
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            dtype = _DTYPES.get(entry["dtype"])
            if dtype is None:
                raise ValueError(f"{loc.describe()}: tensor {name} has unsupported dtype {entry['dtype']}")
            begin, end = entry["data_offsets"]
            shape = tuple(int(s) for s in entry["shape"])
            if end - begin != int(np.prod(shape)) * np.dtype(dtype).itemsize or data_start + end > start + size:
                raise ValueError(f"{loc.describe()}: tensor {name} does not fit its offsets")
            tensors[name] = _Tensor(dtype, shape, data_start + int(begin))
        result = (tensors, header.get("__metadata__", {}) or {})
        with self._lock:
            self._headers[key] = result
        return result

    def tensor(self, loc: FeatureLocation, name: str) -> np.ndarray:
        """Tensor *name* of a feature file, memory-mapped read-only (bfloat16 as uint16 bit patterns)."""
        tensors, _ = self.header(loc)
        t = tensors[name]
        return self._maps.get(
            (loc.path, loc.member, name),
            lambda: np.memmap(loc.path, dtype=t.dtype, mode="r", offset=t.offset, shape=t.shape),
            lambda a: 0,
        )


class AerialFeatures:
    """The DINOv3 feature tiles of one feature set, read in place (see the module docstring).

    Picklable (to its folder and options) and fork-safe: open zips and
    memory maps belong to one process, so it goes into DataLoader workers as
    it is.

    Args:
        root: A dataset root, its ``aerial/`` or ``aerial/features/``, or the
            feature set's folder.
        feature_set: Which set (default: the only one).

    Raises:
        FileNotFoundError: If there are no feature tiles.
        ValueError: If there are several feature sets and *feature_set* is None.
    """

    def __init__(self, root: PathLike, feature_set: Optional[str] = None) -> None:
        self.folder = find_feature_dir(root, feature_set).resolve()
        self.feature_set = self.folder.name
        self.manifest: dict = json.loads((self.folder / FEATURE_MANIFEST).read_text())
        m = self.manifest
        #: The backbone blocks whose tokens are stored, in file order (e.g. ``(4, 11, 17, 23)``).
        self.layers: tuple[int, ...] = tuple(int(x) for x in m["layers"])
        self.channels = int(m["channels"])
        #: Tokens per tile edge (128) and the edge of one token in metres (1.28).
        self.tokens = int(m["tokens_per_tile"])
        self.tile_extent_m = float(m["tile_extent_m"])
        self.token_m = self.tile_extent_m / self.tokens
        self.bucket_size_m = float(m.get("area_bucket_size_m", DEFAULT_AREA_BUCKET_SIZE_M))
        self._grids: dict[str, TileGrid] = {}
        self._tiles: dict[str, dict[tuple[int, int], FeatureLocation]] = {}
        self._info: dict[str, dict[tuple[int, int], dict]] = {}
        for area_id, entry in m["areas"].items():
            self._grids[area_id] = TileGrid(float(entry["origin_xy"][0]), float(entry["origin_xy"][1]),
                                            self.tile_extent_m)
            zip_path = self.folder / f"{area_id}.zip"
            tiles, info = {}, {}
            for t in entry["tiles"]:
                ij = (int(t["i"]), int(t["j"]))
                name = f"tile_{ij[0]}_{ij[1]}.safetensors"
                tiles[ij] = (FeatureLocation(str(zip_path), f"{area_id}/{name}") if zip_path.is_file()
                             else FeatureLocation(str(self.folder / area_id / name), None))
                info[ij] = {k: v for k, v in t.items() if k not in ("i", "j")}
            self._tiles[area_id] = tiles
            self._info[area_id] = info
        self._available = {a: frozenset(t) for a, t in self._tiles.items()}
        self._reader = _FeatureReader()

    def __repr__(self) -> str:
        return (f"AerialFeatures({str(self.folder)!r}: {self.num_tiles} tiles in {len(self._grids)} areas, "
                f"layers {list(self.layers)}, {self.channels} channels)")

    def __getstate__(self) -> dict:
        return {"folder": self.folder}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["folder"])

    # ------------------------------------------------------------------
    # Metadata

    @property
    def area_ids(self) -> list[str]:
        return sorted(self._grids)

    @property
    def num_tiles(self) -> int:
        return sum(len(t) for t in self._tiles.values())

    @property
    def terms(self) -> Optional[Path]:
        """The terms of use (``TERMS.md``), if shipped."""
        path = self.folder / TERMS
        return path if path.is_file() else None

    def grid(self, area_id: str) -> TileGrid:
        return self._grids[area_id]

    def tiles(self, area_id: str) -> frozenset[tuple[int, int]]:
        """The ``(i, j)`` tiles of *area_id* (empty for an unknown area)."""
        return self._available.get(area_id, frozenset())

    def tile_info(self, area_id: str, i: int, j: int) -> dict:
        """What the manifest says about a tile (e.g. ``missing_neighbors``: edges encoded without real context)."""
        return dict(self._info[area_id][(int(i), int(j))])

    def location(self, area_id: str, i: int, j: int) -> FeatureLocation:
        return self._tiles[area_id][(int(i), int(j))]

    def area_of(self, lat: float, lon: float) -> str:
        """The area (10 km bucket) of a WGS84 position, e.g. a scene's map origin."""
        from kitscenes.aerial._areas import bucket_area_id

        return bucket_area_id(lat, lon, self.bucket_size_m)

    def iter_tiles(self, areas: Optional[Iterable[str]] = None):
        """``(area_id, (i, j))`` of every tile (of *areas*), sorted."""
        for area_id in sorted(self._tiles if areas is None else areas):
            for ij in sorted(self._tiles.get(area_id, {})):
                yield area_id, ij

    # ------------------------------------------------------------------
    # Tokens

    def _layer_index(self, layers: Optional[Sequence[int]]) -> list[int]:
        if layers is None:
            return list(range(len(self.layers)))
        index = []
        for layer in layers:
            if int(layer) not in self.layers:
                raise ValueError(f"Layer {layer} is not stored; {self.feature_set} has layers {list(self.layers)}")
            index.append(self.layers.index(int(layer)))
        return index

    def tile(self, area_id: str, i: int, j: int, layers: Optional[Sequence[int]] = None) -> np.ndarray:
        """The patch tokens of tile (i, j): (L, C, 128, 128) float32, row 0 the north edge.

        Args:
            layers: Backbone blocks to read (default all, :attr:`layers`); only those are read.

        Raises:
            KeyError: For a tile the manifest does not list.
        """
        maps = self._reader.tensor(self._tiles[area_id][(int(i), int(j))], "maps")
        return bf16_to_float32(np.stack([maps[k] for k in self._layer_index(layers)]))

    def cls(self, area_id: str, i: int, j: int, layers: Optional[Sequence[int]] = None) -> np.ndarray:
        """The class tokens of tile (i, j), from the same forward as its patch tokens: (L, C) float32."""
        cls = self._reader.tensor(self._tiles[area_id][(int(i), int(j))], "cls")
        return bf16_to_float32(np.stack([cls[k] for k in self._layer_index(layers)]))

    def tile_metadata(self, area_id: str, i: int, j: int) -> dict:
        """The ``__metadata__`` stored in the tile file (values JSON-decoded)."""
        _, meta = self._reader.header(self._tiles[area_id][(int(i), int(j))])
        out = {}
        for k, v in meta.items():
            try:
                out[k] = json.loads(v)
            except (TypeError, ValueError):
                out[k] = v
        return out

    def sampler(self, area_id: str) -> TileSampler:
        """Tile-range and coverage checks on the grid of *area_id* (:class:`kitscenes.aerial._crop.TileSampler`)."""
        return TileSampler(area_id, self._grids[area_id], self._available[area_id], lambda i, j: None,
                           self.feature_set)

    def covering_area(self, bbox: tuple[float, float, float, float], prefer: Optional[str] = None) -> Optional[str]:
        """The first area (*prefer* first) whose tiles cover the EPSG:25832 *bbox*, or None."""
        order = ([prefer] if prefer in self._grids else []) + [a for a in sorted(self._grids) if a != prefer]
        for area_id in order:
            if not self.sampler(area_id).missing_tiles(bbox):
                return area_id
        return None

    def window(self, area_id: str, bbox: tuple[float, float, float, float],
               layers: Optional[Sequence[int]] = None) -> np.ndarray:
        """The tokens of an EPSG:25832 *bbox* on the token grid: (L, C, H, W) float32, row 0 its north edge.

        *bbox* should lie on the token grid (multiples of :attr:`token_m` from the tile origin).

        Raises:
            TileCoverageError: If a tile the window needs is not there.
        """
        sampler = self.sampler(area_id)
        missing = sampler.missing_tiles(bbox)
        if missing:
            raise TileCoverageError(f"Missing feature tiles in {area_id} of {self.feature_set}: {missing}")
        grid, res, n = self._grids[area_id], self.token_m, self.tokens
        min_i, min_j, max_i, max_j = sampler.tile_range(bbox)
        mosaic_min_e = grid.origin_e + min_i * grid.tile_extent_m
        mosaic_max_n = grid.origin_n + (max_j + 1) * grid.tile_extent_m
        width = round((bbox[2] - bbox[0]) / res)
        height = round((bbox[3] - bbox[1]) / res)
        col_start = round((bbox[0] - mosaic_min_e) / res)
        row_start = round((mosaic_max_n - bbox[3]) / res)
        index = self._layer_index(layers)
        out = np.empty((len(index), self.channels, height, width), dtype=np.float32)
        for i in range(min_i, max_i + 1):
            col_off = (i - min_i) * n
            c0, c1 = max(col_start, col_off), min(col_start + width, col_off + n)
            for j in range(min_j, max_j + 1):
                row_off = (max_j - j) * n
                r0, r1 = max(row_start, row_off), min(row_start + height, row_off + n)
                if c0 >= c1 or r0 >= r1:
                    continue
                maps = self._reader.tensor(self._tiles[area_id][(i, j)], "maps")
                for dst, k in enumerate(index):
                    out[dst, :, r0 - row_start:r1 - row_start, c0 - col_start:c1 - col_start] = bf16_to_float32(
                        maps[k, :, r0 - row_off:r1 - row_off, c0 - col_off:c1 - col_off])
        return out


def warp_tokens_ego_aligned(
    source: np.ndarray,
    sample_bbox: tuple[float, float, float, float],
    source_m: float,
    ego_xy: tuple[float, float],
    ego_yaw: float,
    roi_m: tuple[float, float],
    resolution_m: float,
) -> np.ndarray:
    """Bilinear ego-aligned crop of (..., H, W) *source*, the tokens of *sample_bbox* at *source_m* per token.

    The geometry of :func:`kitscenes.aerial._crop.warp_ego_aligned`: output
    ``(..., round(lateral_m / resolution_m), round(longitudinal_m / resolution_m))``,
    ego at the centre, forward to +column, left to -row; token centres at
    ``(k + 0.5) * source_m``. Edges replicate (the sample window holds the crop
    at any heading, so this only touches the outermost half token).
    """
    longitudinal_m, lateral_m = roi_m
    width_px = round(longitudinal_m / resolution_m)
    height_px = round(lateral_m / resolution_m)
    rows, cols = np.meshgrid(np.arange(height_px, dtype=np.float64), np.arange(width_px, dtype=np.float64),
                             indexing="ij")
    x_body = (cols - (width_px - 1) / 2.0) * resolution_m
    y_body = -(rows - (height_px - 1) / 2.0) * resolution_m
    cos_yaw, sin_yaw = math.cos(ego_yaw), math.sin(ego_yaw)
    world_e = ego_xy[0] + cos_yaw * x_body - sin_yaw * y_body
    world_n = ego_xy[1] + sin_yaw * x_body + cos_yaw * y_body
    src_col = (world_e - sample_bbox[0]) / source_m - 0.5
    src_row = (sample_bbox[3] - world_n) / source_m - 0.5
    h, w = source.shape[-2:]
    src_col = np.clip(src_col, 0.0, w - 1.0)
    src_row = np.clip(src_row, 0.0, h - 1.0)
    c0 = np.minimum(np.floor(src_col).astype(np.int64), w - 2) if w > 1 else np.zeros_like(src_col, np.int64)
    r0 = np.minimum(np.floor(src_row).astype(np.int64), h - 2) if h > 1 else np.zeros_like(src_row, np.int64)
    fc = (src_col - c0).astype(np.float32)
    fr = (src_row - r0).astype(np.float32)
    c1 = np.minimum(c0 + 1, w - 1)
    r1 = np.minimum(r0 + 1, h - 1)
    flat = source.reshape(-1, h * w)
    out = (flat[:, r0 * w + c0] * ((1 - fr) * (1 - fc)) + flat[:, r0 * w + c1] * ((1 - fr) * fc)
           + flat[:, r1 * w + c0] * (fr * (1 - fc)) + flat[:, r1 * w + c1] * (fr * fc))
    return out.reshape(source.shape[:-2] + (height_px, width_px)).astype(np.float32, copy=False)


@dataclasses.dataclass(frozen=True)
class NorthUpFeatures:
    """A north-up window of feature tokens around the ego, with the geometry to place it in the ego frame.

    The tokens are the stored ones, unchanged (no interpolation, no rotation):
    row 0 is the north edge, column 0 the west edge, one token per
    :attr:`token_m`. Use the helpers to relate them to the ego frame
    (``x`` forward, ``y`` left, metres): :meth:`token_centers_ego` for
    positional encodings, :meth:`ego_to_tokens` to sample at ego-frame
    points (cross-/deformable attention), :meth:`sampling_grid` for an
    ego-aligned ``torch.nn.functional.grid_sample`` of *learned* features.
    ``np.asarray(window)`` gives :attr:`tokens`. See ``docs/aerial_features.md``.

    Attributes:
        tokens: (L, C, S, S) float32.
        bbox_epsg25832: ``(min_e, min_n, max_e, max_n)`` of the window.
        token_m: Edge of a token in metres (1.28).
        ego_xy: EPSG:25832 position of the ego.
        yaw: Heading of the ego, radians counter-clockwise from east.
        layers: The backbone blocks in :attr:`tokens`.
    """

    tokens: np.ndarray
    bbox_epsg25832: tuple[float, float, float, float]
    token_m: float
    ego_xy: tuple[float, float]
    yaw: float
    layers: tuple[int, ...]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return self.tokens if dtype is None else self.tokens.astype(dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.tokens.shape

    @property
    def ego_rc(self) -> tuple[float, float]:
        """The ego in token coordinates ``(row, col)``: token ``(r, c)`` covers ``[r, r + 1) x [c, c + 1)``."""
        rc = self.ego_to_tokens(np.zeros((1, 2)))[0]
        return float(rc[0]), float(rc[1])

    def ego_to_tokens(self, points_xy: np.ndarray) -> np.ndarray:
        """(N, 2) ego-frame points (x forward, y left; m) -> (N, 2) token coordinates ``(row, col)``.

        Token centres are at ``(r + 0.5, c + 0.5)``.
        """
        p = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        e = self.ego_xy[0] + c * p[:, 0] - s * p[:, 1]
        n = self.ego_xy[1] + s * p[:, 0] + c * p[:, 1]
        return np.stack([(self.bbox_epsg25832[3] - n) / self.token_m, (e - self.bbox_epsg25832[0]) / self.token_m], -1)

    def token_centers_ego(self) -> np.ndarray:
        """(S, S, 2) ego-frame position (x forward, y left; m) of every token centre."""
        size = self.tokens.shape[-1]
        k = (np.arange(size) + 0.5) * self.token_m
        de = self.bbox_epsg25832[0] + k[None, :] - self.ego_xy[0]
        dn = self.bbox_epsg25832[3] - k[:, None] - self.ego_xy[1]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.stack([c * de + s * dn, -s * de + c * dn], -1).astype(np.float32)

    def sampling_grid(self, roi: tuple[float, float], resolution_m: float) -> np.ndarray:
        """(H, W, 2) grid for ``grid_sample(..., align_corners=False)`` of an ego-aligned crop of this window.

        ``H = round(lateral_m / resolution_m)``, ``W = round(longitudinal_m / resolution_m)``, ego at the
        centre, forward to +column, left to -row (the crop geometry of :class:`AerialFeatureCrops`). Meant
        for features a model computed from :attr:`tokens` (north-up) and wants in the ego frame.
        """
        longitudinal_m, lateral_m = roi
        width, height = round(longitudinal_m / resolution_m), round(lateral_m / resolution_m)
        rows, cols = np.meshgrid(np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64),
                                 indexing="ij")
        pts = np.stack([(cols - (width - 1) / 2.0) * resolution_m, -(rows - (height - 1) / 2.0) * resolution_m], -1)
        rc = self.ego_to_tokens(pts.reshape(-1, 2)).reshape(height, width, 2)
        size = self.tokens.shape[-1]
        return np.stack([rc[..., 1] / size * 2 - 1, rc[..., 0] / size * 2 - 1], -1).astype(np.float32)


class AerialFeatureCrops(PoseCrops):
    """Crops of the aerial DINOv3 features around the ego, cut on the fly (like :class:`kitscenes.aerial.AerialCrops`).

    ``crops = ds.use_aerial_features(roi=...)`` makes one and attaches it to
    the dataset. ``crops[frame]`` cuts the crop of a frame
    (:meth:`kitscenes.Frame.aerial_features` is the shortcut for the attached
    crops); :meth:`crop` cuts one at any EPSG:25832 pose. Two orientations:

    - ``"north"`` (the default, for learning): a :class:`NorthUpFeatures`, the
      stored tokens of a fixed-size north-up window that holds the ROI at any
      heading, unchanged, with the pose and helpers to place them in the ego
      frame. The model does the rotation (rotate its outputs, or sample with
      the pose; see ``docs/aerial_features.md``).
    - ``"ego"``: ``(L, C, round(lateral_m / resolution_m), round(longitudinal_m
      / resolution_m))`` float32, ego at the centre, forward to +column, left to
      -row, bilinear between token centres (the geometry of the aerial image
      crops). Rotating the token grid does not rotate what the tokens encode:
      DINOv3 features are not rotation-equivariant, and a head trained
      north-up loses about half its mIoU on such crops (see
      ``docs/aerial_features.md``). Use it where only the layout matters.

    The tiles are channel-first, so a crop reads the requested layers of
    each tile it touches (32 MiB per layer and tile from disk or the page
    cache); ask only for the layers you use.

    Args:
        features: An :class:`AerialFeatures`, or what it takes.
        roi: ``(longitudinal_m, lateral_m)`` of a crop, centred on the ego (``"north"``: covered at any heading).
        layers: Backbone blocks (default: the last stored one; ``features.layers`` for all).
        resolution_m: ``"ego"`` only: size of a crop cell (default: a token, 1.28 m).
        orientation: ``"north"`` (default) or ``"ego"``.
    """

    _what = "aerial_features"
    _placed_on = "the features"
    _tiles_name = "the feature tiles"

    def __init__(
        self,
        features: Union[AerialFeatures, PathLike],
        roi: tuple[float, float],
        *,
        layers: Optional[Sequence[int]] = None,
        resolution_m: Optional[float] = None,
        orientation: str = "north",
    ) -> None:
        if orientation not in ("ego", "north"):
            raise ValueError(f"orientation must be 'ego' or 'north', got {orientation!r}")
        if orientation == "north" and resolution_m is not None:
            raise ValueError("resolution_m goes with orientation='ego'; north-up crops are the stored tokens")
        self.features = features if isinstance(features, AerialFeatures) else AerialFeatures(features)
        self.roi_m = (float(roi[0]), float(roi[1]))
        self.orientation = orientation
        self.layers = tuple(int(x) for x in (layers if layers is not None else self.features.layers[-1:]))
        self.features._layer_index(self.layers)  # raises for a layer that is not stored
        self.resolution_m = float(resolution_m) if resolution_m is not None else self.features.token_m
        #: ``"north"``: tokens from the ego's token to the window edge (window = 2 * half + 1 tokens).
        self.half_tokens = math.ceil(0.5 * math.hypot(*self.roi_m) / self.features.token_m) + 1
        layer_tag = "-".join(str(x) for x in self.layers)
        self.cache_id = (f"{self.features.feature_set}_roi{self.roi_m[0]:g}x{self.roi_m[1]:g}_{orientation}"
                         f"_l{layer_tag}_r{round(self.resolution_m * 100):03d}")
        self._init_process()

    def __repr__(self) -> str:
        return (f"AerialFeatureCrops({self.features!r}, roi={self.roi_m}, layers={list(self.layers)}, "
                f"orientation={self.orientation!r}, resolution_m={self.resolution_m:g})")

    def __getstate__(self) -> dict:
        return {"features": self.features, "roi": self.roi_m, "layers": self.layers,
                "resolution_m": self.resolution_m, "orientation": self.orientation}

    def __setstate__(self, state: dict) -> None:
        north = state["orientation"] == "north"
        self.__init__(state["features"], state["roi"], layers=state["layers"],
                      resolution_m=None if north else state["resolution_m"], orientation=state["orientation"])

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """``(L, C, height, width)`` of a crop (``"north"``: of its tokens, a fixed square)."""
        if self.orientation == "north":
            size = 2 * self.half_tokens + 1
            return (len(self.layers), self.features.channels, size, size)
        return (len(self.layers), self.features.channels, round(self.roi_m[1] / self.resolution_m),
                round(self.roi_m[0] / self.resolution_m))

    # ------------------------------------------------------------------
    # Crops at any pose

    def _sample_bbox(self, xy: tuple[float, float], area_id: Optional[str] = None) -> tuple[float, float, float, float]:
        if self.orientation == "ego":
            return crop_sample_bbox((float(xy[0]), float(xy[1])), self.roi_m, self.features.token_m)
        # north: the fixed window of 2 * half + 1 tokens around the token that holds the ego, on the area's grid
        grid = self.features.grid(area_id if area_id in self.features.area_ids else self.features.area_ids[0])
        m, h = self.features.token_m, self.half_tokens
        ce = math.floor((float(xy[0]) - grid.origin_e) / m)
        cn = math.floor((float(xy[1]) - grid.origin_n) / m)
        return (grid.origin_e + (ce - h) * m, grid.origin_n + (cn - h) * m,
                grid.origin_e + (ce + h + 1) * m, grid.origin_n + (cn + h + 1) * m)

    def covers(self, xy: tuple[float, float], *, area_id: Optional[str] = None) -> bool:
        """Whether the feature tiles hold everything a crop at EPSG:25832 *xy* needs (at any heading)."""
        return self.features.covering_area(self._sample_bbox(xy, area_id), area_id) is not None

    def crop(self, xy: tuple[float, float], yaw: float, *,
             area_id: Optional[str] = None) -> Union[np.ndarray, NorthUpFeatures]:
        """The feature crop at EPSG:25832 *xy* with heading *yaw* (radians, counter-clockwise from east).

        Returns:
            ``"ego"``: (L, C, H, W) float32; ``"north"``: a :class:`NorthUpFeatures` (see the class docstring).

        Raises:
            TileCoverageError: If no area has every tile the crop needs.
        """
        ego_xy = (float(xy[0]), float(xy[1]))
        bbox = self._sample_bbox(ego_xy, area_id)
        area = self.features.covering_area(bbox, area_id)
        if area is None:
            raise TileCoverageError(f"No feature tiles for a crop at EPSG:25832 {ego_xy} in {self.features!r}")
        window = self.features.window(area, bbox, self.layers)
        if self.orientation == "north":
            return NorthUpFeatures(window, bbox, self.features.token_m, ego_xy, float(yaw), self.layers)
        return warp_tokens_ego_aligned(window, bbox, self.features.token_m, ego_xy, float(yaw), self.roi_m,
                                       self.resolution_m)

    # ------------------------------------------------------------------
    # Crops of frames (crops[frame], Frame.aerial_features)

    def _area_of(self, scene: "Scene", poses: "ScenePoses") -> Optional[str]:
        return self.features.area_of(poses.lat, poses.lon)
