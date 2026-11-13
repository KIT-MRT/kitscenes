"""The raw aerial tiles: :class:`AerialTiles`.

The release ships the DOP20 tiles (2048 px, 0.2 m/px, JPEG quality 95,
4:4:4) in ``<dataset root>/aerial/``, in one of two layouts::

    aerial/ATTRIBUTION.md                                     licences and source notes
    aerial/metadata/area_index.json                           scene -> 10 km area
    aerial/metadata/provider_metadata.json                    WMS, licence per state
    aerial/metadata/raw_tile_manifest_dop20_res020_tile2048.json  tile grid and tiles per area
    aerial/raw_tiles/dop20_res020_tile2048/<area>.zip         as downloaded: one zip per area, members
                                                              <area>/tile_<i>_<j>.jpg (+ .json), stored
    aerial/raw_tiles/dop20_res020_tile2048/<area>/tile_<i>_<j>.jpg (+ .json)   or extracted

An area's zip is read in place (members by offset, nothing extracted) and
preferred over an extracted folder. Folders of PNG tiles and tile stores
(:func:`kitscenes.aerial.build_tile_store`: ``.npy``) have the same
``metadata/`` + ``raw_tiles/`` layout, so all of them are read the same
way, alone or several together (e.g. one folder per state).
"""

from __future__ import annotations

import functools
import json
import os
import posixpath
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, NamedTuple, Optional, Sequence, Union

import numpy as np

from kitscenes.aerial._areas import DEFAULT_AREA_BUCKET_SIZE_M
from kitscenes.aerial._tile_io import DEFAULT_TILE_CACHE_MB, SUFFIX_FORMAT, TileReader
from kitscenes.aerial._tiling import TileGrid

if TYPE_CHECKING:
    from kitscenes.aerial._crop import TileSampler

PathLike = Union[str, os.PathLike]

AERIAL_DIR = "aerial"
METADATA_DIR = "metadata"
RAW_TILES_DIR = "raw_tiles"
MANIFEST_PREFIX = "raw_tile_manifest_"
AREA_INDEX = "area_index.json"
PROVIDER_METADATA = "provider_metadata.json"
ATTRIBUTION = "ATTRIBUTION.md"


def raw_cache_ids(aerial_dir: PathLike) -> list[str]:
    """IDs of the raw tile sets in an aerial folder (``metadata/raw_tile_manifest_<id>.json``), sorted."""
    folder = Path(aerial_dir) / METADATA_DIR
    if not folder.is_dir():
        return []
    return sorted(p.name[len(MANIFEST_PREFIX):-len(".json")] for p in folder.glob(f"{MANIFEST_PREFIX}*.json"))


def find_aerial_dir(path: PathLike) -> Path:
    """The aerial folder of *path*: *path* itself if it holds a raw tile manifest, else ``<path>/aerial``.

    Raises:
        FileNotFoundError: If neither does.
    """
    path = Path(path)
    if raw_cache_ids(path):
        return path
    if raw_cache_ids(path / AERIAL_DIR):
        return path / AERIAL_DIR
    raise FileNotFoundError(
        f"No raw aerial tiles in {path}: neither it nor {path / AERIAL_DIR} has "
        f"{METADATA_DIR}/{MANIFEST_PREFIX}<id>.json. They come with every 'kitscenes download ROOT' (the areas of "
        "the selected scenes, unless --no-aerial; e.g. '--splits val --components none' adds those of val to a root "
        "with little else), or pass the folder that holds them."
    )


class TileLocation(NamedTuple):
    """Where one tile is: a file (*member* None) or member *member* of the zip *path*."""

    root: int
    path: str
    member: Optional[str]
    format: str

    def describe(self) -> str:
        return f"{self.path}:{self.member}" if self.member else self.path


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


class AerialTiles:
    """The raw aerial tiles of one or more aerial folders, read in place.

    Each folder holds ``metadata/`` and ``raw_tiles/`` (see the module
    docstring): the release layout (zips or extracted JPEG), PNG tiles
    or a tile store (``.npy``). Folders are
    merged: their areas must use the same tile grid, and a tile is read
    from the first folder that lists it.

    Picklable (to its folders and options) and fork-safe: open zips,
    memory maps and the cache of decoded tiles belong to one process, so
    it goes into DataLoader workers as it is.

    Args:
        roots: A dataset root (its ``aerial/``), an aerial folder, or a list
            of them.
        raw_cache_id: Which tile set, e.g. ``"dop20_res020_tile2048"``
            (default: the only one).
        cache_mb: Decoded JPEG/PNG tiles kept per process, in MiB (12 MiB per
            2048 px tile; 0: none): the bound of the cache, so a DataLoader
            with N workers holds at most N x *cache_mb*. Tile-store tiles are
            memory-mapped, not cached: their pages are the OS page cache,
            shared by all processes and reclaimable.

    Raises:
        FileNotFoundError: If a folder has no raw tile manifest.
        ValueError: If the folders do not fit together (another tile size,
            an area on another grid, a scene in two areas), or hold several
            tile sets and *raw_cache_id* is None.
    """

    def __init__(
        self,
        roots: Union[PathLike, Sequence[PathLike]],
        *,
        raw_cache_id: Optional[str] = None,
        cache_mb: float = DEFAULT_TILE_CACHE_MB,
    ) -> None:
        if isinstance(roots, (str, os.PathLike)):
            roots = [roots]
        self.roots: list[Path] = []
        for root in roots:
            folder = find_aerial_dir(root).resolve()
            if folder not in self.roots:
                self.roots.append(folder)
        if not self.roots:
            raise ValueError("AerialTiles needs at least one folder")
        if raw_cache_id is None:
            ids = sorted({i for r in self.roots for i in raw_cache_ids(r)})
            if len(ids) != 1:
                raise ValueError(f"Pick a raw_cache_id: {', '.join(map(str, self.roots))} hold {ids}")
            raw_cache_id = ids[0]
        self.raw_cache_id = raw_cache_id
        self.cache_mb = float(cache_mb)
        self._load()
        self._reader = TileReader(self.cache_mb)

    # ------------------------------------------------------------------
    # Metadata

    def _load(self) -> None:
        self._grids: dict[str, TileGrid] = {}
        self._tiles: dict[str, dict[tuple[int, int], TileLocation]] = {}
        self.scene_to_area: dict[str, str] = {}
        self.area_info: dict[str, dict] = {}
        self.provider_metadata: dict[str, dict] = {}
        self.bucket_size_m = DEFAULT_AREA_BUCKET_SIZE_M
        self.formats: set[str] = set()
        self.tile_format: dict = {}
        params = None
        for k, root in enumerate(self.roots):
            manifest_path = root / METADATA_DIR / f"{MANIFEST_PREFIX}{self.raw_cache_id}.json"
            data = _read_json(manifest_path)
            if data is None:
                raise FileNotFoundError(f"{root} has no {METADATA_DIR}/{manifest_path.name}")
            these = (data.get("provider"), float(data["resolution_m_per_px"]), int(data["effective_tile_size_px"]),
                     float(data["tile_extent_m"]))
            if params is None:
                params = these
                self.provider, self.resolution_m, self.tile_size_px, self.tile_extent_m = these
                self.requested_tile_size_px = int(data.get("tile_size_px", these[2]))
                self.tile_format = dict(data.get("tile_format") or {})
            elif not (abs(these[1] - params[1]) < 1e-9 and these[2] == params[2]
                      and abs(these[3] - params[3]) < 1e-6):
                raise ValueError(f"{root}: tiles of {these[2]} px at {these[1]} m/px, but {self.roots[0]} has "
                                 f"{params[2]} px at {params[1]} m/px; they cannot be combined")
            tiles_dir = root / RAW_TILES_DIR / self.raw_cache_id
            for area_id, entry in data["areas"].items():
                origin = (float(entry["origin_xy"][0]), float(entry["origin_xy"][1]))
                grid = self._grids.get(area_id)
                if grid is None:
                    self._grids[area_id] = TileGrid(origin_e=origin[0], origin_n=origin[1],
                                                    tile_extent_m=self.tile_extent_m)
                    self._tiles[area_id] = {}
                elif abs(grid.origin_e - origin[0]) > 1e-6 or abs(grid.origin_n - origin[1]) > 1e-6:
                    raise ValueError(
                        f"Area {area_id} has another tile grid in {root} (origin {list(origin)}) than in "
                        f"{self._root_of(area_id)} ({[grid.origin_e, grid.origin_n]}); their tiles cannot be combined."
                    )
                zip_path = tiles_dir / f"{area_id}.zip"
                in_zip = zip_path.is_file()
                area_dir = tiles_dir / area_id
                tiles = self._tiles[area_id]
                for t in entry["tiles"]:
                    ij = (int(t["i"]), int(t["j"]))
                    if ij in tiles:
                        continue
                    stored = str(t.get("path") or "")
                    suffix = posixpath.splitext(stored)[1].lower() or ".png"
                    fmt = SUFFIX_FORMAT.get(suffix)
                    if fmt is None:
                        raise ValueError(f"{manifest_path}: unknown tile format {stored!r}")
                    name = f"tile_{ij[0]}_{ij[1]}{suffix}"
                    if in_zip:
                        location = TileLocation(k, str(zip_path), f"{area_id}/{name}", fmt)
                    else:  # the tile is where the layout puts it
                        location = TileLocation(k, str(area_dir / name), None, fmt)
                    tiles[ij] = location
                    self.formats.add(fmt)
            index = _read_json(root / METADATA_DIR / AREA_INDEX, {})
            if index:
                self.bucket_size_m = float(index.get("bucket_size_m", self.bucket_size_m))
            for scene_id, area_id in index.get("scene_to_area", {}).items():
                if self.scene_to_area.setdefault(scene_id, area_id) != area_id:
                    raise ValueError(f"Scene {scene_id} is in area {area_id} in {root}, "
                                     f"but in {self.scene_to_area[scene_id]} in another folder")
            for area_id, entry in index.get("areas", {}).items():
                mine = self.area_info.setdefault(area_id, {k: v for k, v in entry.items() if k != "scene_ids"})
                mine["scene_ids"] = sorted(set(mine.get("scene_ids", [])) | set(entry.get("scene_ids", [])))
            for state, entry in (_read_json(root / METADATA_DIR / PROVIDER_METADATA, {}) or {}).items():
                self.provider_metadata.setdefault(state, entry)
        self._available = {area_id: frozenset(tiles) for area_id, tiles in self._tiles.items()}

    def _root_of(self, area_id: str) -> Path:
        locations = self._tiles.get(area_id)
        return self.roots[next(iter(locations.values())).root] if locations else self.roots[0]

    def __repr__(self) -> str:
        roots = ", ".join(repr(str(r)) for r in self.roots)
        return (f"AerialTiles([{roots}], {self.raw_cache_id!r}: {self.num_tiles} tiles in {len(self._grids)} areas, "
                f"{'/'.join(sorted(self.formats))})")

    def __getstate__(self) -> dict:
        return {"roots": self.roots, "raw_cache_id": self.raw_cache_id, "cache_mb": self.cache_mb}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["roots"], raw_cache_id=state["raw_cache_id"], cache_mb=state["cache_mb"])

    @property
    def area_ids(self) -> list[str]:
        return sorted(self._grids)

    @property
    def num_tiles(self) -> int:
        return sum(len(t) for t in self._tiles.values())

    @property
    def attribution(self) -> list[Path]:
        """The ``ATTRIBUTION.md`` files of the folders (licences and source notes of the imagery)."""
        return [r / ATTRIBUTION for r in self.roots if (r / ATTRIBUTION).is_file()]

    def grid(self, area_id: str) -> TileGrid:
        return self._grids[area_id]

    def tiles(self, area_id: str) -> frozenset[tuple[int, int]]:
        """The ``(i, j)`` tiles of *area_id* (empty for an unknown area)."""
        return self._available.get(area_id, frozenset())

    def location(self, area_id: str, i: int, j: int) -> TileLocation:
        return self._tiles[area_id][(int(i), int(j))]

    def area_of(self, scene_id: str, lat: Optional[float] = None, lon: Optional[float] = None) -> Optional[str]:
        """The area of a scene: from the area index, else (given its map origin) its 10 km bucket."""
        area_id = self.scene_to_area.get(scene_id)
        if area_id is None and lat is not None and lon is not None:
            from kitscenes.aerial._areas import bucket_area_id

            area_id = bucket_area_id(lat, lon, self.bucket_size_m)
        return area_id

    # ------------------------------------------------------------------
    # Pixels

    def tile(self, area_id: str, i: int, j: int) -> np.ndarray:
        """The (S, S, 3) uint8 RGB pixels of tile (i, j): memory-mapped (``.npy``) or decoded (cached per process).

        Raises:
            KeyError: For a tile the folders do not list.
        """
        loc = self._tiles[area_id][(int(i), int(j))]
        return self._reader.pixels(loc.path, loc.member, loc.format)

    def tile_bytes(self, area_id: str, i: int, j: int) -> bytes:
        """The tile as stored (JPEG, PNG or ``.npy`` bytes)."""
        loc = self._tiles[area_id][(int(i), int(j))]
        return self._reader.read_bytes(loc.path, loc.member)

    def sampler(self, area_id: str) -> "TileSampler":
        """Windows of EPSG:25832 bboxes on the tiles of *area_id* (see :class:`kitscenes.aerial._crop.TileSampler`).

        Made on each call and not kept (it refers back to this object: kept, the cycle would keep
        the decoded tiles alive until the garbage collector runs).
        """
        from kitscenes.aerial._crop import TileSampler

        return TileSampler(area_id, self._grids[area_id], self._available[area_id],
                           functools.partial(self.tile, area_id), self.raw_cache_id)

    def covering_area(self, bbox: tuple[float, float, float, float],
                      prefer: Optional[str] = None) -> Optional[str]:
        """The first area (*prefer* first) whose tiles cover the EPSG:25832 *bbox*, or None."""
        order = ([prefer] if prefer in self._grids else []) + [a for a in sorted(self._grids) if a != prefer]
        for area_id in order:
            if not self.sampler(area_id).missing_tiles(bbox):
                return area_id
        return None

    def iter_tiles(self, areas: Optional[Iterable[str]] = None):
        """``(area_id, (i, j), location)`` of every tile (of *areas*), sorted."""
        for area_id in sorted(self._tiles if areas is None else areas):
            for ij in sorted(self._tiles.get(area_id, {})):
                yield area_id, ij, self._tiles[area_id][ij]
