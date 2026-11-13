"""Downloading scenes in the release archive format (:mod:`kitscenes._archive`), component by component.

The manifest ``data/sequence_archives.csv`` has one row per file:
``sequence_id, split, component, archive_path, archive_sha256,
archive_size_bytes``. ``component`` is the archive's component, ``metadata``
for the loose files of a scene, or one of the rows without a scene:
``city_map``, ``osm_map`` / ``osm_metadata``, ``aerial_tiles`` /
``aerial_metadata``, ``aerial_features`` / ``aerial_features_metadata``.
Files land at ``<root>/<archive_path>``, the layout the API reads directly.
:func:`select_files` says which files a selection takes; see docs/download.md.

A root remembers what it holds in ``dataset_info.json`` (the HF revision the
files come from, the manifest's SHA-256): one root never mixes two versions
of the dataset. Files count as present when their size matches the manifest;
``verify`` checks SHA-256 instead (hashes cached by size and mtime in
``.kitscenes/``). An extracted zip leaves a marker with its SHA-256, so the
next run does not download it again. Moving a root to a new revision removes
every file whose SHA-256 changed (:func:`remove_stale`).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import threading
import posixpath
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Sequence

from kitscenes._archive import COMPONENTS
from kitscenes.constants import CAMERA_NAMES, HD_MAPS_DIR, LIDAR_NAMES, SD_MAPS_DIR
from kitscenes.pack import sha256_file
from kitscenes.splits import _split_name, known_split

logger = logging.getLogger(__name__)

METADATA = "metadata"
#: Component of the Lanelet2 city maps (``lanelet2_hd_maps/<City>.osm``, loose files at the top level).
CITY_MAP = "city_map"
#: Components of the OpenStreetMap SD maps (``openstreetmap_sd_maps/<City>_<YYYY>-W<ww>.osm.pbf``,
#: ``openstreetmap_sd_maps/<set>/<City>.osm`` such as ``paper_2026-05-03/``, and ``ATTRIBUTION.md``; ODbL),
#: loose files at the top level.
SD_MAP = "osm_map"
SD_MAP_METADATA = "osm_metadata"
#: Components of the aerial imagery: one zip of raw tiles per area, and the loose metadata.
AERIAL_TILES = "aerial_tiles"
AERIAL_METADATA = "aerial_metadata"
#: Components of the DINOv3 features of the aerial imagery: one zip per area and feature set, and the
#: set's loose files (manifest, terms of use).
AERIAL_FEATURES = "aerial_features"
AERIAL_FEATURES_METADATA = "aerial_features_metadata"
#: Names that are not ``--components``: KITScenes Aerial has its own flags (``--aerial-features``, ``--no-aerial*``).
_AERIAL_NAMES = ("aerial-tiles", "aerial-features", "aerial", "aerial-metadata", "aerial-features-metadata")
#: The area index: which scene lies in which area (selects the areas of the selected scenes).
AREA_INDEX_PATH = "aerial/metadata/area_index.json"
_AERIAL_ROWS = (AERIAL_TILES, AERIAL_METADATA, AERIAL_FEATURES, AERIAL_FEATURES_METADATA)
_AREA_ZIPS = (AERIAL_TILES, AERIAL_FEATURES)  # one zip per area, extracted into <area>/ next to it
_LOOSE = (CITY_MAP, SD_MAP, SD_MAP_METADATA) + _AERIAL_ROWS  # rows without a scene
DATASET_INFO = "dataset_info.json"
STATE_DIR = ".kitscenes"
#: Per scene: ``<component>.zip.sha256`` for every extracted archive.
EXTRACTED_DIR = ".extracted"

#: The components a partial download chooses from: the cameras and lidars,
#: about 98 % of the data. Everything else always comes along, so nothing
#: that reads a scene breaks.
OPTIONAL_COMPONENTS: frozenset = frozenset(CAMERA_NAMES + LIDAR_NAMES)

#: Named sets of optional components for ``--components``.
PRESETS: dict[str, tuple[str, ...]] = {
    "all": tuple(sorted(OPTIONAL_COMPONENTS)),
    "cameras": CAMERA_NAMES,
    "ring-cameras": tuple(c for c in CAMERA_NAMES if c.startswith("camera_ring")),
    "base-cameras": tuple(c for c in CAMERA_NAMES if c.startswith("camera_base")),
    "front-camera": ("camera_ring_front",),
    "lidars": LIDAR_NAMES,
    "lidar-top": ("lidar_top",),
    "none": (),  # no cameras or lidars: metadata, maps, labels, radar, GNSS/INS
}


@dataclasses.dataclass(frozen=True)
class ArchiveFile:
    """One manifest row."""

    scene_id: str
    split: str
    component: str
    path: str
    sha256: str
    size: int


def _safe_name(value: str, what: str) -> str:
    if not value or value in (".", "..") or value.startswith(".") or any(c in value for c in "/\\\0"):
        raise ValueError(f"Unsafe {what} {value!r} in the manifest.")
    return value


def _safe_path(row: dict) -> str:
    path = row["archive_path"]
    if row["component"] in _AERIAL_ROWS:
        parts = path.split("/")
        bad = ("\\" in path or "\0" in path or posixpath.isabs(path) or posixpath.normpath(path) != path
               or parts[0] != "aerial" or len(parts) < 2 or any(not p or p.startswith(".") for p in parts))
        expected = {AERIAL_TILES: "aerial/raw_tiles/<id>/<area>.zip",
                    AERIAL_FEATURES: "aerial/features/<set>/<area>.zip",
                    AERIAL_FEATURES_METADATA: "aerial/features/<set>/<file>"}.get(row["component"])
        if row["component"] == AERIAL_TILES:
            bad = bad or len(parts) != 4 or parts[1] != "raw_tiles" or not parts[3].endswith(".zip")
        elif row["component"] == AERIAL_FEATURES:
            bad = bad or len(parts) != 4 or parts[1] != "features" or not parts[3].endswith(".zip")
        elif row["component"] == AERIAL_FEATURES_METADATA:
            bad = bad or len(parts) != 4 or parts[1] != "features" or parts[3].endswith(".zip")
        if bad:
            raise ValueError(f"Unsafe archive_path {path!r} in the manifest; expected "
                             + (expected or "aerial/<file>"))
        return path
    if row["component"] == CITY_MAP:
        folder, _, name = path.partition("/")
        ok = (folder == HD_MAPS_DIR and name == posixpath.basename(name) and name
              and not name.startswith(".") and name.endswith(".osm") and "\\" not in name and "\0" not in name)
        if not ok:
            raise ValueError(f"Unsafe archive_path {path!r} in the manifest; expected {HD_MAPS_DIR}/<file>.osm.")
        return path
    if row["component"] in (SD_MAP, SD_MAP_METADATA):
        # openstreetmap_sd_maps/<file> or openstreetmap_sd_maps/<snapshot set>/<file> (e.g. paper_2026-05-03/)
        parts = path.split("/")
        ok = (parts[0] == SD_MAPS_DIR and len(parts) in (2, 3) and "\\" not in path and "\0" not in path
              and all(p and not p.startswith(".") for p in parts[1:]))
        if row["component"] == SD_MAP:
            ok = ok and parts[-1].endswith((".osm", ".osm.pbf"))
        if not ok:
            raise ValueError(f"Unsafe archive_path {path!r} in the manifest; expected {SD_MAPS_DIR}/[<set>/]<file>"
                             + (".osm or .osm.pbf" if row["component"] == SD_MAP else "") + ".")
        return path
    prefix = f"data/{row['split']}/{row['sequence_id']}/"
    if (
        "\\" in path or "\0" in path or posixpath.isabs(path)
        or posixpath.normpath(path) != path or not path.startswith(prefix) or path == prefix
    ):
        raise ValueError(f"Unsafe archive_path {path!r} in the manifest; expected {prefix}<file>.")
    return path


def parse_manifest(rows: Iterable[dict]) -> list[ArchiveFile]:
    """Validate manifest rows (their values become paths).

    Raises:
        ValueError: For an unsafe or unknown value.
    """
    out = []
    for row in rows:
        component = row["component"]
        if component not in (METADATA, *_LOOSE) and component not in COMPONENTS:
            raise ValueError(f"Unknown component {component!r} in the manifest.")
        city = component in _LOOSE
        out.append(ArchiveFile(
            scene_id="" if city else _safe_name(row["sequence_id"], "sequence_id"),
            split="" if city else _safe_name(row["split"], "split"),
            component=component,
            path=_safe_path(row),
            sha256=row["archive_sha256"],
            size=int(row["archive_size_bytes"]),
        ))
    return out


def resolve_components(names: Optional[Sequence[str]]) -> Optional[set[str]]:
    """Camera/lidar names and presets -> the optional components wanted (None = all cameras and lidars).

    Names of components that always come along (``radar``, ``gnss``, the
    labels, ...) are accepted and change nothing. KITScenes Aerial is not
    chosen here (:func:`select_files`; ``--aerial-features``,
    ``--no-aerial-features``, ``--no-aerial``), so its names are refused with
    that hint. The SD maps (the weekly snapshots and the
    paper's ``paper_2026-05-03/``) always come along.

    Raises:
        ValueError: For an unknown name.
    """
    if names is None:
        return None
    out: set[str] = set()
    for name in names:
        if name in PRESETS:
            out.update(PRESETS[name])
        elif name in OPTIONAL_COMPONENTS:
            out.add(name)
        elif name in COMPONENTS or name in (METADATA, CITY_MAP, SD_MAP, SD_MAP_METADATA):
            pass
        else:
            aerial = name in _AERIAL_ROWS or name in _AERIAL_NAMES
            raise ValueError(f"Unknown component or preset {name!r}. Presets: {sorted(PRESETS)}; "
                             f"cameras and lidars: {sorted(OPTIONAL_COMPONENTS)}"
                             + (". KITScenes Aerial is not a component: see --aerial-features, "
                                "--no-aerial-features and --no-aerial" if aerial else ""))
    return out


def aerial_kinds(aerial_tiles: bool = True, aerial_features: bool = True) -> tuple[str, ...]:
    """The aerial zip components a download takes: :data:`AERIAL_TILES` and/or :data:`AERIAL_FEATURES`."""
    return tuple(k for k, on in ((AERIAL_TILES, aerial_tiles), (AERIAL_FEATURES, aerial_features)) if on)


def needs_area_index(files: Sequence[ArchiveFile], *, aerial_features: bool, aerial_tiles: bool = True,
                     **selection) -> bool:
    """Whether :func:`select_files` needs the area index for this selection: aerial zips are wanted and in the
    manifest, and the selection leaves out scenes (:func:`scene_filtered`)."""
    kinds = aerial_kinds(aerial_tiles, aerial_features)
    return any(f.component in kinds for f in files) and scene_filtered(**selection)


def scenes_where(cities: Optional[Sequence[str]] = None, lidar: Optional[str] = None,
                 where: Optional[dict] = None) -> Optional[set[str]]:
    """IDs of the released scenes in *cities* with *lidar* that match *where*, from the shipped scene table;
    None: no filter.

    *where* maps columns of the scene table to criteria as in
    :func:`kitscenes.find_scenes` (``{"weather": ["rain"], "max_speed_mps": (None, 15)}``).
    Names are matched case-insensitively.

    Raises:
        TypeError: For an unknown column.
        ValueError: For an unknown city, lidar or value.
    """
    criteria = {"city": cities, "lidar": lidar, **(where or {})}
    if all(v is None for v in criteria.values()):
        return None
    from kitscenes.splits import find_scenes

    return {info.scene_id for info in find_scenes(**criteria)}


def scene_filtered(splits=None, scene_ids=None, cities=None, lidar=None, max_gb=None, where=None) -> bool:
    """Whether a selection leaves out scenes (then only the aerial areas of the selected scenes are wanted)."""
    return any(x is not None for x in (splits, scene_ids, cities, lidar, max_gb)) or bool(where)


def placed_on_aerial(split: str) -> bool:
    """Whether the scenes of *split* can be placed on the aerial imagery: their poses are in the map frame.

    The test-e2e splits are not (poses relative to the first pose): their
    frames get no aerial crops, so they pull no aerial data into a download.
    """
    info = known_split(split)
    return info is None or info.pose_frame == "local_utm"


def scene_areas(scenes: Iterable[tuple[str, str]], area_index: dict) -> set[str]:
    """The aerial areas of ``(split, scene_id)`` pairs by the area index; scenes not placed on the imagery
    (:func:`placed_on_aerial`) or not in the index have none."""
    to_area = area_index.get("scene_to_area", {})
    return {to_area[sid] for split, sid in scenes if sid in to_area and placed_on_aerial(split)}


def select_files(
    files: Sequence[ArchiveFile],
    *,
    aerial_features: bool,
    aerial_tiles: bool = True,
    splits: Optional[Sequence[str]] = None,
    scene_ids: Optional[Sequence[str]] = None,
    components: Optional[Sequence[str]] = None,
    max_gb: Optional[float] = None,
    cities: Optional[Sequence[str]] = None,
    lidar: Optional[str] = None,
    area_index: Optional[dict] = None,
    where: Optional[dict] = None,
) -> list[ArchiveFile]:
    """The files of the selected scenes: the chosen cameras/lidars, everything else always, and the aerial
    zips of their areas.

    *cities*, *lidar* and *where* select scenes by the shipped scene table
    (:func:`scenes_where`). The aerial zips (*aerial_tiles*: the DOP20
    tiles, *aerial_features*: the DINOv3 features) come by area, with their
    metadata; *area_index* (the release's ``aerial/metadata/area_index.json``)
    is needed when scenes are filtered. Scenes of the test-e2e splits pull
    no area (:func:`placed_on_aerial`).

    *max_gb* keeps whole scenes in order; an area's aerial zips count with
    its first kept scene, tiles before features, and are left out if they do
    not fit.

    Raises:
        ValueError: For unknown names, an empty selection, or aerial data of
            some scenes without *area_index*.
        TypeError: For an unknown scene-table column.
    """
    wanted = resolve_components(components)
    criteria = where
    where = scenes_where(cities, lidar, criteria)
    city_maps = [f for f in files if f.component in (CITY_MAP, SD_MAP, SD_MAP_METADATA)]
    aerial = [f for f in files if f.component in _AERIAL_ROWS]
    files = [f for f in files if f.component not in _LOOSE]
    if splits is not None:
        keys = {_split_name(s) for s in splits}
        available = {f.split for f in files}
        missing = [s for s in splits if _split_name(s) not in available]
        if missing:
            raise ValueError(f"Unknown split(s) {missing}. Available splits: {sorted(available)}")
    if scene_ids is not None:
        known = {f.scene_id for f in files}
        unknown = [s for s in scene_ids if s not in known]
        if unknown:
            raise ValueError(f"Scene ID(s) not found in manifest: {unknown}")
    selected = [
        f for f in files
        if (splits is None or f.split in keys)
        and (scene_ids is None or f.scene_id in scene_ids)
        and (where is None or f.scene_id in where)
        and (f.component not in OPTIONAL_COMPONENTS or wanted is None or f.component in wanted)
    ]
    if where is not None and not selected:
        what = " and ".join(([f"in {', '.join(cities)}"] if cities else []) + ([f"with lidar {lidar}"] if lidar else [])
                            + [f"with {k}={v!r}" for k, v in (criteria or {}).items() if v is not None])
        raise ValueError(f"No selected scene is {what} (by the scene table).")
    kinds = aerial_kinds(aerial_tiles, aerial_features)
    zips: dict[tuple[str, str], list[ArchiveFile]] = {}  # (area, kind) -> its zips (one per feature set)
    for f in aerial:
        if f.component in kinds:
            zips.setdefault((posixpath.basename(f.path)[:-len(".zip")], f.component), []).append(f)
    if zips and area_index is None and scene_filtered(splits, scene_ids, cities, lidar, max_gb, criteria):
        raise ValueError(f"Selecting the aerial data of some scenes needs the area index ({AREA_INDEX_PATH})")
    metadata = {AERIAL_TILES: [f for f in aerial if f.component == AERIAL_METADATA],
                AERIAL_FEATURES: [f for f in aerial if f.component in (AERIAL_METADATA, AERIAL_FEATURES_METADATA)]}
    aerial_files: list[ArchiveFile] = []
    taken: set[str] = set()

    def take(group: list[ArchiveFile]) -> None:
        for f in group:
            if f.path not in taken:
                taken.add(f.path)
                aerial_files.append(f)

    if max_gb is not None:
        by_scene: dict[tuple[str, str], list[ArchiveFile]] = {}
        for f in selected:
            by_scene.setdefault((f.split, f.scene_id), []).append(f)
        kept, total, budget, seen = [], 0, max_gb * 1000 ** 3, set()
        for key, scene_files in by_scene.items():
            size = sum(f.size for f in scene_files)
            if total + size > budget:
                break
            kept.extend(scene_files)
            total += size
            for area in scene_areas([key], area_index or {}) - seen:
                seen.add(area)
                for kind in kinds:  # tiles before features; each only if it still fits
                    group = [f for f in zips.get((area, kind), []) + metadata[kind] if f.path not in taken]
                    cost = sum(f.size for f in group)
                    if zips.get((area, kind)) and total + cost <= budget:
                        take(group)
                        total += cost
        selected = kept
    elif zips:
        areas = None  # every area
        if scene_filtered(splits, scene_ids, cities, lidar, max_gb, criteria):
            areas = scene_areas({(f.split, f.scene_id) for f in selected}, area_index)
        for (area, kind), group in zips.items():
            if areas is None or area in areas:
                take(group + metadata[kind])
    order = (AERIAL_METADATA, AERIAL_FEATURES_METADATA, AERIAL_TILES, AERIAL_FEATURES)  # the small files first
    aerial_files.sort(key=lambda f: (order.index(f.component), f.path))
    return city_maps + aerial_files + selected


def aerial_left_out(files: Sequence[ArchiveFile], selected: Sequence[ArchiveFile], area_index: Optional[dict],
                    aerial_tiles: bool = True, aerial_features: bool = True) -> list[ArchiveFile]:
    """The aerial zips (of the kinds asked for) of the selected scenes' areas that *selected* leaves out: by
    ``max_gb``, or the features of a filtered download (``aerial_tiles=False, aerial_features=True``)."""
    if area_index is None:
        return []
    kinds = aerial_kinds(aerial_tiles, aerial_features)
    areas = scene_areas({(f.split, f.scene_id) for f in selected if f.scene_id}, area_index)
    have = {f.path for f in selected}
    return [f for f in files if f.component in kinds and f.path not in have
            and posixpath.basename(f.path)[:-len(".zip")] in areas]


class HashCache:
    """SHA-256 of files in a root, remembered by (size, mtime) in ``.kitscenes/hashes.json``.

    A file whose size and modification time did not change is not hashed
    again; any change (a same-size corruption by a later write included)
    makes it hashed anew.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / STATE_DIR / "hashes.json"
        self._lock = threading.Lock()
        try:
            self._entries: dict[str, list] = json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError):
            self._entries = {}

    def sha256(self, root: Path, rel: str, *, recompute: bool = False) -> str:
        stat = (root / rel).stat()
        key = [stat.st_size, stat.st_mtime_ns]
        with self._lock:
            entry = self._entries.get(rel)
        if entry and entry[:2] == key and not recompute:
            return entry[2]
        digest = sha256_file(root / rel)
        self.remember(root, rel, digest)
        return digest

    def remember(self, root: Path, rel: str, digest: str) -> None:
        stat = (root / rel).stat()
        with self._lock:
            self._entries[rel] = [stat.st_size, stat.st_mtime_ns, digest]

    def save(self) -> None:
        with self._lock:
            data = json.dumps(self._entries, sort_keys=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.partial")
        tmp.write_text(data)
        os.replace(tmp, self.path)


def _marker(root: Path, f: ArchiveFile) -> Path:
    scene_dir = (root / f.path).parent  # the scene, or raw_tiles/<id>/ (features/<set>/) of an aerial zip
    return scene_dir / EXTRACTED_DIR / (posixpath.basename(f.path) + ".sha256")


def _extractable(f: ArchiveFile) -> bool:
    return f.path.endswith(".zip") and (f.component in COMPONENTS or f.component in _AREA_ZIPS)


def _folders(f: ArchiveFile) -> Sequence[str]:
    """The folders an extracted archive fills next to it: the component's, or the area of an aerial zip."""
    if f.component in _AREA_ZIPS:
        return [posixpath.basename(f.path)[:-len(".zip")]]
    return COMPONENTS[f.component]


def read_marker(marker: Path) -> Optional[tuple[str, list[str]]]:
    """The SHA-256 of an extracted archive and the folders it filled; None without a (readable) marker."""
    try:
        digest, *folders = marker.read_text().split()
    except (OSError, ValueError):  # missing, unreadable or empty
        return None
    if len(digest) != 64:
        return None
    return digest, folders


def is_complete(root: Path, f: ArchiveFile, *, verify: bool = False, hashes: Optional[HashCache] = None,
                recompute: bool = True, recipes: Optional[dict] = None) -> bool:
    """Whether the file is in place.

    By default its size must match the manifest (fast: no reading); with
    *verify* its SHA-256 too (*hashes* remembers them by size and mtime;
    *recompute* False takes a remembered hash, e.g. one made earlier in the
    same run). An archive that was extracted counts as present when its
    marker holds the manifest's SHA-256 (the hash of the zip it was
    extracted from) and the folders it filled are there.

    A camera with a recipe in *recipes* (``{camera: ImageRecipe}``, the
    root's ``--images``) is complete when its scene holds it transcoded by
    that recipe from this very release zip (:mod:`kitscenes.transcode`);
    without one, a transcoded camera is not complete (the release's images
    are wanted).
    """
    if f.component in CAMERA_NAMES and f.scene_id:
        from kitscenes.transcode import variant_complete

        recipe = (recipes or {}).get(f.component)
        state = variant_complete(root, f, recipe, verify=verify)
        if state is not None:
            return state
        if recipe is not None:
            return False  # the release's images (or none): not yet in the recipe
    return release_copy_present(root, f, verify=verify, hashes=hashes, recompute=recompute)


def release_copy_present(root: Path, f: ArchiveFile, *, verify: bool = False, hashes: Optional[HashCache] = None,
                         recompute: bool = True) -> bool:
    """Whether the file is in place as released (as a file, or extracted): :func:`is_complete` without recipes."""
    path = root / f.path
    if path.is_file():
        if path.stat().st_size != f.size:
            return False
        if not verify:
            return True
        digest = hashes.sha256(root, f.path, recompute=recompute) if hashes is not None else sha256_file(path)
        return digest == f.sha256
    if _extractable(f):
        marker = read_marker(_marker(root, f))
        if marker is not None:
            digest, folders = marker
            return digest == f.sha256 and all((path.parent / folder).is_dir() for folder in folders)
    return False


def remove_stale(root: Path, files: Iterable[ArchiveFile]) -> list[ArchiveFile]:
    """Remove what *root* holds of *files* (files whose content changed): the file, or its extraction.

    An extraction (marker plus the component's folders) is removed too:
    the API would read it as data of the new revision. Called before the
    new manifest is put in place, so an interrupted update leaves the old
    manifest and missing files, which the next run downloads again.

    Returns:
        The files that had been extracted (to extract again once downloaded).
    """
    extracted = []
    for f in files:
        path = root / f.path
        if f.component in CAMERA_NAMES and f.scene_id:
            from kitscenes.transcode import drop_variant

            drop_variant(path.parent, f.component)  # its images came from the old zip
        if path.is_file():
            path.unlink()
        if _extractable(f):
            marker = _marker(root, f)
            if marker.is_file():
                if read_marker(marker) is not None:
                    extracted.append(f)
                for folder in _folders(f):
                    shutil.rmtree(path.parent / folder, ignore_errors=True)
                marker.unlink()
    return extracted


def fetch(
    root: Path,
    files: Sequence[ArchiveFile],
    download_one,
    *,
    jobs: int = 8,
    verify: bool = False,
    extract: bool = False,
    hashes: Optional[HashCache] = None,
    transcoding=None,
) -> list[tuple[ArchiveFile, Exception]]:
    """Download *files* into *root* (skipping complete ones), *jobs* at a time; returns the failures.

    ``download_one(f, root) -> Path`` puts one file at ``root / f.path``
    (HuggingFace writes a temporary file and renames it when complete, so
    an interrupted run leaves no half-written file in place). Each file is
    then checked against the manifest: its size (fast; the default), or its
    SHA-256 with *verify*. A file that fails the check is removed and
    reported.
    *hashes*: the hashes of a check made just before in this run (they are
    not computed again); without it *verify* hashes every present file.
    With *extract*, each zip is hashed (or its remembered hash taken) and
    extracted only if it matches the manifest.

    *transcoding* (:class:`kitscenes.transcode.DownloadTranscoding`) re-encodes
    the cameras it has a recipe for: their zips are downloaded into its
    staging buffer (a download waits while the buffer is full), transcoded
    into the scene and deleted; a release zip already in the root is
    transcoded in place. What a scene holds of a camera in another form (the
    release's images when a recipe is wanted, a transcoded variant when it
    is not, or one made from an older release zip) is removed first.
    """
    recompute = hashes is None
    if hashes is None and verify:
        hashes = HashCache(root)
    recipes = transcoding.recipes if transcoding is not None else None
    todo = [f for f in files if not is_complete(root, f, verify=verify, hashes=hashes, recompute=recompute,
                                                recipes=recipes)]
    in_place: list[ArchiveFile] = []
    if transcoding is not None or any(f.component in CAMERA_NAMES for f in todo):
        from kitscenes.transcode import read_variant

        downloads = []
        for f in todo:
            if f.component not in CAMERA_NAMES or not f.scene_id:
                downloads.append(f)
                continue
            recipe = (recipes or {}).get(f.component)
            variant = read_variant((root / f.path).parent, f.component)
            if recipe is not None and variant is None and release_copy_present(
                    root, f, verify=verify, hashes=hashes, recompute=recompute):
                in_place.append(f)  # downloaded before the recipe was set
                continue
            if (recipe is not None and variant is not None and variant.source_sha256 == f.sha256
                    and recipe.reduce > variant.recipe.reduce and recipe.reduce % variant.recipe.reduce == 0):
                in_place.append(f)  # smaller than stored: from the stored images
                continue
            if variant is not None:
                drop_camera(root, f)  # another variant (or one of an older release zip)
            downloads.append(f)
        todo = downloads
    failed: list[tuple[ArchiveFile, Exception]] = []
    lock = threading.Lock()
    transcodes: list = []
    if transcoding is not None:
        transcoding.clean_staging()
        if in_place or any(transcoding.recipe(f) is not None for f in todo):
            transcoding.start()  # now: a transcoder that cannot start fails the run before anything is staged
        for f in in_place:
            transcodes.append((f, transcoding.submit(f)))

    def one(f: ArchiveFile) -> None:
        staged = transcoding is not None and transcoding.recipe(f) is not None
        base = transcoding.staging if staged else root
        if staged:
            transcoding.budget.acquire(f.size)
        target = base / f.path
        try:
            path = Path(download_one(f, base))
            if path != target:  # a fetcher that downloads elsewhere
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, target)
            size = target.stat().st_size
            if size != f.size:
                target.unlink(missing_ok=True)
                raise ValueError(f"Incomplete download of {f.path}: {size} of {f.size} bytes")
            if verify:
                actual = sha256_file(target)
                if actual != f.sha256:
                    target.unlink(missing_ok=True)
                    raise ValueError(f"Checksum mismatch for {f.path}: expected {f.sha256}, got {actual}")
                if not staged:
                    hashes.remember(root, f.path, actual)
        except BaseException:
            if staged:
                target.unlink(missing_ok=True)
                transcoding.budget.release(f.size)
            raise
        if staged:
            try:
                future = transcoding.submit(f, staged=target)  # releases the budget when done
            except BaseException:
                target.unlink(missing_ok=True)
                transcoding.budget.release(f.size)
                raise
            with lock:
                transcodes.append((f, future))
        elif f.path.endswith(".zip"):
            _marker(root, f).unlink(missing_ok=True)  # a new archive: its old extraction is stale

    try:
        pool = ThreadPoolExecutor(max_workers=max(1, jobs))
        try:
            futures = {pool.submit(one, f): f for f in todo}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as err:  # collected and reported
                    failed.append((futures[future], err))
        except BaseException:  # Ctrl-C: finish the running downloads, start no more
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        for f, future in transcodes:
            try:
                future.result()
            except Exception as err:  # collected and reported; the scene keeps what it had
                failed.append((f, err))
        if extract:
            bad = {(f.split, f.scene_id) for f, _ in failed}
            for f in files:
                if _extractable(f) and (f.split, f.scene_id) not in bad and (root / f.path).is_file():
                    try:
                        extract_archive(root, f, hashes=hashes)
                    except Exception as err:
                        failed.append((f, err))
    finally:
        if hashes is not None:
            hashes.save()
    return failed


def extract_archive(root: Path, f: ArchiveFile, *, hashes: Optional[HashCache] = None) -> None:
    """Extract the downloaded archive *f* (a scene component, an area's aerial tiles or features).

    See :func:`extract_component`.
    """
    if f.component in _AREA_ZIPS:
        extract_aerial_tiles(root / f.path, sha256=f.sha256, hashes=hashes)
    else:
        extract_component(root / f.path, sha256=expected_zip_sha256(root, f), hashes=hashes)


def expected_zip_sha256(root: Path, f: ArchiveFile) -> str:
    """The SHA-256 the scene's zip of *f* should have: the manifest's, or for a transcoded camera its record's."""
    if f.component in CAMERA_NAMES and f.scene_id:
        from kitscenes.transcode import read_variant

        variant = read_variant((root / f.path).parent, f.component)
        if variant is not None and variant.sha256:
            return variant.sha256
    return f.sha256


def drop_camera(root: Path, f: ArchiveFile) -> None:
    """Remove what a scene holds of a camera (zip, extracted folder, marker, transcode record) before fetching it anew."""
    from kitscenes.transcode import drop_variant

    path = root / f.path
    drop_variant(path.parent, f.component)
    path.unlink(missing_ok=True)
    marker = _marker(root, f)
    marker.unlink(missing_ok=True)
    for folder in _folders(f):
        shutil.rmtree(path.parent / folder, ignore_errors=True)


def _zip_digest(zip_path: Path, hashes: Optional[HashCache]) -> str:
    if hashes is not None and zip_path.is_relative_to(hashes.root):
        return hashes.sha256(hashes.root, zip_path.relative_to(hashes.root).as_posix(), recompute=False)
    return sha256_file(zip_path)


def _write_marker(marker: Path, digest: str, folders: Sequence[str]) -> None:
    marker.parent.mkdir(exist_ok=True)
    tmp = marker.with_name(marker.name + ".partial")
    tmp.write_text("\n".join([digest, *folders]) + "\n")  # the archive, and the folders it filled
    os.replace(tmp, marker)  # atomic: a crash leaves the old marker or none, never a half-written one


def extract_aerial_tiles(zip_path: Path, *, sha256: Optional[str] = None, hashes: Optional[HashCache] = None) -> None:
    """Extract an area zip (``raw_tiles/<id>/<area>.zip``, ``features/<set>/<area>.zip``) next to it and delete it.

    Like :func:`extract_component`: checked against *sha256* first, extracted
    to a staging folder, moved in place (replacing an older extraction), a
    marker ``raw_tiles/<id>/.extracted/<area>.zip.sha256`` written, and only
    then the zip deleted. The API reads the zip as well (in place), so this
    is optional.

    Raises:
        ValueError: For a zip that does not match *sha256*, or a member
            outside ``<area>/`` (the zip is kept in both cases).
    """
    tiles_dir = zip_path.parent
    area = zip_path.stem
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            norm = posixpath.normpath(name)
            if posixpath.isabs(name) or "\\" in name or not (norm == area or norm.startswith(area + "/")):
                raise ValueError(f"Refusing to extract {name!r} from {zip_path}")
        digest = _zip_digest(zip_path, hashes)
        if sha256 is not None and digest != sha256:
            raise ValueError(f"{zip_path} does not match the manifest (SHA-256 {digest[:12]}..., expected "
                             f"{sha256[:12]}...); not extracted. Download it again with --verify.")
        staging = tiles_dir / f".partial-{area}"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            zf.extractall(staging)
            dst = tiles_dir / area
            if dst.exists():
                shutil.rmtree(dst)
            os.replace(staging / area, dst)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    _write_marker(tiles_dir / EXTRACTED_DIR / (zip_path.name + ".sha256"), digest, [area])
    zip_path.unlink()


def extract_component(zip_path: Path, *, sha256: Optional[str] = None, hashes: Optional[HashCache] = None) -> None:
    """Extract a component zip of a scene folder into the scene and delete it.

    The zip is hashed first (*hashes*: a :class:`HashCache` of the root that
    may remember it) and must match *sha256*, the manifest's, when given.
    Extraction goes to ``<split>/.partial-<scene>-<component>/`` first; the
    component's folders are then moved into the scene (replacing an older
    extraction), a marker ``.extracted/<component>.zip.sha256`` records the
    archive's SHA-256 and the folders, and only then is the zip deleted. An
    interrupted extraction leaves the zip and no half-filled folder.

    Raises:
        ValueError: For a zip that does not match *sha256*, or a member
            outside ``<scene_id>/`` or the component's folders (the zip is
            kept in both cases).
    """
    scene_dir = zip_path.parent
    split_dir = scene_dir.parent
    component = zip_path.stem
    folders = COMPONENTS.get(component)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            norm = posixpath.normpath(name)
            if posixpath.isabs(name) or "\\" in name or not (norm == scene_dir.name or norm.startswith(scene_dir.name + "/")):
                raise ValueError(f"Refusing to extract {name!r} from {zip_path}")
        digest = _zip_digest(zip_path, hashes)
        if sha256 is not None and digest != sha256:
            raise ValueError(f"{zip_path} does not match the manifest (SHA-256 {digest[:12]}..., expected "
                             f"{sha256[:12]}...); not extracted. Download it again: kitscenes download "
                             f"{split_dir.parent.parent} --scenes {scene_dir.name} --verify")
        staging = split_dir / f".partial-{scene_dir.name}-{component}"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            zf.extractall(staging)
            extracted = staging / scene_dir.name
            moved = folders if folders is not None else [p.name for p in extracted.iterdir()]
            leftover = [p for p in extracted.rglob("*") if p.is_file()
                        and not any(p.relative_to(extracted).as_posix().startswith(f + "/") for f in moved)]
            if leftover:  # checked before anything in the scene is touched
                raise ValueError(f"{zip_path}: members outside the component's folders, e.g. {leftover[0]}")
            placed = []
            for folder in moved:
                src = extracted / folder
                if not src.exists():
                    continue  # e.g. a radar this vehicle does not have
                dst = scene_dir / folder
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    shutil.rmtree(dst)
                os.replace(src, dst)
                placed.append(folder)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    _write_marker(scene_dir / EXTRACTED_DIR / (zip_path.name + ".sha256"), digest, placed)
    zip_path.unlink()


class RootBusy(OSError):
    """Another download or extraction is writing to the root."""


@contextlib.contextmanager
def root_lock(root: Path):
    """Hold the root's lock (``.kitscenes/lock``) while writing to it.

    Raises:
        RootBusy: If another process holds it.
    """
    try:
        import fcntl
    except ImportError:  # Windows: no lock
        yield
        return
    path = root / STATE_DIR / "lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RootBusy(f"another kitscenes download or extract is writing to {root}; wait for it to finish") from None
        except OSError as err:  # e.g. a file system without locks
            logger.warning("Cannot lock %s (%s); make sure no other download runs on this root.", path, err)
        yield


# ---------------------------------------------------------------------------
# What a root holds: dataset_info.json
# ---------------------------------------------------------------------------


def read_dataset_info(root: Path) -> Optional[dict]:
    """The root's ``dataset_info.json``, or None."""
    try:
        return json.loads((root / DATASET_INFO).read_text())
    except FileNotFoundError:
        return None


class UnsupportedRelease(ValueError):
    """A dataset release this kitscenes does not read."""


def check_manifest(rows: Sequence[dict]) -> None:
    """Refuse a manifest this kitscenes does not read.

    Raises:
        UnsupportedRelease: For a dataset 1.x manifest (one tar per scene, no ``component`` column).
    """
    if rows and "component" not in rows[0]:
        raise UnsupportedRelease("this is a dataset 1.x release (one tar per scene); this kitscenes reads "
                                 "release 2.0 and later. Read 1.x with the legacy kitscenes 0.1.0.")


def write_dataset_info(root: Path, *, repo_id: str, revision: Optional[str], manifest: Path,
                       rows: Sequence[dict]) -> dict:
    """Record the version a root holds: repo, revision, the manifest's SHA-256, its format, what the release ships."""
    from datetime import datetime, timezone

    from kitscenes._archive import FORMAT_VERSION

    check_manifest(rows)
    files = parse_manifest(rows)
    components = sorted({f.component for f in files} - {METADATA})
    info = {
        "repo_id": repo_id,
        "revision": revision,
        "manifest_sha256": sha256_file(manifest),
        "archive_format": FORMAT_VERSION,
        "scenes": len({(f.split, f.scene_id) for f in files if f.scene_id}),
        "components_in_release": components,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    previous = read_dataset_info(root) or {}
    if previous.get("images"):  # the root's recipes per camera (kitscenes.transcode)
        info["images"] = previous["images"]
    tmp = root / (DATASET_INFO + ".partial")
    tmp.write_text(json.dumps(info, indent=1) + "\n")
    os.replace(tmp, root / DATASET_INFO)
    return info
