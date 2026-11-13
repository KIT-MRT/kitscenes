"""Finding scenes: :func:`open` and :class:`Dataset`.

Example::

    import kitscenes

    ds = kitscenes.open("/data/kitscenes", splits=["train", "map_change_train"])
    scene = ds[0]
    frame = scene.frame(10)
    cloud = frame.lidar("lidar_top")
"""

from __future__ import annotations

import copy
import logging
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Iterator, Optional, Sequence, Union, overload

from kitscenes._cache import PayloadCache
from kitscenes.constants import HD_MAPS_DIR, KITSCENES_ROOT_ENV_VAR, SD_MAPS_DIR
from kitscenes.errors import Availability, DataFormatError, DatasetNotFound, SceneNotFound, SensorUnavailable
from kitscenes.frame import Frame
from kitscenes._layout import DATA_SUBDIR, REFERENCE_TIMESTAMPS
from kitscenes.scene import Scene, SceneTrack
from kitscenes.splits import SplitDefinition, _known_value, canonical_split_name, known_split, split_sort_key
from kitscenes._storage import open_storage

if TYPE_CHECKING:
    from kitscenes.aerial import AerialCache, AerialCrops, AerialFeatureCrops, AerialFeatures, AerialTiles
    from kitscenes.sdmap import SDMap

logger = logging.getLogger(__name__)

#: Number of Scene objects (with their cached metadata) a Dataset keeps by default (``open(scene_cache=)``).
SCENE_CACHE_SIZE = 32


def _is_scene_dir(path: Path) -> bool:
    return (path / REFERENCE_TIMESTAMPS).is_file()


def _subdirs(path: Path) -> list[Path]:
    return sorted(d for d in path.iterdir() if d.is_dir() and not d.name.startswith("."))


def discover_scenes(root: Path) -> list[tuple[str, Path]]:
    """Find the scene folders ``root/data/<split>/<scene_id>/`` as ``(split, scene_path)``, ordered by split and ID.

    A scene folder is one with ``timestamp.reference.txt``; a folder in
    ``data/`` that contains scene folders is a split folder.

    Raises:
        ValueError: If *root* is itself a scene folder.
        DatasetNotFound: If the scenes are not in ``data/<split>/``.
    """
    if _is_scene_dir(root):
        raise ValueError(
            f"{root} is a scene folder. Pass the dataset root (the folder that holds "
            f"{DATA_SUBDIR}/<split>/), or open the scene with kitscenes.Scene({str(root)!r})."
        )
    layout = f"Scenes are expected in <root>/{DATA_SUBDIR}/<split>/<scene_id>/"
    parent = root / DATA_SUBDIR
    if not parent.is_dir():
        raise DatasetNotFound(f"{root} has no {DATA_SUBDIR}/ folder. {layout}: pass the folder that holds {DATA_SUBDIR}/.")

    scenes: list[tuple[str, Path]] = []
    for child in _subdirs(parent):
        if _is_scene_dir(child):
            raise DatasetNotFound(f"{child} is a scene folder outside a split folder. {layout}.")
        members = [d for d in _subdirs(child) if _is_scene_dir(d)]
        if not members:
            continue
        if known_split(child.name) is None:
            logger.warning("Unknown split folder %s; its %d scenes are listed under that name.", child, len(members))
        scenes.extend((child.name, d) for d in members)

    scenes.sort(key=lambda item: (split_sort_key(item[0]), item[1].name))
    return scenes


class Dataset:
    """The scenes found under a dataset root.

    Use :func:`kitscenes.open` to create one. Indexing gives
    :class:`~kitscenes.scene.Scene` objects (by position or ID), slicing and
    :meth:`filter` give sub-datasets, and :meth:`frames` a flat frame list for
    training. A Dataset pickles to its scene list.

    Args:
        root: Dataset root; ``$KITSCENES_ROOT`` when omitted.
        splits: One split or a list (folder names or aliases such as
            ``"test_e2e"``, see :mod:`kitscenes.splits`; with a split
            definition, its split names); None for all.
        cache_mb: Budget of an optional payload cache for decoded images and
            point clouds (0 = no cache, the default). Cached payloads are
            read-only. Useful in notebooks that read the same frame often;
            don't use it in training loops.
        scene_cache: How many opened scenes (with their timeline, poses,
            calibration, archive index and map) the dataset keeps, per
            process; a scene that falls out is opened again when asked for
            (a few ms, plus its map). Raise it for loaders that visit many
            scenes in turn, e.g. a shuffled ``DataLoader`` over map labels.
        verify: Check the CRC-32 of every member read from a release zip
            (costs ~25 % of one core per read; the zips are SHA-256-checked
            on download, so this is for distrust of the storage, e.g. a long
            training run on network storage). Extracted files have no
            checksum to compare and are read as they are.
        sd_snapshot: The OpenStreetMap snapshot ``scene.sd_map`` (and what
            uses it: ``sd_route``, navigation, ``SDMapSpec`` samples) loads:
            ``"scene"`` (default: the scene's city and recording week),
            ``"latest"``, ``"paper_2026-05-03"`` (the 2026-05-03 files of the
            paper, e.g. to reproduce a training made with them) or a week
            (``"2026-W06"``). See :meth:`sd_map`.
        use_imu: Refine the ego motion with the gyro where a scene allows it
            (:attr:`kitscenes.Scene.ego_motion`). False: the poses alone.
        split_definition: Which split each scene is in
            (:class:`~kitscenes.splits.SplitDefinition`, its name or a CSV
            path). None (default): the release folders, which is the default
            definition :data:`~kitscenes.splits.DEFAULT_SPLIT`. With one,
            :attr:`splits`, ``splits=``, :meth:`filter`, :meth:`split_of` and
            ``scene.split`` follow it; scenes on disk it does not list are left
            out. The files stay where the release puts them,
            and each scene's data keeps its release folder's conventions
            (``scene.release_split``, ``scene.split_info``).

    Raises:
        DatasetNotFound: If the root does not exist, has no (matching) scenes, or its
            scenes are not in ``data/<split>/<scene_id>/``.
        ValueError: For an unknown split name, a split definition that breaks
            the rules, or a scene folder as root.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        splits: Union[str, Sequence[str], None] = None,
        *,
        cache_mb: int = 0,
        scene_cache: int = SCENE_CACHE_SIZE,
        verify: bool = False,
        sd_snapshot: str = "scene",
        use_imu: bool = True,
        split_definition: Union[SplitDefinition, str, Path, None] = None,
    ) -> None:
        if root is None:
            root = os.environ.get(KITSCENES_ROOT_ENV_VAR)
            if root is None:
                raise ValueError(
                    f"No root given and ${KITSCENES_ROOT_ENV_VAR} is not set. "
                    "Pass the dataset root or set the environment variable."
                )
        root = Path(root)
        if not root.is_dir():
            raise DatasetNotFound(f"Dataset root not found: {root}")
        definition = None if split_definition is None else SplitDefinition.load(split_definition)
        wanted = _split_names(splits, definition)
        found = discover_scenes(root)
        if definition is not None:
            listed = [(s, p) for s, p in found if p.name in definition]
            if len(listed) < len(found):
                logger.info("%d scene(s) under %s are not in split definition %s; left out.",
                            len(found) - len(listed), root, definition.name)
            order = {sid: k for k, sid in enumerate(definition.assignments)}
            found = sorted(listed, key=lambda item: order[item[1].name])
        split_of = (lambda folder, sid: folder) if definition is None else (lambda folder, sid: definition.split_of(sid))
        selected = [(s, p) for s, p in found if wanted is None or split_of(s, p.name) in wanted]
        if not selected:
            what = "scenes" if wanted is None else f"scenes of split(s) {sorted(wanted)}"
            available = sorted({split_of(s, p.name) for s, p in found},
                               key=split_sort_key if definition is None else definition.splits.index)
            raise DatasetNotFound(
                f"No {what} under {root}. A scene is a folder with {REFERENCE_TIMESTAMPS} "
                f"in {DATA_SUBDIR}/<split>/. Splits found: {available or 'none'}"
                + (f" (split definition {definition.name})." if definition is not None else ".")
            )
        entries: dict[str, tuple[str, Path]] = {}
        for split, path in selected:
            if path.name in entries:
                logger.warning("Scene %s is in several splits; using %s, ignoring %s.", path.name, entries[path.name][1], path)
                continue
            entries[path.name] = (split, path)
        if not (isinstance(scene_cache, int) and scene_cache >= 1):
            raise ValueError(f"scene_cache must be an int >= 1, got {scene_cache!r}")
        if not isinstance(sd_snapshot, str) or not sd_snapshot:
            raise ValueError(f"sd_snapshot must be a snapshot name such as 'scene', got {sd_snapshot!r}")
        self.root = root
        #: The OpenStreetMap snapshot ``scene.sd_map`` loads (see :class:`Dataset`).
        self.sd_snapshot = sd_snapshot
        #: Whether the scenes' ego motion may use the gyro (see :class:`Dataset`).
        self.use_imu = bool(use_imu)
        #: How many opened scenes the dataset keeps (see :class:`Dataset`).
        self.scene_cache = scene_cache
        #: Whether members read from release zips are CRC-32 checked (see :class:`Dataset`).
        self.verify = bool(verify)
        #: The split definition the dataset follows (see :class:`Dataset`), or None: the release folders.
        self.split_definition = definition
        self.cache = PayloadCache(cache_mb * 1024 ** 2) if cache_mb > 0 else None
        #: The aerial crops frames read (an AerialCrops or AerialCache: :meth:`use_aerial`), or None.
        self.aerial: Union["AerialCache", "AerialCrops", None] = None
        #: The aerial feature crops frames read (:meth:`use_aerial_features`), or None.
        self.aerial_features: Optional["AerialFeatureCrops"] = None
        self._entries = entries
        self._ids = list(entries)
        self._scenes: OrderedDict[str, Scene] = OrderedDict()
        self._lock = Lock()

    def __getstate__(self) -> dict:
        """Everything but the opened scenes and the lock (also what a sub-dataset starts from)."""
        return {k: v for k, v in self.__dict__.items() if k not in ("_scenes", "_lock")}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._scenes = OrderedDict()
        self._lock = Lock()

    def __repr__(self) -> str:
        definition = f", split_definition={self.split_definition.name!r}" if self.split_definition else ""
        return f"Dataset(root={str(self.root)!r}, splits={self.splits}, scenes={len(self)}{definition})"

    # ------------------------------------------------------------------

    @property
    def scene_ids(self) -> list[str]:
        return list(self._ids)

    @property
    def splits(self) -> list[str]:
        """Splits with scenes in this dataset: the split folders in table order, or the split definition's
        in its order."""
        if self.split_definition is not None:
            names = {self.split_definition.split_of(sid) for sid in self._ids}
            return [s for s in self.split_definition.splits if s in names]
        return sorted({split for split, _ in self._entries.values()}, key=split_sort_key)

    def split_of(self, scene_id: str) -> str:
        """Split of a scene: its folder, or the split definition's."""
        folder = self._entry(scene_id)[0]
        return folder if self.split_definition is None else self.split_definition.split_of(scene_id)

    def _entry(self, scene_id: str) -> tuple[str, Path]:
        try:
            return self._entries[scene_id]
        except KeyError:
            raise SceneNotFound(f"No scene {scene_id!r} in {self.root} (splits {self.splits}).") from None

    def scene(self, scene_id: str) -> Scene:
        """The scene with this ID; the dataset keeps the last :attr:`scene_cache` scenes."""
        with self._lock:
            scene = self._scenes.get(scene_id)
            if scene is not None:
                self._scenes.move_to_end(scene_id)
                return scene
        folder, path = self._entry(scene_id)
        scene = Scene(open_storage(path, verify=self.verify), scene_id, self.split_of(scene_id), release_split=folder,
                      cache=self.cache, aerial=self.aerial, sd_snapshot=self.sd_snapshot,
                      aerial_features=self.aerial_features, use_imu=self.use_imu)
        with self._lock:
            self._scenes[scene_id] = scene
            while len(self._scenes) > self.scene_cache:
                self._scenes.popitem(last=False)
        return scene

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[Scene]:
        for scene_id in self._ids:
            yield self.scene(scene_id)

    def __contains__(self, scene_id: object) -> bool:
        return scene_id in self._entries

    @overload
    def __getitem__(self, key: Union[int, str]) -> Scene: ...
    @overload
    def __getitem__(self, key: slice) -> "Dataset": ...

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._subset(self._ids[key])
        if isinstance(key, str):
            return self.scene(key)
        try:
            scene_id = self._ids[key]
        except IndexError:
            raise IndexError(f"Scene index {key} is out of range: the dataset has {len(self)} scenes "
                             f"(splits {self.splits}).") from None
        return self.scene(scene_id)

    def _subset(self, ids: Sequence[str]) -> "Dataset":
        """The dataset with only the scenes *ids*: the same options, cache and aerial crops."""
        subset = copy.copy(self)  # through __getstate__: its own opened scenes and lock
        subset._entries = {i: self._entries[i] for i in ids}
        subset._ids = list(subset._entries)
        return subset

    def filter(
        self,
        *,
        splits: Union[str, Sequence[str], None] = None,
        scene_ids: Optional[Sequence[str]] = None,
        city: Union[str, Sequence[str], None] = None,
        lidar: Union[str, Sequence[str], None] = None,
        **where: object,
    ) -> "Dataset":
        """The scenes matching all given criteria (an empty result is allowed here).

        Besides the split folders and IDs, any column of the scene table
        (:class:`kitscenes.splits.SceneInfo`) selects scenes, as in
        :func:`kitscenes.find_scenes`: a value or a list of values for the
        categories (``weather="rain"``, ``time_of_day=["evening", "night"]``),
        True / False for ``dark`` and ``sun_below_horizon``, a value, a list
        or a range ``(low, high)`` (None: open) for the numbers
        (``max_speed_mps=(None, 15)``, ``road_motorway_share=(0.5, None)``)::

            ds.filter(splits="train", city="Karlsruhe", weather=["rain", "overcast"], dark=False)

        A scene whose value is None (unknown, or withheld for its split:
        :func:`kitscenes.splits.withheld_columns`, e.g. the road types of
        test-e2e) does not match a criterion on that column; scenes the table
        does not list match none.

        Args:
            splits: Splits: folders (names or aliases), or the split definition's names.
            scene_ids: Scene IDs.
            city: ``"Karlsruhe"``, ``"Frankfurt"``, ``"Sindelfingen"`` (or a
                list; any case), from the scene table.
            lidar: ``"velodyne"`` or ``"hesai"`` (any case), from the scene table.
            **where: Other columns of the scene table.

        Raises:
            SceneNotFound: For an ID that is not in the dataset.
            TypeError: For an unknown column, or a value of the wrong type.
            ValueError: For a value the scene table does not know.
        """
        from kitscenes.splits import find_scenes

        criteria = {"city": city, "lidar": lidar, **where}
        if any(v is not None for v in criteria.values()):
            listed = {info.scene_id for info in find_scenes(**criteria)}
            ids = self._ids if scene_ids is None else list(scene_ids)
            return self.filter(splits=splits, scene_ids=[i for i in ids if i in listed])
        wanted = _split_names(splits, self.split_definition)
        if scene_ids is None:
            ids = self._ids
        else:
            ids = list(scene_ids)
            for scene_id in ids:
                self._entry(scene_id)  # raises SceneNotFound
        return self._subset([i for i in ids if wanted is None or self.split_of(i) in wanted])

    def city_maps(self) -> dict[str, Path]:
        """Lanelet2 HD city maps in the dataset's top-level ``lanelet2_hd_maps/`` folder, by city.

        Load one in a scene's frame with ``scene.load_map(path)``.
        """
        folder = self.root / HD_MAPS_DIR
        return {p.stem: p for p in sorted(folder.glob("*.osm"))} if folder.is_dir() else {}

    def sd_maps(self) -> dict[str, dict[str, Path]]:
        """OpenStreetMap SD maps in the top-level ``openstreetmap_sd_maps/`` folder: ``{city: {snapshot: path}}``.

        Snapshots per city, the weeks first (oldest to latest), then the named sets::

            {"Karlsruhe": {"2025-W33": .../Karlsruhe_2025-W33.osm.pbf, "2026-W06": ..., "2026-W12": ...,
                           "paper_2026-05-03": .../paper_2026-05-03/Karlsruhe.osm}, ...}

        - ``<City>_<YYYY>-W<ww>.osm.pbf``: one per city and ISO recording week,
          taken at the Monday 00:00 UTC of the week (snapshot name ``"2026-W06"``;
          PBF without contributor metadata);
        - ``paper_2026-05-03/<City>.osm``: the Overpass snapshots of
          2026-05-03 used by the paper (XML), named ``"paper_2026-05-03"``.

        © OpenStreetMap contributors, ODbL 1.0: see ``openstreetmap_sd_maps/ATTRIBUTION.md``; data derived from
        them falls under the ODbL too.
        """
        from kitscenes.sdmap._files import sd_map_files

        return sd_map_files(self.root / SD_MAPS_DIR)

    def sd_map(self, city: str, *, snapshot: Optional[str] = None) -> "SDMap":
        """An OpenStreetMap SD map of *city* (:class:`kitscenes.sdmap.SDMap`) in the city's map frame.

        That is the frame of ``poses.txt`` of every scene of the city (all
        share the city's ``maps/origin.json``, :data:`kitscenes.map.origin.CITY_ORIGINS`;
        the test splits too). One map per file and process, parsed on first
        use and cached on disk (see :mod:`kitscenes.sdmap`); a scene's
        ``scene.sd_map`` is the same object when it uses the same file.

        Args:
            city: ``"Karlsruhe"``, ``"Frankfurt"`` or ``"Sindelfingen"`` (any case).
            snapshot: Which snapshot (:meth:`sd_maps`): None (default) or
                ``"latest"`` for the city's latest week, a week ``"2026-W06"`` (also
                ``"Karlsruhe_2026-W06.osm.pbf"`` as the scene table writes it),
                or ``"paper_2026-05-03"`` (``"paper"``) for the snapshots of the
                paper. A scene's own snapshot is ``scene.info.sd_snapshot``.

        Raises:
            ValueError: For an unknown city, or a snapshot file of another city.
            MapUnavailable: ``NOT_DOWNLOADED`` without the file.
        """
        from kitscenes.errors import MapUnavailable
        from kitscenes.map.origin import CITY_ORIGINS
        from kitscenes.sdmap._files import pick_snapshot
        from kitscenes.sdmap.sd_map import shared_sd_map

        name = _known_value("city", city, set(CITY_ORIGINS))
        snapshots = self.sd_maps().get(name, {})
        hit = pick_snapshot(snapshots, snapshot, name)
        if hit is None:
            raise MapUnavailable("sd_map", None, Availability.NOT_DOWNLOADED,
                                 f"no SD map {name} {snapshot or 'latest'} in {self.root / SD_MAPS_DIR} "
                                 f"(snapshots of {name}: {', '.join(snapshots) or 'none'})",
                                 f"the SD maps come with every download: kitscenes download {self.root}")
        return shared_sd_map(hit[1], CITY_ORIGINS[name], name)

    # ------------------------------------------------------------------
    # Aerial imagery (kitscenes.aerial)
    # ------------------------------------------------------------------

    def prepare_aerials(
        self,
        roi: tuple[float, float],
        *,
        tiles: Union["AerialTiles", str, Path, Sequence[Union[str, Path]], None] = None,
        cache_dir: Union[str, Path, None] = None,
        version: str = "v1",
        backend: str = "auto",
        verbose: bool = False,
        workers: Optional[int] = None,
        tile_cache_mb: float = 256,
    ) -> "AerialCache":
        """Cut one ego-aligned aerial crop per frame into a crop cache on disk (PNG), then attach it.

        Optional: :meth:`use_aerial` with ``roi=`` cuts the same crops on the
        fly, with no cache. This is for tools that read image files. Covers
        the frames of this dataset that the raw tiles cover and cuts only
        what is missing, so a rerun over a complete cache writes nothing.
        Frames whose tiles are missing get no crop (a warning names them).
        See :mod:`kitscenes.aerial`.

        Args:
            roi: ``(longitudinal_m, lateral_m)`` of a crop, centred on the ego.
            tiles: The raw tiles, only read: an
                :class:`~kitscenes.aerial.AerialTiles`, or folders that hold
                them (default ``<root>/aerial``, the release tiles).
            cache_dir: The writable folder of the crop cache (default
                ``<root>/aerial``; another one when the root is read-only).
            version: Tag in the cache ID, to keep variants apart.
            backend: Interpolation, see :class:`kitscenes.aerial.AerialCrops`.
            verbose: Progress bars.
            workers: Crop processes, one scene at a time each (default: the CPU
                count, at most 8). Memory: each holds up to *tile_cache_mb* of
                decoded tiles plus ~150 MiB.
            tile_cache_mb: Decoded raw tiles kept per process (12 MiB per 2048 px
                tile; the bound of the cache, not a reservation).

        Returns:
            The cache, also in :attr:`aerial`; its ``coverage``
            (:class:`~kitscenes.aerial.AerialCoverage`) says which frames got
            no crop and why (also in ``<cache_dir>/metadata/coverage_<cache_id>.json``).

        Raises:
            ValueError: If no scene has a map origin and poses, or if the
                tile folders do not fit together.
            FileNotFoundError: Without raw tiles.
        """
        from kitscenes.aerial.prepare import prepare_aerials

        cache = prepare_aerials(
            self, roi, tiles=tiles, cache_dir=cache_dir, version=version, backend=backend, verbose=verbose,
            workers=workers, tile_cache_mb=tile_cache_mb,
        )
        return self.use_aerial(cache)

    def use_aerial(
        self,
        cache: Union["AerialCache", "AerialCrops", str, None] = None,
        *,
        roi: Optional[tuple[float, float]] = None,
        tiles: Union["AerialTiles", str, Path, Sequence[Union[str, Path]], None] = None,
        backend: str = "auto",
        cache_dir: Union[str, Path, None] = None,
    ) -> Union["AerialCache", "AerialCrops"]:
        """The aerial crops of this dataset: ``crops[frame]`` gives the crop of a frame.

        Either crops cut on the fly from the raw tiles (*roi*, or an
        :class:`~kitscenes.aerial.AerialCrops`), or a prepared crop cache
        (:meth:`prepare_aerials`; by ID, or the only one in *cache_dir*)::

            crops = ds.use_aerial(roi=(60, 30))                               # from <root>/aerial
            crops = ds.use_aerial(roi=(60, 30), tiles="/scratch/tile_store")  # from a tile store: ms per crop
            crops = ds.use_aerial("dop20_res020_roi60x30_ego_v1", cache_dir="/scratch/aerial")
            crop = crops[frame]                                               # (H, W, 3) uint8 RGB

        The crops are also attached to the dataset, for the shortcut
        :meth:`kitscenes.Frame.aerial` (the last call wins).

        Args:
            cache: An :class:`~kitscenes.aerial.AerialCrops` or
                :class:`~kitscenes.aerial.AerialCache`, a prepared cache's ID
                (see ``kitscenes.aerial.cache_ids(cache_dir)``), or None.
            roi: ``(longitudinal_m, lateral_m)``: crops on the fly, from *tiles*.
            tiles: With *roi*: an :class:`~kitscenes.aerial.AerialTiles` or
                folders that hold tiles (default ``<root>/aerial``).
            backend: With *roi*: the interpolation (:class:`~kitscenes.aerial.AerialCrops`).
            cache_dir: Without *roi*: the folder of the prepared cache
                (default ``<root>/aerial``).

        Returns:
            The crops (an :class:`~kitscenes.aerial.AerialCrops` or
            :class:`~kitscenes.aerial.AerialCache`), also in :attr:`aerial`.

        Raises:
            AerialUnavailable: ``NOT_PREPARED`` if there is no such cache.
            FileNotFoundError: With *roi*, if there are no raw tiles.
            ValueError: If *cache* is None, *roi* is None and the folder has several caches.
        """
        from kitscenes.aerial import AerialCache, AerialCrops

        if roi is not None:
            if cache is not None:
                raise ValueError("Pass either a cache or roi=, not both")
            cache = AerialCrops(tiles if tiles is not None else self.root, roi, backend=backend)
        elif tiles is not None:
            raise ValueError("tiles= goes with roi= (crops on the fly)")
        if not isinstance(cache, (AerialCache, AerialCrops)):
            from kitscenes.aerial.cache import aerial_root

            cache = AerialCache.open(Path(cache_dir) if cache_dir is not None else aerial_root(self.root), cache)
        with self._lock:
            self.aerial = cache
            for scene in self._scenes.values():
                scene._aerial = cache
        return cache

    def use_aerial_features(
        self,
        crops: Optional["AerialFeatureCrops"] = None,
        *,
        roi: Optional[tuple[float, float]] = None,
        layers: Optional[Sequence[int]] = None,
        resolution_m: Optional[float] = None,
        features: Union["AerialFeatures", str, Path, None] = None,
        feature_set: Optional[str] = None,
        orientation: str = "north",
    ) -> "AerialFeatureCrops":
        """The DINOv3 feature crops of the aerial imagery: ``crops[frame]`` gives the crop of a frame.

        Cut on the fly from the feature tiles (:mod:`kitscenes.aerial.features`;
        they come with a full ``kitscenes download ROOT``, with a filtered one only with ``--aerial-features``)::

            crops = ds.use_aerial_features(roi=(60, 30))                      # north-up: stored tokens + pose
            crops = ds.use_aerial_features(roi=(60, 30), orientation="ego", layers=(4, 11, 17, 23))
            features = crops[frame]

        The crops are also attached to the dataset, for the shortcut
        :meth:`kitscenes.Frame.aerial_features` (the last call wins).

        ``orientation="north"`` gives the stored north-up tokens around the ego
        with the pose (:class:`~kitscenes.aerial.NorthUpFeatures`); the model
        rotates (its outputs, or by sampling with the pose). ``"ego"``
        resamples the tokens into the ego frame, which does not rotate what
        they encode (DINOv3 features are not rotation-equivariant); see
        ``docs/aerial_features.md``.

        Args:
            crops: An :class:`~kitscenes.aerial.AerialFeatureCrops` (instead of the arguments below).
            roi: ``(longitudinal_m, lateral_m)`` of a crop, centred on the ego.
            layers: Backbone blocks (default the last stored one).
            resolution_m: ``"ego"`` only: crop cell size (default one token, 1.28 m).
            orientation: ``"north"`` (default) or ``"ego"``.
            features: An :class:`~kitscenes.aerial.AerialFeatures` or the folder
                that holds the feature tiles (default ``<root>/aerial/features``).
            feature_set: Which set, if there are several.

        Returns:
            The crops, also in :attr:`aerial_features`.

        Raises:
            FileNotFoundError: If there are no feature tiles.
            ValueError: Without *roi* and *crops*, or with both.
        """
        from kitscenes.aerial import AerialFeatureCrops, AerialFeatures

        if (crops is None) == (roi is None):
            raise ValueError("Pass either crops or roi=")
        if crops is None:
            if not isinstance(features, AerialFeatures):
                features = AerialFeatures(features if features is not None else self.root, feature_set)
            crops = AerialFeatureCrops(features, roi, layers=layers, resolution_m=resolution_m,
                                       orientation=orientation)
        with self._lock:
            self.aerial_features = crops
            for scene in self._scenes.values():
                scene._aerial_features = crops
        return crops

    def read_tracks(self, *, workers: int = 32) -> tuple[dict[str, "SceneTrack"], dict[str, str]]:
        """Every scene's :attr:`~kitscenes.scene.Scene.track`, read in a thread pool.

        A few small files per scene: on a network mount the cost is the round
        trips, which the threads overlap. Opens no sensor data.

        Returns:
            ``(tracks, failures)``: ``{scene_id: SceneTrack}`` in dataset
            order, and ``{scene_id: "ErrorType: message"}`` for scenes that
            could not be read: an ``OSError`` (e.g. a network timeout; worth
            retrying, not caching) or a :class:`~kitscenes.errors.DataFormatError`
            (a malformed file).
        """
        from concurrent.futures import ThreadPoolExecutor

        def one(scene_id: str):
            try:
                return scene_id, self[scene_id].track, None
            except (OSError, DataFormatError) as err:
                return scene_id, None, f"{type(err).__name__}: {err}"

        tracks, failures = {}, {}
        with ThreadPoolExecutor(max(1, workers)) as pool:
            for scene_id, track, error in pool.map(one, self._ids):
                if error is None:
                    tracks[scene_id] = track
                else:
                    failures[scene_id] = error
        return tracks, failures

    def frames(self, require: Union[str, Sequence[str]] = (), *, workers: int = 4) -> "FrameList":
        """All frames of all scenes, as a flat list (for training: :class:`kitscenes.SampleDataset`).

        Reads each scene's timeline and, for *require*, its file lists (in
        the release format: the scene's ``index.parquet``), in a thread pool.

        Args:
            require: Sensors (one name or several) that must have data; frames
                where one is missing (dropped) are left out. A required
                sensor that is not in this copy raises.
            workers: Threads that open the scenes (they mostly wait on reads;
                more help on a cold or network file system).

        Raises:
            SensorUnavailable: With ``state`` ``NOT_DOWNLOADED``; the message names the download command.
        """
        from concurrent.futures import ThreadPoolExecutor

        require = (require,) if isinstance(require, str) else tuple(require)

        def one(scene_id: str) -> tuple["Scene", list[int], list[str]]:
            scene = self.scene(scene_id)
            keep, absent = None, []
            for sensor in require:
                if scene.availability(sensor) is Availability.NOT_DOWNLOADED:
                    absent.append(sensor)
                present = set(scene.frame_indices(sensor).tolist())
                keep = present if keep is None else keep & present
            n = scene.num_frames
            return scene, list(range(n)) if keep is None else sorted(i for i in keep if i < n), absent

        refs: list[tuple[str, int]] = []
        missing: list[tuple["Scene", str]] = []
        with ThreadPoolExecutor(max(1, workers)) as pool:
            for scene, indices, absent in pool.map(one, self._ids):
                refs.extend((scene.id, i) for i in indices)
                missing.extend((scene, sensor) for sensor in absent)
        if missing:
            from kitscenes.scene import component_of

            scenes = {s.id for s, _ in missing}
            sensors = sorted({n for _, n in missing})
            components = sorted({component_of(n) for n in sensors})
            first = missing[0][0]
            root = first._dataset_root or "$KITSCENES_ROOT"
            detail = (f"{', '.join(sensors)} not downloaded in {len(scenes)} scene(s) (e.g. {first.id}), "
                      f"so their frames would be left out")
            hint = f"get them with: kitscenes download {root} --components {' '.join(components)}"
            raise SensorUnavailable(sensors[0], None, Availability.NOT_DOWNLOADED, detail, hint)
        return FrameList(self, refs)


class FrameList(Sequence[Frame]):
    """Frames of a dataset by position; holds only scene IDs and indices, so it pickles small.

    The frames are views that read on access: for a torch ``DataLoader`` wrap
    them in :class:`kitscenes.SampleDataset` (``SampleDataset(ds, spec,
    frames)``), which turns each into a dict of arrays, with
    ``collate_fn=kitscenes.collate``.
    """

    def __init__(self, dataset: Dataset, refs: Sequence[tuple[str, int]]) -> None:
        self.dataset = dataset
        self.refs = list(refs)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return FrameList(self.dataset, self.refs[i])
        scene_id, index = self.refs[i]
        return self.dataset.scene(scene_id).frame(index)

    def __repr__(self) -> str:
        return f"FrameList(frames={len(self)}, scenes={len({s for s, _ in self.refs})})"


def _split_names(splits: Union[str, Sequence[str], None],
                 definition: Optional[SplitDefinition] = None) -> Optional[set[str]]:
    if splits is None:
        return None
    names = [splits] if isinstance(splits, str) else list(splits)
    if definition is not None:
        definition.scene_ids(names)  # ValueError for a split it does not have
        return set(names)
    return {canonical_split_name(name) for name in names}


def open(  # noqa: A001 - kitscenes.open() mirrors io.open()-style entry points
    root: Path | str | None = None,
    splits: Union[str, Sequence[str], None] = None,
    *,
    cache_mb: int = 0,
    scene_cache: int = SCENE_CACHE_SIZE,
    verify: bool = False,
    sd_snapshot: str = "scene",
    use_imu: bool = True,
    split_definition: Union[SplitDefinition, str, Path, None] = None,
) -> Dataset:
    """Open a dataset root. See :class:`Dataset`."""
    return Dataset(root, splits, cache_mb=cache_mb, scene_cache=scene_cache, verify=verify, sd_snapshot=sd_snapshot,
                   use_imu=use_imu, split_definition=split_definition)
