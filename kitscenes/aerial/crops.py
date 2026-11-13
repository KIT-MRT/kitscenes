"""Aerial crops cut when asked for: :class:`AerialCrops`, and which frames get one: :class:`AerialCoverage`."""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Union

import numpy as np

from kitscenes.aerial._crop import TileCoverageError, crop_sample_bbox, resolve_backend, warp_ego_aligned
from kitscenes.aerial._frame_crops import PoseCrops
from kitscenes.aerial._ids import make_aerial_cache_id
from kitscenes.aerial.config import AerialCacheConfig
from kitscenes.aerial.tiles import AerialTiles, PathLike
from kitscenes.errors import MapUnavailable

if TYPE_CHECKING:
    from kitscenes.aerial._poses import ScenePoses
    from kitscenes.scene import Scene

logger = logging.getLogger(__name__)


class AerialCrops(PoseCrops):
    """Ego-aligned aerial crops cut on the fly from raw tiles, without a prepared crop cache.

    ``crops = ds.use_aerial(roi=...)`` makes one and attaches it to the
    dataset. ``crops[frame]`` cuts the crop of a frame
    (:meth:`kitscenes.Frame.aerial` is the shortcut for the attached crops);
    :meth:`crop` cuts one at any EPSG:25832 pose. The
    crops have the geometry, interpolation and orientation of a prepared
    crop cache with the same :attr:`cache_id`
    (``dop20_res020_roi60x30_ego_v1``): ``(round(lateral_m / res),
    round(longitudinal_m / res), 3)`` uint8 RGB, ego at the centre pixel,
    forward to +column, left to -row.

    How fast depends on the tiles: from a tile store (``.npy``,
    :func:`kitscenes.aerial.build_tile_store`) a crop takes a few
    milliseconds in any order; from JPEG or PNG tiles each tile a crop
    touches is decoded first (JPEG ~50 ms, PNG ~300 ms per 2048 px tile) and
    kept in a bounded cache (``AerialTiles(cache_mb=...)``), which helps
    when frames come in order. Memory per process: at most the tile
    cache (``cache_mb``, default 256 MiB, JPEG/PNG only) plus one sample
    window (~1 MiB); a tile store needs no cache.

    Picklable (to its tiles and parameters) and fork-safe: each process
    opens what it reads itself, so it goes into DataLoader workers with the
    dataset.

    Args:
        tiles: An :class:`~kitscenes.aerial.AerialTiles`, or what it takes (a
            dataset root, an aerial folder or a list of them).
        roi: ``(longitudinal_m, lateral_m)`` of a crop, centred on the ego.
        backend: Interpolation (bilinear either way): ``"cv2"`` (OpenCV,
            ~20x faster; a value differs from ``"scipy"`` by at most 1
            level, about 1 in 10^4 values), ``"scipy"`` (the reference) or
            ``"auto"`` (cv2 if installed).
        version: Tag in :attr:`cache_id`, as in ``prepare_aerials``.

    Raises:
        ValueError: For an unknown backend.
    """

    #: Crops are ego-aligned (forward is +column); part of :attr:`cache_id`.
    orientation = "ego"

    def __init__(
        self,
        tiles: Union[AerialTiles, PathLike, Sequence[PathLike]],
        roi: tuple[float, float],
        *,
        backend: str = "auto",
        version: str = "v1",
    ) -> None:
        self.tiles = tiles if isinstance(tiles, AerialTiles) else AerialTiles(tiles)
        self.roi_m = (float(roi[0]), float(roi[1]))
        self.backend = resolve_backend(backend)
        self.version = version
        self.resolution_m = self.tiles.resolution_m
        self.cache_id = make_aerial_cache_id(self.tiles.provider, self.resolution_m, self.roi_m, self.orientation, version)
        self.config = AerialCacheConfig(
            provider=self.tiles.provider, resolution_m=self.resolution_m, roi_m=self.roi_m,
            tile_size_px=self.tiles.requested_tile_size_px, orientation=self.orientation, version=version,
            cache_id=self.cache_id, raw_cache_id=self.tiles.raw_cache_id,
        )
        self._init_process()

    def __repr__(self) -> str:
        return f"AerialCrops({self.tiles!r}, roi={self.roi_m}, backend={self.backend!r})"

    def __getstate__(self) -> dict:
        return {"tiles": self.tiles, "roi": self.roi_m, "backend": self.backend, "version": self.version}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["tiles"], state["roi"], backend=state["backend"], version=state["version"])

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(height_px, width_px, 3)`` of a crop."""
        return (round(self.roi_m[1] / self.resolution_m), round(self.roi_m[0] / self.resolution_m), 3)

    # ------------------------------------------------------------------
    # Crops at any pose

    def covers(self, xy: tuple[float, float], *, area_id: Optional[str] = None) -> bool:
        """Whether the tiles hold everything a crop at EPSG:25832 *xy* needs (at any heading)."""
        bbox = crop_sample_bbox((float(xy[0]), float(xy[1])), self.roi_m, self.resolution_m)
        return self.tiles.covering_area(bbox, area_id) is not None

    def crop(self, xy: tuple[float, float], yaw: float, *, area_id: Optional[str] = None) -> np.ndarray:
        """The ego-aligned crop at EPSG:25832 *xy* (easting, northing) with heading *yaw* (radians, from east).

        Args:
            area_id: The area to read first (any area that covers the crop is used).

        Returns:
            (H, W, 3) uint8 RGB; ego at the centre pixel, forward to +column, left to -row.

        Raises:
            TileCoverageError: If no area has every tile the crop needs.
        """
        ego_xy = (float(xy[0]), float(xy[1]))
        bbox = crop_sample_bbox(ego_xy, self.roi_m, self.resolution_m)
        area = self.tiles.covering_area(bbox, area_id)
        if area is None:
            raise TileCoverageError(f"No raw tiles for a crop at EPSG:25832 {ego_xy} in {self.tiles!r}")
        window = self.tiles.sampler(area).sample_bbox(bbox, self.resolution_m)
        return warp_ego_aligned(window, bbox, ego_xy, float(yaw), self.roi_m, self.resolution_m, self.backend)

    # ------------------------------------------------------------------
    # Crops of frames (crops[frame], Frame.aerial)

    def _area_of(self, scene: "Scene", poses: "ScenePoses") -> Optional[str]:
        return self.tiles.area_of(scene.id, poses.lat, poses.lon)

    def coverage(self, scenes: Iterable["Scene"]) -> "AerialCoverage":
        """Which frames of *scenes* (e.g. a dataset) get a crop, and why the others do not."""
        report = {}
        for scene in scenes:
            report[scene.id] = self._scene_coverage(scene)
        return AerialCoverage(self.cache_id, report)

    def _scene_coverage(self, scene: "Scene") -> "SceneCoverage":
        try:
            return self._coverage_of(scene, self.scene_poses(scene), None)
        except MapUnavailable as err:
            return self._coverage_of(scene, None, f"no map origin ({err.state.value})")
        except ValueError as err:
            return self._coverage_of(scene, None, str(err))

    def _coverage_of(self, scene: "Scene", poses: Optional["ScenePoses"], reason: Optional[str]) -> "SceneCoverage":
        num_frames = len(scene.timestamps_ns)
        if poses is None:
            empty = np.zeros(0, dtype=np.int64)
            return SceneCoverage(scene.id, scene.release_split, None, num_frames, empty, empty, empty, reason)
        area_id = self.tiles.area_of(scene.id, poses.lat, poses.lon)
        covered = np.array([self.covers(xy, area_id=area_id) for xy in poses.xy], dtype=bool)
        without_pose = np.setdiff1d(np.arange(num_frames), poses.indices).astype(np.int64)
        return SceneCoverage(scene.id, scene.release_split, area_id, num_frames, poses.indices[covered],
                             poses.indices[~covered], without_pose, None)


@dataclasses.dataclass(frozen=True)
class SceneCoverage:
    """The aerial crops of one scene.

    Attributes:
        scene_id, split: The scene and its release folder.
        area_id: Its 10 km area (None without a map origin or poses).
        num_frames: Its frames.
        covered: Frame indices that get a crop.
        uncovered: Frames with a pose whose crop needs a tile that is not there.
        without_pose: Frames whose time lies outside the poses.
        reason: Why the scene gets no crop at all (no map origin, no poses), else None.
    """

    scene_id: str
    split: Optional[str]
    area_id: Optional[str]
    num_frames: int
    covered: np.ndarray
    uncovered: np.ndarray
    without_pose: np.ndarray
    reason: Optional[str]

    @property
    def complete(self) -> bool:
        """Every frame with a pose gets a crop (scenes without a map origin count as complete: they get none)."""
        if self.reason is not None:
            return self.reason.startswith("no map origin")
        return not len(self.uncovered)

    def to_dict(self) -> dict:
        return {"split": self.split, "area": self.area_id, "num_frames": self.num_frames,
                "covered": len(self.covered), "uncovered_frames": self.uncovered.tolist(),
                "frames_without_pose": self.without_pose.tolist(), "reason": self.reason}


@dataclasses.dataclass(frozen=True)
class AerialCoverage:
    """Which frames of a set of scenes get an aerial crop (:meth:`AerialCrops.coverage`)."""

    cache_id: str
    scenes: dict[str, SceneCoverage]

    @property
    def complete(self) -> bool:
        """Every frame with a pose in a scene with a map origin gets a crop."""
        return all(s.complete for s in self.scenes.values())

    @property
    def num_covered(self) -> int:
        return sum(len(s.covered) for s in self.scenes.values())

    @property
    def num_uncovered(self) -> int:
        return sum(len(s.uncovered) for s in self.scenes.values())

    def incomplete(self) -> dict[str, SceneCoverage]:
        """The scenes with frames that have a pose but no crop, or with poses that cannot be used."""
        return {k: s for k, s in self.scenes.items() if not s.complete}

    def summary(self) -> str:
        no_origin = sum(1 for s in self.scenes.values() if s.reason and s.reason.startswith("no map origin"))
        return (f"{self.cache_id}: {self.num_covered} frames with a crop in {len(self.scenes)} scenes; "
                f"{self.num_uncovered} frames lack raw tiles ({len(self.incomplete())} scenes incomplete); "
                f"{no_origin} scenes without a map origin")

    def to_json(self, path: Optional[PathLike] = None) -> str:
        """The report as JSON (written to *path* if given)."""
        text = json.dumps({"cache_id": self.cache_id, "complete": self.complete,
                           "scenes": {k: s.to_dict() for k, s in sorted(self.scenes.items())}}, indent=1)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(text + "\n")
        return text
