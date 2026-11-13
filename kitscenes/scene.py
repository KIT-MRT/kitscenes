"""One recorded scene: its timeline, poses, sensor rig, file index and map.

A :class:`Scene` holds metadata only, loaded lazily and cached on the
object; sensor data is read through :meth:`Scene.frame`. Scenes pickle to
their storage, ID and split, so they can be sent to DataLoader workers.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Literal, Optional, Sequence, Union, overload

import numpy as np

from kitscenes._repr import compact_repr
from kitscenes.constants import LidarNameLike, SensorNameLike, sensor_name
from kitscenes.errors import Availability, Missing, Unavailable
from kitscenes._layout import (
    CALIBRATION,
    GROUND_SEGMENTATION,
    INSTANCE_MASKS,
    MAP_ORIGIN,
    MAP_OSM,
    POSES,
    REFERENCE_TIMESTAMPS,
    frame_index_of,
    sensor_kind,
)
from kitscenes.motion import ExtrapolationError, InterpolationGapError, Trajectory
from kitscenes._poses import parse_poses
from kitscenes.rig import Rig
from kitscenes.schema import EgoPose
from kitscenes.splits import PoseFrame, SceneInfo, SplitInfo, known_split
from kitscenes._storage import DirStorage, Storage, ZipStorage
from kitscenes._timestamps import parse_timestamps

if TYPE_CHECKING:
    import pyarrow as pa

    from kitscenes.aerial import AerialCache, AerialCrops, AerialFeatureCrops
    from kitscenes._cache import PayloadCache
    from kitscenes.ego_motion import EgoMotionCompensation
    from kitscenes.ego_trajectory import EgoState
    from kitscenes.frame import Frame
    from kitscenes.image_corrections import ImageCorrection
    from kitscenes.map import MapOrigin, SceneMap
    from kitscenes.points import PointCloud
    from kitscenes.sdmap import SDMap, SDRoute
    from kitscenes.transcode import ImageVariant

logger = logging.getLogger(__name__)



@dataclasses.dataclass(frozen=True, eq=False)
class SceneTrack:
    """Where a scene drove: its poses and what places them, read without opening any sensor data.

    Attributes:
        scene_id, split: The scene and its split (:attr:`Scene.split`).
        city, lidar: From the scene table of :data:`kitscenes.splits.RELEASE` (None for scenes it does not list).
        origin: The scene's map frame (``maps/origin.json``); None where the
            split withholds maps or the file is missing. Without it the
            positions cannot be placed on the earth.
        t_ns: (N,) int64 pose times in nanoseconds since the scene start.
        xyz: (N, 3) base-frame positions in the scene's world frame.
        quat: (N, 4) orientations, quaternions (qx, qy, qz, qw).
    """

    __repr__ = compact_repr

    scene_id: str
    split: Optional[str]
    city: Optional[str]
    lidar: Optional[str]
    origin: Optional["MapOrigin"]
    t_ns: np.ndarray
    xyz: np.ndarray
    quat: np.ndarray

    def __len__(self) -> int:
        return len(self.t_ns)

    def pose(self, i: int) -> EgoPose:
        """Recorded pose *i* (a line of ``poses.txt``) as an :class:`~kitscenes.schema.EgoPose`."""
        return EgoPose(int(self.t_ns[i]), self.xyz[i], self.quat[i])

    def __iter__(self) -> Iterator[EgoPose]:
        return (self.pose(i) for i in range(len(self)))

    def _require_origin(self) -> "MapOrigin":
        if self.origin is None:
            raise ValueError(f"Scene {self.scene_id} has no map origin; its poses cannot be placed on the earth.")
        return self.origin

    def utm(self) -> np.ndarray:
        """(N, 3) positions in absolute UTM (zone: ``origin.epsg``).

        Raises:
            ValueError: Without :attr:`origin`.
        """
        return self._require_origin().to_utm(self.xyz)

    def latlon(self) -> np.ndarray:
        """(N, 2) latitude, longitude in degrees.

        Raises:
            ValueError: Without :attr:`origin`.
        """
        return self._require_origin().to_latlon(self.xyz)


class SceneFiles:
    """Where a scene's files are and what they hold as stored: ``scene.files``.

    Attributes:
        storage: Where the files are read from: a folder or the release zips.
        calib_file: The ``calib.json`` used instead of the scene's, or None (see :class:`Scene`).
    """

    def __init__(self, scene: "Scene", storage: Storage, calib_file: Optional[Path | str]) -> None:
        self._scene = scene
        self.storage = storage
        self.calib_file = calib_file

    def __repr__(self) -> str:
        return f"SceneFiles({self.storage!r})"

    @property
    def path(self) -> Optional[Path]:
        """Scene folder, for folder storage."""
        return self.storage.root if isinstance(self.storage, DirStorage) else None

    @cached_property
    def image_variants(self) -> dict[str, "ImageVariant"]:
        """``{camera: record}`` of the cameras whose images were re-encoded (:mod:`kitscenes.transcode`)."""
        from kitscenes.transcode import read_variants

        return read_variants(self.storage)

    @cached_property
    def rig_original(self) -> Rig:
        """The sensors as ``calibration/calib.json`` has them: the cameras at the release's image size.

        Raises:
            CalibrationError: As :attr:`kitscenes.Scene.rig`.
        """
        from kitscenes.errors import CalibrationError

        source = self.storage.describe(CALIBRATION)
        if self.calib_file is not None:
            source = str(self.calib_file)
            rig = Rig.from_json_bytes(Path(self.calib_file).read_bytes(), source)
        elif self.storage.exists(CALIBRATION):
            rig = Rig.from_json_bytes(self.storage.read_bytes(CALIBRATION), source)
        else:
            rig = Rig.empty()
        uncalibrated = [n for n in self._scene.sensors if n not in rig.sensor_names]
        if uncalibrated:
            raise CalibrationError(f"{source} has no entry for {', '.join(uncalibrated)}, which have data in scene {self._scene.id}")
        return rig

    @cached_property
    def image_corrections(self) -> dict[str, "ImageCorrection"]:
        """``{camera: correction}`` from ``calibration/image_corrections.json``, empty without the file.

        :meth:`kitscenes.Frame.image` applies them, see :mod:`kitscenes.image_corrections`.
        They are at the release's image size.

        Raises:
            CalibrationError: If the file is malformed.
        """
        from kitscenes.image_corrections import load_image_corrections

        return load_image_corrections(self.storage)



class Scene:
    """A scene of the dataset.

    Args:
        storage: Where the scene's files are (a folder path is accepted).
        scene_id: Scene ID (UUID); defaults to the folder name.
        split: The scene's split (a dataset opened with a split definition
            passes the definition's); by default :attr:`release_split`.
        release_split: The release folder the scene is in; by default
            *split* if that is a release folder, else the scene folder's
            parent if that is one (``data/<split>/<scene_id>``).
        cache: Optional payload cache shared by the frames (see
            :class:`kitscenes._cache.PayloadCache`); not pickled.
        aerial: The aerial crops :meth:`Frame.aerial` reads, an AerialCrops or
            AerialCache (set by the dataset, see :meth:`kitscenes.Dataset.use_aerial`).
        aerial_features: The aerial feature crops :meth:`Frame.aerial_features` reads, an
            AerialFeatureCrops (set by the dataset, see :meth:`kitscenes.Dataset.use_aerial_features`).
        sd_snapshot: Which OpenStreetMap snapshot :attr:`sd_map` loads:
            ``"scene"`` (default: the scene's city and recording week, the
            scene table's ``sd_snapshot``), ``"latest"`` (the city's latest),
            ``"paper_2026-05-03"`` (the snapshots of the paper) or a week
            (``"2026-W06"``); see :meth:`kitscenes.Dataset.sd_map`.
        use_imu: Refine the ego motion with the gyro where the scene has IMU data
            and a valid ``gnss_ins_imu`` calibration (see :attr:`ego_motion`).
            False: the poses alone, for everything.
        calib_file: A ``calib.json`` to use instead of the scene's, for all of its
            entries, e.g. to test a new calibration. It must fit the scene; this
            is not checked.
    """

    def __init__(
        self,
        storage: Storage | Path | str,
        scene_id: Optional[str] = None,
        split: Optional[str] = None,
        *,
        release_split: Optional[str] = None,
        cache: Optional["PayloadCache"] = None,
        aerial: Union["AerialCache", "AerialCrops", None] = None,
        sd_snapshot: str = "scene",
        aerial_features: Optional["AerialFeatureCrops"] = None,
        use_imu: bool = True,
        calib_file: Optional[Path | str] = None,
    ) -> None:
        if not isinstance(sd_snapshot, str) or not sd_snapshot:
            raise ValueError(f"sd_snapshot must be a snapshot name such as 'scene', got {sd_snapshot!r}")
        if not isinstance(storage, Storage):
            from kitscenes._storage import open_storage

            storage = open_storage(storage)
        #: The scene's files and how they are stored (:class:`SceneFiles`): ``path``, ``storage``, ``calib_file``,
        #: ``image_variants``, ``rig_original``, ``image_corrections``.
        self.files = SceneFiles(self, storage, calib_file)
        if scene_id is None:
            if not isinstance(storage, DirStorage):
                raise ValueError("scene_id is required for non-folder storage")
            scene_id = storage.root.name
        self.id = scene_id
        if release_split is None:
            info = known_split(split) if split is not None else None
            if info is None:
                parent = getattr(storage, "root", None)
                info = known_split(Path(parent).parent.name) if parent is not None else None
            release_split = info.name if info is not None else None
        self._release_split = release_split
        #: The scene's split: :attr:`release_split`, or the split definition's (``kitscenes.open(split_definition=)``).
        self.split = split if split is not None else release_split
        self._cache = cache
        self._aerial = aerial
        self._aerial_features = aerial_features
        #: Which OpenStreetMap snapshot :attr:`sd_map` loads (see :class:`Scene`).
        self.sd_snapshot = sd_snapshot
        #: Whether the ego motion may use the gyro (see :class:`Scene`).
        self.use_imu = bool(use_imu)
        self._frame_index: dict[str, np.ndarray] = {}
        self._messages: dict[tuple[str, str, str], object] = {}
        self._warned: set[str] = set()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"Scene(id={self.id!r}, split={self.split!r}, storage={self.files.storage!r})"

    def __getstate__(self) -> dict:
        # no cached map or metadata
        return {"storage": self.files.storage, "id": self.id, "split": self.split, "release_split": self._release_split,
                "aerial": self._aerial,
                "sd_snapshot": self.sd_snapshot, "aerial_features": self._aerial_features,
                "use_imu": self.use_imu, "calib_file": self.files.calib_file}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["storage"], state["id"], state["split"], release_split=state["release_split"],
                      aerial=state["aerial"], sd_snapshot=state["sd_snapshot"],
                      aerial_features=state["aerial_features"], use_imu=state["use_imu"],
                      calib_file=state["calib_file"])

    # ------------------------------------------------------------------
    # Identity and timeline
    # ------------------------------------------------------------------

    @property
    def info(self) -> Optional["SceneInfo"]:
        """City, lidar generation and split of the scene from the scene table (None if it does not list it)."""
        from kitscenes.splits import scene_info

        return scene_info(self.id)

    @property
    def release_split(self) -> Optional[str]:
        """The release folder of the scene (``data/<release_split>/``): the folder it is in, or for a scene
        outside one, the folder the scene table lists it in (:attr:`info`). None if neither knows the scene.

        Its data follows this folder's conventions (:attr:`split_info`), whatever split :attr:`split` names.
        """
        if self._release_split is not None:
            return self._release_split
        listed = self.info
        return listed.split if listed is not None else None

    @property
    def split_info(self) -> Optional[SplitInfo]:
        """Conventions of the scene's release folder (maps and labels shipped, pose frame), :attr:`release_split`.

        For a folder that is no release folder, from the one the scene table
        lists the scene in (:attr:`info`). None if neither knows the scene.
        """
        info = known_split(self._release_split) if self._release_split is not None else None
        if info is None:
            listed = self.info
            info = known_split(listed.split) if listed is not None else None
        return info

    @cached_property
    def timestamps_ns(self) -> np.ndarray:
        """(T,) int64 frame times in nanoseconds since the scene start (read-only).

        Frame ``i`` of every sensor was captured for ``timestamps_ns[i]``.
        """
        ts = parse_timestamps(
            self.files.storage.read_bytes(REFERENCE_TIMESTAMPS).decode(),
            self.files.storage.describe(REFERENCE_TIMESTAMPS),
        )
        ts.flags.writeable = False
        return ts

    @property
    def num_frames(self) -> int:
        return len(self.timestamps_ns)

    def __len__(self) -> int:
        return self.num_frames

    @property
    def duration_s(self) -> float:
        ts = self.timestamps_ns
        return float(ts[-1] - ts[0]) / 1e9 if len(ts) > 1 else 0.0

    # ------------------------------------------------------------------
    # Poses
    # ------------------------------------------------------------------

    @cached_property
    def _poses(self) -> tuple[EgoPose, ...]:
        """The recorded ego poses, one per line of ``poses.txt`` (empty without one); public as :attr:`track`."""
        try:  # one request, no existence check first (on a network mount each is a round trip)
            data = self.files.storage.read_bytes(POSES)
        except FileNotFoundError:
            return ()
        poses = tuple(parse_poses(data.decode(), self.files.storage.describe(POSES)))
        try:
            frames = self.num_frames
        except FileNotFoundError:  # no timeline: the poses are still the poses
            frames = len(poses)
        if poses and len(poses) != frames:
            logger.warning("Scene %s: %d reference timestamps but %d poses.", self.id, frames, len(poses))
        return poses

    @property
    def pose_frame(self) -> Optional[PoseFrame]:
        """World frame of the poses: ``"local_utm"`` (the map frame) or ``"first_pose"``.

        From the release folder (:attr:`split_info`). None without poses, or for a scene no release folder knows.
        """
        info = self.split_info
        return info.pose_frame if info is not None and self._poses else None

    @property
    def _trajectory(self) -> tuple[Optional[Trajectory], Optional[str]]:
        """The ego motion (:attr:`ego_motion`) and None, or None and why there is none."""
        ego_motion = self.ego_motion
        return ego_motion.trajectory, None if ego_motion.trajectory is not None else ego_motion.reason

    @cached_property
    def ego_motion(self) -> "EgoMotionCompensation":
        """The scene's ego motion, built on first use (:class:`~kitscenes.ego_motion.EgoMotionCompensation`).

        ``source`` says what it comes from and ``reason`` why not from more:
        ``"imu"`` (poses refined with the gyro), ``"poses"`` (no IMU data as in
        the test-e2e splits, an invalid ``gnss_ins_imu``, or ``use_imu=False``) or
        None (no usable poses: clouds are not compensated). Everything that
        needs the ego motion uses it: clouds, ``frame.pose``, :meth:`ego`,
        :meth:`accumulate`. :attr:`track` stays what ``poses.txt`` says.
        """
        from kitscenes.ego_motion import EgoMotionCompensation

        return EgoMotionCompensation(self)

    @cached_property
    def _pose_trajectory(self) -> tuple[Optional[Trajectory], Optional[str]]:
        """The pose interpolation (:class:`kitscenes.motion.Trajectory`) and None, or None and why there is none."""
        poses = self._poses
        if len(poses) < 2:
            return None, "no poses.txt" if not poses else "only one pose"
        try:
            return Trajectory.from_ego_poses(poses), None
        except ValueError as err:
            return None, f"invalid poses: {err}"

    def ego(self, t_ns) -> "EgoState":
        """The ego vehicle's pose and motion at time(s) *t_ns* (ns since the scene start, like ``frame.t_ns``).

        The motion is from central differences, which read the poses up to
        0.2 s after *t_ns*; ``frame.ego(causal=True)`` uses the past only.

        Pose (``T_world_base``, ``xyz``, ``quat``), velocity, angular velocity
        and acceleration in the base frame, speed, yaw rate and curvature
        (:class:`~kitscenes.ego_trajectory.EgoState`), from the same ego motion
        as for deskewing the lidar (:attr:`ego_motion`), and extrapolated up to
        0.2 s past the ends of the poses. A scalar time gives fields without a leading axis, an
        array of M times fields of length M.

        Raises:
            PoseUnavailable: Without usable poses (``NOT_DOWNLOADED``,
                ``INVALID``), or ``DROPPED`` for a time too far outside the
                poses or in a gap between them.
        """
        from kitscenes.ego_trajectory import ego_state

        err = self._pose_unavailable(None)
        if err is not None:
            raise err
        try:
            return ego_state(self._trajectory[0], t_ns)
        except (ExtrapolationError, InterpolationGapError) as exc:
            from kitscenes.errors import PoseUnavailable

            raise PoseUnavailable("pose", None, Availability.DROPPED, str(exc)) from None

    @property
    def road_height_m(self) -> float:
        """Height of the base origin above the road in metres, a constant per lidar generation
        (:data:`kitscenes.constants.ROAD_HEIGHT_M`)."""
        from kitscenes.constants import DEFAULT_ROAD_HEIGHT_M, ROAD_HEIGHT_M

        info = self.info
        return ROAD_HEIGHT_M.get(info.lidar, DEFAULT_ROAD_HEIGHT_M) if info is not None else DEFAULT_ROAD_HEIGHT_M

    @property
    def T_road_base(self) -> np.ndarray:
        """(4, 4) base frame -> road frame: a shift by :attr:`road_height_m` along z (a new array).
        See docs/conventions.md, "The road frame"."""
        T = np.eye(4)
        T[2, 3] = self.road_height_m
        return T

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    @cached_property
    def rig(self) -> Rig:
        """Calibrated sensors from ``calibration/calib.json``, the cameras as their images are stored.

        A camera whose images were downscaled (:mod:`kitscenes.transcode`,
        ``scene.files.image_variants``) has its ``K`` and ``image_size`` scaled to
        them, so everything that projects into its images stays right;
        ``scene.files.rig_original`` has the release's calibration.

        Raises:
            CalibrationError: If ``calib.json`` is malformed, has an invalid
                entry, or has no entry for a sensor with data in this copy
                (also if it is missing while there is sensor data).
        """
        import dataclasses

        from kitscenes.errors import CalibrationError

        rig = self.files.rig_original
        variants = self.files.image_variants
        if not variants:
            return rig
        cameras = dict(rig.cameras)
        for name, variant in variants.items():
            camera = cameras.get(name)
            if camera is None:
                continue
            if tuple(camera.image_size) != tuple(variant.original_size):
                raise CalibrationError(
                    f"Scene {self.id}: {name} was transcoded from {variant.original_size[0]}x{variant.original_size[1]} "
                    f"images, but calib.json says {camera.image_size[0]}x{camera.image_size[1]}")
            cameras[name] = camera.scaled(variant.scale)
        return dataclasses.replace(rig, cameras=cameras)

    @cached_property
    def sensors(self) -> tuple[str, ...]:
        """Sensors with data in this copy of the scene (a folder), in canonical order.

        Remembered, but a sensor downloaded later is noticed the first time
        it is asked for (:meth:`availability`).
        """
        from kitscenes.constants import ALL_SENSOR_NAMES

        return tuple(n for n in ALL_SENSOR_NAMES if self.files.storage.is_dir(n))

    def _refresh(self) -> None:
        """Forget what this scene found on disk (sensors, frame lists), e.g. after downloading more components."""
        with self._lock:
            for name in ("sensors", "_archive_index", "rig", "ego_motion"):  # calibration/ is not downloaded later
                self.__dict__.pop(name, None)
            self.files.__dict__.pop("image_variants", None)
            self._frame_index.clear()
            if isinstance(self.files.storage, ZipStorage):  # a component may have been re-written in place
                self.files.storage.close()
                self.files.storage._index = None

    @cached_property
    def _archive_index(self):
        """The scene's parsed ``index.parquet`` (archive layout, extracted or not), None without one."""
        from kitscenes._archive import INDEX_FILE, ArchiveIndex
        from kitscenes._storage import ZipStorage

        try:
            if isinstance(self.files.storage, ZipStorage):
                return self.files.storage.index  # parsed once, shared with the storage
            if self.files.storage.exists(INDEX_FILE):
                from kitscenes._storage import index_overlays

                overlays = index_overlays(self.files.storage.root) if isinstance(self.files.storage, DirStorage) else ()
                return ArchiveIndex.from_bytes(self.files.storage.read_bytes(INDEX_FILE), overlays)
        except Exception as err:  # a broken index: fall back
            logger.warning("Scene %s: unreadable %s (%s)", self.id, INDEX_FILE, err)
        return None

    @property
    def _dataset_root(self) -> Optional[Path]:
        """The dataset root this scene sits in (``<root>/data/<split>/<id>``), None otherwise."""
        path = getattr(self.files.storage, "root", None)
        if path is None or Path(path).parent.parent.name != "data":
            return None
        return Path(path).parent.parent.parent

    def _download_hint(self, component: str) -> str:
        """The command that downloads *component* (an archive component) of this scene."""
        root = self._dataset_root or "$KITSCENES_ROOT"
        return f"get it with: kitscenes download {root} --scenes {self.id} --components {component}"

    def frame_indices(self, sensor: SensorNameLike) -> np.ndarray:
        """Sorted frame indices that have a file for *sensor* (read-only array)."""
        sensor = sensor_name(sensor)
        sensor_kind(sensor)  # KeyError for unknown names
        with self._lock:
            indices = self._frame_index.get(sensor)
        if indices is None:
            found = (frame_index_of(m) for m in self.files.storage.list(sensor))
            indices = np.array(sorted(i for i in found if i is not None), dtype=np.int64)
            indices.flags.writeable = False
            with self._lock:
                self._frame_index[sensor] = indices
        return indices

    def availability(self, what: Union[SensorNameLike, str], index: Optional[int] = None) -> Availability:
        """Whether some data is there (at frame *index*), and if not, why.

        The accessors raise the same state in an :class:`~kitscenes.errors.Unavailable`.

        Args:
            what: A sensor name (``"lidar_top"``), ``"ground_labels/<lidar>"``,
                ``"instances/<camera>"``, ``"map"``, ``"sd_map"`` (the
                OpenStreetMap SD map, :attr:`sd_map`), ``"pose"``, ``"aerial"``
                (an aerial crop, see :meth:`kitscenes.Frame.aerial`) or a GNSS
                topic ``"<device>/<topic>"`` (``"gnss/navsatfix"``,
                ``"gnss_ins/imu"``, see :meth:`message_topics`).
            index: Frame index; None asks about the scene as a whole.

        Raises:
            KeyError: For an unknown name.
            IndexError: For an index outside the timeline.
        """
        err = self._unavailable(sensor_name(what), index)
        return Availability.PRESENT if err is None else err.state

    def _unavailable(self, what: str, index: Optional[int] = None) -> Optional["Unavailable"]:
        """The error an accessor raises for *what* (at frame *index*), None if it is there."""
        from kitscenes.constants import ALL_SENSOR_NAMES, CAMERA_NAMES, LIDAR_NAMES
        from kitscenes.messages import DEVICES

        kind, _, name = what.partition("/")
        if what in ALL_SENSOR_NAMES:
            check = lambda: self._sensor_unavailable(what, index)  # noqa: E731
        elif kind == "ground_labels" and name in LIDAR_NAMES:
            check = lambda: self._labels_unavailable(what, GROUND_SEGMENTATION, name, index)  # noqa: E731
        elif kind == "instances" and name in CAMERA_NAMES:
            check = lambda: self._labels_unavailable(what, INSTANCE_MASKS, name, index)  # noqa: E731
        elif what == "map":
            check = lambda: self._map_unavailable(MAP_OSM)  # noqa: E731
        elif what == "sd_map":
            check = self._sd_map_unavailable
        elif what == "pose":
            check = lambda: self._pose_unavailable(index)  # noqa: E731
        elif what == "aerial":
            check = lambda: self._aerial_unavailable(index)  # noqa: E731
        elif what == "aerial_features":
            check = lambda: self._aerial_unavailable(index, features=True)  # noqa: E731
        elif kind in DEVICES and name:
            check = lambda: self._topic_unavailable(kind, name)  # noqa: E731
        else:
            raise KeyError(
                f"Unknown data {what!r}: expected a sensor name ({', '.join(ALL_SENSOR_NAMES)}), "
                "'ground_labels/<lidar>', 'instances/<camera>', 'map', 'sd_map', 'pose', 'aerial', "
                "'aerial_features' or "
                "'<gnss device>/<topic>'"
            )
        if index is not None:
            self._check_index(index)
        return check()

    def _sensor_unavailable(self, sensor: str, index: Optional[int]) -> Optional["Unavailable"]:
        from kitscenes.errors import SensorUnavailable

        if sensor not in self.sensors and self.files.storage.is_dir(sensor):
            self._refresh()  # downloaded since the scene looked
        if sensor not in self.sensors:  # every scene has all sensors: this copy lacks it (a partial download)
            return SensorUnavailable(sensor, index, Availability.NOT_DOWNLOADED,
                                     f"{self.files.storage.describe(sensor)} is not in this copy",
                                     self._download_hint(component_of(sensor)))
        if index is None:
            return None
        indices = self.frame_indices(sensor)
        pos = np.searchsorted(indices, index)
        if pos < len(indices) and indices[pos] == index:
            return None
        return SensorUnavailable(sensor, index, Availability.DROPPED,
                                 f"{self.files.storage.describe(sensor)} has no file for this frame")

    def _labels_unavailable(self, what: str, product: str, sensor: str, index: Optional[int]) -> Optional["Unavailable"]:
        """Ground labels (*product* ``GROUND_SEGMENTATION``) or instance masks of *sensor*."""
        from kitscenes.errors import LabelsUnavailable
        from kitscenes._layout import ground_label_member, instance_member

        info = self.split_info
        if info is not None and not info.has_labels:
            return LabelsUnavailable(what, index, Availability.WITHHELD_BY_SPLIT, f"the {info.name} split ships no labels")
        folder = f"{product}/{sensor}"
        if not self.files.storage.is_dir(folder):
            # Not in index.parquet, or the product is here (it downloads as one archive) without it: not in the release.
            archive = self._archive_index
            shipped = bool(archive.folder_components(folder)) if archive is not None else not self.files.storage.is_dir(product)
            if not shipped:
                covered = "the front and rear ring cameras" if product == INSTANCE_MASKS else "other lidars"
                return LabelsUnavailable(what, index, Availability.NOT_IN_RELEASE,
                                         f"the release has none for {sensor} in this scene (only for {covered})")
            return LabelsUnavailable(what, index, Availability.NOT_DOWNLOADED,
                                     f"{self.files.storage.describe(folder)} is not in this copy",
                                     self._download_hint(component_of(folder)))
        if index is None:
            return None
        member = ground_label_member(sensor, index) if product == GROUND_SEGMENTATION else instance_member(sensor, index)
        if self.files.storage.exists(member):
            return None
        return LabelsUnavailable(what, index, Availability.DROPPED, f"no {self.files.storage.describe(member)}")

    def _map_unavailable(self, member: str, lanelet2: bool = True) -> Optional["Unavailable"]:
        """Why the map file *member* can't be loaded (*lanelet2*: also check for lanelet2)."""
        from kitscenes.errors import MapUnavailable

        info = self.split_info
        if info is not None and not info.has_maps:
            return MapUnavailable("map", None, Availability.WITHHELD_BY_SPLIT, f"the {info.name} split ships no maps")
        if not self.files.storage.exists(member):
            return MapUnavailable("map", None, Availability.NOT_DOWNLOADED, f"no {self.files.storage.describe(member)}",
                                  self._download_hint("none"))
        if lanelet2:
            from kitscenes.map import _lanelet2

            ok, why = _lanelet2.available()
            if not ok:
                return MapUnavailable("map", None, Availability.NOT_INSTALLED,
                                      f"loading maps needs lanelet2>={_lanelet2.LANELET2_MIN_VERSION} ({why})",
                                      'install it with: pip install "kitscenes[map]"')
        return None

    def _pose_unavailable(self, index: Optional[int]) -> Optional["Unavailable"]:
        from kitscenes.errors import PoseUnavailable

        trajectory, why = self._pose_trajectory  # same limits as the refined one, without building it
        if trajectory is None:
            if not self.files.storage.exists(POSES):
                return PoseUnavailable("pose", index, Availability.NOT_DOWNLOADED, f"no {self.files.storage.describe(POSES)}",
                                       self._download_hint("none"))
            return PoseUnavailable("pose", index, Availability.INVALID, f"{self.files.storage.describe(POSES)}: {why}")
        if index is None:
            return None
        try:
            trajectory._check_query(np.array([self.timestamps_ns[index] / 1e9]))
        except (ExtrapolationError, InterpolationGapError) as err:
            return PoseUnavailable("pose", index, Availability.DROPPED, str(err))
        return None

    def _aerial_unavailable(self, index: Optional[int], features: bool = False, source=None) -> Optional["Unavailable"]:
        """Why this scene (or frame) has no crop in *source* (default: the crops attached to the dataset)."""
        from kitscenes.errors import AerialUnavailable

        what = "aerial_features" if features else "aerial"
        info = self.split_info
        if info is not None and not info.has_maps:
            # placed with the city's origin (kitscenes.aerial.placement_origin) if the poses are in the map frame
            from kitscenes.map.origin import CITY_ORIGINS

            city = self.info.city if self.info is not None else None
            if info.pose_frame != "local_utm":
                return AerialUnavailable(what, index, Availability.WITHHELD_BY_SPLIT,
                                         f"the {info.name} split's poses are relative to the scene's first pose, "
                                         "so its scenes cannot be placed on the aerial imagery")
            if city not in CITY_ORIGINS:
                return AerialUnavailable(what, index, Availability.WITHHELD_BY_SPLIT,
                                         f"the {info.name} split ships no map origin and the scene's city is unknown, "
                                         "so it cannot be placed on the aerial imagery")
        cache = source if source is not None else (self._aerial_features if features else self._aerial)
        if cache is None:
            from kitscenes.aerial.cache import NOT_PREPARED_MESSAGE

            message = ("no aerial feature crops attached: ds.use_aerial_features(roi=(60, 30))" if features
                       else NOT_PREPARED_MESSAGE)
            return AerialUnavailable(what, index, Availability.NOT_PREPARED, message)
        if index is None or cache._has_crop_for(self, index):
            return None
        return AerialUnavailable(what, index, Availability.DROPPED,
                                 f"no crop for this frame of scene {self.id} in {cache!r}")

    def _topic_unavailable(self, device: str, topic: str) -> Optional["Unavailable"]:
        """None if the topic file is there; an unknown topic of a device that is there is a KeyError."""
        from kitscenes.errors import SensorUnavailable
        from kitscenes.messages import list_topics, topic_member

        what = f"{device}/{topic}"
        member = topic_member(device, topic)  # KeyError for an unknown device
        info = self.split_info
        if info is not None and not info.has_gnss:
            return SensorUnavailable(what, None, Availability.WITHHELD_BY_SPLIT, f"the {info.name} split ships no GNSS")
        if self.files.storage.exists(member):
            return None
        if not self.files.storage.is_dir(device):
            return SensorUnavailable(what, None, Availability.NOT_DOWNLOADED,
                                     f"{self.files.storage.describe(device)} is not in this copy",
                                     self._download_hint(component_of(member)))
        raise KeyError(f"Unknown {device} topic {topic!r}; this scene has {list_topics(self.files.storage, device)}")

    def message_topics(self, device: str) -> list[str]:
        """Message topics of a GNSS device in this copy of the scene (empty if none)."""
        from kitscenes.messages import list_topics

        return list_topics(self.files.storage, device)

    @overload
    def messages(self, device: str, topic: str, rate: str = ..., *, missing: Literal["raise"] = ...) -> "pa.Table": ...
    @overload
    def messages(
        self, device: str, topic: str, rate: str = ..., *, missing: Literal["none"],
    ) -> Optional["pa.Table"]: ...

    def messages(self, device: str, topic: str, rate: str = "native", *, missing: Missing = "raise") -> Optional["pa.Table"]:
        """One GNSS/INS message topic as a ``pyarrow.Table`` (cached), see :mod:`kitscenes.messages`.

        Args:
            device: ``"gnss"`` or ``"gnss_ins"``.
            topic: e.g. ``"navsatfix"``, ``"imu"``, ``"insnavgeod"``, ``"pose"``; see :meth:`message_topics`.
            rate: ``"native"`` (every message, as stored in ``<device>/<topic>.parquet``)
                or ``"frame"`` (the messages assigned to frames, ``index`` = the
                frame index; topics slower than the frames have fewer rows).
            missing: ``"raise"`` or ``"none"`` (return None if unavailable).

        Returns:
            Columns ``index``, ``t_ns``, ``t_sync_ns``, ``frame_index``, then
            the message fields (:mod:`kitscenes.messages`).

        Raises:
            SensorUnavailable: ``WITHHELD_BY_SPLIT`` (test-e2e splits) or
                ``NOT_DOWNLOADED`` (the device's GNSS is not in this copy).
            KeyError: For an unknown device, or a topic the device does not have.
            ValueError: For an unknown rate.
        """
        from kitscenes.messages import check_rate, frame_rows, read_messages

        _check_missing(missing)
        check_rate(rate)
        key = (device, topic, rate)
        with self._lock:
            table = self._messages.get(key)
        if table is not None:
            return table
        if rate == "frame":  # derived from the cached native table: one read per topic
            native = self.messages(device, topic, missing=missing)
            table = None if native is None else frame_rows(native)
        else:
            err = self._topic_unavailable(device, topic)
            if err is not None:
                if missing == "none":
                    return None
                raise err
            table = read_messages(self.files.storage, device, topic)
        if table is None:
            return None
        with self._lock:
            self._messages[key] = table
        return table

    @overload
    def imu(self, frame: str = ..., *, missing: Literal["raise"] = ...) -> dict[str, np.ndarray]: ...
    @overload
    def imu(self, frame: str = ..., *, missing: Literal["none"]) -> Optional[dict[str, np.ndarray]]: ...

    def imu(self, frame: str = "imu", *, missing: Missing = "raise") -> Optional[dict[str, np.ndarray]]:
        """The INS receiver's IMU at its native rate (50 Hz).

        Args:
            frame: ``"imu"`` (as recorded) or ``"base"`` (axes of the vehicle
                base frame: the ``gnss_ins_imu`` rotation in ``calib.json`` if it
                is valid, :attr:`kitscenes.rig.Rig.imu_to_base`, else the nominal
                axes :data:`kitscenes.messages.IMU_TO_BASE`).
            missing: ``"raise"`` or ``"none"`` (return None if unavailable).

        Returns:
            ``t_ns`` (N,), ``angular_velocity`` (N, 3) rad/s and
            ``linear_acceleration`` (N, 3) m/s² (specific force: about +9.8 on
            the up axis at rest), plus ``orientation`` (N, 4) quaternion
            ``x, y, z, w`` of the receiver's attitude for ``frame="imu"``.

        Raises:
            SensorUnavailable: As :meth:`messages` (``"gnss_ins/imu"``).
        """
        from kitscenes.errors import CalibrationError
        from kitscenes.messages import IMU_TO_BASE, vectors

        if frame not in ("imu", "base"):
            raise ValueError(f"frame must be 'imu' or 'base', got {frame!r}")
        table = self.messages("gnss_ins", "imu", missing=missing)
        if table is None:
            return None
        out = {
            "t_ns": table.column("t_ns").to_numpy(),
            "angular_velocity": vectors(table, "angular_velocity"),
            "linear_acceleration": vectors(table, "linear_acceleration"),
        }
        if frame == "base":
            try:
                rotation = self.rig.imu_to_base
            except CalibrationError:  # the IMU itself does not need the calibration
                rotation = None
            rotation = IMU_TO_BASE if rotation is None else rotation
            out["angular_velocity"] = out["angular_velocity"] @ rotation.T
            out["linear_acceleration"] = out["linear_acceleration"] @ rotation.T
        else:
            out["orientation"] = np.column_stack([vectors(table, "orientation"), table.column("orientation.w").to_numpy()])
        return out

    def _check_index(self, index: int) -> None:
        if not 0 <= index < self.num_frames:
            raise IndexError(f"Frame index {index} is outside scene {self.id} ({self.num_frames} frames).")

    def _warn_once(self, key: str, message: str, *args: object) -> None:
        """Log a warning once per scene object and key."""
        with self._lock:
            if key in self._warned:
                return
            self._warned.add(key)
        logger.warning(message, *args)

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------

    @property
    def track(self) -> SceneTrack:
        """Every recorded pose (``poses.txt``) as arrays, with city, lidar and map origin; reads no sensor data.

        In the frame of :attr:`pose_frame`. ``track.pose(i)`` and iterating
        give :class:`~kitscenes.schema.EgoPose` objects; for the pose at any
        time use :meth:`ego` or ``frame.pose``.
        """
        info = self.info
        try:
            origin = self.origin
        except Unavailable:
            origin = None
        poses = self._poses
        return SceneTrack(
            scene_id=self.id, split=self.split, city=info.city if info else None, lidar=info.lidar if info else None,
            origin=origin,
            t_ns=np.array([p.t_ns for p in poses], dtype=np.int64),
            xyz=np.array([p.xyz for p in poses], dtype=np.float64).reshape(-1, 3),
            quat=np.array([p.quat for p in poses], dtype=np.float64).reshape(-1, 4),
        )

    @cached_property
    def origin(self) -> "MapOrigin":
        """Origin of the scene's map frame (``maps/origin.json``).

        Raises:
            MapUnavailable: Like :attr:`map` (but lanelet2 is not needed).
        """
        from kitscenes.map import MapOrigin

        err = self._map_unavailable(MAP_ORIGIN, lanelet2=False)
        if err is not None:
            raise err
        return MapOrigin.from_json(self.files.storage.read_bytes(MAP_ORIGIN))

    @property
    def map(self) -> "SceneMap":
        """The scene's HD map (:class:`kitscenes.map.SceneMap`), loaded on first query.

        Raises:
            MapUnavailable: With ``state`` ``WITHHELD_BY_SPLIT`` (the split
                ships no maps), ``NOT_DOWNLOADED`` (no ``maps/`` in this copy)
                or ``NOT_INSTALLED`` (no supported lanelet2).
        """
        try:
            return self.__dict__["_map"]
        except KeyError:
            pass
        err = self._map_unavailable(MAP_OSM)
        if err is not None:
            raise err
        scene_map = self.load_map(self.files.storage.root / MAP_OSM)
        self.__dict__["_map"] = scene_map
        return scene_map

    def load_map(self, osm_path: Path | str) -> "SceneMap":
        """Any Lanelet2 map (e.g. a city map, see :meth:`kitscenes.Dataset.city_maps`) in this scene's frame.

        Raises:
            MapUnavailable: If the scene has no map origin, or lanelet2 is missing.
        """
        from kitscenes.map import SceneMap

        err = self._map_unavailable(MAP_ORIGIN)
        if err is not None:
            raise err
        return SceneMap(osm_path, self.origin)

    # ------------------------------------------------------------------
    # SD map (OpenStreetMap)
    # ------------------------------------------------------------------

    def _sd_map_place(self) -> tuple[Optional[tuple["MapOrigin", Optional[str]]], Optional["Unavailable"]]:
        """``((origin, city), None)`` that places the scene on its city's SD map, or ``(None, why not)``."""
        from kitscenes.errors import MapUnavailable
        from kitscenes.map.origin import CITY_ORIGINS
        from kitscenes.sdmap.sd_map import city_of_origin

        info = self.split_info
        if info is not None and info.pose_frame == "first_pose":
            return None, MapUnavailable(
                "sd_map", None, Availability.WITHHELD_BY_SPLIT,
                f"the {info.name} split's poses are relative to the scene's first pose, "
                "so the scene cannot be placed on an SD map")
        listed = self.info
        city = listed.city if listed is not None else None
        try:
            origin = self.origin
        except MapUnavailable:
            if city not in CITY_ORIGINS:
                return None, MapUnavailable("sd_map", None, Availability.WITHHELD_BY_SPLIT,
                                            "the scene has no map origin and its city is unknown, "
                                            "so it cannot be placed on an SD map")
            origin = CITY_ORIGINS[city]
        return (origin, city or city_of_origin(origin)), None

    def _sd_map_unavailable(self) -> Optional["Unavailable"]:
        place, err = self._sd_map_place()
        return err if err is not None else self._sd_map_path(place[1])[1]

    def _sd_map_path(self, city: Optional[str]) -> tuple[Optional[Path], Optional["Unavailable"]]:
        """The SD-map file :attr:`sd_snapshot` selects for *city*, or why there is none (resolved once)."""
        key = ("_sd_map_file", city, self.sd_snapshot)
        hit = self.__dict__.get("_sd_map_file")
        if hit is not None and hit[0] == key:
            return hit[1]
        result = self._find_sd_map_path(city)
        self.__dict__["_sd_map_file"] = (key, result)
        return result

    def _find_sd_map_path(self, city: Optional[str]) -> tuple[Optional[Path], Optional["Unavailable"]]:
        from kitscenes.constants import SD_MAPS_DIR
        from kitscenes.errors import MapUnavailable
        from kitscenes.sdmap._files import SCENE, pick_snapshot, sd_map_files

        root = self._dataset_root
        if city is None:
            return None, MapUnavailable("sd_map", None, Availability.NOT_DOWNLOADED,
                                        "the scene's city is unknown; load a file with scene.load_sd_map(path)")
        if root is None:
            return None, MapUnavailable("sd_map", None, Availability.NOT_DOWNLOADED,
                                        "the scene is not in a dataset folder (<root>/data/<split>/<id>); "
                                        "load the city's file with scene.load_sd_map(path)")
        folder = root / SD_MAPS_DIR
        snapshots = sd_map_files(folder).get(city, {})
        if self.sd_snapshot == SCENE:
            listed = self.info.sd_snapshot if self.info is not None else None
            hit = pick_snapshot(snapshots, listed, city)  # None: an unlisted scene gets the latest
            wanted = listed or f"{city}'s latest snapshot"
        else:
            hit = pick_snapshot(snapshots, self.sd_snapshot, city)
            wanted = f"{city} {self.sd_snapshot}"
        if hit is None:
            have = ", ".join(snapshots) or "none"
            return None, MapUnavailable("sd_map", None, Availability.NOT_DOWNLOADED,
                                        f"no SD map {wanted} in {folder} (snapshots of {city}: {have})",
                                        f"the SD maps come with every download: kitscenes download {root}")
        return hit[1], None

    @property
    def sd_map(self) -> "SDMap":
        """The OpenStreetMap SD map of the scene (:class:`kitscenes.sdmap.SDMap`) in the frame of its poses.

        The snapshot of the scene's city and recording week,
        ``<root>/openstreetmap_sd_maps/<City>_<YYYY>-W<ww>.osm.pbf`` (the scene
        table's ``sd_snapshot``), or the one :attr:`sd_snapshot` names (e.g.
        ``kitscenes.open(root, sd_snapshot="paper_2026-05-03")`` for the
        2026-05-03 files of the paper).
        Placed with the scene's ``maps/origin.json``, or in the test and
        map_change_test splits (no maps, poses in the city frame) with the
        city's origin. One object per file, shared by every scene that uses it
        (and :meth:`kitscenes.Dataset.sd_map`), parsed on first use.

        Raises:
            MapUnavailable: ``WITHHELD_BY_SPLIT`` in the test-e2e splits (poses
                relative to the first pose), ``NOT_DOWNLOADED`` without the file.
        """
        from kitscenes.sdmap.sd_map import shared_sd_map

        place, err = self._sd_map_place()
        if err is not None:
            raise err
        origin, city = place
        path, err = self._sd_map_path(city)
        if err is not None:
            raise err
        return shared_sd_map(path, origin, city)

    def load_sd_map(self, osm_path: Path | str) -> "SDMap":
        """Any OpenStreetMap file as an :class:`kitscenes.sdmap.SDMap` in this scene's frame.

        Raises:
            MapUnavailable: ``WITHHELD_BY_SPLIT`` if the scene cannot be placed (test-e2e).
        """
        from kitscenes.sdmap.sd_map import shared_sd_map

        place, err = self._sd_map_place()
        if err is not None:
            raise err
        return shared_sd_map(osm_path, place[0], place[1])

    @property
    def sd_route(self) -> Optional["SDRoute"]:
        """The route the scene drove on the SD map (:meth:`kitscenes.sdmap.SDMap.route_along` over its
        poses), None if no pose matches a road (e.g. outside the snapshot's area). Computed once per scene.

        Raises:
            MapUnavailable: Like :attr:`sd_map`.
        """
        try:
            return self.__dict__["_sd_route"]
        except KeyError:
            pass
        smap = self.sd_map
        track = self.track
        if len(track) < 2:
            route = None
        else:
            from scipy.spatial.transform import Rotation

            yaw = Rotation.from_quat(track.quat).as_euler("ZYX")[:, 0]
            route = smap.route_along(track.xyz[:, :2], yaw)
        self.__dict__["_sd_route"] = route
        return route

    def accumulate(
        self,
        frames: Iterable[int],
        lidars: Optional[Sequence[LidarNameLike]] = None,
        *,
        frame: str = "world",
        reference: Optional[int] = None,
        voxel_size: Optional[float] = None,
        max_range: Optional[float] = None,
        device=None,
        workers: Optional[int] = None,
    ) -> "PointCloud":
        """Lidar points of several frames in one frame; see :func:`kitscenes.accumulate.accumulate` for the arguments."""
        from kitscenes.accumulate import accumulate

        return accumulate(self, frames, lidars, frame=frame, reference=reference, voxel_size=voxel_size,
                          max_range=max_range, device=device, workers=workers)

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def frame(self, index: int) -> "Frame":
        """The frame at *index* (negative indices count from the end)."""
        from kitscenes.frame import Frame

        if index < 0:
            index += self.num_frames
        return Frame(self, index)

    def __getitem__(self, index: Union[int, slice]) -> Union["Frame", list["Frame"]]:
        """``scene[i]`` is :meth:`frame` ``(i)``; ``scene[a:b]`` a list of frames."""
        if isinstance(index, slice):
            return [self.frame(i) for i in range(*index.indices(self.num_frames))]
        return self.frame(index)

    def __iter__(self) -> Iterator["Frame"]:
        for index in range(self.num_frames):
            yield self.frame(index)


def component_of(folder: str) -> str:
    """The archive component that holds a sensor or label folder (``radar_left`` -> ``radar``)."""
    from kitscenes._archive import COMPONENTS

    folder = folder.rstrip("/")
    for component, folders in COMPONENTS.items():
        if any(folder == f or folder.startswith(f + "/") for f in folders):
            return component
    return folder.split("/", 1)[0]


def _check_missing(missing: str) -> None:
    if missing not in ("raise", "none"):
        raise ValueError(f"missing must be 'raise' or 'none', got {missing!r}")
