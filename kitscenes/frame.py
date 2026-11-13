"""One frame of a scene: all sensor data captured for one reference time.

A :class:`Frame` is a view ``(scene, index)``; it holds no data. Accessors
decode on every call and return arrays the caller owns, unless the scene's
dataset was opened with a cache (then they are shared and read-only).

Conventions: times are integer nanoseconds since the scene start; points
are in the vehicle's base frame at the frame time unless ``frame=`` says
otherwise (``"road"``, ``"sensor"``, ``"world"``); everything is metres and
seconds.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Sequence, TypeVar, Union, overload

import numpy as np

from kitscenes.ego_trajectory import DEFAULT_STEP_S, EgoTrajectory, relative_trajectory, slot_times
from kitscenes.decode import _decode_reduced
from kitscenes.constants import CameraNameLike, LidarNameLike, RadarNameLike, SensorNameLike, sensor_name, sensor_names
from kitscenes.errors import Availability, Missing, Unavailable
from kitscenes.image_corrections import stored_correction, warp_image
from kitscenes._layout import ground_label_member, sensor_kind, sensor_member
from kitscenes.motion import (
    DeskewInputs,
    apply_deskew_inputs,
    compensate_radar_points,
    deskew_inputs,
    measured_point_mask,
)
from kitscenes._parquet import (
    _no_return_mask,
    load_ground_mask_parquet,
    load_lidar_columns,
    load_lidar_sweep,
    load_radar_columns,
)
from kitscenes.points import PointCloud, RadarCloud, ScanLayout
from kitscenes.rig import invert_transform, transform_points
from kitscenes.schema import EgoPose, SensorFile

if TYPE_CHECKING:
    from kitscenes.ego_trajectory import EgoState
    from kitscenes.instances import InstanceMasks
    from kitscenes.scene import Scene
    from kitscenes.navigation import CommandRule, Navigation

CoordinateFrame = Literal["base", "road", "sensor", "world"]
_FRAMES = ("base", "road", "sensor", "world")
_EGO_FRAMES = ("base", "road")
_MISSING = ("raise", "none")
_T = TypeVar("_T")


class Frame:
    """The data of one scene at one reference-timeline index.

    Args:
        scene: The scene.
        index: Frame index, ``0 <= index < scene.num_frames``.
    """

    __slots__ = ("scene", "index")

    def __init__(self, scene: "Scene", index: int) -> None:
        scene._check_index(index)
        self.scene = scene
        self.index = int(index)

    def __repr__(self) -> str:
        return f"Frame(scene={self.scene.id!r}, index={self.index}, t_ns={self.t_ns})"

    def __getstate__(self) -> tuple:
        return (self.scene, self.index)

    def __setstate__(self, state: tuple) -> None:
        self.scene, self.index = state

    # ------------------------------------------------------------------
    # Time and pose
    # ------------------------------------------------------------------

    @property
    def t_ns(self) -> int:
        """Frame time in nanoseconds since the scene start."""
        return int(self.scene.timestamps_ns[self.index])

    @property
    def t_s(self) -> float:
        return self.t_ns / 1e9

    @property
    def token(self) -> str:
        """``"<scene id>_<index:04d>"``: the frame's ID in info files, labels and submissions.

        The frame index, zero-padded to four digits: ``"17a14cb5-4d50-727b-ae2a-5c15d88be042_0007"``.
        Nothing but the scene id may be taken from it (``token.rsplit("_", 1)[0]``).
        """
        return f"{self.scene.id}_{self.index:04d}"

    @property
    def pose(self) -> EgoPose:
        """Ego pose at the frame time, in the frame of :attr:`Scene.pose_frame`.

        From :attr:`Scene.ego_motion`: interpolated between the poses, with the
        rotation refined by the gyro where the scene allows it.

        Raises:
            PoseUnavailable: ``DROPPED`` if the frame time lies too far outside
                the poses or in a gap between them, ``INVALID`` or
                ``NOT_DOWNLOADED`` if the scene has no usable poses
                (``frame.availability("pose")`` asks first).
        """
        err = self.scene._unavailable("pose", self.index)
        if err is not None:
            raise err
        rot, trans = self.scene._trajectory[0].interpolate(np.array([self.t_s]))
        return EgoPose(self.t_ns, trans[0], rot.as_quat()[0])

    @property
    def T_world_base(self) -> np.ndarray:
        """(4, 4) pose matrix at the frame time: maps base-frame points into the world frame.

        Raises:
            PoseUnavailable: As :attr:`pose`.
        """
        return self.pose.T_world_base

    @property
    def T_world_road(self) -> np.ndarray:
        """(4, 4) pose of the road frame at the frame time: maps road-frame points into the world frame.

        ``T_world_base @ invert_transform(scene.T_road_base)``: the base pose
        moved down onto the road (see :attr:`Scene.T_road_base`).

        Raises:
            PoseUnavailable: As :attr:`pose`.
        """
        return self.T_world_base @ invert_transform(self.scene.T_road_base)

    def ego(self, causal: bool = False) -> "EgoState":
        """Pose, velocity, acceleration, yaw rate and curvature at the frame time.

        The default is ``scene.ego(frame.t_ns)``: central differences, which
        read the poses up to 0.2 s after the frame. A planner must not see
        those: use ``causal=True`` for a model input.

        Args:
            causal: Use the poses up to the frame time only: the motion is
                :func:`kitscenes.e2e.past_motion` of the last 0.5 s, what
                ``sample["ego"]`` holds. Of ``angular_velocity`` only the yaw
                rate is set (roll and pitch rate are NaN); with fewer than
                0.2 s of past poses (the scene start) the motion is NaN.

        Raises:
            PoseUnavailable: As :attr:`pose`.
        """
        self._require("pose")
        if not causal:
            return self.scene.ego(self.t_ns)
        from kitscenes.e2e import EGO_WINDOW_S, past_motion
        from kitscenes.ego_trajectory import EgoState

        pose = self.pose
        motion = past_motion(self.past(EGO_WINDOW_S))
        nan = np.full(3, np.nan)
        return EgoState(
            t_ns=np.int64(self.t_ns), T_world_base=pose.T_world_base, xyz=pose.xyz, quat=pose.quat,
            velocity=nan if motion is None else motion["velocity"],
            angular_velocity=np.array([np.nan, np.nan, np.nan if motion is None else motion["yaw_rate"]]),
            acceleration=nan if motion is None else motion["acceleration"],
        )

    @overload
    def future(
        self, horizon_s: float = ..., *, step_s: float = ..., frame: str = ..., missing: Literal["raise"] = ...,
    ) -> EgoTrajectory: ...
    @overload
    def future(
        self, horizon_s: float = ..., *, step_s: float = ..., frame: str = ..., missing: Literal["none"],
    ) -> Optional[EgoTrajectory]: ...

    def future(
        self, horizon_s: float = 3.0, *, step_s: float = DEFAULT_STEP_S, frame: str = "base",
        missing: Missing = "raise",
    ) -> Optional[EgoTrajectory]:
        """Where the vehicle went in the *horizon_s* seconds after this frame, relative to its pose here.

        K = ``round(horizon_s / step_s)`` slots, slot k (0-based) at
        ``t_ns + (k + 1) * step`` (the frame itself is not a slot), from the same interpolation of the poses as :attr:`pose`; slots the
        poses do not cover (past the end of the scene, in a gap) are NaN and
        not ``valid``. See :class:`~kitscenes.ego_trajectory.EgoTrajectory`.

        Args:
            frame: ``"base"`` (default) or ``"road"``: the trajectory of that
                frame's origin, relative to that frame at :attr:`t_ns`
                (:attr:`Scene.T_road_base`).

        Raises:
            PoseUnavailable: Exactly when :attr:`pose` does (unless ``missing="none"``).
        """
        _check_choice(frame, _EGO_FRAMES, "frame")
        return _or_none(missing, lambda: self._relative_trajectory(horizon_s, step_s, frame, past=False))

    @overload
    def past(
        self, horizon_s: float = ..., *, step_s: float = ..., frame: str = ..., missing: Literal["raise"] = ...,
    ) -> EgoTrajectory: ...
    @overload
    def past(
        self, horizon_s: float = ..., *, step_s: float = ..., frame: str = ..., missing: Literal["none"],
    ) -> Optional[EgoTrajectory]: ...

    def past(
        self, horizon_s: float = 3.0, *, step_s: float = DEFAULT_STEP_S, frame: str = "base",
        missing: Missing = "raise",
    ) -> Optional[EgoTrajectory]:
        """Where the vehicle came from in the *horizon_s* seconds before this frame, relative to its pose here.

        Like :meth:`future`, backwards in time: the slots are at
        ``t_ns - K * step, ..., t_ns - step`` (the oldest first); slots
        before the first pose are not ``valid``.

        Raises:
            PoseUnavailable: Exactly when :attr:`pose` does (unless ``missing="none"``).
        """
        _check_choice(frame, _EGO_FRAMES, "frame")
        return _or_none(missing, lambda: self._relative_trajectory(horizon_s, step_s, frame, past=True))

    def _relative_trajectory(self, horizon_s: float, step_s: float, frame: str, past: bool) -> EgoTrajectory:
        times = slot_times(self.t_ns, horizon_s, step_s, past=past)  # ValueError before PoseUnavailable
        self._require("pose")
        T = invert_transform(self.scene.T_road_base) if frame == "road" else None
        return relative_trajectory(self.scene._trajectory[0], self.t_ns, times, index=self.index,
                                   T_base_frame=T, frame=frame)

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def availability(self, what: Union[SensorNameLike, str]) -> Availability:
        """Whether some data is there at this frame, and if not, why; see :meth:`Scene.availability`.

        *what* is a sensor name, ``"ground_labels/<lidar>"``,
        ``"instances/<camera>"``, ``"pose"``, ``"aerial"``, ``"aerial_features"``,
        ``"map"`` or a GNSS topic.
        """
        return self.scene.availability(sensor_name(what), self.index)

    def sensors(self, kind: Optional[str] = None) -> tuple[str, ...]:
        """Sensors with data at this frame, optionally only ``"camera"``, ``"lidar"`` or ``"radar"``."""
        return tuple(
            s for s in self.scene.sensors
            if (kind is None or sensor_kind(s) == kind) and self.availability(s) is Availability.PRESENT
        )

    def _require(self, what: str, kind: Optional[str] = None) -> str:
        """Raise the :class:`Unavailable` of *what* at this frame, if it is not there.

        Returns the member name of the sensor file (for a *kind* of sensor).

        Raises:
            KeyError: If *what* is no sensor of this *kind*.
        """
        if kind is not None and sensor_kind(what) != kind:
            raise KeyError(f"{what!r} is not a {kind}")
        err = self.scene._unavailable(what, self.index)
        if err is not None:
            raise err
        return sensor_member(what, self.index) if kind is not None else ""

    def _T_from_base(self, frame: str) -> np.ndarray:
        """``T_world_base`` or ``T_road_base`` at this frame (raises PoseUnavailable for the world without a pose)."""
        return self.T_world_base if frame == "world" else self.scene.T_road_base

    def _cached(self, key: tuple, load):
        cache = self.scene._cache
        if cache is None:
            return load()
        return cache.get_or_load((self.scene.id, self.index) + key, load)

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    @overload
    def sensor_file(self, sensor: SensorNameLike, *, missing: Literal["raise"] = ...) -> SensorFile: ...
    @overload
    def sensor_file(self, sensor: SensorNameLike, *, missing: Literal["none"]) -> Optional[SensorFile]: ...

    def sensor_file(self, sensor: SensorNameLike, *, missing: Missing = "raise") -> Optional[SensorFile]:
        """Where the stored file of a sensor (camera, lidar, radar) is at this frame, for path-based loaders.

        A :class:`~kitscenes.schema.SensorFile`: the file path of an extracted
        scene, or the zip, member name, byte offset and size in the release
        archive format; ``.read()`` gives the bytes (for a camera the same
        as :meth:`image_bytes`)::

            f = frame.sensor_file("camera_ring_front")
            f.path, f.offset, f.size, f.zip_member   # e.g. .../camera_ring_front.zip, 1638400, 912345, "<scene>/camera_ring_front/0000000042.jpg"

        Raises:
            SensorUnavailable: If the sensor has no file here (``DROPPED``,
                ``NOT_DOWNLOADED``, ...; unless ``missing="none"``).
            KeyError: For a name that is no sensor.
            TypeError: For a scene storage without files on disk.
        """
        sensor = sensor_name(sensor)
        kind = sensor_kind(sensor)

        def load() -> SensorFile:
            member = self._require(sensor, kind)
            locate = getattr(self.scene.files.storage, "locate", None)
            if locate is None:
                raise TypeError(f"{type(self.scene.files.storage).__name__} keeps no files on disk; it cannot locate {member}")
            try:
                path, offset, size, zip_member = locate(member)
            except FileNotFoundError:  # deleted since the scene looked: say why, as for any missing file
                self.scene._refresh()
                err = self.scene._unavailable(sensor, self.index)
                if err is not None:
                    raise err from None
                raise
            return SensorFile(sensor, self.index, member, path, offset, size, zip_member)

        return _or_none(missing, load)

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    @overload
    def image_bytes(self, camera: CameraNameLike, *, missing: Literal["raise"] = ...) -> bytes: ...
    @overload
    def image_bytes(self, camera: CameraNameLike, *, missing: Literal["none"]) -> Optional[bytes]: ...

    def image_bytes(self, camera: CameraNameLike, *, missing: Missing = "raise") -> Optional[bytes]:
        """The encoded JPEG, e.g. for decoding on a GPU.

        It is the stored file, so without the image correction of :meth:`image`
        (:func:`kitscenes.decode.decode_cameras` applies it).

        Raises:
            SensorUnavailable: If the camera has no image here (unless ``missing="none"``).
        """
        camera = sensor_name(camera)
        return _or_none(missing, lambda: self._image_bytes(camera))

    def _image_bytes(self, camera: str) -> bytes:
        self._require(camera, "camera")
        return self.scene.files.storage.read_bytes(sensor_member(camera, self.index))

    @overload
    def image(
        self, camera: CameraNameLike, *, reduce: int = ..., image_correction: bool = ..., missing: Literal["raise"] = ...,
    ) -> np.ndarray: ...
    @overload
    def image(
        self, camera: CameraNameLike, *, reduce: int = ..., image_correction: bool = ..., missing: Literal["none"],
    ) -> Optional[np.ndarray]: ...

    def image(
        self, camera: CameraNameLike, *, reduce: int = 1, image_correction: bool = True, missing: Missing = "raise",
    ) -> Optional[np.ndarray]:
        """The decoded image as an (H, W, 3) uint8 RGB array.

        Args:
            camera: Camera name.
            reduce: Decode at 1/2, 1/4 or 1/8 size (JPEG DCT scaling: several
                times faster than decoding at full size and resizing). Project
                into it with ``scene.rig.cameras[camera].scaled(1 / reduce)``.
                Relative to the stored image: for a camera transcoded to 1/2
                (``scene.files.image_variants``) ``reduce=2`` gives 1/4
                of the release's size, as ``scene.rig`` describes the stored one.
            image_correction: Warp the image onto the calibration if the scene ships
                a correction for this camera (:mod:`kitscenes.image_corrections`).
                False gives the stored pixels.
            missing: ``"raise"`` or ``"none"`` (return None if unavailable).

        Raises:
            SensorUnavailable: If the camera has no image here (unless ``missing="none"``).
            ImageSizeError: If the image does not have the size of ``scene.rig``.
        """
        if reduce not in (1, 2, 4, 8):
            raise ValueError("reduce must be 1, 2, 4 or 8")
        camera = sensor_name(camera)
        data = self.image_bytes(camera, missing=missing)
        if data is None:
            return None
        correct = image_correction and camera in self.scene.files.image_corrections
        return self._cached(("image", camera, reduce, correct), lambda: self._decoded(camera, data, reduce, correct))

    def _decoded(self, camera: str, data: bytes, reduce: int, correct: bool) -> np.ndarray:
        image = self._check_size(camera, _decode_reduced(data, reduce), reduce)
        corr = stored_correction(self.scene, camera) if correct else None
        if corr is None or corr.is_identity:
            return image
        return warp_image(image, corr.scaled(1 / reduce))

    def _check_size(self, camera: str, image: np.ndarray, reduce: int) -> np.ndarray:
        """*image* if it has the size of ``scene.rig`` (reduced), else :class:`~kitscenes.errors.ImageSizeError`."""
        from kitscenes.errors import CalibrationError, ImageSizeError

        try:
            calibrated = self.scene.rig.cameras.get(camera)
        except CalibrationError:  # reported where the rig is used
            return image
        if calibrated is None:
            return image
        w, h = calibrated.image_size
        want = (-(-h // reduce), -(-w // reduce))
        if image.shape[:2] != want:
            got = f"{image.shape[1] * reduce}x{image.shape[0] * reduce}"
            variant = self.scene.files.image_variants.get(camera)
            raise ImageSizeError(
                f"Scene {self.scene.id}: {camera} image {self.index} is {got}, but the calibration says {w}x{h}"
                + (f" (transcoded {variant.recipe})" if variant else "")
                + ". Images resized outside kitscenes have no record of it, so projections into them would be wrong: "
                "re-encode them with 'kitscenes transcode', or download the release's images again.")
        return image

    # ------------------------------------------------------------------
    # Lidar
    # ------------------------------------------------------------------

    @overload
    def lidar(
        self, name: LidarNameLike, *, frame: CoordinateFrame = ..., deskew: bool = ..., rows: bool = ...,
        labels: bool = ..., no_returns: bool = ..., missing: Literal["raise"] = ...,
    ) -> PointCloud: ...
    @overload
    def lidar(
        self, name: LidarNameLike, *, frame: CoordinateFrame = ..., deskew: bool = ..., rows: bool = ...,
        labels: bool = ..., no_returns: bool = ..., missing: Literal["none"],
    ) -> Optional[PointCloud]: ...

    def lidar(
        self,
        name: LidarNameLike,
        *,
        frame: CoordinateFrame = "base",
        deskew: bool = True,
        rows: bool = False,
        labels: bool = False,
        no_returns: bool = False,
        missing: Missing = "raise",
    ) -> Optional[PointCloud]:
        """One lidar sweep: the measured points, in file order.

        Args:
            name: Lidar name.
            frame: ``"base"`` (vehicle frame at :attr:`t_ns`), ``"road"``
                (:attr:`Scene.T_road_base`), ``"sensor"``, or ``"world"``
                (the scene's world frame, through :attr:`pose`).
            deskew: Compensate the ego motion during the sweep
                (:mod:`kitscenes.motion`), with the poses refined by the gyro
                or the poses alone (:attr:`kitscenes.Scene.ego_motion` says
                which). Without usable poses the points are only transformed
                (``motion_compensated=False``, ``reason`` says why; logged
                once per scene).
            rows: Return every stored row instead of the measured points
                only. Rows that hold no point stay at the origin with
                :attr:`PointCloud.measured` False: OT128 slots that never
                fired (a quarter of the rows) or near-field returns the
                extraction zeroed, empty Velodyne grid cells.
            labels: Add the ground-segmentation labels as a bool column
                ``ground`` (``cloud["ground"]``), aligned with the cloud's
                rows. If the lidar has a sweep but no labels here, this raises
                :class:`~kitscenes.errors.LabelsUnavailable`; with
                ``missing="none"`` the cloud then comes without the column.
            no_returns: Keep the no-return rows too (needs ``rows=True``):
                fired rays that returned nothing (Hesai and Velodyne; stored
                as (0, 0, 0) with reflectivity -1 and their real firing
                time). They stay at the origin and are flagged by
                :attr:`PointCloud.no_return`.
            missing: ``"raise"`` or ``"none"`` (return None if the lidar has no sweep here).

        Returns:
            The cloud; ``attrs["row"]`` is each point's row in the stored
            sweep and ``cloud.layout`` maps rows to (ring, column, return)
            for the Hesai and Velodyne lidars (see
            :class:`~kitscenes.points.ScanLayout`, :meth:`PointCloud.range_image`).

        Raises:
            SensorUnavailable: If the lidar has no sweep here (unless ``missing="none"``).
            LabelsUnavailable: For ``labels=True`` without ground labels (unless ``missing="none"``).
            PoseUnavailable: For ``frame="world"`` without a pose at this frame (unless ``missing="none"``).
            CalibrationError: If the scene's ``calib.json`` is malformed or lacks a sensor (see :attr:`kitscenes.Scene.rig`).
            ValueError: For ``no_returns=True`` without ``rows=True``.
        """
        _check_choice(frame, _FRAMES, "frame")
        _check_rows(rows, no_returns)
        name = sensor_name(name)
        return _or_none(missing, lambda: self._lidar(name, frame, deskew, rows, labels, no_returns, missing))

    def _lidar(
        self, name: str, frame: str, deskew: bool, rows: bool = False, labels: bool = False,
        no_returns: bool = False, missing: str = "raise",
    ) -> PointCloud:
        self._require(name, "lidar")
        if frame in ("world", "road"):
            T = self._T_from_base(frame)  # raises PoseUnavailable before anything is read
            return self._lidar(name, "base", deskew, rows, labels, no_returns, missing).transformed(T, frame)
        member = sensor_member(name, self.index)
        ground = self._ground_or_none(name, no_returns, missing) if labels else None
        if ground is None:
            def load() -> PointCloud:
                cloud = self._load_lidar(name, member, frame, deskew, no_returns)
                return cloud if rows else cloud.compact()

            return self._cached(("lidar", name, frame, deskew, rows, no_returns), load)
        # The labels have one entry per stored row, so they are attached before the rows without a point go.
        cloud = self._cached(
            ("lidar", name, frame, deskew, True, no_returns),
            lambda: self._load_lidar(name, member, frame, deskew, no_returns),
        )
        cloud = dataclasses.replace(cloud, attrs={**cloud.attrs, "ground": ground})
        return cloud if rows else cloud.compact()

    def _ground_or_none(self, name: str, no_returns: bool, missing: str) -> Optional[np.ndarray]:
        """The ground labels of the stored rows; None without labels for ``missing="none"``."""
        from kitscenes.errors import LabelsUnavailable

        try:
            return self._ground_mask(name, no_returns)
        except LabelsUnavailable:
            if missing == "none":
                return None
            raise

    def _lidar_inputs(
        self, name: str, member: str, deskew: bool, no_returns: bool = False,
    ) -> tuple[DeskewInputs, dict, np.ndarray, Optional[ScanLayout]]:
        cols, layout = load_lidar_sweep(self.scene.files.storage.read_bytes(member), drop_no_returns=not no_returns)
        xyz = np.column_stack([cols.pop("x"), cols.pop("y"), cols.pop("z")]).astype(np.float32, copy=False)
        T = self.scene.rig.sensor(name).T_base_sensor
        trajectory, unavailable = self.scene._trajectory if deskew else (None, "deskewing disabled")
        inputs = deskew_inputs(xyz, cols.get("timestamp"), T, trajectory, self.t_s, unavailable_reason=unavailable)
        return inputs, cols, T, layout

    def _load_lidar(self, name: str, member: str, frame: str, deskew: bool, no_returns: bool) -> PointCloud:
        """The stored rows of a sweep in the base or the sensor frame."""
        inputs, cols, T, layout = self._lidar_inputs(name, member, deskew, no_returns)
        if deskew and not inputs.deskewed:
            self.scene._warn_once(
                f"deskew:{inputs.reason}",
                "Lidar sweeps of scene %s are not motion compensated (%s); points are only transformed.",
                self.scene.id, inputs.reason,
            )
        if frame == "sensor" and not inputs.deskewed:
            xyz = inputs.xyz.copy()
        else:
            xyz = apply_deskew_inputs(inputs)
            if frame == "sensor":
                xyz = transform_points(invert_transform(T), xyz)
                xyz[~inputs.measured] = 0.0
        return PointCloud(
            xyz=xyz, attrs=cols, measured=inputs.measured, frame=frame, t_ns=self.t_ns, sensor=name,
            motion_compensated=inputs.deskewed, reason=inputs.reason, layout=layout,
        )

    @overload
    def lidar_deskew_inputs(
        self, name: LidarNameLike, *, frame: str = ..., missing: Literal["raise"] = ...,
    ) -> DeskewInputs: ...
    @overload
    def lidar_deskew_inputs(
        self, name: LidarNameLike, *, frame: str = ..., missing: Literal["none"],
    ) -> Optional[DeskewInputs]: ...

    def lidar_deskew_inputs(
        self, name: LidarNameLike, *, frame: str = "base", missing: Missing = "raise",
    ) -> Optional[DeskewInputs]:
        """Raw points plus per-point transforms, to deskew elsewhere (e.g. on a GPU).

        Apply with :func:`kitscenes.motion_torch.deskew_on_device` or
        :func:`kitscenes.motion.apply_deskew_inputs`; the result equals
        ``lidar(name, frame=frame, rows=True).xyz``. Cheap compared to deskewing itself.

        Args:
            frame: ``"base"`` or ``"road"``: the frame the table maps into.

        Raises:
            SensorUnavailable: If the lidar has no sweep here (unless ``missing="none"``).
        """
        _check_choice(frame, _EGO_FRAMES, "frame")
        name = sensor_name(name)

        def load() -> DeskewInputs:
            self._require(name, "lidar")
            inputs = self._lidar_inputs(name, sensor_member(name, self.index), True)[0]
            if frame == "road":  # a pure z shift: add it to the translation of every row
                table = inputs.table.copy()
                table[:, 11] += np.float32(self.scene.road_height_m)
                inputs = dataclasses.replace(inputs, table=table)
            return inputs

        return _or_none(missing, load)

    @overload
    def fused_lidar(
        self, names: Optional[Sequence[LidarNameLike]] = ..., *, frame: CoordinateFrame = ..., deskew: bool = ...,
        labels: bool = ..., missing: Literal["raise"] = ...,
    ) -> PointCloud: ...
    @overload
    def fused_lidar(
        self, names: Optional[Sequence[LidarNameLike]] = ..., *, frame: CoordinateFrame = ..., deskew: bool = ...,
        labels: bool = ..., missing: Literal["none"],
    ) -> Optional[PointCloud]: ...

    def fused_lidar(
        self,
        names: Optional[Sequence[LidarNameLike]] = None,
        *,
        frame: CoordinateFrame = "base",
        deskew: bool = True,
        labels: bool = False,
        missing: Missing = "raise",
    ) -> Optional[PointCloud]:
        """Several lidars fused into one cloud of measured points.

        The cloud has no ``layout``; ``attrs["row"]`` still gives each
        point's row in its stored sweep. ``sensor`` lists the fused lidars
        and ``source_index`` points into it; ``motion_compensated`` is True
        only if all were.

        Args:
            names: Lidars to fuse; by default all with a sweep at this frame.
            frame: ``"base"``, ``"road"`` or ``"world"`` (see :meth:`lidar`).
            labels: Add the ground labels as a bool column ``ground``, as in
                :meth:`lidar`. If one of the lidars has no labels here, this
                raises :class:`~kitscenes.errors.LabelsUnavailable`; with
                ``missing="none"`` the cloud then comes without the column.
            missing: ``"none"`` leaves out the lidars without a sweep here
                (None if that is all of them, or there is no pose for
                ``frame="world"``); ``"raise"`` (default) raises for them.

        Raises:
            SensorUnavailable: If a requested lidar has no sweep here.
            LabelsUnavailable: For ``labels=True`` if a lidar has no ground labels (unless ``missing="none"``).
            PoseUnavailable: For ``frame="world"`` without a pose at this frame.
        """
        _check_choice(frame, ("base", "road", "world"), "frame")
        _check_choice(missing, _MISSING, "missing")
        names = self.sensors("lidar") if names is None else sensor_names(names)
        if missing == "none":
            names = tuple(n for n in names if self.availability(n) is Availability.PRESENT)
            if not names:
                return None
        return _or_none(missing, lambda: self._fused(names, frame, deskew, labels, missing))

    def _fused(
        self, names: Sequence[str], frame: str, deskew: bool, labels: bool = False, missing: str = "raise",
    ) -> PointCloud:
        from kitscenes._fusion import fuse_point_clouds

        T = self._T_from_base(frame) if frame != "base" else None
        if labels:
            for name in names:
                self._require(name, "lidar")
            labels = all(self._ground_or_none(name, False, missing) is not None for name in names)
        fused = fuse_point_clouds([self._lidar(n, "base", deskew, labels=labels) for n in names], t_ns=self.t_ns)
        return fused if T is None else fused.transformed(T, frame)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    @overload
    def ground_mask(
        self, lidar: LidarNameLike, *, rows: bool = ..., no_returns: bool = ..., missing: Literal["raise"] = ...,
    ) -> np.ndarray: ...
    @overload
    def ground_mask(
        self, lidar: LidarNameLike, *, rows: bool = ..., no_returns: bool = ..., missing: Literal["none"],
    ) -> Optional[np.ndarray]: ...

    def ground_mask(
        self, lidar: LidarNameLike, *, rows: bool = False, no_returns: bool = False, missing: Missing = "raise",
    ) -> Optional[np.ndarray]:
        """Ground-segmentation labels of a sweep: (N,) bool, aligned with ``frame.lidar(lidar)``.

        The mask lines up with the cloud of :meth:`lidar` called with the
        same *rows* and *no_returns*; ``frame.lidar(lidar, labels=True)``
        gives the same labels as the cloud's ``ground`` column.

        Args:
            rows: One label per stored row (``lidar(lidar, rows=True)``)
                instead of one per measured point. The label of a row
                without a point means nothing.
            no_returns: Align with ``lidar(lidar, rows=True, no_returns=True)``
                (needs ``rows=True``): the no-return rows have no label and get False.
            missing: ``"raise"`` or ``"none"`` (return None if unavailable).

        Raises:
            LabelsUnavailable: With ``state`` ``WITHHELD_BY_SPLIT`` (test
                splits), ``NOT_IN_RELEASE`` (a lidar without labels),
                ``NOT_DOWNLOADED``, ``DROPPED``, or ``INVALID`` if the labels
                fit no frame offset or not this sweep's point count (unless
                ``missing="none"``).
            SensorUnavailable: If the lidar itself has no sweep here (unless ``missing="none"``).
            ValueError: For ``no_returns=True`` without ``rows=True``.
        """
        _check_rows(rows, no_returns)
        lidar = sensor_name(lidar)
        return _or_none(missing, lambda: self._ground_mask(lidar, no_returns, rows))

    def _ground_mask(self, lidar: str, no_returns: bool = False, rows: bool = True) -> np.ndarray:
        from kitscenes.errors import LabelsUnavailable

        self._require(lidar, "lidar")
        what = f"ground_labels/{lidar}"
        self._require(what)
        member = sensor_member(lidar, self.index)

        def load() -> np.ndarray:
            label = ground_label_member(lidar, self.index)
            mask = load_ground_mask_parquet(self.scene.files.storage.read_bytes(label))
            cols = load_lidar_columns(
                self.scene.files.storage.read_bytes(member), drop_no_returns=False, columns=("x", "y", "z", "reflectivity"),
            )
            labelled = ~_no_return_mask(cols["reflectivity"], cols["x"], cols["y"], cols["z"])
            if len(mask) != int(labelled.sum()):
                raise LabelsUnavailable(
                    what, self.index, Availability.INVALID, f"{len(mask)} labels for {int(labelled.sum())} points")
            if not rows:
                xyz = np.column_stack([cols["x"], cols["y"], cols["z"]]).astype(np.float32, copy=False)
                return mask[measured_point_mask(xyz[labelled])]
            if not no_returns:
                return mask
            full = np.zeros(len(labelled), dtype=bool)  # no-return rows: no label
            full[labelled] = mask
            return full

        return self._cached(("ground", lidar, rows, no_returns), load)

    @overload
    def instances(self, camera: CameraNameLike, *, missing: Literal["raise"] = ...) -> "InstanceMasks": ...
    @overload
    def instances(self, camera: CameraNameLike, *, missing: Literal["none"]) -> Optional["InstanceMasks"]: ...

    def instances(self, camera: CameraNameLike, *, missing: Missing = "raise") -> Optional["InstanceMasks"]:
        """Instance masks of a camera image (seamseg on Mapillary Vistas; the front and
        rear ring cameras only), see :mod:`kitscenes.instances`.

        The masks have the release's image size (``masks.image_size``). For a
        camera whose images were downscaled (``scene.files.image_variants``)
        get them at the stored size with ``masks.mask(i, size=...)`` /
        ``masks.label_images(size=...)`` (``size=scene.rig.cameras[camera].image_size``).

        Raises:
            LabelsUnavailable: With ``state`` ``WITHHELD_BY_SPLIT`` (test splits),
                ``NOT_IN_RELEASE`` (cameras without masks), ``NOT_DOWNLOADED`` or
                ``DROPPED`` (unless ``missing="none"``).
        """
        from kitscenes.instances import InstanceMasks
        from kitscenes._layout import instance_member

        camera = sensor_name(camera)
        if sensor_kind(camera) != "camera":
            raise KeyError(f"{camera!r} is not a camera")

        def load() -> "InstanceMasks":
            self._require(f"instances/{camera}")
            return InstanceMasks.from_json(self.scene.files.storage.read_bytes(instance_member(camera, self.index)), camera)

        return _or_none(missing, load)

    # ------------------------------------------------------------------
    # Radar
    # ------------------------------------------------------------------

    @overload
    def radar(
        self, name: RadarNameLike, *, frame: CoordinateFrame = ..., compensate: bool = ..., missing: Literal["raise"] = ...,
    ) -> RadarCloud: ...
    @overload
    def radar(
        self, name: RadarNameLike, *, frame: CoordinateFrame = ..., compensate: bool = ..., missing: Literal["none"],
    ) -> Optional[RadarCloud]: ...

    def radar(
        self,
        name: RadarNameLike,
        *,
        frame: CoordinateFrame = "base",
        compensate: bool = True,
        missing: Missing = "raise",
    ) -> Optional[RadarCloud]:
        """One radar cloud, ego-motion compensated to the frame time.

        Positions are moved from the capture time (~19 ms before the frame) to
        :attr:`t_ns`, and ``range_rate`` gets the ego radial velocity removed;
        see :func:`kitscenes.motion.compensate_radar_points`. ``range``,
        ``azimuth`` and ``elevation`` stay relative to the radar.

        Args:
            frame: ``"base"``, ``"road"``, ``"sensor"`` or ``"world"`` (see :meth:`lidar`).
            compensate: Compensate positions and Doppler.
            missing: ``"raise"`` or ``"none"`` (return None if unavailable).

        Raises:
            SensorUnavailable: If the radar has no cloud here (unless ``missing="none"``).
            PoseUnavailable: For ``frame="world"`` without a pose at this frame (unless ``missing="none"``).
            CalibrationError: If the scene's ``calib.json`` is malformed or lacks a sensor (see :attr:`kitscenes.Scene.rig`).
        """
        _check_choice(frame, _FRAMES, "frame")
        name = sensor_name(name)
        return _or_none(missing, lambda: self._radar(name, frame, compensate))

    def _radar(self, name: str, frame: str, compensate: bool) -> RadarCloud:
        self._require(name, "radar")
        if frame in ("world", "road"):
            T = self._T_from_base(frame)
            return self._radar(name, "base", compensate).transformed(T, frame)
        member = sensor_member(name, self.index)
        return self._cached(("radar", name, frame, compensate), lambda: self._load_radar(name, member, frame, compensate))

    def _load_radar(self, name: str, member: str, frame: str, compensate: bool) -> RadarCloud:
        cols = load_radar_columns(self.scene.files.storage.read_bytes(member))
        raw = np.empty(len(cols["x"]), dtype=[(k, v.dtype) for k, v in cols.items()])
        for k, v in cols.items():
            raw[k] = v
        T = self.scene.rig.sensor(name).T_base_sensor
        trajectory, why = self.scene._trajectory
        if compensate and trajectory is not None:
            out, positions_ok, doppler_ok, reason = compensate_radar_points(
                raw, T, trajectory, self.t_s, target_frame=frame,
            )
        else:
            # Only the extrinsic (compensate_radar_points would return sensor-frame points here).
            reason = why if compensate else "compensation disabled"
            out, positions_ok, doppler_ok = raw, False, False
            if frame == "base":
                xyz_sensor = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float64)
                xyz_base = transform_points(T, xyz_sensor)
                xyz_base[np.linalg.norm(xyz_sensor, axis=1) <= 1e-6] = 0.0  # unmeasured rows stay at 0
                out = raw.copy()
                for i, axis in enumerate("xyz"):
                    out[axis] = xyz_base[:, i]
        if compensate and reason is not None:
            self.scene._warn_once(
                f"radar:{reason.split(':', 1)[0]}",
                "Radar clouds of scene %s are not fully compensated (%s; first: %s).", self.scene.id, reason, name,
            )
        xyz = np.column_stack([out["x"], out["y"], out["z"]]).astype(np.float32)
        attrs = {k: np.array(out[k]) for k in cols if k not in ("x", "y", "z")}
        measured = np.linalg.norm(np.column_stack([cols["x"], cols["y"], cols["z"]]), axis=1) > 1e-6
        capture = None
        if "timestamp" in cols and len(cols["timestamp"]):
            capture = int(round(float(np.median(cols["timestamp"])) * 1e9))
        return RadarCloud(
            xyz=xyz, attrs=attrs, measured=measured, frame=frame, t_ns=self.t_ns, sensor=name,
            motion_compensated=positions_ok, reason=reason,
            doppler_compensated=doppler_ok, capture_t_ns=capture,
        )

    # ------------------------------------------------------------------
    # Navigation (kitscenes.navigation)
    # ------------------------------------------------------------------

    def navigation(self, rule: "CommandRule", *, source: str = "auto", targets: str = "goal",
                   goal: str = "sd_route", max_distance_m: float = 50.0,
                   spacing_m: Optional[float] = None) -> "Navigation":
        """The driving command and target points of this frame by *rule* (:mod:`kitscenes.navigation`).

        Args:
            rule: A :class:`~kitscenes.navigation.CommandRule`, e.g.
                ``kitscenes.navigation.NAVSIM``, ``NUSCENES`` or ``KITSCENES`` (junction-aware, needs the SD map).
            source: ``"path"`` (the recorded path), ``"sd_route"`` (the driven
                route on the SD map, :attr:`Scene.sd_route`) or ``"auto"``
                (the SD route for maneuver rules, else the path).
            targets: ``"goal"`` (default): one target point, the end of the
                scene's route, the same for every frame: a destination as in
                route-following benchmarks, without the route's shape. ``"route"``:
                route points up to *max_distance_m* ahead (the exits of the turns and
                the point at the lookahead, or every *spacing_m*), plus the route
                polyline and the next maneuver. These follow the driven route, so
                within the prediction horizon they reveal the future path: use them
                for analysis, not as a model input for trajectory prediction.
            goal: Where the goal point comes from (``targets="goal"``), the same for every rule.
                ``"sd_route"`` (default): the end of the driven SD route on the OSM road
                centre line, a route-level destination as a navigation system gives it (no
                lane, no exact stopping point; empty where the scene has no SD route).
                ``"path"``: the exact last recorded position (lane-level).
            max_distance_m: How far ahead the target points and the route reach (``"route"`` only).
            spacing_m: Target points every *spacing_m* instead of at the maneuvers and the end (``"route"`` only).

        Returns:
            A :class:`~kitscenes.navigation.Navigation`; its command is UNKNOWN
            where the rule cannot tell (the path ends within the lookahead).

        Raises:
            PoseUnavailable: As :attr:`pose`.
            MapUnavailable: For the SD route in a scene that cannot be placed on the SD map (test-e2e).
        """
        from kitscenes.navigation import frame_navigation

        return frame_navigation(self, rule, source=source, targets=targets, goal=goal,
                                max_distance_m=max_distance_m, spacing_m=spacing_m)

    # ------------------------------------------------------------------
    # Aerial imagery
    # ------------------------------------------------------------------

    @overload
    def aerial(self, *, missing: Literal["raise"] = ...) -> np.ndarray: ...
    @overload
    def aerial(self, *, missing: Literal["none"]) -> Optional[np.ndarray]: ...

    def aerial(self, *, missing: Missing = "raise") -> Optional[np.ndarray]:
        """A shortcut for ``crops[frame]`` with the crops attached by ``crops = ds.use_aerial(...)``
        (:meth:`kitscenes.Dataset.use_aerial`): the ego-aligned aerial crop of this frame, (H, W, 3) uint8 RGB.

        The ego position (:attr:`pose`) is the centre pixel, forward points to
        +column (right), left to -row (up). The crops are cut on the fly from
        the raw tiles (:class:`kitscenes.aerial.AerialCrops`) or read from a
        prepared crop cache (:meth:`kitscenes.Dataset.prepare_aerials`).

        Args:
            missing: ``"raise"`` or ``"none"`` (return None if this frame has no crop).

        Raises:
            AerialUnavailable: With ``state`` ``NOT_PREPARED`` (no cache
                attached), ``WITHHELD_BY_SPLIT`` (test-e2e splits: poses
                relative to the first pose) or ``DROPPED`` (no crop for this frame), unless
                ``missing="none"``.
        """
        def load() -> np.ndarray:
            source = self.scene._aerial
            if source is None:
                self._require("aerial")  # raises: withheld by the split, or nothing attached
            return source.at(self)

        return _or_none(missing, load)

    @overload
    def aerial_features(self, *, missing: Literal["raise"] = ...) -> Any: ...
    @overload
    def aerial_features(self, *, missing: Literal["none"]) -> Optional[Any]: ...

    def aerial_features(self, *, missing: Missing = "raise") -> Optional[Any]:
        """A shortcut for ``crops[frame]`` with the crops attached by ``crops = ds.use_aerial_features(...)``
        (:meth:`kitscenes.Dataset.use_aerial_features`): the aerial DINOv3 features around this frame's pose.

        ``orientation="north"`` (the default): a :class:`kitscenes.aerial.NorthUpFeatures`,
        the stored north-up tokens with the pose and helpers to place them in
        the ego frame. ``"ego"``: an (L, C, H, W) float32 crop with the
        geometry of :meth:`aerial` (ego at the centre, forward to +column, left
        to -row), the tokens resampled (not re-encoded) into the ego frame.
        See ``docs/aerial_features.md``.

        Args:
            missing: ``"raise"`` or ``"none"`` (return None if this frame has no crop).

        Raises:
            AerialUnavailable: With ``state`` ``NOT_PREPARED`` (nothing attached),
                ``WITHHELD_BY_SPLIT`` (test-e2e splits) or ``DROPPED`` (no crop
                for this frame), unless ``missing="none"``.
        """
        def load() -> Any:
            source = self.scene._aerial_features
            if source is None:
                self._require("aerial_features")  # raises: withheld by the split, or nothing attached
            return source.at(self)

        return _or_none(missing, load)


def _check_choice(value: str, choices: tuple[str, ...], name: str) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def _check_rows(rows: bool, no_returns: bool) -> None:
    if no_returns and not rows:
        raise ValueError("no_returns=True needs rows=True: the no-return rows are rows without a point.")


def _or_none(missing: str, load: Callable[[], _T]) -> Optional[_T]:
    """``load()``; for ``missing="none"`` None instead of an :class:`Unavailable`."""
    _check_choice(missing, _MISSING, "missing")
    try:
        return load()
    except Unavailable:
        if missing == "none":
            return None
        raise
