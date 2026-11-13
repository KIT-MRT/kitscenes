"""Small shared data types: :class:`EgoPose`, :class:`SensorFile` and :class:`Provenance`."""

from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from kitscenes._repr import compact_repr


class Provenance(str, Enum):
    """How a piece of derived data came to exist.

    Every derived-data type in the API (fused point clouds, map projections,
    depth, trajectories, ...) should carry one of these so downstream
    benchmark code can't mistake a map projection or a model prediction for
    human-annotated ground truth.
    """

    SOURCE = "source" # Recorded directly by a vehicle sensor
    MAP_DERIVED = "map_derived" # Projected from the Lanelet2 HD map (e.g. lane centerlines)
    GEOMETRY_DERIVED = "geometry_derived" # Computed from calibration and geometry (e.g. depth, point cloud)
    MODEL_DERIVED = "model_derived" # Produced by a machine learning model (e.g. segmentation, detection, tracking)


@dataclasses.dataclass(frozen=True, eq=False)
class EgoPose:
    """Pose of the vehicle's base frame at one time: ``T_world_base``.

    The same names as :class:`kitscenes.EgoState` (``xyz``, ``quat``,
    ``T_world_base``), which adds the motion.

    Attributes:
        t_ns: Time in nanoseconds since the scene start.
        xyz: (3,) x, y, z position in metres in the scene's world frame. In
            scenes with maps that is the map's local UTM frame (relative to
            ``maps/origin.json``); in the test-e2e splits it is relative to
            the first pose (see :attr:`kitscenes.splits.SplitInfo.pose_frame`).
        quat: (4,) orientation quaternion (qx, qy, qz, qw).
    """

    __repr__ = compact_repr

    t_ns: int
    xyz: np.ndarray  # (3,) float64, read-only
    quat: np.ndarray  # (4,) float64, quaternion, read-only

    def __post_init__(self) -> None:
        # Poses are shared by every user of a cached scene: read-only copies.
        for name in ("xyz", "quat"):
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            array.flags.writeable = False
            object.__setattr__(self, name, array)

    @property
    def T_world_base(self) -> np.ndarray:
        """(4, 4) transform from the base frame into the world frame (a new array)."""
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(self.quat).as_matrix()
        T[:3, 3] = self.xyz
        return T

    @property
    def yaw(self) -> float:
        """Heading in radians: the direction of the base x axis in the world x-y plane.

        ``atan2(R[1, 0], R[0, 0])``, the yaw of the intrinsic z-y'-x''
        (yaw, pitch, roll) decomposition ``R = Rz(yaw) Ry(pitch) Rx(roll)``.
        """
        return float(Rotation.from_quat(self.quat).as_euler("ZYX")[0])


@dataclasses.dataclass(frozen=True)
class SensorFile:
    """Where the stored file of one sensor frame is (:meth:`kitscenes.Frame.sensor_file`).

    For path-based loaders and exporters: the bytes of the file (a camera's
    JPEG, a lidar's or radar's parquet) are ``size`` bytes at ``offset`` in
    ``path``. In an extracted scene ``path`` is the file itself (offset 0);
    in the release archive format it is the component's zip (uncompressed:
    the bytes are stored as they are, at the offset from ``index.parquet``,
    for cameras and lidars a multiple of 4096) and ``zip_member`` the name of
    the member in it. Plain data: it pickles, and :meth:`read` needs no scene.

    Attributes:
        sensor: Sensor name.
        index: Frame index.
        member: Path in the scene folder, e.g. ``camera_ring_front/0000000042.jpg``.
        path: The file holding the bytes: the extracted file, or ``<scene>/<component>.zip``.
        offset: First byte of the data in :attr:`path` (0 for an extracted file).
        size: Number of bytes.
        zip_member: Name of the member in the zip (``<scene_id>/<member>``), None for an extracted file.
    """

    sensor: str
    index: int
    member: str
    path: Path
    offset: int
    size: int
    zip_member: Optional[str] = None

    @property
    def in_zip(self) -> bool:
        """Whether the bytes are read from a zip (else :attr:`path` is the file itself)."""
        return self.zip_member is not None

    def read(self) -> bytes:
        """The file's bytes (one read of :attr:`size` bytes at :attr:`offset`).

        Raises:
            OSError: If :attr:`path` is gone, or an extracted file is shorter than expected.
            kitscenes.errors.ChecksumError: If the zip ends before the member does (a truncated archive).
        """
        with open(self.path, "rb") as fh:
            fh.seek(self.offset)
            data = fh.read(self.size)
        if len(data) != self.size:
            message = f"Short read of {self.member} from {self.path} ({len(data)} of {self.size} bytes)"
            if self.in_zip:
                from kitscenes.errors import ChecksumError

                raise ChecksumError(message + ": the zip is truncated; download the scene again")
            raise OSError(message)
        return data
