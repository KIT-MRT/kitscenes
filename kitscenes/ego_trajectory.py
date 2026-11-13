"""Ego motion: the state at any time (:class:`EgoState`) and a frame's future and past (:class:`EgoTrajectory`).

:class:`EgoState` is what :meth:`kitscenes.Scene.ego` returns: the pose,
velocity, acceleration, yaw rate and curvature at any time(s) within the
poses, from the same interpolation that deskews the lidar
(:class:`kitscenes.motion.Trajectory`).

A frame's future and past (:meth:`kitscenes.Frame.future`,
:meth:`kitscenes.Frame.past`) are where the recording vehicle went next and
came from, at fixed time slots and re-expressed *relative to the frame's
pose*, which is what makes them usable as a planning, prediction or history
input: world-frame coordinates are neither translation- nor
rotation-invariant, so they cannot be learned from or compared across
scenes directly. :func:`relative_trajectory` does the work on a
:class:`kitscenes.motion.Trajectory`, so it can be unit tested without a
scene on disk.

Two coordinate frames are involved:

* **world**: the frame of ``poses.txt`` (:attr:`kitscenes.Scene.pose_frame`):
  the map's local UTM frame, metres relative to the origin in
  ``maps/origin.json`` (see :meth:`kitscenes.map.MapOrigin.to_utm`), or in
  the test-e2e splits relative to the first pose.
* **ego**: the vehicle base frame at the anchor (the frame time), so it sits at
  the origin with +x pointing the way the vehicle was facing, +y left and
  +z up. This is the frame a model target normally wants. With
  ``frame="road"`` it is the road frame instead, and the positions are those
  of its origin (docs/conventions.md, "The road frame").

The ego frame is the vehicle base frame, so a trajectory goes into a camera
image like any other base-frame geometry: ``camera.project(t.xyz_ego[t.valid])``
for the waypoints, :func:`kitscenes.projection.project_lines` for the path as a
line (clipped at the camera).

Times are integer nanoseconds since the scene start, like ``Frame.t_ns``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from kitscenes._repr import compact_repr
from kitscenes.schema import Provenance

if TYPE_CHECKING:
    from kitscenes.motion import Trajectory

__all__ = ["EgoState", "EgoTrajectory", "relative_trajectory", "DEFAULT_STEP_S"]


@dataclass(frozen=True, eq=False)
class EgoTrajectory:
    """Ego poses at fixed time slots around one frame, relative to its pose.

    What :meth:`kitscenes.Frame.future` and :meth:`kitscenes.Frame.past`
    return: for the future, slot k (0-based) is the pose ``(k + 1) * step``
    after the frame time; for the past (the oldest first), ``(K - k) * step``
    before it; from the same interpolation of ``poses.txt`` as ``frame.pose``. A slot
    the recorded poses do not cover (past the start or end of the scene, or
    in a gap the interpolation refuses) is NaN and ``valid`` False; there are
    always K slots.

    Attributes:
        index: Frame index of the anchor (-1 if there is none).
        anchor_t_ns: The anchor time (the frame time).
        t_ns: (K,) int64 slot times, ns since the scene start, ascending
            (for the past: the oldest first, the slot just before the frame last).
        xyz_ego: (K, 3) float64 positions in the base frame at the anchor.
        quat_ego: (K, 4) float64 orientations relative to it, quaternion (qx, qy, qz, qw).
        xyz_world: (K, 3) float64 positions in the world frame (see the module docs).
        valid: (K,) bool: the poses cover the slot.
        frame: ``"base"`` or ``"road"``: the ego frame (and whose origin the positions are of).
        provenance: ``GEOMETRY_DERIVED``.
    """

    __repr__ = compact_repr

    index: int
    anchor_t_ns: int
    t_ns: np.ndarray
    xyz_ego: np.ndarray
    quat_ego: np.ndarray
    xyz_world: np.ndarray
    valid: np.ndarray
    frame: str = "base"
    provenance: Provenance = Provenance.GEOMETRY_DERIVED

    def __len__(self) -> int:
        return len(self.t_ns)


@dataclass(frozen=True, eq=False)
class EgoState:
    """The ego vehicle's pose and motion at one or more times (:meth:`kitscenes.Scene.ego`).

    For a scalar time the fields have no leading axis (``T_world_base`` is
    (4, 4), ``speed`` a 0-d array); for M times they lead with M.

    Attributes:
        t_ns: Times in nanoseconds since the scene start.
        T_world_base: (M, 4, 4) pose matrices: map base-frame points into the world frame.
        xyz: (M, 3) position in the world frame.
        quat: (M, 4) orientation, quaternion (qx, qy, qz, qw).
        velocity: (M, 3) m/s, in the base frame at that time (x forward, y left, z up).
        angular_velocity: (M, 3) rad/s, in the base frame.
        acceleration: (M, 3) m/s², the rate of change of the world-frame
            velocity in the axes of the base frame (no gravity, unlike the IMU).

    Velocities are central differences of the poses (10 Hz) and the
    acceleration differentiates them once more, both interpolated linearly
    to the query times: smooth enough for an ego state, not a substitute for
    the IMU (:meth:`kitscenes.Scene.imu`).
    """

    __repr__ = compact_repr

    t_ns: np.ndarray
    T_world_base: np.ndarray
    xyz: np.ndarray
    quat: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    acceleration: np.ndarray

    @property
    def speed(self) -> np.ndarray:
        """Speed in m/s (norm of :attr:`velocity`)."""
        return np.linalg.norm(self.velocity, axis=-1)

    @property
    def yaw_rate(self) -> np.ndarray:
        """Yaw rate in rad/s (the z component of :attr:`angular_velocity`), positive turning left."""
        return self.angular_velocity[..., 2]

    @property
    def curvature(self) -> np.ndarray:
        """Path curvature in 1/m, ``yaw_rate / planar speed``; positive turning left, NaN below 0.5 m/s."""
        planar = np.hypot(self.velocity[..., 0], self.velocity[..., 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(planar >= MIN_CURVATURE_SPEED_MPS, self.yaw_rate / planar, np.nan)


#: Below this planar speed (m/s) :attr:`EgoState.curvature` is NaN: the yaw rate of a car
#: that barely moves says little about its path.
MIN_CURVATURE_SPEED_MPS = 0.5


def ego_state(trajectory: "Trajectory", t_ns) -> EgoState:
    """The :class:`EgoState` of *trajectory* at time(s) *t_ns* (ns since the scene start).

    Raises:
        ExtrapolationError, InterpolationGapError: For a time too far outside the poses or in a gap.
    """
    t_ns = np.asarray(t_ns)
    scalar = t_ns.ndim == 0
    t_ns = np.atleast_1d(t_ns).astype(np.int64)
    t_s = t_ns / 1e9
    rot, trans = trajectory.interpolate_matrices(t_s)
    T = np.tile(np.eye(4), (len(t_s), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = rot, trans
    v, w = trajectory.body_velocities(t_s)
    a = trajectory.body_accelerations(t_s)
    fields = dict(t_ns=t_ns, T_world_base=T, xyz=trans, quat=Rotation.from_matrix(rot).as_quat(),
                  velocity=v, angular_velocity=w, acceleration=a)
    if scalar:
        fields = {k: f[0] for k, f in fields.items()}
    return EgoState(**fields)


#: Default spacing of trajectory slots: the 10 Hz frame rate.
DEFAULT_STEP_S = 0.1

#: How far outside the recorded poses a slot may lie and still count as covered: the
#: poses and the frame times of the release differ by < 0.05 ms, the frames are 100 ms apart.
SLOT_TOLERANCE_NS = 5_000_000


def num_slots(horizon_s: float, step_s: float = DEFAULT_STEP_S) -> int:
    """The number of slots of a *horizon_s* trajectory every *step_s*: ``round(horizon_s / step_s)``.

    Raises:
        ValueError: For a negative horizon or a step that is not positive.
    """
    if not step_s > 0:
        raise ValueError(f"step_s must be positive, got {step_s!r}")
    if not horizon_s >= 0:
        raise ValueError(f"horizon_s must be >= 0, got {horizon_s!r}")
    return int(round(horizon_s / step_s))


def slot_times(anchor_t_ns: int, horizon_s: float, step_s: float = DEFAULT_STEP_S, *, past: bool = False) -> np.ndarray:
    """(K,) int64 slot times: ``anchor + k * step`` for k = 1..K, or for the past ``anchor - k * step``
    in time order (the oldest first, ``anchor - step`` last)."""
    k = np.arange(1, num_slots(horizon_s, step_s) + 1, dtype=np.int64)
    offsets = k * int(round(step_s * 1e9))
    return int(anchor_t_ns) - offsets[::-1] if past else int(anchor_t_ns) + offsets


def relative_trajectory(
    trajectory: "Trajectory", anchor_t_ns: int, t_ns: np.ndarray, *, index: int = -1,
    tolerance_ns: int = SLOT_TOLERANCE_NS, T_base_frame: Optional[np.ndarray] = None, frame: str = "base",
) -> EgoTrajectory:
    """The poses at times *t_ns* relative to the pose at *anchor_t_ns*, from the interpolated *trajectory*.

    A slot is valid where the recorded poses cover its time (at most
    *tolerance_ns* outside them, in no gap the interpolation refuses, see
    :meth:`kitscenes.motion.Trajectory.covers`); invalid slots are NaN.

    *T_base_frame* (4, 4) places another frame *frame* rigidly on the
    vehicle (e.g. the road frame: ``invert_transform(scene.T_road_base)``);
    the result is then that frame's trajectory, relative to itself at the
    anchor (``T_frame0_framek = T_frame_base T_base0_basek T_base_frame``).

    Raises:
        ExtrapolationError, InterpolationGapError: If the anchor itself has no pose
            (the same rule as ``Frame.pose``).
    """
    rot_anchor, trans_anchor = trajectory.interpolate_matrices(np.array([int(anchor_t_ns) / 1e9]))
    t_ns = np.asarray(t_ns, dtype=np.int64).reshape(-1)
    valid = trajectory.covers(t_ns / 1e9, margin_s=tolerance_ns / 1e9)
    xyz_world = np.full((len(t_ns), 3), np.nan)
    xyz_ego = np.full((len(t_ns), 3), np.nan)
    quat_ego = np.full((len(t_ns), 4), np.nan)
    if valid.any():
        rot, trans = trajectory.interpolate_matrices(t_ns[valid] / 1e9)
        if T_base_frame is not None:  # the poses of the other frame: T_world_frame = T_world_base T_base_frame
            R_bf, t_bf = np.asarray(T_base_frame[:3, :3], np.float64), np.asarray(T_base_frame[:3, 3], np.float64)
            trans_anchor = trans_anchor + rot_anchor @ t_bf
            rot_anchor = rot_anchor @ R_bf
            trans = trans + rot @ t_bf
            rot = rot @ R_bf
        R_anchor_world = rot_anchor[0].T  # world -> the ego frame at the anchor
        xyz_world[valid] = trans
        xyz_ego[valid] = (trans - trans_anchor[0]) @ R_anchor_world.T
        quat_ego[valid] = Rotation.from_matrix(R_anchor_world @ rot).as_quat()
    return EgoTrajectory(
        index=int(index), anchor_t_ns=int(anchor_t_ns), t_ns=t_ns, xyz_ego=xyz_ego, quat_ego=quat_ego,
        xyz_world=xyz_world, valid=valid, frame=frame,
    )
