"""Ego trajectory interpolation and per-point LiDAR motion compensation.

For a point ``p_s`` of a sensor measured at time ``t`` and the frame's reference
time ``t_target``::

    T_rel(t) = T_world_base(t_target)^-1 * T_world_base(t)
    p_base'  = T_rel(t) * T_base_sensor * p_s

Poses are interpolated linearly (lerp of translation, slerp of rotation)
between the bracketing poses of ``poses.txt``. Outside the pose range the
SE(3) twist of the first/last pose pair is continued (constant body velocity),
for at most :data:`MAX_EXTRAPOLATION_S`. The Hesai ``lidar_top`` scans
0…+100 ms after the reference time, so the last frame of every scene needs
this extrapolation.

Points sharing a firing time need the same transform. :func:`compensate_points`
evaluates the trajectory once per unique timestamp (exact), or once per time
bin of ``time_bin_s``, which avoids sorting the timestamps. The default bin of
0.02 ms stayed below 2 mm error on the release test scenes even for points at
~440 m range while turning (0.1 ms: up to ~9 mm there), below the 5 mm storage
grid, at about 40-70 % of the exact method's CPU time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from kitscenes._repr import compact_repr
from kitscenes.schema import EgoPose

# Extrapolating further than this beyond the first/last pose raises ExtrapolationError.
MAX_EXTRAPOLATION_S = 0.2
# Interpolating inside a pose interval longer than this raises InterpolationGapError.
MAX_POSE_GAP_S = 0.2
# Default time bin for compensate_points (seconds). None means exact per unique timestamp.
DEFAULT_TIME_BIN_S: Optional[float] = 2e-5

TARGET_FRAMES = ("sensor", "base", "world")


class ExtrapolationError(ValueError):
    """Raised when a query time is too far outside the pose time range."""


class InterpolationGapError(ValueError):
    """Raised when a query time falls into a too large gap between two poses."""


def _skew(v: np.ndarray) -> np.ndarray:
    k = np.zeros(v.shape[:-1] + (3, 3))
    k[..., 0, 1], k[..., 0, 2] = -v[..., 2], v[..., 1]
    k[..., 1, 0], k[..., 1, 2] = v[..., 2], -v[..., 0]
    k[..., 2, 0], k[..., 2, 1] = -v[..., 1], v[..., 0]
    return k


def _so3_left_jacobian(rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec, axis=-1)[..., None, None]
    k = _skew(rotvec)
    small = theta < 1e-6
    theta_safe = np.where(small, 1.0, theta)
    a = np.where(small, 0.5 - theta**2 / 24.0, (1.0 - np.cos(theta_safe)) / theta_safe**2)
    b = np.where(small, 1.0 / 6.0 - theta**2 / 120.0, (theta_safe - np.sin(theta_safe)) / theta_safe**3)
    return np.eye(3) + a * k + b * (k @ k)


def _rodrigues(rotvec: np.ndarray) -> np.ndarray:
    """Rotation matrices (M, 3, 3) from rotation vectors (M, 3)."""
    theta = np.linalg.norm(rotvec, axis=-1)[:, None, None]
    k = _skew(rotvec)
    small = theta < 1e-8
    theta_safe = np.where(small, 1.0, theta)
    a = np.where(small, 1.0 - theta**2 / 6.0, np.sin(theta_safe) / theta_safe)
    b = np.where(small, 0.5 - theta**2 / 24.0, (1.0 - np.cos(theta_safe)) / theta_safe**2)
    return np.eye(3) + a * k + b * (k @ k)


def _se3_log(rotation: Rotation, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotvec = rotation.as_rotvec()
    rho = np.linalg.solve(_so3_left_jacobian(rotvec), translation[..., None])[..., 0]
    return rotvec, rho


def _se3_exp(rotvec: np.ndarray, rho: np.ndarray) -> tuple[Rotation, np.ndarray]:
    return Rotation.from_rotvec(rotvec), (_so3_left_jacobian(rotvec) @ rho[..., None])[..., 0]


class Trajectory:
    """Ego poses ``T_world_base(t)`` with linear interpolation and constant-velocity extrapolation.

    Args:
        times_s: (N,) strictly increasing pose times in seconds, in the same
            time base as the per-point LiDAR timestamps.
        translations: (N, 3) positions.
        quaternions_xyzw: (N, 4) rotations.
        max_extrapolation_s: Largest allowed distance of a query time before
            the first / after the last pose.
        max_gap_s: Largest allowed pose interval a query time may fall into.
    """

    def __init__(
        self,
        times_s: np.ndarray,
        translations: np.ndarray,
        quaternions_xyzw: np.ndarray,
        max_extrapolation_s: float = MAX_EXTRAPOLATION_S,
        max_gap_s: float = MAX_POSE_GAP_S,
    ) -> None:
        times_s = np.asarray(times_s, dtype=np.float64)
        translations = np.asarray(translations, dtype=np.float64)
        quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=np.float64)
        if times_s.ndim != 1 or len(times_s) < 2:
            raise ValueError("A trajectory needs at least two poses.")
        if translations.shape != (len(times_s), 3) or quaternions_xyzw.shape != (len(times_s), 4):
            raise ValueError("Shapes of times, translations and quaternions do not match.")
        if np.any(np.diff(times_s) <= 0):
            raise ValueError("Pose timestamps must be strictly increasing.")
        self.times = times_s
        self.translations = translations
        self.rotations = Rotation.from_quat(quaternions_xyzw)  # normalizes (poses.txt has 4 decimals)
        self.max_extrapolation_s = max_extrapolation_s
        self.max_gap_s = max_gap_s
        # Matrix form of the poses and the per-interval rotation increments, for the
        # vectorized path in relative_transforms (scipy Rotation objects are slow there).
        self._rot_mats = self.rotations.as_matrix()
        self._delta_rotvec = (self.rotations[:-1].inv() * self.rotations[1:]).as_rotvec()
        # A scene's trajectory is shared by all its frames: its arrays are read-only (copies of the inputs).
        self.times, self.translations = self.times.copy(), self.translations.copy()
        for array in (self.times, self.translations, self._rot_mats, self._delta_rotvec):
            array.flags.writeable = False
        self._velocity_cache: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._start_twist = self._pair_twist(0, 1)
        self._end_twist = self._pair_twist(len(times_s) - 2, len(times_s) - 1)

    @classmethod
    def from_ego_poses(cls, poses: Sequence[EgoPose], **kwargs) -> "Trajectory":
        """Build a trajectory from loaded :class:`~kitscenes.schema.EgoPose` objects."""
        return cls(
            np.array([p.t_ns for p in poses], dtype=np.float64) / 1e9,
            np.array([p.xyz for p in poses], dtype=np.float64),
            np.array([p.quat for p in poses], dtype=np.float64),
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.times)

    def _pair_twist(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        rot_ij = self.rotations[i].inv() * self.rotations[j]
        trans_ij = self.rotations[i].inv().apply(self.translations[j] - self.translations[i])
        rotvec, rho = _se3_log(rot_ij, trans_ij)
        dt = self.times[j] - self.times[i]
        return rotvec / dt, rho / dt

    def _check_query(self, t: np.ndarray) -> None:
        if not len(t):
            return
        before = self.times[0] - t.min()
        after = t.max() - self.times[-1]
        if before > self.max_extrapolation_s or after > self.max_extrapolation_s:
            raise ExtrapolationError(
                f"Query times [{t.min():.6f}, {t.max():.6f}] s require extrapolating "
                f"{max(before, after):.4f} s beyond the poses [{self.times[0]:.6f}, {self.times[-1]:.6f}] s "
                f"(allowed: {self.max_extrapolation_s} s)."
            )
        inside = t[(t >= self.times[0]) & (t <= self.times[-1])]
        if len(inside):
            idx = np.clip(np.searchsorted(self.times, inside, side="right") - 1, 0, len(self.times) - 2)
            gaps = self.times[idx + 1] - self.times[idx]
            in_gap = (gaps > self.max_gap_s) & (inside > self.times[idx]) & (inside < self.times[idx + 1])
            if np.any(in_gap):
                bad = idx[in_gap][0]
                raise InterpolationGapError(
                    f"Query time falls into a pose gap of {self.times[bad + 1] - self.times[bad]:.4f} s between "
                    f"{self.times[bad]:.6f} s and {self.times[bad + 1]:.6f} s (allowed: {self.max_gap_s} s)."
                )

    def covers(self, t: np.ndarray, margin_s: float = 0.0) -> np.ndarray:
        """Which times ``t`` (seconds) the poses cover: at most *margin_s* outside
        their range and in no pose interval longer than ``max_gap_s``. (M,) bool.

        Unlike :meth:`interpolate`, which extrapolates up to
        ``max_extrapolation_s``, this asks where the poses themselves are.
        """
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        ok = (t >= self.times[0] - margin_s) & (t <= self.times[-1] + margin_s)
        idx = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 2)
        in_gap = ((self.times[idx + 1] - self.times[idx]) > self.max_gap_s) \
            & (t > self.times[idx]) & (t < self.times[idx + 1])
        return ok & ~in_gap

    def _interpolate_inside(self, t: np.ndarray) -> tuple[Rotation, np.ndarray]:
        idx = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 2)
        t0, t1 = self.times[idx], self.times[idx + 1]
        alpha = (t - t0) / (t1 - t0)
        translations = (1.0 - alpha)[:, None] * self.translations[idx] + alpha[:, None] * self.translations[idx + 1]
        r0 = self.rotations[idx]
        delta = (r0.inv() * self.rotations[idx + 1]).as_rotvec()
        return r0 * Rotation.from_rotvec(alpha[:, None] * delta), translations

    def _extrapolate(
        self, t: np.ndarray, anchor: int, twist: tuple[np.ndarray, np.ndarray],
    ) -> tuple[Rotation, np.ndarray]:
        tau = (t - self.times[anchor])[:, None]
        rot_delta, trans_delta = _se3_exp(tau * twist[0], tau * twist[1])
        return (
            self.rotations[anchor] * rot_delta,
            self.translations[anchor] + self.rotations[anchor].apply(trans_delta),
        )

    def interpolate(self, t: np.ndarray) -> tuple[Rotation, np.ndarray]:
        """Poses ``T_world_base`` at times ``t`` (seconds).

        Returns:
            ``(rotations, translations)``: a ``Rotation`` of length ``M`` and an ``(M, 3)`` array.

        Raises:
            ExtrapolationError: If a time lies too far outside the pose range.
            InterpolationGapError: If a time falls into a too large pose gap.
        """
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        self._check_query(t)
        quats = np.empty((len(t), 4))
        translations = np.empty((len(t), 3))
        before, after = t < self.times[0], t > self.times[-1]
        inside = ~(before | after)
        if np.any(inside):
            rot, trans = self._interpolate_inside(t[inside])
            quats[inside], translations[inside] = rot.as_quat(), trans
        for mask, anchor, twist in (
            (before, 0, self._start_twist),
            (after, len(self.times) - 1, self._end_twist),
        ):
            if np.any(mask):
                rot, trans = self._extrapolate(t[mask], anchor, twist)
                quats[mask], translations[mask] = rot.as_quat(), trans
        return Rotation.from_quat(quats), translations

    def interpolate_matrices(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Like :meth:`interpolate`, but returns ``(R (M, 3, 3), trans (M, 3))`` and is vectorized in numpy."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        self._check_query(t)
        rot = np.empty((len(t), 3, 3))
        trans = np.empty((len(t), 3))
        before, after = t < self.times[0], t > self.times[-1]
        inside = ~(before | after)
        if np.any(inside):
            ti = t[inside]
            idx = np.clip(np.searchsorted(self.times, ti, side="right") - 1, 0, len(self.times) - 2)
            alpha = (ti - self.times[idx]) / (self.times[idx + 1] - self.times[idx])
            trans[inside] = (1.0 - alpha)[:, None] * self.translations[idx] + alpha[:, None] * self.translations[idx + 1]
            rot[inside] = self._rot_mats[idx] @ _rodrigues(alpha[:, None] * self._delta_rotvec[idx])
        for mask, anchor, twist in (
            (before, 0, self._start_twist),
            (after, len(self.times) - 1, self._end_twist),
        ):
            if np.any(mask):
                r, tr = self._extrapolate(t[mask], anchor, twist)
                rot[mask], trans[mask] = r.as_matrix(), tr
        return rot, trans

    def body_velocities(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linear and angular ego velocity in the base frame at times ``t``.

        Central differences at the poses, linearly interpolated to ``t``
        (constant beyond the ends). On the release radar data this fits the
        Doppler of static returns slightly better than the interval velocity of
        the linear trajectory or the nearest pose's central difference.

        Returns:
            ``(v_body, w_body)``, both ``(M, 3)``.
        """
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        self._check_query(t)
        if self._velocity_cache is None:
            dt = np.diff(self.times)
            w_interval = self._delta_rotvec / dt[:, None]  # body frame of the interval start
            v_world = np.gradient(self.translations, self.times, axis=0)
            w_pose = np.vstack([w_interval[:1], 0.5 * (w_interval[1:] + w_interval[:-1]), w_interval[-1:]])
            self._velocity_cache = (v_world, w_pose)
        v_world, w_pose = self._velocity_cache
        tc = np.clip(t, self.times[0], self.times[-1])
        idx = np.clip(np.searchsorted(self.times, tc, side="right") - 1, 0, len(self.times) - 2)
        alpha = ((tc - self.times[idx]) / (self.times[idx + 1] - self.times[idx]))[:, None]
        v = (1 - alpha) * v_world[idx] + alpha * v_world[idx + 1]
        w = (1 - alpha) * w_pose[idx] + alpha * w_pose[idx + 1]
        rot, _ = self.interpolate_matrices(t)
        return np.einsum("nji,nj->ni", rot, v), w  # world -> body: R^T v

    def body_accelerations(self, t: np.ndarray) -> np.ndarray:
        """Linear ego acceleration at times ``t``: the derivative of the world-frame
        velocity of :meth:`body_velocities` (differences of its values at the
        poses, linearly interpolated), in the axes of the base frame. ``(M, 3)``.
        """
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        self.body_velocities(t[:0])  # fills the velocity cache
        v_world = self._velocity_cache[0]
        a_world = np.gradient(v_world, self.times, axis=0) if len(self.times) > 2 else np.zeros_like(v_world)
        tc = np.clip(t, self.times[0], self.times[-1])
        idx = np.clip(np.searchsorted(self.times, tc, side="right") - 1, 0, len(self.times) - 2)
        alpha = ((tc - self.times[idx]) / (self.times[idx + 1] - self.times[idx]))[:, None]
        a = (1 - alpha) * a_world[idx] + alpha * a_world[idx + 1]
        rot, _ = self.interpolate_matrices(t)
        return np.einsum("nji,nj->ni", rot, a)

    def relative_transforms(self, t: np.ndarray, t_target: float) -> tuple[np.ndarray, np.ndarray]:
        """``T_rel(t) = T_world_base(t_target)^-1 * T_world_base(t)`` as matrices.

        Maps the base frame at ``t`` into the base frame at ``t_target``.

        Returns:
            ``(R, trans)`` with shapes ``(M, 3, 3)`` and ``(M, 3)``.
        """
        rot_target, trans_target = self.interpolate_matrices(np.array([t_target]))
        rot_t, trans_t = self.interpolate_matrices(t)
        rot_target_inv = rot_target[0].T
        return rot_target_inv @ rot_t, (trans_t - trans_target[0]) @ rot_target_inv.T


def measured_point_mask(xyz: np.ndarray) -> np.ndarray:
    """Rows that hold a measured point: anything not at the sensor origin.

    Rows at the origin are slots without a point: firing slots that never
    fired (OT128; nominal column time), OT128 returns closer than 1.4 m that
    the extraction zeroed, empty Velodyne grid cells (timestamp 0), and
    fired rays without a return (reflectivity -1, real firing time; dropped
    on load unless ``no_returns=True``). They are never moved, so they stay at
    the origin in every frame. The timestamp is no criterion: the first point
    of a sequence is a real measurement at time 0.
    """
    xyz = np.asarray(xyz)
    return (xyz[:, 0] != 0) | (xyz[:, 1] != 0) | (xyz[:, 2] != 0)


def point_transforms(
    timestamps_s: np.ndarray,
    trajectory: Trajectory,
    t_target_s: float,
    time_bin_s: Optional[float] = DEFAULT_TIME_BIN_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-point motion transforms, shared by points with the same (binned) time.

    Returns:
        ``(R, trans, index)``: ``R`` (M, 3, 3) and ``trans`` (M, 3) map the base
        frame at a sample time into the base frame at ``t_target_s``; point
        ``i`` uses sample ``index[i]``. Small enough to compute on the CPU and
        apply on a GPU (see :func:`apply_point_transforms`).
    """
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    if len(timestamps_s) == 0:
        return np.zeros((0, 3, 3)), np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    if time_bin_s is None:
        sample_times, index = np.unique(timestamps_s, return_inverse=True)
    else:
        t_min = float(timestamps_s.min())
        n_bins = int(np.floor((float(timestamps_s.max()) - t_min) / time_bin_s)) + 1
        index = np.minimum(((timestamps_s - t_min) / time_bin_s).astype(np.int64), n_bins - 1)
        sample_times = t_min + (np.arange(n_bins) + 0.5) * time_bin_s
    rot, trans = trajectory.relative_transforms(sample_times, t_target_s)
    return rot, trans, index


def apply_point_transforms(
    points_base: np.ndarray,
    rot: np.ndarray,
    trans: np.ndarray,
    index: np.ndarray,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """``rot[index[i]] @ points_base[i] + trans[index[i]]`` for all points.

    Computed in ``dtype`` (float32 by default: ~2x faster than float64 and
    < 0.1 mm error for points within a few hundred metres of the vehicle;
    the stored clouds are float32 on a 5 mm grid anyway).
    """
    table = np.concatenate([rot.reshape(len(rot), 9), trans], axis=1).astype(dtype)[index]  # (N, 12)
    q = np.asarray(points_base, dtype=dtype)
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    out = np.empty((len(q), 3), dtype=dtype)
    out[:, 0] = table[:, 0] * x + table[:, 1] * y + table[:, 2] * z + table[:, 9]
    out[:, 1] = table[:, 3] * x + table[:, 4] * y + table[:, 5] * z + table[:, 10]
    out[:, 2] = table[:, 6] * x + table[:, 7] * y + table[:, 8] * z + table[:, 11]
    return out


def compensate_points(
    xyz: np.ndarray,
    timestamps_s: np.ndarray,
    T_base_sensor: Optional[np.ndarray],
    trajectory: Trajectory,
    t_target_s: float,
    *,
    target_frame: str = "base",
    time_bin_s: Optional[float] = DEFAULT_TIME_BIN_S,
    compute_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Motion-compensate the points of one sensor to ``t_target_s``.

    Args:
        xyz: (N, 3) points in the sensor frame (or the base frame if
            ``T_base_sensor`` is None).
        timestamps_s: (N,) per-point times in the trajectory's time base.
        T_base_sensor: (4, 4) sensor-to-base extrinsic (``T_to_reference`` in
            ``calib.json``), or None if ``xyz`` is already in the base frame.
        trajectory: Ego poses.
        t_target_s: Time to compensate to, normally the frame's reference time.
        target_frame: ``"base"`` (reference frame at ``t_target_s``),
            ``"sensor"``, or ``"world"``.
        time_bin_s: Time bin for pose evaluation; None for exact per-timestamp.
        compute_dtype: Precision of the per-point transform (see
            :func:`apply_point_transforms`); ``np.float64`` for evaluation.

    Returns:
        (N, 3) float64 points in ``target_frame``. Rows without a measured
        point (see :func:`measured_point_mask`) stay at the origin.

    Raises:
        ExtrapolationError, InterpolationGapError: From the trajectory.
    """
    if target_frame not in TARGET_FRAMES:
        raise ValueError(f"Unknown target frame {target_frame!r}, expected one of {TARGET_FRAMES}.")
    xyz = np.asarray(xyz, dtype=np.float64)
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or timestamps_s.shape != (len(xyz),):
        raise ValueError(f"Expected xyz (N, 3) and timestamps (N,), got {xyz.shape} and {timestamps_s.shape}.")
    if target_frame == "sensor" and T_base_sensor is None:
        raise ValueError("target_frame='sensor' needs T_base_sensor.")

    out = np.zeros_like(xyz)
    measured = measured_point_mask(xyz)
    if not np.any(measured):
        return out

    points = xyz[measured]
    if T_base_sensor is not None:
        rot_bs, trans_bs = T_base_sensor[:3, :3], T_base_sensor[:3, 3]
        points = points @ rot_bs.T + trans_bs
    rot, trans, index = point_transforms(timestamps_s[measured], trajectory, t_target_s, time_bin_s)
    points = apply_point_transforms(points, rot, trans, index, dtype=compute_dtype).astype(np.float64)

    if target_frame == "sensor":
        out[measured] = (points - trans_bs) @ rot_bs
    elif target_frame == "world":
        rot_wb, trans_wb = trajectory.interpolate(np.array([t_target_s]))
        out[measured] = points @ rot_wb.as_matrix()[0].T + trans_wb[0]
    else:
        out[measured] = points
    return out


@dataclass(frozen=True)
class DeskewInputs:
    """Everything a device needs to deskew one sweep into the base frame at its reference time.

    ``out[i] = table[index[i], :9].reshape(3, 3) @ xyz[i] + table[index[i], 9:]`` for points
    with ``measured[i]``; the other rows stay at the origin. The sensor extrinsic is
    folded into ``table``, so ``xyz`` is the raw sensor-frame cloud.

    Attributes:
        xyz: (N, 3) float32 raw points in the sensor frame.
        index: (N,) int32 row of ``table`` per point (0 for unmeasured rows).
        measured: (N,) bool, rows that hold a measured point (:func:`measured_point_mask`).
        table: (M, 12) float32 rows ``[R (row-major 3x3), t]``, sensor -> base at t_target.
        deskewed: False if motion compensation was impossible; then ``table`` only
            applies the extrinsic (base frame without motion correction).
        reason: Why deskewing was impossible, else None.
    """

    __repr__ = compact_repr

    xyz: np.ndarray
    index: np.ndarray
    measured: np.ndarray
    table: np.ndarray
    deskewed: bool
    reason: Optional[str] = None


def deskew_inputs(
    xyz: np.ndarray,
    timestamps_s: Optional[np.ndarray],
    T_base_sensor: Optional[np.ndarray],
    trajectory: Optional[Trajectory],
    t_target_s: float,
    *,
    time_bin_s: Optional[float] = DEFAULT_TIME_BIN_S,
    unavailable_reason: Optional[str] = None,
) -> DeskewInputs:
    """Build :class:`DeskewInputs` (CPU, cheap: O(N) binning plus ~M pose evaluations).

    Never raises on trajectory problems: if the sweep cannot be compensated
    (no trajectory, no timestamps, extrapolation or gap errors) the result has
    ``deskewed=False`` and an extrinsic-only table.
    """
    xyz32 = np.ascontiguousarray(xyz, dtype=np.float32)
    measured = measured_point_mask(xyz32)
    T = np.eye(4) if T_base_sensor is None else np.asarray(T_base_sensor, dtype=np.float64)
    rot_s, trans_s = T[:3, :3], T[:3, 3]
    index = np.zeros(len(xyz32), dtype=np.int32)

    def extrinsic_only(reason: str) -> DeskewInputs:
        table = np.concatenate([rot_s.reshape(9), trans_s])[None].astype(np.float32)
        return DeskewInputs(xyz32, index, measured, table, False, reason)

    if trajectory is None:
        return extrinsic_only(unavailable_reason or "no ego trajectory")
    if timestamps_s is None:
        return extrinsic_only("sweep has no per-point timestamps")
    if not np.any(measured):  # nothing to move; trivially deskewed
        table = np.concatenate([rot_s.reshape(9), trans_s])[None].astype(np.float32)
        return DeskewInputs(xyz32, index, measured, table, True, None)
    try:
        rot, trans, idx = point_transforms(
            np.asarray(timestamps_s, dtype=np.float64)[measured], trajectory, t_target_s, time_bin_s,
        )
    except (ExtrapolationError, InterpolationGapError) as err:
        return extrinsic_only(str(err))
    # Fold the extrinsic in: R_rel (R_s p + t_s) + t_rel.
    rot_c = rot @ rot_s
    trans_c = rot @ trans_s + trans
    table = np.concatenate([rot_c.reshape(len(rot_c), 9), trans_c], axis=1).astype(np.float32)
    index[measured] = idx
    return DeskewInputs(xyz32, index, measured, table, True, None)


def apply_deskew_inputs(inputs: DeskewInputs) -> np.ndarray:
    """CPU (numpy, float32) application of :class:`DeskewInputs`; see :mod:`kitscenes.motion_torch` for GPUs."""
    out = np.zeros_like(inputs.xyz)
    v = inputs.measured
    if np.any(v):
        rows = inputs.table[inputs.index[v]]
        x, y, z = inputs.xyz[v, 0], inputs.xyz[v, 1], inputs.xyz[v, 2]
        out[v, 0] = rows[:, 0] * x + rows[:, 1] * y + rows[:, 2] * z + rows[:, 9]
        out[v, 1] = rows[:, 3] * x + rows[:, 4] * y + rows[:, 5] * z + rows[:, 10]
        out[v, 2] = rows[:, 6] * x + rows[:, 7] * y + rows[:, 8] * z + rows[:, 11]
    return out


def compensate_radar_points(
    raw: np.ndarray,
    T_base_sensor: Optional[np.ndarray],
    trajectory: Optional[Trajectory],
    t_target_s: float,
    *,
    target_frame: str = "sensor",
) -> tuple[np.ndarray, bool, bool, Optional[str]]:
    """Ego-motion compensation of one radar cloud (structured array as stored).

    - **Positions**: each detection is moved rigidly from its ``timestamp`` (one
      per cloud; radars are not triggered, release clouds are ~19 ms before the
      reference time) to ``t_target_s``. ``range``, ``azimuth`` and ``elevation``
      are recomputed from the compensated sensor-frame xyz (``atan2(y, x)``,
      ``asin(z / range)``), and stay relative to the radar for
      ``target_frame="base"``.
    - **Doppler**: ``range_rate`` gets the ego radial velocity added,
      ``u · (v + w × p)`` with the line of sight ``u`` and position ``p`` of the
      measurement (base frame) and the ego velocity at the capture time, so
      static returns read ~0 m/s. Standard deviations stay unchanged.

    Returns:
        ``(points, positions_compensated, doppler_compensated, reason)``. If the
        capture times can't be used (missing, or outside the pose range), the
        positions stay as measured and the Doppler uses the velocity at
        ``t_target_s``; ``reason`` says why.
    """
    if target_frame not in ("sensor", "base"):
        raise ValueError(f"target_frame must be 'sensor' or 'base', got {target_frame!r}")
    out = raw.copy()
    if trajectory is None:
        return out, False, False, "missing or insufficient poses.txt data"
    if len(raw) == 0:
        return out, True, True, None

    T = np.eye(4) if T_base_sensor is None else np.asarray(T_base_sensor, dtype=np.float64)
    rot_s, trans_s = T[:3, :3], T[:3, 3]
    xyz = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float64)
    rng = np.linalg.norm(xyz, axis=1)
    valid = rng > 1e-6
    names = raw.dtype.names or ()
    times = raw["timestamp"].astype(np.float64) if "timestamp" in names else None

    reason = None
    capture_ok = times is not None
    if times is None:
        reason = "radar cloud has no timestamps"
    else:
        try:
            trajectory._check_query(np.array([times.min(), times.max()]))
        except (ExtrapolationError, InterpolationGapError) as err:
            capture_ok, reason = False, f"capture time unusable: {err}"

    doppler_done = False
    if "range_rate" in names and np.any(valid):
        t_vel = times[valid] if capture_ok else np.full(int(valid.sum()), t_target_s)
        try:
            v, w = trajectory.body_velocities(t_vel)
        except (ExtrapolationError, InterpolationGapError) as err:
            reason = reason or f"no ego velocity: {err}"
        else:
            p = xyz[valid] @ rot_s.T + trans_s
            u = (xyz[valid] / rng[valid, None]) @ rot_s.T
            rr = raw["range_rate"].astype(np.float64)
            rr[valid] += np.einsum("ij,ij->i", u, v + np.cross(w, p))
            out["range_rate"] = rr.astype(out.dtype["range_rate"])
            doppler_done = True

    positions_done = False
    sensor_xyz = xyz
    if capture_ok:
        sensor_xyz = compensate_points(
            xyz, times, T, trajectory, t_target_s,
            target_frame="sensor", time_bin_s=None, compute_dtype=np.float64,
        )
        positions_done = True
        r = np.linalg.norm(sensor_xyz, axis=1)
        ok = r > 1e-6
        for field, values in (
            ("range", r),
            ("azimuth", np.arctan2(sensor_xyz[:, 1], sensor_xyz[:, 0])),
            ("elevation", np.arcsin(np.clip(sensor_xyz[:, 2] / np.where(ok, r, 1.0), -1.0, 1.0))),
        ):
            if field in names:
                column = out[field].astype(np.float64)
                column[ok] = values[ok]
                out[field] = column.astype(out.dtype[field])

    if target_frame == "base":
        sensor_xyz = np.where(valid[:, None], sensor_xyz @ rot_s.T + trans_s, 0.0)
    if positions_done or target_frame == "base":
        for axis, column in zip(("x", "y", "z"), sensor_xyz.T):
            out[axis] = column.astype(out.dtype[axis])
    return out, positions_done, doppler_done, reason
