"""IMU-refined ego motion of a scene.

The rotations in ``poses.txt`` carry roll/pitch errors of a few mrad on dynamic
scenes: the lidar SLAM deskews each scan with a constant-velocity model, which
is wrong under body rocking at 1-5 Hz. The gyro measures this motion. The
refined attitude takes the part faster than :data:`REFINEMENT_WINDOW_S` from the
integrated 50 Hz gyro and the slower part from ``poses.txt``, so it does not
drift away from the poses. Translations are not changed.

1. The IMU axes come from ``gnss_ins_imu`` in ``calib.json``
   (:attr:`kitscenes.rig.Rig.imu_to_base`); only the gyro bias is estimated per scene.
2. The bias-corrected gyro is integrated and corrected on the world side towards
   ``poses.txt`` with a least-squares cubic B-spline (knots every
   :data:`REFINEMENT_WINDOW_S`).

:attr:`kitscenes.Scene.ego_motion` builds this once per scene. A scene without
IMU data (the test-e2e splits), with an invalid ``gnss_ins_imu`` or opened with
``use_imu=False`` falls back to the poses alone, a scene without usable poses to
no trajectory: :attr:`EgoMotionCompensation.source` says which one it is and
:attr:`EgoMotionCompensation.reason` why.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.interpolate import make_lsq_spline
from scipy.spatial.transform import Rotation

from kitscenes.motion import Trajectory, apply_deskew_inputs, deskew_inputs
from kitscenes.points import PointCloud, RadarCloud
from kitscenes.rig import IMU_ENTRY, MAX_IMU_DEVIATION_DEG, imu_deviation_deg, invert_transform, transform_points

if TYPE_CHECKING:
    from kitscenes.scene import Scene

logger = logging.getLogger(__name__)

#: Attitude changes slower than this come from ``poses.txt``, faster ones from the gyro (knot spacing, seconds).
REFINEMENT_WINDOW_S = 1.0
#: Largest allowed gap between IMU messages within the poses (seconds).
MAX_IMU_GAP_S = 0.1
#: How far the poses may exceed the IMU messages at either end (seconds).
IMU_COVERAGE_TOLERANCE_S = 0.01
#: A residual above this in roll or pitch means that the poses deviate noticeably from the gyro (mrad, ``qa["flagged"]``).
QA_FLAG_MRAD = 1.0

_COMPENSATE_FRAMES = ("base", "sensor", "world")


def integrate_gyro(times_s: np.ndarray, rates: np.ndarray) -> Rotation:
    """Body attitude at every IMU message (identity at the first), trapezoidal integration of the body rates."""
    steps = Rotation.from_rotvec((rates[:-1] + rates[1:]) / 2 * np.diff(times_s)[:, None]).as_quat()
    attitude = np.empty((len(times_s), 4))
    attitude[0] = current = (0.0, 0.0, 0.0, 1.0)
    for i, (x2, y2, z2, w2) in enumerate(steps, 1):  # the Hamilton product, far faster here than scalar Rotations
        x1, y1, z1, w1 = current
        current = (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                   w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)
        attitude[i] = current
    return Rotation.from_quat(attitude)


def _increments(attitude: Rotation) -> np.ndarray:
    return (attitude[:-1].inv() * attitude[1:]).as_rotvec()


def _at(times_s: np.ndarray, attitude: Rotation, t: np.ndarray) -> Rotation:
    """*attitude* at the times *t*, which may exceed *times_s* by the coverage tolerance (then the end value)."""
    t = np.clip(t, times_s[0], times_s[-1])
    idx = np.clip(np.searchsorted(times_s, t, side="right") - 1, 0, len(times_s) - 2)
    alpha = (t - times_s[idx]) / (times_s[idx + 1] - times_s[idx])
    return attitude[idx] * Rotation.from_rotvec(alpha[:, None] * (attitude[idx].inv() * attitude[idx + 1]).as_rotvec())


def estimate_gyro_bias(trajectory: Trajectory, times_s: np.ndarray, rates: np.ndarray, imu_to_base: Rotation) -> np.ndarray:
    """Gyro bias (rad/s, IMU axes) for a known IMU rotation: least squares on ``inc_pose ~ R * (inc_imu - bias * dt)``."""
    t = trajectory.times
    inc_imu = _increments(_at(times_s, integrate_gyro(times_s, rates), t))
    inc_pose = _increments(trajectory.rotations)
    dt = np.diff(t)[:, None]
    return imu_to_base.inv().apply(np.sum((imu_to_base.apply(inc_imu) - inc_pose) * dt, axis=0) / np.sum(dt ** 2))


def rotation_residual_mrad(rotations: Rotation, gyro_attitude: Rotation) -> np.ndarray:
    """RMS (roll, pitch, yaw) of the difference between pose and gyro rotation increments, mrad per pose interval."""
    r = _increments(gyro_attitude) - _increments(rotations)
    r = r[5:-5] if len(r) > 20 else r
    return np.sqrt(np.mean((r - r.mean(axis=0)) ** 2, axis=0)) * 1e3


class ImuTrajectory(Trajectory):
    """A :class:`~kitscenes.motion.Trajectory` with the refined attitude, also between the poses.

    Translations and the extrapolation and gap limits are those of the poses.

    Attributes:
        qa: ``imu_to_base_deviation_from_nominal_deg``, ``gyro_bias_rad_s`` (IMU
            axes), ``residual_before_mrad`` and ``residual_after_mrad`` (roll,
            pitch, yaw; pose against gyro increments), and ``flagged``: the poses
            deviate noticeably from the gyro. That is a remark on the poses, not an
            error.
    """

    def __init__(
        self, poses: Trajectory, imu_times_s: np.ndarray, imu_attitude: Rotation, end_rates: np.ndarray, qa: dict,
    ) -> None:
        # The attitude as a trajectory of its own: same interpolation, and past the IMU data the constant body rate
        # end_rates (2, 3) of its first and last message. Its limits are never reached: the queries are checked
        # against the poses first.
        self._attitude = Trajectory(imu_times_s, np.zeros((len(imu_times_s), 3)), imu_attitude.as_quat(),
                                    max_extrapolation_s=np.inf, max_gap_s=np.inf)
        self._attitude._start_twist = (end_rates[0], np.zeros(3))
        self._attitude._end_twist = (end_rates[1], np.zeros(3))
        self.qa = qa
        super().__init__(
            poses.times, poses.translations, self._attitude.interpolate(poses.times)[0].as_quat(),
            max_extrapolation_s=poses.max_extrapolation_s, max_gap_s=poses.max_gap_s,
        )

    def interpolate(self, t: np.ndarray) -> tuple[Rotation, np.ndarray]:
        translations = super().interpolate(t)[1]  # also enforces the limits
        return self._attitude.interpolate(t)[0], translations

    def interpolate_matrices(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        translations = super().interpolate_matrices(t)[1]
        return self._attitude.interpolate_matrices(t)[0], translations


def refine_trajectory(
    poses: Trajectory, imu_times_s: np.ndarray, imu_rates: np.ndarray, imu_to_base: np.ndarray,
    window_s: float = REFINEMENT_WINDOW_S,
) -> ImuTrajectory:
    """The trajectory with the refined attitude.

    Args:
        poses: The poses.
        imu_times_s: (N,) IMU message times, in the time base of the poses.
        imu_rates: (N, 3) angular velocities in rad/s in the IMU axes.
        imu_to_base: (3, 3) rotation from the IMU axes to the base frame's.
        window_s: See :data:`REFINEMENT_WINDOW_S`.

    Raises:
        ValueError: If the IMU data is unusable; the message says why.
    """
    times = np.asarray(imu_times_s, dtype=np.float64)
    rates = np.asarray(imu_rates, dtype=np.float64)
    t = poses.times
    if len(times) < 2:
        raise ValueError("fewer than two IMU messages")
    if np.any(np.diff(times) <= 0) or not np.all(np.isfinite(rates)):
        raise ValueError("IMU data is not strictly increasing in time or has non-finite rates")
    if times[0] > t[0] + IMU_COVERAGE_TOLERANCE_S or times[-1] < t[-1] - IMU_COVERAGE_TOLERANCE_S:
        raise ValueError(f"IMU data [{times[0]:.3f}, {times[-1]:.3f}] s does not cover the poses [{t[0]:.3f}, {t[-1]:.3f}] s")
    margin = poses.max_extrapolation_s
    gaps = np.diff(times[(times >= t[0] - margin) & (times <= t[-1] + margin)])
    if len(gaps) and gaps.max() > MAX_IMU_GAP_S:
        raise ValueError(f"IMU data has a gap of {gaps.max():.3f} s (allowed {MAX_IMU_GAP_S} s)")

    rotation = Rotation.from_matrix(np.array(imu_to_base, dtype=np.float64))  # a copy: older scipy refuses read-only input
    bias = estimate_gyro_bias(poses, times, rates, rotation)
    body_rates = rotation.apply(rates - bias)
    gyro = integrate_gyro(times, body_rates)

    # World-side correction L(t) = R_pose(t) * R_gyro(t)^-1: constant frame offset + slow gyro drift (+ fast pose
    # error, which the low-pass removes).
    gyro_at_poses = _at(times, gyro, t)
    correction = poses.rotations * gyro_at_poses.inv()
    mean = correction.mean()
    if len(t) >= 8:
        knots = np.r_[[t[0]] * 4, np.arange(t[0] + window_s, t[-1] - window_s / 2, window_s), [t[-1]] * 4]
        spline = make_lsq_spline(t, (mean.inv() * correction).as_rotvec(), knots, k=3)
        smooth = mean * Rotation.from_rotvec(spline(np.clip(times, t[0], t[-1])))
    else:  # too short for the spline: a constant correction
        smooth = Rotation.concatenate([mean] * len(times))
    refined = smooth * gyro

    before = rotation_residual_mrad(poses.rotations, gyro_at_poses)
    qa = {
        "imu_to_base_deviation_from_nominal_deg": imu_deviation_deg(imu_to_base),
        "gyro_bias_rad_s": bias.tolist(),
        "residual_before_mrad": before.tolist(),
        "residual_after_mrad": rotation_residual_mrad(_at(times, refined, t), gyro_at_poses).tolist(),
        "flagged": bool(before[:2].max() > QA_FLAG_MRAD),
    }
    return ImuTrajectory(poses, times, refined, body_rates[[0, -1]], qa)


class EgoMotionCompensation:
    """The ego motion of a scene, built once (:attr:`kitscenes.Scene.ego_motion`).

    Every part of the API that needs the ego motion uses it: ``frame.lidar``,
    ``frame.radar``, ``frame.pose``, ``scene.ego``, ``scene.accumulate`` and so on.
    Use it directly to see what a scene got, or to compensate a cloud you already
    hold::

        raw = frame.lidar("lidar_top", frame="sensor", deskew=False)
        cloud = scene.ego_motion.compensate(raw)

    Never raises for missing data; see :attr:`source`.

    Attributes:
        trajectory: The trajectory, or None without usable poses.
        source: ``"imu"`` (gyro and poses), ``"poses"`` (the poses alone) or
            None (no ego motion: clouds are not compensated, ``motion_compensated=False``).
        reason: Why the source is not ``"imu"``, else None.
        qa: The checks of the refinement (:class:`ImuTrajectory`), empty unless the source is ``"imu"``.
    """

    def __init__(self, scene: "Scene") -> None:
        self.scene = scene
        self.trajectory, self.reason = scene._pose_trajectory
        self.source: Optional[str] = None if self.trajectory is None else "poses"
        self.qa: dict = {}
        if self.trajectory is None:
            logger.info("Scene %s has no ego motion: %s", scene.id, self.reason)
            return
        self.reason = "use_imu=False" if not scene.use_imu else self._refine()
        if self.reason is not None and scene.use_imu:
            logger.info("Scene %s: ego motion from the poses alone (%s)", scene.id, self.reason)

    def _refine(self) -> Optional[str]:
        """Replace the trajectory by the refined one; returns why that is not possible, else None."""
        try:
            rig = self.scene.rig
            imu_to_base = rig.imu_to_base
            if imu_to_base is None:
                if IMU_ENTRY not in rig.transforms:
                    return f"no {IMU_ENTRY} in the calibration"
                deviation = imu_deviation_deg(rig.transforms[IMU_ENTRY][:3, :3])
                return (f"{IMU_ENTRY} in the calibration is {deviation:.1f} deg from the nominal IMU axes "
                        f"(allowed {MAX_IMU_DEVIATION_DEG} deg)")
            imu = self.scene.imu("imu", missing="none")
            if imu is None:
                return "no IMU data"
            trajectory = refine_trajectory(self.trajectory, imu["t_ns"] / 1e9, imu["angular_velocity"], imu_to_base)
        except ValueError as err:  # also DataFormatError, CalibrationError
            return str(err)
        self.trajectory, self.source, self.qa = trajectory, "imu", trajectory.qa
        return None

    def __repr__(self) -> str:
        return f"EgoMotionCompensation(scene={self.scene.id!r}, source={self.source!r}, reason={self.reason!r})"

    def compensate(self, cloud: PointCloud, *, frame: str = "base") -> PointCloud:
        """Deskew a lidar sweep read without deskewing (``deskew=False``), to its frame time.

        The cloud says everything needed: sensor, frame time, point times.

        Args:
            cloud: One sweep of this scene in the ``"sensor"`` or ``"base"`` frame.
            frame: ``"base"``, ``"sensor"`` or ``"world"``.

        Returns:
            A new cloud; without ego motion the points are only transformed, with
            ``motion_compensated=False`` and the ``reason``.

        Raises:
            ValueError: If the cloud cannot be compensated here: already deskewed,
                radar (``frame.radar`` compensates it), fused, or in another frame.
            PoseUnavailable: For ``frame="world"`` without a pose at this time.
        """
        if frame not in _COMPENSATE_FRAMES:
            raise ValueError(f"frame must be one of {_COMPENSATE_FRAMES}, got {frame!r}")
        if isinstance(cloud, RadarCloud):
            raise ValueError("compensate() takes lidar sweeps; frame.radar() compensates radar")
        if not isinstance(cloud.sensor, str):
            raise ValueError(f"compensate() takes one sweep, not a fused cloud of {cloud.sensor}")
        if cloud.motion_compensated:
            raise ValueError("The cloud is already deskewed; read it with deskew=False")
        if cloud.frame not in ("sensor", "base"):
            raise ValueError(f"The cloud must be in the 'sensor' or 'base' frame, not {cloud.frame!r}")

        T = self.scene.rig.sensor(cloud.sensor).T_base_sensor
        xyz = cloud.xyz
        if cloud.frame == "base":
            xyz = np.zeros_like(cloud.xyz)
            xyz[cloud.measured] = transform_points(invert_transform(T), cloud.xyz[cloud.measured])
        t_s = cloud.t_ns / 1e9
        inputs = deskew_inputs(xyz, cloud.attrs.get("timestamp"), T, self.trajectory, t_s, unavailable_reason=self.reason)
        out = dataclasses.replace(cloud, xyz=apply_deskew_inputs(inputs), frame="base",
                                  motion_compensated=inputs.deskewed, reason=inputs.reason)
        if frame == "sensor":
            return out.transformed(invert_transform(T), "sensor")
        if frame == "world":
            if self.trajectory is None:
                raise self.scene._pose_unavailable(None)
            rotation, translation = self.trajectory.interpolate_matrices(np.array([t_s]))
            T_world_base = np.eye(4)
            T_world_base[:3, :3], T_world_base[:3, 3] = rotation[0], translation[0]
            return out.transformed(T_world_base, "world")
        return out
