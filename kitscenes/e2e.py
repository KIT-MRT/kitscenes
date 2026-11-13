"""End-to-end planning helpers: a causal ego state and the high-level driving command.

Experimental: the benchmark's navigation input is not final; the test-e2e
scenes will ship their commands.

The trajectories themselves come from :meth:`kitscenes.Frame.future` or
``SampleSpec(future_s=...)``; ``step_s`` / ``trajectory_step_s`` picks the
benchmark sampling directly, e.g. 3 s at 2 Hz (UniAD, VAD)::

    frame.future(3.0, step_s=0.5)                         # 6 slots, 0.5 s .. 3.0 s
    kitscenes.SampleSpec(future_s=5.0, trajectory_step_s=0.25)   # 20 slots, 5 s at 4 Hz

:func:`driving_command` follows the UniAD / VAD convention on nuScenes: the
command is derived from the ground-truth future itself (where the vehicle
went), so it leaks part of the answer to a planner that is conditioned on
it. In test-e2e the future is withheld, so it cannot be computed there.

:func:`past_motion` is the ego state of the samples (``sample["ego"]``) and
of ``frame.ego(causal=True)``: from past poses only. ``frame.ego()``
differentiates centrally and so reads the poses after the frame (0.1 s,
0.2 s for the acceleration).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from kitscenes.ego_trajectory import DEFAULT_STEP_S

if TYPE_CHECKING:
    from kitscenes.ego_trajectory import EgoTrajectory
    from kitscenes.navigation import Command

__all__ = ["past_motion", "driving_command", "COMMAND_THRESHOLD_M", "COMMAND_HORIZON_S", "EGO_WINDOW_S"]

#: The lateral offset (m) at the horizon beyond which the command is a turn (UniAD, VAD: 2 m at 3 s).
COMMAND_THRESHOLD_M = 2.0
#: The horizon (s) the command looks at (UniAD, VAD: the 6th waypoint at 2 Hz).
COMMAND_HORIZON_S = 3.0
#: The past (s) :func:`past_motion` is fitted to in the samples: 6 poses at 10 Hz and the frame's own.
EGO_WINDOW_S = 0.5


def past_motion(past: "EgoTrajectory") -> Optional[dict]:
    """The ego motion at the frame time from the past only: a least-squares quadratic over time.

    Fitted to the valid slots of *past* (``frame.past(EGO_WINDOW_S)``,
    in its ego frame) and the frame's own pose, evaluated at the frame time:
    ``{"velocity": (3,), "acceleration": (3,), "yaw_rate": float, "speed": float}``
    in that frame's axes (m/s, m/s², rad/s; yaw rate positive turning left).
    None with fewer than two valid slots. The quadratic lags behind sharp
    changes and in a tight turn misreads the arc (0.6 m/s² along x at 10 m/s
    on a 20 m radius); on a sample of release scenes it is within 0.1 m/s (median) of the
    central differences of :meth:`kitscenes.Frame.ego`.
    """
    from scipy.spatial.transform import Rotation

    ok = np.asarray(past.valid, dtype=bool)
    if ok.sum() < 2:
        return None
    tau = np.append((past.t_ns[ok] - past.anchor_t_ns) / 1e9, 0.0)
    xyz = np.vstack([past.xyz_ego[ok], np.zeros(3)])
    yaw = np.unwrap(np.append(Rotation.from_quat(past.quat_ego[ok]).as_euler("ZYX")[:, 0], 0.0))
    V = np.vander(tau, 3)  # tau^2, tau, 1
    c_xyz = np.linalg.lstsq(V, xyz, rcond=None)[0]
    c_yaw = np.linalg.lstsq(V, yaw, rcond=None)[0]
    return {"velocity": c_xyz[1], "acceleration": 2 * c_xyz[0], "yaw_rate": float(c_yaw[1]),
            "speed": float(np.linalg.norm(c_xyz[1]))}


def driving_command(xyz: np.ndarray, valid: Optional[np.ndarray] = None, *, step_s: float = DEFAULT_STEP_S,
                    horizon_s: float = COMMAND_HORIZON_S, threshold_m: float = COMMAND_THRESHOLD_M) -> "Command":
    """The driving command of a future trajectory, a :class:`kitscenes.navigation.Command`.

    From the lateral offset y (left positive) of the last valid slot up to
    *horizon_s*: ``y >= threshold_m`` is LEFT, ``y <= -threshold_m`` RIGHT,
    else STRAIGHT (UniAD's and VAD's rule, in their frames x >= 2 m is right;
    the rule of :data:`kitscenes.navigation.NUSCENES`). UNKNOWN if no slot up
    to the horizon is valid. ``command.onehot("vad")`` gives UniAD's and
    VAD's one-hot order.

    Args:
        xyz: (K, 2+) future positions in the ego frame at the frame time
            (``EgoTrajectory.xyz_ego``, ``sample["future"]["xyz_ego"]``),
            slot k at ``(k + 1) * step_s``.
        valid: (K,) bool, None for all valid.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    k = min(len(xyz), int(round(horizon_s / step_s)))
    ok = np.ones(k, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)[:k]
    from kitscenes.navigation import Command, _from_offset

    if not ok.any():
        return Command.UNKNOWN
    return _from_offset(float(xyz[np.flatnonzero(ok)[-1], 1]), threshold_m)
