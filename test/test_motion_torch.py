"""Tests for kitscenes.motion_torch (skipped without torch)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kitscenes.motion import Trajectory, apply_deskew_inputs, compensate_points, deskew_inputs  # noqa: E402
from kitscenes.motion_torch import collate_deskew_inputs, deskew_on_device  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _trajectory() -> Trajectory:
    t = np.arange(0.0, 2.0001, 0.1)
    yaw = 0.4 * t
    quats = np.column_stack([np.zeros_like(t), np.zeros_like(t), np.sin(yaw / 2), np.cos(yaw / 2)])
    return Trajectory(t, np.column_stack([8.0 * t, 0.5 * t, np.zeros_like(t)]), quats)


def _sweep(seed: int, n: int, t0: float):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-80, 80, (n, 3)).astype(np.float32)
    xyz[::17] = 0.0  # rows without a point
    ts = np.sort(rng.uniform(t0, t0 + 0.1, n))
    T = np.eye(4)
    T[:3, 3] = [1.2, -0.3, 1.8]
    return xyz, ts, T


@pytest.mark.parametrize("device", DEVICES)
def test_device_matches_numpy_and_float64_reference(device: str) -> None:
    traj = _trajectory()
    sweeps = [_sweep(s, 5000 + 1000 * s, 0.5 + 0.3 * s) for s in range(3)]
    inputs = [deskew_inputs(x, t, T, traj, t[0] + 0.05) for x, t, T in sweeps]
    outs = deskew_on_device(inputs, device=device)
    for (x, t, T), inp, out in zip(sweeps, inputs, outs):
        got = out.cpu().numpy()
        np.testing.assert_allclose(got, apply_deskew_inputs(inp), atol=1e-4)
        ref = compensate_points(x, t, T, traj, t[0] + 0.05, time_bin_s=None, compute_dtype=np.float64)
        np.testing.assert_allclose(got, ref, atol=2e-3)  # binning + float32
        np.testing.assert_array_equal(got[~inp.measured], 0.0)


def test_collate_offsets_table_rows() -> None:
    traj = _trajectory()
    a, b = (deskew_inputs(*_sweep(s, 100, 0.5), traj, 0.55) for s in (0, 1))
    batch = collate_deskew_inputs([a, b])
    assert batch["lengths"] == [100, 100]
    assert int(batch["index"][100:].min()) >= len(a.table)


def test_failed_sweep_gets_extrinsic_only_table() -> None:
    xyz, ts, T = _sweep(0, 50, 5.0)  # far beyond the poses
    inp = deskew_inputs(xyz, ts, T, _trajectory(), 5.05)
    assert not inp.deskewed and "extrapolating" in inp.reason
    out = deskew_on_device([inp], device="cpu")[0].numpy()
    np.testing.assert_allclose(out[inp.measured], xyz[inp.measured] + T[:3, 3], atol=1e-5)
