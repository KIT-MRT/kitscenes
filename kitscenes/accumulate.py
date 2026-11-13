"""Several frames of lidar in one frame of reference.

:func:`accumulate` deskews every sweep (as :meth:`kitscenes.frame.Frame.lidar`)
and moves it with its frame's pose into the world frame, or on into the base
frame of one reference frame. Static structure lines up; moving objects smear
along their path. Rows without a measurement are dropped.

On the CPU (default), the frame pose and the output transform are folded
into each sweep's small deskew table (:class:`kitscenes.motion.DeskewInputs`),
so every point is transformed once; sweeps are read, prepared and
transformed in a thread pool. With ``device=`` the CPU only reads and
decodes: the time binning, the pose lookup (one table of the scene's poses on
a 0.02 ms grid, :class:`kitscenes._sweeps.PoseTable`, shared with the voxel maps), the transform and the
thinning run batched on a torch device such as a GPU.
"""

from __future__ import annotations

import dataclasses
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Sequence

import numpy as np

from kitscenes._sweeps import BATCH_POINTS, TABLE_MARGIN_S, pack_cells
from kitscenes.constants import LidarNameLike, sensor_names
from kitscenes.motion import DEFAULT_TIME_BIN_S
from kitscenes.points import PointCloud
from kitscenes.rig import invert_transform
from kitscenes.schema import Provenance

if TYPE_CHECKING:
    from kitscenes.frame import Frame
    from kitscenes.scene import Scene

__all__ = ["accumulate"]


@dataclasses.dataclass
class _Sweep:
    """One measured sweep (CPU path), ready to transform: ``table[index]`` maps ``xyz`` into the output frame."""

    frame_index: int
    lidar: str
    xyz: np.ndarray  # (N, 3) float32, raw sensor frame, measured rows only
    index: np.ndarray  # (N,) int32 rows of table
    table: np.ndarray  # (M, 12) float32 [R row-major, t], sensor -> output (relative to the origin)
    center: np.ndarray  # (3,) the frame's base origin in output coordinates
    attrs: dict
    reason: Optional[str]  # why the sweep is not deskewed, else None
    # The transformed points and the rows they came from (None: all).
    out: Optional[np.ndarray] = None
    kept: Optional[np.ndarray] = None
    keys: Optional[np.ndarray] = None


def accumulate(
    scene: "Scene",
    frames: Iterable[int],
    lidars: Optional[Sequence[LidarNameLike]] = None,
    *,
    frame: str = "world",
    reference: Optional[int] = None,
    voxel_size: Optional[float] = None,
    max_range: Optional[float] = None,
    device=None,
    workers: Optional[int] = None,
) -> PointCloud:
    """Lidar points of several frames in one frame.

    Args:
        scene: The scene.
        frames: Frame indices to accumulate.
        lidars: Lidars per frame (default: all with a sweep at that frame).
        frame: ``"world"`` (the frame of the poses), ``"base"`` (the base
            frame of *reference* at its time) or ``"road"`` (its road frame,
            :attr:`kitscenes.Scene.T_road_base`).
        reference: Frame whose base or road frame to use (default: the first
            of *frames*); the result's ``t_ns`` is its time.
        voxel_size: Keep one point per cube of this size in metres (the first
            in frame order), to bound the size of long accumulations. The
            cubes are aligned with the output frame.
        max_range: Drop points farther than this from their own frame's base origin.
        device: Experimental. A torch device (e.g. ``"cuda"``) for the time binning, pose
            lookup, transform and thinning; needs ``torch``. Default: numpy
            in the worker threads. The device path looks poses up on a fixed
            0.02 ms grid (interpolated from the trajectory every 1 ms) instead
            of per-sweep bins: the two differ by up to 0.01 ms of motion (0.3
            mm at 30 m/s) plus about 0.5 mm at 100 m range, and with
            *voxel_size* a few points per million land in a neighbouring cube. Sweeps whose times leave the trajectory are placed with the
            frame pose, not deskewed, on both paths.
        workers: Threads reading (and on the CPU preparing) sweeps (default:
            up to 4 on the CPU path, 16 with a device).

    Returns:
        A cloud with ``frame_index`` (int32), ``source`` (index into
        ``sensor``) and the shared lidar attributes per point. World clouds
        have float64 coordinates, base and road clouds float32.

    Raises:
        ValueError: If the scene has no usable poses, or for a bad *frame*.
        PoseUnavailable: If a frame lies outside the poses or in a gap between them.
        SensorUnavailable: If a requested lidar has no sweep at a frame.
    """
    if frame not in ("world", "base", "road"):
        raise ValueError(f"frame must be 'world', 'base' or 'road', got {frame!r}")
    frames = list(frames)
    if not frames:
        raise ValueError("No frames to accumulate.")
    if scene._trajectory[0] is None:
        raise ValueError(f"Scene {scene.id} has no usable poses ({scene._trajectory[1]}); nothing to accumulate into.")
    reference = frames[0] if reference is None else reference
    scene.rig  # load once here, not racing in the threads

    # Output transform A (world -> output) and the origin the float32 points are relative to.
    if frame in ("base", "road"):
        A, origin = invert_transform(_pose(scene, reference)), np.zeros(3)
        if frame == "road":
            A = scene.T_road_base @ A
    else:
        origin = _pose(scene, frames[0])[:3, 3].astype(np.float64)
        if voxel_size is not None:
            origin = np.round(origin / voxel_size) * voxel_size  # keep the cubes aligned with the world frame
        A = np.eye(4)
        A[:3, 3] = -origin

    jobs, names = [], []
    for index in frames:
        f = scene.frame(index)
        for name in (f.sensors("lidar") if lidars is None else sensor_names(lidars)):
            jobs.append((f, name, f._require(name, "lidar")))
            if name not in names:
                names.append(name)
    source_of = {name: i for i, name in enumerate(names)}
    if device is not None:
        return _accumulate_on_device(
            scene, frames, [(f.index, name) for f, name, _ in jobs], names, A, origin, frame, reference,
            voxel_size, max_range, device, workers,
        )

    prepare = lambda job: _prepare(job, A, voxel_size, max_range)
    chunks: list[tuple[list[_Sweep], np.ndarray, np.ndarray, object]] = []  # sweeps, points, kept rows, keys
    reasons: list[str] = []
    for sweep in _in_order(prepare, jobs, workers or min(4, os.cpu_count() or 1)):
        if sweep.reason is not None:
            _not_deskewed(scene, reasons, sweep.frame_index, sweep.lidar, sweep.reason)
        chunks.append(([sweep], sweep.out, sweep.kept, sweep.keys))
        sweep.out = sweep.kept = sweep.keys = None

    xyz = np.concatenate([c[1] for c in chunks]) if chunks else np.zeros((0, 3), np.float32)
    shared = [k for k in (chunks[0][0][0].attrs if chunks else {}) if all(k in s.attrs for c in chunks for s in c[0])]
    attrs = {k: np.concatenate([_rows(c[0], c[2], lambda s: s.attrs[k]) for c in chunks]) for k in shared}
    attrs["frame_index"] = np.concatenate(
        [_rows(c[0], c[2], lambda s: np.full(len(s.xyz), s.frame_index, np.int32)) for c in chunks]
    ) if chunks else np.zeros(0, np.int32)
    source = np.concatenate(
        [_rows(c[0], c[2], lambda s: np.full(len(s.xyz), source_of[s.lidar], np.uint8)) for c in chunks]
    ) if chunks else np.zeros(0, np.uint8)
    if voxel_size is not None and len(xyz):  # chunks are thinned on their own; now across chunks
        first = _first_per_key([c[3] for c in chunks])
        xyz, source = xyz[first], source[first]
        attrs = {k: v[first] for k, v in attrs.items()}

    return PointCloud(
        xyz=xyz if frame != "world" else xyz.astype(np.float64) + origin,
        attrs=attrs, measured=np.ones(len(xyz), bool), frame=frame, t_ns=scene.frame(reference).t_ns,
        sensor=tuple(names), motion_compensated=not reasons, reason="; ".join(reasons) or None,
        source_index=source, provenance=Provenance.GEOMETRY_DERIVED,
    )


def _pose(scene: "Scene", index: int) -> np.ndarray:
    """``T_world_base`` at frame *index*; raises PoseUnavailable where the poses do not reach."""
    return scene.frame(index).T_world_base


def _not_deskewed(scene: "Scene", reasons: list, frame_index: int, lidar: str, reason: str) -> None:
    scene._warn_once(
        f"deskew:{reason}",
        "Lidar sweeps of scene %s are not motion compensated (%s); points are only transformed.",
        scene.id, reason,
    )
    reasons.append(f"frame {frame_index} {lidar}: {reason}")


def _in_order(fn, jobs: list, workers: int) -> Iterator:
    """``map(fn, jobs)`` on a thread pool, in order, with a bounded number of jobs in flight."""
    if workers <= 1:
        yield from map(fn, jobs)
        return
    with ThreadPoolExecutor(workers) as pool:
        pending: deque = deque()
        for job in jobs:
            pending.append(pool.submit(fn, job))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _prepare(job: tuple, A: np.ndarray, voxel_size: Optional[float], max_range: Optional[float]) -> _Sweep:
    """Read one sweep and compose its deskew table with the frame pose and *A* (worker thread)."""
    f, name, member = job
    inputs, cols, _, _ = f._lidar_inputs(name, member, True)
    P = A @ _pose(f.scene, f.index)  # base at the frame time -> output
    table = inputs.table.astype(np.float64)
    rot = P[:3, :3] @ table[:, :9].reshape(-1, 3, 3)
    trans = table[:, 9:] @ P[:3, :3].T + P[:3, 3]
    v = inputs.measured
    sweep = _Sweep(
        frame_index=f.index, lidar=name, xyz=inputs.xyz[v], index=inputs.index[v],
        table=np.concatenate([rot.reshape(-1, 9), trans], axis=1).astype(np.float32),
        center=P[:3, 3].copy(), attrs={k: c[v] for k, c in cols.items() if k != "row"},  # as the device path
        reason=None if inputs.deskewed else inputs.reason,
    )
    sweep.out, sweep.kept, sweep.keys = _cpu_chunk(sweep, voxel_size, max_range)
    return sweep


def _cpu_chunk(s: _Sweep, voxel_size: Optional[float], max_range: Optional[float]):
    rows = s.table[s.index]
    x, y, z = s.xyz[:, 0], s.xyz[:, 1], s.xyz[:, 2]
    out = np.empty_like(s.xyz)
    for axis in range(3):
        out[:, axis] = rows[:, 3 * axis] * x + rows[:, 3 * axis + 1] * y + rows[:, 3 * axis + 2] * z + rows[:, 9 + axis]
    kept = None
    if max_range is not None:
        kept = np.flatnonzero(np.linalg.norm(out - s.center.astype(np.float32), axis=1) <= max_range)
    keys = None
    if voxel_size is not None:
        keys = pack_cells(np.floor((out if kept is None else out[kept]) / np.float32(voxel_size)).astype(np.int64))
        _, first = np.unique(keys, return_index=True)  # first occurrence of every key
        first.sort()
        kept, keys = (first if kept is None else kept[first]), keys[first]
    return (out if kept is None else out[kept]), kept, keys


def _first_per_key_torch(keys):
    import torch

    unique, inverse = torch.unique(keys, return_inverse=True)
    first = torch.full((len(unique),), len(keys), dtype=torch.long, device=keys.device)
    first.scatter_reduce_(0, inverse, torch.arange(len(keys), device=keys.device), reduce="amin")
    return torch.sort(first).values


def _first_per_key(parts: list) -> np.ndarray:
    """Positions of the first point of every key in the concatenated chunks, in order."""
    if isinstance(parts[0], np.ndarray):
        _, first = np.unique(np.concatenate(parts), return_index=True)
        first.sort()
        return first
    import torch

    return _first_per_key_torch(torch.cat(parts)).cpu().numpy()


def _rows(sweeps: list[_Sweep], kept: Optional[np.ndarray], column) -> np.ndarray:
    """``column(sweep)`` of the chunk's sweeps, concatenated, at the kept rows (None: all)."""
    values = [column(s) for s in sweeps]
    joined = values[0] if len(values) == 1 else np.concatenate(values)
    return joined if kept is None else joined[kept]


def _accumulate_on_device(
    scene: "Scene", frames: list[int], sweeps: list[tuple[int, str]], names: list[str], A: np.ndarray,
    origin: np.ndarray, frame: str, reference: int, voxel_size: Optional[float], max_range: Optional[float],
    device, workers: Optional[int],
) -> PointCloud:
    """:func:`accumulate` with binning, pose lookup, transform and thinning on a torch device."""
    import torch

    from kitscenes._sweeps import DeviceTables, PoseTable, concat_raw, read_frame

    dev = torch.device(device)
    extrinsics = [scene.rig.sensor(name).T_base_sensor for name in names]
    times = scene.timestamps_ns[frames] / 1e9
    poses = PoseTable.build(
        scene._trajectory[0], times.min() - TABLE_MARGIN_S, times.max() + TABLE_MARGIN_S, DEFAULT_TIME_BIN_S, A,
    )
    tables = DeviceTables(poses, extrinsics, dev, frame_times=dict(zip(frames, times)))
    centers = {i: (A @ _pose(scene, i))[:3, 3] for i in frames} if max_range is not None else None
    by_frame: dict[int, list[str]] = {}
    for index, name in sweeps:
        by_frame.setdefault(index, []).append(name)
    order = [i for i in dict.fromkeys(frames) if i in by_frame]
    read = lambda i: read_frame(
        scene, i, by_frame[i], columns=None, sensor_index=[names.index(n) for n in by_frame[i]], skip_missing=False,
    )

    reasons: list[str] = []
    parts = []  # (raw, points, rows, keys)

    def flush(batch: list) -> None:
        raw = concat_raw(batch)
        out, rows, bad, _ = tables.transform([raw])
        for i in bad:
            _not_deskewed(scene, reasons, *raw.sweeps[i], "point times outside the usable poses")
        for index, lidar in raw.untimed:
            _not_deskewed(scene, reasons, index, lidar, "sweep has no per-point timestamps")
        sweep_of = torch.repeat_interleave(
            torch.arange(len(raw.lengths), device=dev), torch.tensor(raw.lengths, device=dev), output_size=len(raw.t),
        ).index_select(0, rows)
        keep = torch.arange(len(rows), device=dev)
        if max_range is not None:
            c = torch.tensor(np.stack([centers[i] for i, _ in raw.sweeps]), dtype=torch.float32, device=dev)
            keep = torch.nonzero(torch.linalg.norm(out - c.index_select(0, sweep_of), dim=1) <= max_range)[:, 0]
        keys = None
        if voxel_size is not None:
            keys = pack_cells(torch.floor(out.index_select(0, keep) / voxel_size).long())
            first = _first_per_key_torch(keys)
            keep, keys = keep[first], keys[first]
        parts.append((raw, out.index_select(0, keep).cpu().numpy(), rows.index_select(0, keep).cpu().numpy(), keys))

    batch: list = []
    for raw in _in_order(read, order, workers or min(16, os.cpu_count() or 1)):
        batch.append(raw)
        if sum(len(r.t) for r in batch) >= BATCH_POINTS:
            flush(batch)
            batch = []
    if batch:
        flush(batch)

    if parts:
        xyz = np.concatenate([p[1] for p in parts])
        shared = [k for k in parts[0][0].attrs if all(k in p[0].attrs for p in parts)]
        attrs = {k: np.concatenate([p[0].attrs[k][p[2]] for p in parts]) for k in shared}
        attrs["frame_index"] = np.concatenate([
            np.repeat(np.array([i for i, _ in p[0].sweeps], np.int32), p[0].lengths)[p[2]] for p in parts
        ])
        source = np.concatenate([p[0].sensor[p[2]] for p in parts])
    else:
        xyz, attrs, source = np.zeros((0, 3), np.float32), {"frame_index": np.zeros(0, np.int32)}, np.zeros(0, np.uint8)
    if voxel_size is not None and len(xyz):  # batches are thinned on their own; now across batches
        first = _first_per_key([p[3] for p in parts])
        xyz, source = xyz[first], source[first]
        attrs = {k: v[first] for k, v in attrs.items()}
    return PointCloud(
        xyz=xyz if frame != "world" else xyz.astype(np.float64) + origin,
        attrs=attrs, measured=np.ones(len(xyz), bool), frame=frame, t_ns=scene.frame(reference).t_ns,
        sensor=tuple(names), motion_compensated=not reasons, reason="; ".join(reasons) or None,
        source_index=source, provenance=Provenance.GEOMETRY_DERIVED,
    )
