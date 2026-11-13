"""Lidar sweeps and pose tables on a torch device, shared by two consumers.

:func:`kitscenes.accumulate.accumulate` (its ``device=`` path) and
:class:`kitscenes.mapping.VoxelMap` both read the stored sweeps unconverted
(:func:`read_frame`), look up ``T_out_base(t)`` per point in a
:class:`PoseTable` and move the points on the device
(:func:`transform_on_device`, :class:`DeviceTables`). Voxel keys
(:func:`pack_cells`) are shared too. Internal: the public names are
re-exported by :mod:`kitscenes.mapping`. Only the device functions need torch.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kitscenes._layout import sensor_member

if TYPE_CHECKING:
    import torch

    from kitscenes.motion import Trajectory
    from kitscenes.scene import Scene

#: Time margin of a pose table around the frame times (sweeps reach about 0.1 s before or after).
TABLE_MARGIN_S = 0.3
#: Rows per device batch (~1 GB of device memory while it is transformed).
BATCH_POINTS = 1 << 23
#: Voxel keys pack three signed 21-bit cell numbers into one int64.
CELL_LIMIT = 1 << 20
CELL_MASK = (1 << 21) - 1


# ----------------------------------------------------------------------
# Poses
# ----------------------------------------------------------------------


@dataclasses.dataclass
class PoseTable:
    """``T_out_base(t)`` of one scene on a uniform time grid.

    Row ``i`` holds the pose at ``t0 + (i + 0.5) * dt`` as ``[R row-major, t]``.
    The trajectory is evaluated every ``sample_s`` seconds (``samples``) and
    the rows in between are interpolated linearly as matrices, on the device
    (:meth:`on_device`): off a rotation by about ``(rate x sample_s)^2 / 8``;
    0.4 mm at 100 m on a real scene with 1 ms samples. Rows next to samples
    outside the trajectory's extrapolation and gap limits are invalid.
    """

    t0: float
    dt: float
    rows: int
    sample_s: float
    samples: np.ndarray  # (S, 12) float64 at t0 + j * sample_s
    samples_valid: np.ndarray  # (S,) bool

    @classmethod
    def build(
        cls, trajectory: "Trajectory", t_start: float, t_end: float, dt: float, A: Optional[np.ndarray] = None,
        *, sample_s: float = 1e-3,
    ) -> "PoseTable":
        """Poses from *t_start* to *t_end* seconds, each followed by the (4, 4) transform *A* if given."""
        rows = max(1, int(math.ceil((t_end - t_start) / dt)))
        sample_s = min(sample_s, rows * dt)
        times = t_start + np.arange(int(math.ceil(rows * dt / sample_s)) + 2) * sample_s
        samples, valid = _evaluate(trajectory, times, A)
        return cls(t_start, dt, rows, sample_s, samples, valid)

    def _interpolation(self, xp):
        """Per row: the sample before it and the weight of the one after (numpy or torch namespace)."""
        u = (xp.arange(self.rows, dtype=xp.float64) + 0.5) * (self.dt / self.sample_s)
        j = xp.clip(xp.floor(u), 0, len(self.samples) - 2)
        return j, u - j

    @property
    def table(self) -> np.ndarray:
        """(rows, 12) float64 (on the CPU; :meth:`on_device` is the fast path)."""
        j, a = self._interpolation(np)
        j = j.astype(np.int64)
        return (1.0 - a)[:, None] * self.samples[j] + a[:, None] * self.samples[j + 1]

    @property
    def valid(self) -> np.ndarray:
        j, _ = self._interpolation(np)
        j = j.astype(np.int64)
        return self.samples_valid[j] & self.samples_valid[j + 1]

    def on_device(self, device) -> tuple["torch.Tensor", "torch.Tensor"]:
        """``(table (rows, 12) float32, valid (rows,) bool)``, interpolated on *device*."""
        import torch

        samples = torch.from_numpy(self.samples).to(device)
        samples_valid = torch.from_numpy(self.samples_valid).to(device)
        u = (torch.arange(self.rows, dtype=torch.float64, device=device) + 0.5) * (self.dt / self.sample_s)
        j = torch.clamp(torch.floor(u), 0, len(self.samples) - 2)
        a = (u - j)[:, None]
        j = j.long()
        table = ((1.0 - a) * samples[j] + a * samples[j + 1]).float()
        return table, samples_valid[j] & samples_valid[j + 1]


def _evaluate(trajectory: "Trajectory", t: np.ndarray, A: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Table rows ``A @ T_world_base(t)`` (zeros where the trajectory can't be evaluated) and their validity."""
    valid = _queryable(trajectory, t)
    table = np.zeros((len(t), 12))
    if valid.any():
        rot, trans = trajectory.interpolate_matrices(t[valid])
        if A is not None:
            rot = A[:3, :3] @ rot
            trans = trans @ A[:3, :3].T + A[:3, 3]
        table[valid] = np.concatenate([rot.reshape(-1, 9), trans], axis=1)
    return table, valid


def sensor_table(extrinsics: Sequence[np.ndarray], scale: float) -> np.ndarray:
    """(L, 12) float32 rows ``T_base_sensor @ diag(scale)``: grid coordinates -> base frame."""
    return np.stack([np.concatenate([(T[:3, :3] * scale).reshape(9), T[:3, 3]]) for T in extrinsics]).astype(np.float32)


def _queryable(trajectory: "Trajectory", t: np.ndarray) -> np.ndarray:
    """Times the trajectory can evaluate (see ``Trajectory._check_query``), vectorized."""
    times = trajectory.times
    ok = (t >= times[0] - trajectory.max_extrapolation_s) & (t <= times[-1] + trajectory.max_extrapolation_s)
    idx = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
    in_gap = (times[idx + 1] - times[idx] > trajectory.max_gap_s) & (t > times[idx]) & (t < times[idx + 1])
    return ok & ~in_gap


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------


@dataclasses.dataclass
class RawSweeps:
    """The stored rows of several lidar sweeps, concatenated (one frame, or a batch of frames).

    Attributes:
        xyz: (N, 3) int32 grid coordinates in each sensor's frame (all stored rows, also those at
            the origin: no-returns and slots without a point; :func:`transform_on_device` skips them).
        t: (N,) float64 point times in seconds on the scene clock.
        sensor: (N,) uint8 index into the scene's lidar list.
        lengths: Rows per sweep; ``sweeps`` names them as (frame index, lidar).
        attrs: Other decoded columns, (N,) each.
        resolution: Metres per grid unit.
        untimed: Sweeps without per-point times (their rows carry the frame time).
        images: Camera images of the frame, ``{camera: (H, W, 3) uint8}`` decoded at reduced
            size, or ``{camera: bytes}`` (the JPEG, to decode on the device).
        class_images: Seamseg masks of the frame as interval tables,
            ``{camera: (starts, ends, rank, classes)}`` (see
            :meth:`kitscenes.instances.InstanceMasks.interval_table`; classes per interval).
    """

    xyz: np.ndarray
    t: np.ndarray
    sensor: np.ndarray
    lengths: list[int]
    sweeps: list[tuple[int, str]]
    attrs: dict
    resolution: float
    untimed: list[tuple[int, str]]
    images: dict = dataclasses.field(default_factory=dict)
    class_images: dict = dataclasses.field(default_factory=dict)


def read_frame(
    scene: "Scene", index: int, lidars: Sequence[str], *, columns: Optional[Sequence[str]] = ("reflectivity",),
    sensor_index: Optional[Sequence[int]] = None, skip_missing: bool = True,
    images: Sequence[str] = (), reduce: Optional[int] = 2, class_images: Sequence[str] = (),
) -> RawSweeps:
    """Read the lidar sweeps of one frame without converting them (one thread; used by the workers).

    Args:
        columns: Attributes to decode besides ``x, y, z, timestamp`` (None: all, ``timestamp`` included).
        sensor_index: The value of ``RawSweeps.sensor`` per lidar (default: its position in *lidars*).
        skip_missing: Leave out lidars without a file at this frame (else raise).
        images: Cameras whose images to decode too (at 1/*reduce* size), where present;
            with ``reduce=None`` their JPEG bytes, undecoded.
        class_images: Cameras whose seamseg masks to read too (as interval tables), where
            present and as large as the camera image.
    """
    xyz, t, sensor, lengths, sweeps, untimed = [], [], [], [], [], []
    attrs: dict[str, list] = {}
    resolution = None
    t_frame = scene.timestamps_ns[index] / 1e9
    for sid, lidar in zip(range(len(lidars)) if sensor_index is None else sensor_index, lidars):
        try:
            data = scene.files.storage.read_bytes(sensor_member(lidar, index))
        except FileNotFoundError:
            if skip_missing:
                continue
            raise
        pf = pq.ParquetFile(pa.BufferReader(data))
        names = pf.schema_arrow.names
        extra = [c for c in names if c not in ("x", "y", "z") and (columns is None or c in columns)]
        wanted = [c for c in names if c in ("x", "y", "z", "timestamp") or c in extra]
        table = pf.read(columns=wanted, use_threads=False)
        res = float((pf.schema_arrow.metadata or {}).get(b"discretization_resolution", b"0.005"))
        grid = np.column_stack([table.column(c).to_numpy() for c in "xyz"])
        if grid.dtype.kind != "i":  # float coordinates: put them on the grid
            grid = np.round(grid / res).astype(np.int32)
        if resolution is None:
            resolution = res
        elif res != resolution:
            grid = np.round(grid * (res / resolution)).astype(np.int32)
        xyz.append(grid.astype(np.int32, copy=False))
        if "timestamp" in names:
            t.append(table.column("timestamp").to_numpy())
        else:
            t.append(np.full(len(grid), t_frame))
            untimed.append((index, lidar))
        sensor.append(np.full(len(grid), sid, np.uint8))
        lengths.append(len(grid))
        sweeps.append((index, lidar))
        for c in extra:
            attrs.setdefault(c, []).append(table.column(c).to_numpy())
    keep = [k for k, v in attrs.items() if len(v) == len(lengths)]  # columns every sweep has
    raw = RawSweeps(
        xyz=np.concatenate(xyz) if xyz else np.zeros((0, 3), np.int32),
        t=np.concatenate(t) if t else np.zeros(0),
        sensor=np.concatenate(sensor) if sensor else np.zeros(0, np.uint8),
        lengths=lengths, sweeps=sweeps, attrs={k: np.concatenate(attrs[k]) for k in keep},
        resolution=resolution or 0.005, untimed=untimed,
    )
    if images or class_images:
        frame = scene.frame(index)
        for camera in images:
            if reduce is None:
                image = frame.image_bytes(camera, missing="none")
            else:
                image = frame.image(camera, reduce=reduce, missing="none")
            if image is not None:
                raw.images[camera] = image
        for camera in class_images:
            masks = frame.instances(camera, missing="none")
            if masks is None:
                continue
            if tuple(masks.image_size) != tuple(scene.files.rig_original.cameras[camera].image_size):  # release size
                continue  # pixels of the camera would not index these masks
            starts, ends, rank, instance = masks.interval_table()
            raw.class_images[camera] = (starts, ends, rank, masks.classes[instance].astype(np.int64))
    return raw


def concat_raw(parts: Sequence[RawSweeps]) -> RawSweeps:
    shared = [k for k in parts[0].attrs if all(k in p.attrs for p in parts)]
    for p in parts[1:]:
        if p.resolution != parts[0].resolution:
            raise ValueError("Sweeps with different storage resolutions in one batch.")
    return RawSweeps(
        xyz=np.concatenate([p.xyz for p in parts]), t=np.concatenate([p.t for p in parts]),
        sensor=np.concatenate([p.sensor for p in parts]), lengths=[n for p in parts for n in p.lengths],
        sweeps=[s for p in parts for s in p.sweeps], attrs={k: np.concatenate([p.attrs[k] for p in parts]) for k in shared},
        resolution=parts[0].resolution, untimed=[u for p in parts for u in p.untimed],
    )


# ----------------------------------------------------------------------
# Device transform and voxel keys
# ----------------------------------------------------------------------


def transform_on_device(parts: Sequence[RawSweeps], poses: "torch.Tensor", valid_rows: "torch.Tensor",
                        sensors: "torch.Tensor", t0: float, dt: float, frame_times: Optional[dict[int, float]] = None,
                        *, ego_box: Optional[Sequence[float]] = None, max_range: Optional[float] = None):
    """Move the measured rows of *parts* into the output frame: ``poses[bin(t)] @ sensors[sensor] @ xyz``.

    *poses* is a :class:`PoseTable` as a (M, 12) float32 tensor, *sensors* a
    :func:`sensor_table` for the parts' resolution, on the same device. The
    parts are concatenated on the device. Points inside *ego_box* (half
    extents around the base origin) or farther than *max_range* from it in
    x, y are dropped, both in the base frame.

    Returns:
        ``(out, rows, bad, measured)``: (K, 3) float32 points on the table's
        device, (K,) int64 indices of their rows in the concatenated parts,
        the sweeps (indices into the concatenated ``sweeps``) whose times left
        the table, and the number of measured rows. With *frame_times*
        ({frame index: seconds}) such sweeps are placed with the pose at the
        frame time instead (not deskewed, like the exact path); without, their
        out-of-table points are dropped.
    """
    import torch

    dev = poses.device
    m = len(valid_rows)
    upload = lambda arrays: torch.cat([torch.from_numpy(a).to(dev, non_blocking=True) for a in arrays])
    xyz = upload([p.xyz for p in parts])
    t = upload([p.t for p in parts])
    sensor = upload([p.sensor for p in parts]).long()
    lengths = [n for p in parts for n in p.lengths]
    sweeps = [s for p in parts for s in p.sweeps]
    measured = (xyz != 0).any(dim=1)
    b = torch.floor((t - t0) / dt).long()
    ok = measured & (b >= 0) & (b < m)
    ok &= valid_rows[b.clamp(0, m - 1)]
    sweep_of = torch.repeat_interleave(
        torch.arange(len(lengths), device=dev), torch.tensor(lengths, device=dev), output_size=len(xyz),
    )
    counts = torch.zeros(len(lengths) + 1, dtype=torch.long, device=dev).index_add_(0, sweep_of, (measured & ~ok).long())
    counts[-1] = measured.sum()
    counts = counts.tolist()  # the one sync per batch
    bad, n_measured = [i for i, n in enumerate(counts[:-1]) if n], counts[-1]
    if bad and frame_times is not None:
        frame_t = torch.tensor([frame_times[sweeps[i][0]] for i in range(len(sweeps))], dtype=torch.float64, device=dev)
        redo = torch.zeros(len(lengths), dtype=torch.bool, device=dev)
        redo[bad] = True
        redo = redo[sweep_of]
        b = torch.where(redo, torch.floor((frame_t[sweep_of] - t0) / dt).long(), b)
        ok = measured & (b >= 0) & (b < m)
        ok &= valid_rows[b.clamp(0, m - 1)]
    rows = torch.nonzero(ok)[:, 0]
    del t, measured, ok, sweep_of
    p = _apply_rows(sensors.index_select(0, sensor[rows]), xyz.index_select(0, rows).float())
    if ego_box is not None or max_range is not None:
        keep = torch.ones(len(p), dtype=torch.bool, device=dev)
        if ego_box is not None:
            box = torch.tensor(ego_box, dtype=p.dtype, device=dev)
            keep &= ~(p.abs() < box).all(dim=1)
        if max_range is not None:
            keep &= p[:, 0] ** 2 + p[:, 1] ** 2 <= max_range ** 2
        rows, p = rows[keep], p[keep]
    return _apply_rows(poses.index_select(0, b[rows]), p), rows, bad, n_measured


def _apply_rows(r: "torch.Tensor", p: "torch.Tensor") -> "torch.Tensor":
    """Per-point ``R @ p + t`` from (N, 12) rows ``[R row-major, t]``."""
    x, y, z = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    return r[:, 0:9:3] * x + r[:, 1:9:3] * y + r[:, 2:9:3] * z + r[:, 9:12]


def pack_cells(cells):
    """(N, 3) int64 cell numbers (numpy or torch) -> (N,) int64 keys."""
    if len(cells) and int(abs(cells).max()) >= CELL_LIMIT:
        raise ValueError(f"The map spans more than {CELL_LIMIT} voxels from its anchor; use a larger voxel_size.")
    c = cells + CELL_LIMIT
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def unpack_cells(keys: np.ndarray) -> np.ndarray:
    return np.stack([(keys >> 42) & CELL_MASK, (keys >> 21) & CELL_MASK, keys & CELL_MASK], axis=1) - CELL_LIMIT


class DeviceTables:
    """A scene's pose table and lidar extrinsics on the device (per storage resolution)."""

    def __init__(
        self, poses: PoseTable, extrinsics: Sequence[np.ndarray], device, frame_times: Optional[dict] = None,
        *, ego_box: Optional[Sequence[float]] = None, max_range: Optional[float] = None,
    ) -> None:
        import torch

        self.poses, self.extrinsics, self.device, self.frame_times = poses, extrinsics, device, frame_times
        self.ego_box, self.max_range = ego_box, max_range
        self.table, self.valid = poses.on_device(device)
        self._sensors: dict[float, "torch.Tensor"] = {}

    def transform(self, parts: Sequence[RawSweeps]):
        import torch

        resolution = parts[0].resolution
        if any(p.resolution != resolution for p in parts):
            raise ValueError("Sweeps with different storage resolutions in one batch.")
        if resolution not in self._sensors:
            self._sensors[resolution] = torch.from_numpy(sensor_table(self.extrinsics, resolution)).to(self.device)
        return transform_on_device(
            parts, self.table, self.valid, self._sensors[resolution], self.poses.t0, self.poses.dt, self.frame_times,
            ego_box=self.ego_box, max_range=self.max_range,
        )
