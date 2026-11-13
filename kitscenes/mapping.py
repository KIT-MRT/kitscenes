"""Voxelized lidar maps of many frames or scenes, on a torch device.

Points are moved with the pose at their own time (a pose table on a 0.1 ms
grid: at most half a step of motion off, a few millimetres). Per occupied
voxel the map keeps the centroid, the point count, the number of frames that
hit it, the longest run of consecutive frames and the mean reflectivity, so
memory grows with the mapped surface, not with the number of frames.
``to_cloud(min_run=3)`` drops moving objects: a static surface is hit frame
after frame, a passing car occupies a voxel for an instant. Scenes are placed
on a voxel grid fixed in UTM, so maps of single scenes merge exactly
(:meth:`VoxelMap.save`, :meth:`VoxelMap.merge`). See docs/lidar_maps.md.

Experimental: :class:`VoxelMap`'s API and save format may still change.
Needs ``torch`` (any device; ``"cpu"`` works, slower)::

    from kitscenes.mapping import VoxelMap, keyframes

    vmap = VoxelMap(voxel_size=0.05, device="cuda")
    for scene in scenes:                        # e.g. overlapping drives of one city
        vmap.add_scene(scene, keyframes(scene, min_distance=0.5))
    cloud = vmap.to_cloud(min_run=3)            # PointCloud in the first scene's map frame, traffic removed
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Sequence

import numpy as np

from kitscenes._sweeps import (
    BATCH_POINTS,
    CELL_LIMIT,
    CELL_MASK,
    TABLE_MARGIN_S,
    DeviceTables,
    PoseTable,
    RawSweeps,
    pack_cells,
    read_frame,
    unpack_cells,
)
from kitscenes._layout import sensor_kind
from kitscenes.constants import LidarNameLike, sensor_names
from kitscenes.points import PointCloud
from kitscenes.rig import invert_transform
from kitscenes.schema import Provenance

if TYPE_CHECKING:
    import torch

    from kitscenes.map import MapOrigin
    from kitscenes.rig import Camera
    from kitscenes.scene import Scene

#: Grid anchors lie on this lattice in UTM (metres), so maps of different scenes share their voxels.
ANCHOR_STEP = 1000.0
#: Voxel entries per device slice when merging runs (~1 GB of device memory).
_MERGE_SLICE = 1 << 24
#: Merge on the device when the unmerged runs hold this many entries, or as many as the merged run.
_MERGE_MIN = 1 << 22
#: "First seen" codes: scene index << 20 | frame index; run ordinals: scene index << 20 | position in add_scene's frames.
_FRAME_BITS = 20
#: Time step of the pose table in seconds; the deskew error is at most half a step of motion.
_TABLE_BIN_S = 1e-4
from kitscenes.constants import EGO_BOX  # noqa: E402  (re-exported: returns off the car's own body)

__all__ = ["VoxelMap", "keyframes", "ANCHOR_STEP", "EGO_BOX"]


# ----------------------------------------------------------------------
# Frames and poses
# ----------------------------------------------------------------------


def keyframes(
    scene: "Scene",
    frames: Optional[Iterable[int]] = None,
    *,
    min_distance: float = 1.0,
    min_angle_deg: float = 10.0,
) -> list[int]:
    """The frames where the car moved *min_distance* metres or turned *min_angle_deg* since the last kept one.

    The first frame is always kept. A car waiting at a light adds frames but no new view.

    Raises:
        PoseUnavailable: Without usable poses, or for a frame outside them.
    """
    frames = list(range(len(scene)) if frames is None else frames)
    if not frames:
        return []
    T = scene.ego(scene.timestamps_ns[frames]).T_world_base
    kept, last = [frames[0]], T[0]
    for index, pose in zip(frames[1:], T[1:]):
        moved = np.linalg.norm(pose[:3, 3] - last[:3, 3])
        cos = (np.trace(last[:3, :3].T @ pose[:3, :3]) - 1.0) / 2.0
        if moved >= min_distance or np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) >= min_angle_deg:
            kept.append(index)
            last = pose
    return kept


# ----------------------------------------------------------------------
# The map
# ----------------------------------------------------------------------

#: Statistics that add up (``rgb_sum``/``rgb_count`` only in maps with colour; ``class_seen``, the
#: points a segmented camera saw, only in maps with classes).
_SUMS = ("count", "offset_sum", "reflectivity_sum", "frames", "rgb_sum", "rgb_count", "class_seen")
_RUN_FIELDS = ("run_first", "run_last", "run_prefix", "run_suffix", "run_best")
#: Class votes, in maps with classes: (voxel key, class) pairs sorted by both, with counts.
_VOTES = ("vote_keys", "vote_classes", "vote_counts")
#: Version of the .npz written by VoxelMap.save.
_FORMAT = 5


class VoxelMap:
    """A voxelized lidar map of one or more scenes, built on a torch device.

    Per occupied voxel it keeps the number of points, the sum of their offsets
    in the voxel (for the centroid), the reflectivity sum, the number of
    distinct frames that hit it and the first frame that did. Voxels are
    aligned with a grid fixed in UTM (see :data:`ANCHOR_STEP`), so maps built
    separately merge exactly. Finished scenes are kept in host memory (68
    bytes per voxel, 84 with colour, plus about 2 per voxel for class votes);
    the device holds the scene being added.

    Args:
        voxel_size: Edge length in metres.
        device: Torch device for the transform and the voxel statistics.
        origin: Map frame of the output (default: the first scene's map frame).
        colour: Also average the ring camera colour of the points
            (:mod:`kitscenes.painting`).
        gpu_decode: Decode the camera images on the device at full size
            (``torchvision.io.decode_jpeg``; the workers only read the
            files). Default: on a CUDA device with torchvision installed.
            Otherwise they are decoded on the CPU at 1/*reduce* size.
        reduce: Size of the images decoded on the CPU, as in
            :meth:`kitscenes.Frame.image`: 1, 2, 4 or 8.
        classes: Also keep the majority Mapillary Vistas class of the
            points, from the seamseg masks of the front and rear cameras
            (:meth:`to_cloud`).
        max_device_voxels: Move the scene being added to host memory when its
            merged statistics exceed this many voxels. Device memory peaks at
            about 220 bytes per voxel of this budget (3.5 GB by default).
    """

    def __init__(
        self, voxel_size: float = 0.05, *, device="cuda", origin: Optional["MapOrigin"] = None,
        colour: bool = False, classes: bool = False, gpu_decode: Optional[bool] = None, reduce: int = 2,
        max_device_voxels: int = 1 << 24,
    ) -> None:
        from kitscenes._optional import require_torch

        torch = require_torch("kitscenes.mapping.VoxelMap")

        if reduce not in (1, 2, 4, 8):
            raise ValueError("reduce must be 1, 2, 4 or 8")
        self.colour, self.classes, self.reduce = colour, classes, reduce
        if gpu_decode is None:
            gpu_decode = torch.device(device).type == "cuda" and _has_torchvision()
        self.gpu_decode = gpu_decode
        self.voxel_size = float(voxel_size)
        self.device = torch.device(device)
        self.origin = origin
        self.anchor = None if origin is None else _anchor(origin)
        self.max_device_voxels = max_device_voxels
        self.scenes: list[str] = []
        self.lidars: list[str] = []
        self._runs: list[dict] = []  # host (numpy), each sorted by key
        self._device: list[dict] = []  # device (torch) runs of the scene being added

    def __len__(self) -> int:
        self._compact()
        return len(self._runs[0]["keys"]) if self._runs else 0

    def __repr__(self) -> str:
        return f"VoxelMap({len(self)} voxels of {self.voxel_size} m, {len(self.scenes)} scene(s), {self.device})"

    # -- building ---------------------------------------------------------

    def add_scene(
        self,
        scene: "Scene",
        frames: Optional[Iterable[int]] = None,
        lidars: Optional[Sequence[LidarNameLike]] = None,
        *,
        ego_box: Optional[Sequence[float]] = EGO_BOX,
        max_range: Optional[float] = None,
        workers: Optional[int] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        """Add the lidar of *frames* (default: all) of a scene.

        Args:
            frames: Frame indices in time order; "consecutive" (the ``run``
                statistics) means consecutive in this list, so with
                :func:`keyframes` it means consecutive keyframes.
            lidars: Lidars to use (default: all of the scene).
            ego_box: Drop returns within these half extents (metres) of the
                base origin, the car's own body (None keeps them).
            max_range: Drop returns farther than this from the car in x, y.
            workers: Reading threads (default: up to 16).
            progress: Called as ``progress(done, total)`` after each frame is read.

        Returns:
            Statistics: frames, rows read, measured points, points mapped
            (measured and inside the trajectory), seconds.

        Raises:
            ValueError: Without usable poses; for a scene without a map frame
                in a map of several scenes (test-e2e scenes have no
                ``maps/origin.json``), or in another UTM zone.
        """
        import torch

        from kitscenes.accumulate import _in_order

        start = time.perf_counter()
        if scene._trajectory[0] is None:
            raise ValueError(f"Scene {scene.id} has no usable poses ({scene._trajectory[1]}).")
        frames = list(range(len(scene)) if frames is None else frames)
        if any(b <= a for a, b in zip(frames, frames[1:])):
            raise ValueError("frames must be increasing (the run statistics follow their order).")
        lidars = [s for s in scene.sensors if sensor_kind(s) == "lidar"] if lidars is None else list(sensor_names(lidars))
        scene_index = self._place(scene)
        A = np.eye(4)
        A[:2, 3] = self._scene_offset(scene)
        extrinsics = []
        for name in lidars:
            extrinsics.append(scene.rig.sensor(name).T_base_sensor)
            if name not in self.lidars:
                self.lidars.append(name)
        stats = {"frames": len(frames), "rows": 0, "measured": 0, "mapped": 0}
        if not frames:
            return stats | {"seconds": time.perf_counter() - start}

        times = scene.timestamps_ns[frames] / 1e9
        poses = PoseTable.build(scene._trajectory[0], times.min() - TABLE_MARGIN_S, times.max() + TABLE_MARGIN_S, _TABLE_BIN_S, A)
        from kitscenes.painting import RING_CAMERAS, SEGMENTED_CAMERAS

        cameras = scene.files.rig_original.cameras  # geometry at the release's size: masks are; images by size ratio
        paint_with = [c for c in RING_CAMERAS if c in cameras and c in scene.sensors] if self.colour else []
        label_with = [c for c in SEGMENTED_CAMERAS if c in cameras] if self.classes else []
        read = lambda index: read_frame(
            scene, index, lidars, columns=("reflectivity",), images=paint_with,
            reduce=None if self.gpu_decode else self.reduce, class_images=label_with,
        )
        job = _SceneJob(scene, A, scene_index, start_ordinal=0)
        workers = workers or min(16, os.cpu_count() or 1)
        tables = DeviceTables(poses, extrinsics, self.device, ego_box=ego_box, max_range=max_range)
        batch: list[RawSweeps] = []
        batch_frames: list[int] = []
        for ordinal, (index, raw) in enumerate(zip(frames, _in_order(read, frames, workers))):
            if not batch:
                job.start_ordinal = ordinal
            batch.append(raw)
            batch_frames.append(index)
            if progress is not None:
                progress(ordinal + 1, len(frames))
            if sum(len(r.t) for r in batch) >= BATCH_POINTS:
                self._add_batch(batch, batch_frames, tables, job, stats)
                batch, batch_frames = [], []
        if batch:
            self._add_batch(batch, batch_frames, tables, job, stats)
        self._spill()
        return stats | {"seconds": time.perf_counter() - start}

    def _add_batch(self, batch, batch_frames, tables, job: "_SceneJob", stats) -> None:
        import torch

        scene_index = job.scene_index
        out, rows, _, measured = tables.transform(batch)
        n_rows = sum(len(r.t) for r in batch)
        stats["rows"] += n_rows
        stats["measured"] += measured
        stats["mapped"] += len(rows)

        # Frame of every mapped point, as a position in this batch.
        local = torch.repeat_interleave(
            torch.arange(len(batch), device=self.device), torch.tensor([len(r.t) for r in batch], device=self.device),
            output_size=n_rows,
        ).index_select(0, rows)
        cells = torch.floor(out / self.voxel_size).long()
        keys = pack_cells(cells)
        unique, inverse = torch.unique(keys, return_inverse=True)
        n = len(unique)
        offsets = out - cells.float() * self.voxel_size
        run = {
            "keys": unique,
            "count": torch.bincount(inverse, minlength=n).int(),
            "offset_sum": torch.zeros((n, 3), device=self.device).index_add_(0, inverse, offsets),
            "reflectivity_sum": torch.zeros(n, device=self.device),
        }
        if all("reflectivity" in r.attrs for r in batch):
            refl = torch.cat([torch.from_numpy(r.attrs["reflectivity"].astype(np.float32, copy=False)).to(self.device) for r in batch])
            run["reflectivity_sum"].index_add_(0, inverse, refl.index_select(0, rows))
        # Distinct (voxel, frame) pairs, sorted by voxel, then frame.
        pairs = torch.unique(inverse * len(batch) + local)
        run.update(_frame_runs(pairs // len(batch), pairs % len(batch), n, (scene_index << _FRAME_BITS) + job.start_ordinal))
        codes = torch.tensor([(scene_index << _FRAME_BITS) | f for f in batch_frames], dtype=torch.long, device=self.device)
        run["first"] = torch.full((n,), torch.iinfo(torch.long).max, dtype=torch.long, device=self.device).scatter_reduce_(
            0, inverse, codes.index_select(0, local), reduce="amin",
        )
        if self.colour or self.classes:
            run.update(self._camera_statistics(out, inverse, local, unique, batch, batch_frames, job))
        self._device.append(run)
        if sum(len(r["keys"]) for r in self._device[1:]) > max(len(self._device[0]["keys"]), _MERGE_MIN):
            self._device = [_reduce(self._device)]
            if len(self._device[0]["keys"]) > self.max_device_voxels:
                self._spill()

    def _camera_statistics(self, out, inverse, local, unique, batch, batch_frames, job: "_SceneJob") -> dict:
        """Colour sums and class votes of a batch, frame by frame (each frame's points are contiguous).

        Nothing here waits for the device: masks become weights and
        ``torch.where``, and all matrices and images go up in a few transfers.
        """
        import torch

        from kitscenes.instances import UNLABELLED

        n, dev = len(unique), self.device
        stats: dict = {}
        if self.colour:
            stats["rgb_sum"] = torch.zeros((n, 3), device=dev)
            stats["rgb_count"] = torch.zeros(n, dtype=torch.int32, device=dev)
        if self.classes:
            stats["class_seen"] = torch.zeros(n, dtype=torch.int32, device=dev)
        cameras = job.scene.files.rig_original.cameras  # the release's size (see above): stored images any size
        names = list(dict.fromkeys(c for r in batch for c in (*r.images, *r.class_images)))
        pairs = []
        queries: list = []  # (frame j, camera name, flat pixel per point or -1)
        if names:
            to_camera = np.stack([
                [(cameras[c].T_sensor_base @ invert_transform(job.A @ job.scene.frame(i).pose.T_world_base))[:3] for c in names]
                for i in batch_frames
            ])  # (frames, cameras, 3, 4): grid frame -> camera
            mats = torch.from_numpy(to_camera.astype(np.float32)).to(dev)
            images = _decode_on_device([r.images for r in batch], dev, job.scene) if self.gpu_decode else [r.images for r in batch]
            bounds = torch.searchsorted(local, torch.arange(len(batch) + 1, device=dev)).tolist()
            for j, raw in enumerate(batch):
                if bounds[j] == bounds[j + 1]:
                    continue
                points, voxel = out[bounds[j]:bounds[j + 1]], inverse[bounds[j]:bounds[j + 1]]
                # Each camera's view of this frame, shared by colour and classes (points behind nearer ones included).
                views = {c: _project_on_device(points, mats[j, k], cameras[c])
                         for k, c in enumerate(names) if c in images[j] or c in raw.class_images}
                if images[j]:
                    rgb, seen = _paint_on_device(images[j], cameras, views)
                    stats["rgb_sum"].index_add_(0, voxel, rgb.float() * seen[:, None])
                    stats["rgb_count"].index_add_(0, voxel, seen.int())
                for name in raw.class_images:
                    u, v, _, seen = views[name]
                    queries.append((j, name, _flat_pixel(cameras[name], u, v, seen)))
        if queries:
            size = max(cameras[name].image_size[0] * cameras[name].image_size[1] for _, name, _ in queries)
            classes, seen = _label_batch(queries, batch, bounds, len(out), size, dev)
            pairs.append(inverse * 256 + classes.long())
            stats["class_seen"].index_add_(0, inverse, seen.int())
        if self.classes:
            pairs = torch.cat(pairs) if pairs else torch.zeros(0, dtype=torch.long, device=dev)
            pairs, counts = torch.unique(pairs[pairs % 256 != UNLABELLED], return_counts=True)
            stats.update(vote_keys=unique[pairs // 256], vote_classes=(pairs % 256).short(), vote_counts=counts.int())
        return stats

    def _spill(self) -> None:
        """Move the device runs, merged, to host memory."""
        if self._device:
            self._runs.append({k: v.cpu().numpy() for k, v in _reduce(self._device).items()})
            self._device = []

    def merge(self, other: "VoxelMap") -> "VoxelMap":
        """Add another map (same voxel size; built from other frames, or frame counts double)."""
        if other.voxel_size != self.voxel_size:
            raise ValueError(f"Voxel sizes differ: {self.voxel_size} vs {other.voxel_size}")
        if (other.colour, other.classes) != (self.colour, self.classes):
            raise ValueError("Maps with and without colour or classes cannot be merged.")
        if not other._runs:
            return self
        if not self.scenes and not self._runs:
            if self.origin is None:
                self.origin, self.anchor = other.origin, other.anchor
        elif self.origin is None or other.origin is None or self.origin.epsg != other.origin.epsg:
            raise ValueError("Maps in different frames (UTM zones, or a scene without maps/origin.json) cannot be merged.")
        other._compact()
        run = dict(other._runs[0])
        if self.anchor is not None and np.any(other.anchor != self.anchor):
            shift = (other.anchor - self.anchor) / self.voxel_size
            if np.any(np.abs(shift - np.round(shift)) > 1e-6):
                raise ValueError(f"The voxel size {self.voxel_size} does not divide the anchor step {ANCHOR_STEP} m.")
            for field in ("keys", "vote_keys") if self.classes else ("keys",):
                cells = unpack_cells(run[field])
                cells[:, :2] += np.round(shift).astype(np.int64)
                run[field] = pack_cells(cells)  # same shift for all: the keys stay sorted
        base = len(self.scenes) << _FRAME_BITS  # the scene indices move up by len(self.scenes)
        for field in ("first", "run_first", "run_last"):
            run[field] = run[field] + base
        self.scenes += other.scenes
        self.lidars += [name for name in other.lidars if name not in self.lidars]
        self._runs.append(run)  # merged with the others when the map is read
        return self

    def _compact(self) -> None:
        """Merge the host runs into one (for :meth:`save`, :meth:`merge` and ``len``)."""
        self._spill()
        if len(self._runs) > 1:
            out = [{k: v.cpu().numpy() for k, v in part.items()} for part in self._merged_slices()]
            self._runs = [{k: np.concatenate([o[k] for o in out]) for k in out[0]}]

    def _merged_slices(self):
        """The merged statistics as device runs of about :data:`_MERGE_SLICE` voxels each, in key order.

        Every host run is sorted by key, so splitting all of them at the same
        keys gives slices that can be merged independently.
        """
        import torch

        self._spill()
        runs = [r for r in self._runs if len(r["keys"])]
        if not runs:
            return
        total = sum(len(r["keys"]) for r in runs)
        slices = max(1, math.ceil(total / _MERGE_SLICE))
        splitters = np.zeros(0, np.int64)
        if slices > 1:
            step = max(1, total // (slices * 64))
            sample = np.sort(np.concatenate([r["keys"][::step] for r in runs]))
            splitters = np.unique(sample[np.linspace(0, len(sample), slices + 1)[1:-1].astype(np.int64)])
        cut = lambda keys: np.concatenate([[0], np.searchsorted(keys, splitters), [len(keys)]])
        bounds = [{"keys": cut(r["keys"]), "vote_keys": cut(r["vote_keys"]) if "vote_keys" in r else None} for r in runs]
        for i in range(len(splitters) + 1):
            part = [
                {k: torch.from_numpy(v[b["vote_keys" if k in _VOTES else "keys"][i]:b["vote_keys" if k in _VOTES else "keys"][i + 1]]).to(self.device)
                 for k, v in r.items()}
                for r, b in zip(runs, bounds) if b["keys"][i + 1] > b["keys"][i]
            ]
            if part:
                yield _reduce(part)

    def _place(self, scene: "Scene") -> int:
        """Register the scene; set the map frame from the first one."""
        from kitscenes.errors import Unavailable

        try:
            origin = scene.origin
        except Unavailable:
            origin = None
        if not self.scenes and self.anchor is None:
            self.origin = origin
            self.anchor = None if origin is None else _anchor(origin)
        elif origin is None or self.origin is None:
            raise ValueError(
                f"Scene {scene.id} cannot join this map: only scenes with maps/origin.json share a frame "
                "(a scene without one can only be mapped alone)."
            )
        elif origin.epsg != self.origin.epsg:
            raise ValueError(f"Scene {scene.id} is in UTM zone EPSG:{origin.epsg}, the map in EPSG:{self.origin.epsg}.")
        if len(self.scenes) >= 1 << (63 - _FRAME_BITS) or len(scene) >= 1 << _FRAME_BITS:
            raise ValueError("Too many scenes or frames for the frame codes.")
        self.scenes.append(scene.id)
        return len(self.scenes) - 1

    def _scene_offset(self, scene: "Scene") -> np.ndarray:
        """(2,) scene map frame -> grid frame (relative to the anchor)."""
        if self.anchor is None:
            return np.zeros(2)
        return scene.origin.utm_offset - self.anchor

    # -- output -----------------------------------------------------------

    def to_cloud(self, *, min_run: int = 1, min_frames: int = 1) -> PointCloud:
        """Voxel centroids in the map frame (of :attr:`origin`), float64.

        Args:
            min_run: Keep voxels hit in at least this many consecutive frames
                of one scene; 3 removes the traffic (and foliage), 1 keeps all.
            min_frames: Keep voxels hit in at least this many frames in total.

        Returns:
            A cloud with per-point ``count``, ``frames``, ``run`` (longest
            consecutive run), ``reflectivity`` (mean), ``scene_index`` (into
            :attr:`scenes`) and ``frame_index`` of the first frame that hit
            the voxel; with colour ``rgb`` (uint8, mean of the points a camera
            saw, :data:`kitscenes.painting.UNSEEN` grey if none) and with
            classes ``class`` (uint8 Vistas trainId) and ``class_votes``
            (int32, the points behind it). The class is the plain majority of
            the voxel's points that a segmented camera saw: each votes for
            the class :func:`kitscenes.painting.label` gives it, or for
            ``UNLABELLED`` where no mask covers it (the masks label objects,
            not road or buildings). Ties go to the higher id, so to
            ``UNLABELLED``; a voxel no segmented camera saw is ``UNLABELLED``
            with 0 votes. As in painting, a point behind an object in a
            camera's view votes for that object.
        """
        from kitscenes.instances import UNLABELLED
        from kitscenes.painting import UNSEEN

        import torch

        shift = None if self.anchor is None else torch.tensor(
            np.append(self.anchor - self.origin.utm_offset, 0.0), dtype=torch.float64, device=self.device,
        )
        names = ["xyz", "count", "frames", "run", "reflectivity", "scene_index", "frame_index"]
        names += ["rgb"] * self.colour + ["class", "class_votes"] * self.classes
        columns: dict[str, list] = {k: [] for k in names}
        for run in self._merged_slices():  # finished on the device, slice by slice
            if self.classes:
                run["class"], run["class_votes"] = _majority(run, UNLABELLED)
            keep = (run["frames"] >= min_frames) & (run["run_best"] >= min_run)
            run = {k: v[keep] for k, v in run.items() if k not in _VOTES}
            count = run["count"].double()
            keys = run["keys"]
            cells = torch.stack([(keys >> 42) & CELL_MASK, (keys >> 21) & CELL_MASK, keys & CELL_MASK], dim=1) - CELL_LIMIT
            xyz = cells.double() * self.voxel_size + run["offset_sum"].double() / count[:, None]
            if shift is not None:
                xyz += shift
            for name, value in (
                ("xyz", xyz), ("count", run["count"]), ("frames", run["frames"]), ("run", run["run_best"]),
                ("reflectivity", (run["reflectivity_sum"] / count).float()),
                ("scene_index", (run["first"] >> _FRAME_BITS).to(torch.int16)),
                ("frame_index", (run["first"] & ((1 << _FRAME_BITS) - 1)).int()),
            ):
                columns[name].append(value.cpu().numpy())
            if self.colour:
                seen = run["rgb_count"] > 0
                mean = run["rgb_sum"] / torch.clamp(run["rgb_count"], min=1)[:, None].float()
                grey = torch.tensor(UNSEEN, dtype=mean.dtype, device=mean.device)
                columns["rgb"].append(torch.where(seen[:, None], mean + 0.5, grey).clamp(0, 255).to(torch.uint8).cpu().numpy())
            if self.classes:
                columns["class"].append(run["class"].cpu().numpy())
                columns["class_votes"].append(run["class_votes"].cpu().numpy())
        empty = {"xyz": np.zeros((0, 3)), "count": np.zeros(0, np.int32), "frames": np.zeros(0, np.int32), "run": np.zeros(0, np.int32),
                 "reflectivity": np.zeros(0, np.float32), "scene_index": np.zeros(0, np.int16), "frame_index": np.zeros(0, np.int32),
                 "rgb": np.zeros((0, 3), np.uint8), "class": np.zeros(0, np.uint8),
                 "class_votes": np.zeros(0, np.int32)}
        out = {k: (np.concatenate(v) if len(v) > 1 else v[0]) if v else empty[k] for k, v in columns.items()}
        xyz = out.pop("xyz")
        return PointCloud(
            xyz=xyz, measured=np.ones(len(xyz), bool), frame="world", t_ns=0, sensor=tuple(self.lidars), attrs=out,
            motion_compensated=True, provenance=Provenance.GEOMETRY_DERIVED,
        )

    def to_tiles(self, size: float = 50.0, **filters) -> dict[tuple[int, int], PointCloud]:
        """:meth:`to_cloud` split into square tiles of *size* metres on a grid fixed in UTM.

        Keys are the tile's UTM cell ``(floor(easting / size), floor(northing / size))``
        (for a map without ``maps/origin.json``, cells of its own frame), so the
        tiles of different maps line up. *filters* go to :meth:`to_cloud`.
        """
        cloud = self.to_cloud(**filters)
        offset = np.zeros(2) if self.origin is None else self.origin.utm_offset
        return _split_tiles(cloud, size, offset)

    def save(self, path: Path | str) -> Path:
        """Write the map to an ``.npz`` file (load with :meth:`load`, combine with :meth:`merge`)."""
        self._compact()
        path = Path(path)
        arrays = self._runs[0] if self._runs else {}
        meta = {
            "format": _FORMAT, "voxel_size": self.voxel_size, "scenes": self.scenes, "lidars": self.lidars,
            "colour": self.colour, "classes": self.classes, "reduce": self.reduce,
            "origin": None if self.origin is None else [self.origin.lat, self.origin.lon],
            "anchor": None if self.anchor is None else self.anchor.tolist(),
        }
        np.savez(path, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), **arrays)
        return path

    @classmethod
    def load(cls, path: Path | str, *, device="cuda") -> "VoxelMap":
        from kitscenes.map import MapOrigin

        with np.load(path) as data:
            meta = json.loads(data["meta"].tobytes())
            if meta.get("format") != _FORMAT:
                raise ValueError(f"{path} is a VoxelMap of format {meta.get('format')}; this version reads {_FORMAT}.")
            origin = None if meta["origin"] is None else MapOrigin(*meta["origin"])
            vmap = cls(meta["voxel_size"], device=device, origin=origin, colour=meta["colour"], classes=meta["classes"],
                       reduce=meta["reduce"])
            vmap.anchor = None if meta["anchor"] is None else np.array(meta["anchor"])
            vmap.scenes, vmap.lidars = list(meta["scenes"]), list(meta["lidars"])
            if "keys" in data:
                vmap._runs = [{k: data[k] for k in data.files if k != "meta"}]
        return vmap


def _reduce(runs: Sequence[dict]) -> dict:
    """Combine device runs (torch, possibly sharing keys), given in time order, into one sorted run.

    Pairwise in a tree, keeping the order, because the run statistics are
    combined in time order (:func:`_combine`).
    """
    runs = list(runs)
    while len(runs) > 1:
        runs = [_combine(runs[i], runs[i + 1]) if i + 1 < len(runs) else runs[i] for i in range(0, len(runs), 2)]
    return runs[0]


def _combine(a: dict, b: dict) -> dict:
    """One run of two; *a* holds earlier frames than *b* wherever they share a voxel."""
    import torch

    unique, inverse = torch.unique(torch.cat([a["keys"], b["keys"]]), return_inverse=True)
    n, ia, ib = len(unique), inverse[:len(a["keys"])], inverse[len(a["keys"]):]
    dev = unique.device
    out = {"keys": unique}
    for field in (f for f in _SUMS if f in a):
        values = a[field]
        out[field] = torch.zeros((n,) + values.shape[1:], dtype=values.dtype, device=dev).index_add_(0, ia, values).index_add_(0, ib, b[field])
    if "vote_keys" in a:
        out.update(_merge_votes([a, b]))
    out["first"] = torch.full((n,), torch.iinfo(torch.long).max, dtype=torch.long, device=dev)
    out["first"].scatter_reduce_(0, ia, a["first"], reduce="amin").scatter_reduce_(0, ib, b["first"], reduce="amin")

    def spread(run, index, field):
        full = torch.zeros(n, dtype=run[field].dtype, device=dev)
        full[index] = run[field]
        return full

    in_a = torch.zeros(n, dtype=torch.bool, device=dev)
    in_a[ia] = True
    in_b = torch.zeros(n, dtype=torch.bool, device=dev)
    in_b[ib] = True
    A = {f: spread(a, ia, f) for f in _RUN_FIELDS}
    B = {f: spread(b, ib, f) for f in _RUN_FIELDS}
    both = in_a & in_b
    adjacent = both & (A["run_last"] + 1 == B["run_first"])
    a_whole = A["run_prefix"] == A["run_last"] - A["run_first"] + 1  # one unbroken run from first to last
    b_whole = B["run_prefix"] == B["run_last"] - B["run_first"] + 1
    zero = torch.zeros_like(A["run_prefix"])
    joined = torch.where(adjacent, A["run_suffix"] + B["run_prefix"], zero)
    out["run_best"] = torch.maximum(torch.maximum(A["run_best"], B["run_best"]), joined)
    out["run_prefix"] = torch.where(
        both, A["run_prefix"] + torch.where(adjacent & a_whole, B["run_prefix"], zero), torch.where(in_a, A["run_prefix"], B["run_prefix"]),
    )
    out["run_suffix"] = torch.where(
        both, B["run_suffix"] + torch.where(adjacent & b_whole, A["run_suffix"], zero), torch.where(in_b, B["run_suffix"], A["run_suffix"]),
    )
    out["run_first"] = torch.where(in_a, A["run_first"], B["run_first"])
    out["run_last"] = torch.where(in_b, B["run_last"], A["run_last"])
    return out


def _merge_votes(runs: Sequence[dict]) -> dict:
    """Sum class votes of several runs: (key, class) pairs sorted by key, then class."""
    import torch

    keys = torch.cat([r["vote_keys"] for r in runs])
    classes = torch.cat([r["vote_classes"] for r in runs])
    counts = torch.cat([r["vote_counts"] for r in runs])
    order = torch.argsort(classes, stable=True)
    order = order[torch.argsort(keys[order], stable=True)]
    keys, classes, counts = keys[order], classes[order], counts[order]
    new = torch.ones(len(keys), dtype=torch.bool, device=keys.device)
    new[1:] = (keys[1:] != keys[:-1]) | (classes[1:] != classes[:-1])
    group = torch.cumsum(new.long(), 0) - 1
    summed = torch.zeros(int(new.sum()), dtype=counts.dtype, device=keys.device).index_add_(0, group, counts)
    return {"vote_keys": keys[new], "vote_classes": classes[new], "vote_counts": summed}


def _majority(run: dict, unlabelled: int):
    """Per voxel of *run* the class with the most votes and its votes.

    Every point a segmented camera saw votes: for its class (the vote
    tables), or for *unlabelled* (``class_seen`` minus those votes). Ties go
    to the higher class, so to *unlabelled* (255); a voxel without votes is
    *unlabelled* with 0.
    """
    import torch

    keys, classes, counts = run["vote_keys"], run["vote_classes"], run["vote_counts"]
    out = torch.full((len(run["keys"]),), unlabelled, dtype=torch.uint8, device=run["keys"].device)
    votes = run["class_seen"].int().clone()  # unlabelled votes, once the class votes are taken off
    if not len(keys):
        return out, votes
    votes.index_add_(0, torch.searchsorted(run["keys"], keys), -counts.int())
    order = torch.argsort(counts, stable=True)  # votes come sorted by key, then class
    order = order[torch.argsort(keys[order], stable=True)]
    keys, classes, counts = keys[order], classes[order], counts[order]
    last = torch.ones(len(keys), dtype=torch.bool, device=keys.device)
    last[:-1] = keys[1:] != keys[:-1]
    at = torch.searchsorted(run["keys"], keys[last])
    best = counts[last].int()
    wins = best > votes[at]
    out[at[wins]] = classes[last][wins].to(torch.uint8)
    votes[at[wins]] = best[wins]
    return out, votes


def _project_on_device(points, to_camera, camera: "Camera"):
    """Pixels (u, v) and depth of grid-frame points in a full-size camera image, and which ones are in it.

    The torch twin of :meth:`kitscenes.rig.Camera.project` (same rules; a
    test holds them equal). *to_camera* is the (3, 4) grid -> camera
    transform on the device.
    """
    import torch

    from kitscenes.projection import MIN_DEPTH

    q = points @ to_camera[:, :3].T + to_camera[:, 3]
    depth = q[:, 2]
    front = depth >= MIN_DEPTH
    safe = torch.where(front, depth, torch.ones_like(depth))
    K = camera.K
    u = float(K[0, 0]) * q[:, 0] / safe + float(K[0, 1]) * q[:, 1] / safe + float(K[0, 2])
    v = float(K[1, 1]) * q[:, 1] / safe + float(K[1, 2])
    width, height = camera.image_size
    return u, v, depth, front & (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)


def _rounded(x, size: int, scale: float = 1.0):
    """Pixel index ``floor((x + 0.5) * scale)``, clamped (:func:`kitscenes.projection.pixel_indices`)."""
    import torch

    return torch.floor((x + 0.5) * scale).clamp(0, size - 1).long()


def _pixel(u, v, image_hw, camera: "Camera"):
    """Row and column in an image of *image_hw* size, as :func:`kitscenes.projection.pixel_indices`."""
    h, w = image_hw
    return _rounded(v, h, h / camera.image_size[1]), _rounded(u, w, w / camera.image_size[0])


def _paint_on_device(images: dict, cameras: dict, views: dict):
    """:func:`kitscenes.painting.paint` on the device: (rgb uint8 (N, 3), seen), from each camera's *views* (:func:`_project_on_device`)."""
    import torch

    u0 = next(iter(views.values()))[0]
    rgb = torch.zeros((len(u0), 3), dtype=torch.uint8, device=u0.device)
    taken = torch.zeros(len(u0), dtype=torch.bool, device=u0.device)
    for name, image in images.items():  # in painting.RING_CAMERAS order
        camera = cameras[name]
        u, v, _, seen = views[name]
        todo = seen & ~taken
        pixels = image if isinstance(image, torch.Tensor) else torch.from_numpy(image).to(u.device)
        rows, cols = _pixel(u, v, pixels.shape[:2], camera)
        rgb = torch.where(todo[:, None], pixels[rows, cols], rgb)
        taken |= todo
    return rgb, taken


def _flat_pixel(camera: "Camera", u, v, seen):
    """Column-major flat pixel index (``col * height + row``) of each point in a camera image, -1 where unseen."""
    import torch

    width, height = camera.image_size
    flat = _rounded(u, width) * height + _rounded(v, height)
    return torch.where(seen, flat, torch.full_like(flat, -1))


def _label_batch(queries: list, batch: list, bounds: list, n: int, size: int, device):
    """The classes of :func:`kitscenes.painting.label` for every point of a batch, in one join,
    and which points a segmented camera saw at all.

    Each (frame, camera) gets its own key range of *size* (the largest image
    in pixels), so the queries of all of them and the mask intervals of all
    of them are matched by one sort and two binary searches; the most
    confident covering instance wins (``rank * 256 + class``, maximum).
    """
    import torch

    from kitscenes.instances import UNLABELLED

    query_keys, interval_starts, interval_ends, interval_values, owners = [], [], [], [], []
    for slot, (j, name, flat) in enumerate(queries):
        base = slot * size
        query_keys.append(torch.where(flat >= 0, flat + base, flat))
        starts, ends, rank, classes = batch[j].class_images[name]
        interval_starts.append(starts + base)
        interval_ends.append(ends + base)
        interval_values.append(rank * 256 + classes)
    keys = torch.cat(query_keys)
    starts = torch.from_numpy(np.concatenate(interval_starts)).to(device)
    ends = torch.from_numpy(np.concatenate(interval_ends)).to(device)
    values = torch.from_numpy(np.concatenate(interval_values)).to(device)
    order = torch.argsort(keys)
    first = torch.searchsorted(keys[order], starts)
    hits = torch.searchsorted(keys[order], ends) - first
    interval = torch.repeat_interleave(torch.arange(len(starts), device=device), hits)  # the one sync
    within = torch.arange(len(interval), device=device) - torch.repeat_interleave(torch.cumsum(hits, 0) - hits, hits)
    best = torch.full((len(keys),), -1, dtype=torch.long, device=device)
    best.scatter_reduce_(0, order[first[interval] + within], values[interval], reduce="amax")
    label = torch.where(best >= 0, best % 256, torch.full_like(best, UNLABELLED)).to(torch.uint8)
    # Back to points: query slot k covers the points of its frame; cameras in order, the first label wins.
    classes = torch.full((n,), UNLABELLED, dtype=torch.uint8, device=device)
    seen = torch.zeros(n, dtype=torch.bool, device=device)
    offset = 0
    for j, name, flat in queries:
        part = label[offset:offset + len(flat)]
        offset += len(flat)
        span = classes[bounds[j]:bounds[j + 1]]
        classes[bounds[j]:bounds[j + 1]] = torch.where(span == UNLABELLED, part, span)
        seen[bounds[j]:bounds[j + 1]] |= flat >= 0
    return classes, seen


def _decode_on_device(images: list[dict], device, scene: "Scene") -> list[dict]:
    """JPEG bytes ``[{camera: bytes}]`` -> ``[{camera: (H, W, 3) uint8 tensor}]``, in batched nvJPEG calls.

    With the image corrections of *scene*, as :meth:`kitscenes.Frame.image` on the CPU path.
    """
    from kitscenes import decode
    from kitscenes.image_corrections import stored_correction

    flat = [(j, camera, data) for j, frame in enumerate(images) for camera, data in frame.items()]
    out: list[dict] = [{} for _ in images]
    for (j, camera, _), image in zip(flat, decode._decode_on_device([d for _, _, d in flat], device)):
        corr = stored_correction(scene, camera)
        if corr is not None:
            image = decode.apply_image_corrections(image[None], [corr])[0]
        out[j][camera] = image.permute(1, 2, 0)
    return out


@dataclasses.dataclass
class _SceneJob:
    """What a batch needs of the scene being added."""

    scene: "Scene"
    A: np.ndarray  # scene map frame -> grid frame
    scene_index: int
    start_ordinal: int  # of the batch being collected


def _frame_runs(voxel: "torch.Tensor", frame: "torch.Tensor", n: int, ordinal0: int) -> dict:
    """Frame and run statistics of one batch from its distinct (voxel, frame) pairs, sorted by voxel then frame.

    *frame* is the position in the batch; *ordinal0* the tagged ordinal of
    the batch's first frame.
    """
    import torch

    dev = voxel.device
    new_voxel = torch.ones(len(voxel), dtype=torch.bool, device=dev)
    new_voxel[1:] = voxel[1:] != voxel[:-1]
    new_run = new_voxel.clone()
    new_run[1:] |= frame[1:] != frame[:-1] + 1
    run_id = torch.cumsum(new_run.long(), 0) - 1
    length = torch.bincount(run_id)[run_id].int()  # of the run each pair belongs to
    first = torch.nonzero(new_voxel)[:, 0]  # one per voxel, in voxel order
    last = torch.cat([first[1:] - 1, torch.tensor([len(voxel) - 1], device=dev)])
    return {
        "frames": (last - first + 1).int(),
        "run_first": frame[first] + ordinal0,
        "run_last": frame[last] + ordinal0,
        "run_prefix": length[first],
        "run_suffix": length[last],
        "run_best": torch.zeros(n, dtype=torch.int32, device=dev).scatter_reduce_(0, voxel, length, reduce="amax"),
    }


def _split_tiles(cloud: PointCloud, size: float, offset: Sequence[float] = (0.0, 0.0)) -> dict[tuple[int, int], PointCloud]:
    """Split a cloud into square tiles: ``{(i, j): part}`` with ``i = floor((x + offset_x) / size)``, same for ``j``."""
    cells = np.floor((cloud.xyz[:, :2] + np.asarray(offset)) / size).astype(np.int64)
    order = np.lexsort((cells[:, 1], cells[:, 0]))
    cells = cells[order]
    bounds = np.flatnonzero(np.any(np.diff(cells, axis=0) != 0, axis=1)) + 1
    tiles = {}
    for start, end in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(cells)]])):
        if end > start:
            tiles[tuple(int(c) for c in cells[start])] = cloud.select(order[start:end])
    return tiles


def _has_torchvision() -> bool:
    import importlib.util

    return importlib.util.find_spec("torchvision") is not None


def _anchor(origin: "MapOrigin") -> np.ndarray:
    return np.round(origin.utm_offset / ANCHOR_STEP) * ANCHOR_STEP
