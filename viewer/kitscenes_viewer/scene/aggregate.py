"""The whole scene as one static point cloud, built by `kitscenes.mapping.VoxelMap`.

One point per voxel, kept when the voxel was seen in `min_run` consecutive
frames: a static surface is hit frame after frame, a moving vehicle occupies
a voxel for an instant. Counting consecutive frames, not hits, matters on a
long scene, where different cars crossing one patch of road add up. The same
test removes wind-blown foliage; `min_run=1` keeps everything.

Points are in the scene's metric frame; the viewer's ENU anchor is applied
later and can change.

`attributes` chooses what the cloud carries besides geometry: "geometry"
(position, intensity), "colour" (+ mean camera colour) or "all" (+ majority
seamseg class). A camera paints and labels every point that lands in its
image; a point no mask covers votes "unlabelled". Colour and classes about
double the build time on a CPU, so without a GPU the default is geometry.

Every point also carries how many lidar points its voxel averaged and the
votes behind its class, so the clouds of several scenes merge voxel by voxel
(`merge_aggregates`): kitscenes puts every scene's voxels on one grid fixed
in UTM.
"""

import dataclasses
import time
from typing import Optional

import numpy as np

from kitscenes.instances import UNLABELLED
from kitscenes.painting import UNSEEN

# Bumped when the cached payload gains or changes a field, so an aggregate
# with another payload is rebuilt instead of misread.
AGGREGATE_FORMAT = 1
ATTRIBUTES = ("geometry", "colour", "all")
AGGREGATE_VOXEL_M = 0.15   # voxel size of the whole-scene cloud
AGGREGATE_MIN_RUN = 3      # consecutive frames a voxel must be seen in to count as static
AGGREGATE_RANGE_M = 60.0   # only returns within this range of the car
UNSEEN_GREY = UNSEEN[0]


def _cell_keys(cells):
    """(N, 3) int64 voxel cells -> (N,) int64 keys, one per cell (21 bits per axis)."""
    c = cells - cells.min(axis=0)
    if int(c.max()) >= 1 << 21:
        raise ValueError("the aggregate spans more than 2**21 voxels on an axis; use a larger voxel")
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def aggregate_cache_name(voxel=AGGREGATE_VOXEL_M, min_run=AGGREGATE_MIN_RUN, max_range_m=AGGREGATE_RANGE_M,
                         attributes="all"):
    """File name of a cached aggregate. The rule is in the name, so a different
    one builds a new file instead of serving an aggregate filtered some other
    way -- and a city can tell which scenes are ready without opening them."""
    return ("aggregate-v%.3f-c%d-r%.0f-%s-f%d.npz"
            % (voxel, min_run, max_range_m, attributes, AGGREGATE_FORMAT))


def resolve_attributes(attributes, device=None):
    """"auto" -> "all" with a GPU, "geometry" without one."""
    if attributes != "auto":
        if attributes not in ATTRIBUTES:
            raise ValueError("attributes must be auto or one of %s" % (ATTRIBUTES,))
        return attributes
    return "all" if (device or default_device()) == "cuda" else "geometry"


@dataclasses.dataclass
class Aggregate:
    """A static cloud: one point per voxel, in some metric frame."""

    xyz: np.ndarray                    # (N, 3) float64
    intensity: np.ndarray              # (N,) float32, mean reflectivity
    count: np.ndarray                  # (N,) int32, lidar points in the voxel
    classes: Optional[np.ndarray] = None  # (N,) uint8 Vistas trainIds, the majority (UNLABELLED included)
    rgb: Optional[np.ndarray] = None      # (N, 3) uint8
    stats: dict = dataclasses.field(default_factory=dict)
    class_votes: Optional[np.ndarray] = None  # (N,) int32, points voting for `classes`

    def __len__(self):
        return len(self.xyz)

    def select(self, keep):
        pick = lambda a: None if a is None else a[keep]  # noqa: E731
        return dataclasses.replace(self, xyz=self.xyz[keep], intensity=self.intensity[keep], count=self.count[keep],
                                   classes=pick(self.classes), rgb=pick(self.rgb), class_votes=pick(self.class_votes))

    def moved(self, offset):
        """The same cloud, translated by `offset` (the frame changes, the voxels do not)."""
        return dataclasses.replace(self, xyz=self.xyz + np.asarray(offset, np.float64))


def read_aggregate(path):
    """An aggregate saved by `SceneAggregator.save_cached`."""
    with np.load(path, allow_pickle=False) as data:
        stats = {k[len("stat_"):]: v.item() for k, v in data.items() if k.startswith("stat_")}
        optional = lambda name: data[name] if name in data else None  # noqa: E731
        return Aggregate(
            xyz=data["xyz"].astype(np.float64), intensity=data["intensity"], count=data["count"],
            classes=optional("classes"), rgb=optional("rgb"), stats=stats,
            class_votes=optional("class_votes"))


def merge_aggregates(parts, voxel, offset=(0.0, 0.0, 0.0)):
    """Several clouds in one frame -> one, voxel by voxel.

    Voxels are the cells ``floor((xyz + offset) / voxel)``; with `offset`
    the frame's distance from the kitscenes grid anchor they are exactly the
    voxels the parts were built on, so each part has at most one point per
    cell. Position, intensity and colour are averaged weighted by the points
    behind each voxel (colour over the parts a camera saw it in); the class
    is the one with most votes over all parts (each part votes for its own
    class only, "unlabelled" included; ties go to the higher id, so to
    unlabelled).
    """
    parts = [p for p in parts if len(p)]
    if not parts:
        return Aggregate(np.zeros((0, 3)), np.zeros(0, np.float32), np.zeros(0, np.int32))
    xyz = np.concatenate([p.xyz for p in parts])
    count = np.concatenate([p.count for p in parts]).astype(np.int64)
    cells = np.floor((xyz + np.asarray(offset, np.float64)) / voxel).astype(np.int64)
    keys, inverse = np.unique(_cell_keys(cells), return_inverse=True)
    n = len(keys)
    weight = count.astype(np.float64)
    total = np.bincount(inverse, weight, minlength=n)
    mean = lambda values: np.bincount(inverse, weight * values, minlength=n) / total  # noqa: E731
    out_xyz = np.stack([mean(xyz[:, k]) for k in range(3)], axis=1)
    out_intensity = mean(np.concatenate([p.intensity for p in parts]).astype(np.float64)).astype(np.float32)
    out_count = np.bincount(inverse, count, minlength=n).astype(np.int32)
    rgb = None
    if any(p.rgb is not None for p in parts):
        colours = np.concatenate([p.rgb if p.rgb is not None else np.full((len(p), 3), UNSEEN_GREY, np.uint8)
                                  for p in parts]).astype(np.float64)
        seen = weight * ~(colours == UNSEEN_GREY).all(axis=1)
        seen_total = np.bincount(inverse, seen, minlength=n)
        rgb = np.full((n, 3), UNSEEN_GREY, np.uint8)
        has = seen_total > 0
        for k in range(3):
            rgb[has, k] = np.clip(np.bincount(inverse, seen * colours[:, k], minlength=n)[has]
                                  / seen_total[has] + 0.5, 0, 255).astype(np.uint8)
    classes = class_votes = None
    if any(p.classes is not None for p in parts):
        labels, part_votes = (np.concatenate(column).astype(np.int64) for column in zip(*map(_votes, parts)))
        voted = part_votes > 0
        pairs, pair_index = np.unique(inverse[voted] * 256 + labels[voted], return_inverse=True)
        votes = np.bincount(pair_index, part_votes[voted]).astype(np.int64)
        order = np.lexsort((pairs % 256, votes, pairs // 256))  # by voxel, most votes (then higher class) last
        winner = np.ones(len(order), bool)
        winner[:-1] = (pairs // 256)[order][1:] != (pairs // 256)[order][:-1]
        classes = np.full(n, UNLABELLED, np.uint8)
        class_votes = np.zeros(n, np.int32)
        at = (pairs // 256)[order][winner]
        classes[at] = (pairs % 256)[order][winner]
        class_votes[at] = votes[order][winner]
    return Aggregate(out_xyz, out_intensity, out_count, classes, rgb, dict(parts[0].stats), class_votes=class_votes)


def _votes(part):
    """(classes, votes) of a part; a part without classes votes for nothing."""
    if part.classes is None:
        return np.full(len(part), UNLABELLED, np.uint8), np.zeros(len(part), np.int32)
    return part.classes, part.class_votes


def default_device():
    """"cuda" when torch sees a GPU, else "cpu"; None without torch."""
    try:
        import torch
    except ImportError:
        return None
    return "cuda" if torch.cuda.is_available() else "cpu"


class SceneAggregator:
    """Builds (and caches) the static aggregate of a whole scene."""

    def __init__(self, source, disk_cache=None, voxel=AGGREGATE_VOXEL_M, max_range_m=AGGREGATE_RANGE_M,
                 min_run=AGGREGATE_MIN_RUN, ego_box=(3.6, 1.5, 2.6), device=None, attributes="all", log=print):
        self.source = source
        self.disk = disk_cache
        self.voxel = float(voxel)
        self.max_range_m = float(max_range_m)
        self.min_run = int(min_run)
        self.ego_box = ego_box
        self.device = device
        self.attributes = resolve_attributes(attributes, device)
        self.log = log
        self.progress = {"state": "idle", "done": 0, "total": 0, "points": 0}

    def build(self):
        """Read every frame once into a voxel map; an `Aggregate` in the scene frame."""
        device = self.device or default_device()
        if device is None:
            raise RuntimeError("the whole-scene cloud needs torch (pip install torch)")
        from kitscenes.mapping import VoxelMap

        frames = list(range(self.source.frame_count))
        self.progress.update(state="scanning", done=0, total=len(frames), points=0)
        t0 = time.perf_counter()
        colour = self.attributes in ("colour", "all") and bool(self.source.cameras)
        classes = self.attributes == "all" and self.source.has_masks()
        vmap = VoxelMap(self.voxel, device=device, colour=colour, classes=classes)
        vmap.add_scene(self.source.scene, frames, ego_box=self.ego_box, max_range=self.max_range_m,
                       progress=lambda done, total: self.progress.update(done=done))
        occupied = len(vmap)
        cloud = vmap.to_cloud(min_run=self.min_run)
        xyz = cloud.xyz
        stats = {
            "frames": len(frames),
            "voxels": int(occupied),
            "static_voxels": int(len(xyz)),
            "points": int(len(xyz)),
            "voxel_m": self.voxel,
            "min_run": self.min_run,
            "max_range_m": self.max_range_m,
            "attributes": self.attributes,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        self.progress.update(state="ready", points=int(len(xyz)))
        self.log("scene: aggregate %d of %d voxels static in %.1f s on %s (%s)"
                 % (len(xyz), occupied, stats["seconds"], device, describe(stats)))
        return Aggregate(xyz=xyz, intensity=cloud.attrs["reflectivity"].astype(np.float32),
                         count=cloud.attrs["count"].astype(np.int32), classes=cloud.attrs.get("class"),
                         rgb=cloud.attrs.get("rgb"), stats=stats, class_votes=cloud.attrs.get("class_votes"))

    # -- disk cache --------------------------------------------------------
    def cache_path(self):
        """Where this exact aggregate lives; the parameters are in the name."""
        if not (self.disk and self.disk.enabled):
            return None
        return self.disk.dir / aggregate_cache_name(self.voxel, self.min_run, self.max_range_m, self.attributes)

    def load_cached(self):
        path = self.cache_path()
        if not path or not path.is_file():
            return None
        try:
            aggregate = read_aggregate(path)
            self.progress.update(state="ready", points=len(aggregate),
                                 done=self.source.frame_count, total=self.source.frame_count)
            return aggregate
        except Exception as exc:
            self.log("scene: ignoring unreadable aggregate cache (%r)" % exc)
            return None

    def save_cached(self, aggregate):
        path = self.cache_path()
        if not path:
            return
        try:
            payload = {"xyz": aggregate.xyz.astype(np.float32), "intensity": aggregate.intensity,
                       "count": np.asarray(aggregate.count, np.int32)}
            if aggregate.classes is not None:
                payload["classes"] = np.asarray(aggregate.classes, np.uint8)
                if aggregate.class_votes is not None:
                    payload["class_votes"] = np.asarray(aggregate.class_votes, np.int32)
            if aggregate.rgb is not None:
                payload["rgb"] = np.asarray(aggregate.rgb, np.uint8)
            payload.update({"stat_" + k: np.array(v) for k, v in aggregate.stats.items()})
            # np.savez appends ".npz" to a *name* but not to an open file, and
            # the rename has to find exactly what was written.
            tmp = path.with_name(path.name + ".part")
            with open(tmp, "wb") as handle:
                np.savez(handle, **payload)
            tmp.replace(path)
        except OSError as exc:
            self.log("scene: could not cache the aggregate (%r)" % exc)


def describe(stats):
    return ("%.2f m voxels, seen in >=%d consecutive frames, within %.0f m, %s"
            % (stats["voxel_m"], stats["min_run"], stats["max_range_m"], stats.get("attributes", "all")))


def chunk_by_tile(xyz, intensity, classes=None, colour=None, tile_m=128.0,
                  max_points=400_000):
    """Split an aggregate into spatial chunks the browser can stream and cull.

    Tiles keep what is drawn together close together, so the viewer can drop
    far chunks; the point cap stops one dense tile becoming a 40 MB download.
    """
    if not len(xyz):
        return []
    tile = np.floor(xyz[:, :2] / tile_m).astype(np.int64)
    key = (tile[:, 0] + (1 << 20)) | ((tile[:, 1] + (1 << 20)) << 21)
    order = np.argsort(key, kind="stable")
    key = key[order]
    xyz = xyz[order]
    intensity = intensity[order]
    classes = classes[order] if classes is not None else None
    colour = colour[order] if colour is not None else None
    bounds = np.flatnonzero(np.diff(key)) + 1
    split_classes = np.split(classes, bounds) if classes is not None else None
    split_colour = np.split(colour, bounds) if colour is not None else None
    chunks = []
    for index, (part_xyz, part_i) in enumerate(zip(np.split(xyz, bounds),
                                                   np.split(intensity, bounds))):
        part_c = split_classes[index] if split_classes is not None else None
        part_rgb = split_colour[index] if split_colour is not None else None
        for start in range(0, len(part_xyz), max_points):
            piece = part_xyz[start:start + max_points]
            chunks.append({
                "xyz": piece,
                "intensity": part_i[start:start + max_points],
                "classes": part_c[start:start + max_points] if part_c is not None else None,
                "rgb": part_rgb[start:start + max_points] if part_rgb is not None else None,
                "centre": piece.mean(axis=0).tolist(),
                "radius": float(np.linalg.norm(piece - piece.mean(axis=0), axis=1).max())
                if len(piece) else 0.0,
                "count": int(len(piece)),
            })
    return chunks
