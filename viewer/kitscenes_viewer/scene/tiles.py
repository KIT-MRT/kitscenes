"""The static clouds of a city's scenes, merged into square tiles.

Five drives down one street are one cloud here, not five stacked copies: every
scene's whole-scene cloud (`aggregate.py`) lies on kitscenes' voxel grid,
which is fixed in UTM, so the clouds of all scenes that reach a tile merge
voxel by voxel (`merge_aggregates`): positions, intensities and colours
weighted by the lidar points behind each voxel, classes by points voted.
Traffic is already out of each scene's cloud, so a voxel is in the merged one
when any scene saw it as static.

Tiles are `TILE_M` squares of the city frame (the shared projector origin
of the city's scenes). A voxel belongs to the tile its grid cell lies in, so
the same voxel from two scenes lands in the same tile even where their
centroids differ. Only scenes whose cloud is built (on disk) take part; a
tile's `version` names them, so the browser fetches a tile again when one
more of its scenes is ready.
"""

import hashlib
import threading
from collections import OrderedDict

import numpy as np

from kitscenes.mapping import ANCHOR_STEP

from .aggregate import AGGREGATE_RANGE_M, AGGREGATE_VOXEL_M, merge_aggregates, read_aggregate

TILE_M = 128.0


def _map_origin(lat_lon):
    from kitscenes.map import MapOrigin
    return MapOrigin(float(lat_lon[0]), float(lat_lon[1]))


class CityTiles:
    """Which built scene clouds reach which tile, and the merged tiles.

    Args:
        scenes: `CityScene`s (origin, poses, id).
        aggregate_file: scene -> path of its built cloud, or None while it is not built.
        origin: (lat, lon) of the city frame, the scenes' shared projector origin.
        reach_m: how far a scene's cloud reaches from its trajectory (the aggregate range).
    """

    def __init__(self, scenes, aggregate_file, origin, reach_m=AGGREGATE_RANGE_M, memory=16, log=print):
        self.origin = _map_origin(origin)
        utm = self.origin.utm_offset
        # City frame -> the frame kitscenes' grid is anchored in.
        self.grid_offset = np.append(utm - np.round(utm / ANCHOR_STEP) * ANCHOR_STEP, 0.0)
        self.aggregate_file = aggregate_file
        self.log = log
        self.memory = max(1, int(memory))
        self._clouds = OrderedDict()   # scene id -> Aggregate in the city frame (LRU)
        self._lock = threading.Lock()
        self.scenes = {}               # id -> (scene, (2,) shift into the city frame, bbox)
        for scene in scenes:
            other = _map_origin(scene.origin)
            if other.epsg != self.origin.epsg:
                log("city: %s is in another UTM zone; left out of the merged tiles" % scene.id)
                continue
            shift = other.utm_offset - utm
            xy = np.asarray(scene.poses["t"])[:, :2] + shift
            bbox = np.concatenate([xy.min(axis=0) - reach_m, xy.max(axis=0) + reach_m])
            self.scenes[scene.id] = (scene, shift, bbox)

    def city_xy(self, lat, lon):
        """(x, y) of a lat/lon in the city frame."""
        return self.origin.from_latlon(lat, lon)[0]

    def _ready(self):
        """Scenes whose cloud is built: id -> path."""
        out = {}
        for scene_id, (scene, _shift, _bbox) in self.scenes.items():
            path = self.aggregate_file(scene)
            if path is not None and path.is_file():
                out[scene_id] = path
        return out

    def near(self, x, y, radius):
        """Tiles within `radius` of (x, y) (city frame) that some built cloud reaches.

        Returns [{key, centre (x, y, z) in the city frame, radius, version, ids}]
        (`ids`: the built scenes whose cloud reaches the tile).
        """
        ready = self._ready()
        lo = np.floor((np.array([x, y]) - radius) / TILE_M).astype(int)
        hi = np.floor((np.array([x, y]) + radius) / TILE_M).astype(int)
        half_diagonal = TILE_M / np.sqrt(2.0)
        tiles = []
        for i in range(lo[0], hi[0] + 1):
            for j in range(lo[1], hi[1] + 1):
                cx, cy = (i + 0.5) * TILE_M, (j + 0.5) * TILE_M
                if np.hypot(cx - x, cy - y) > radius + half_diagonal:
                    continue
                cell = np.array([i * TILE_M, j * TILE_M, (i + 1) * TILE_M, (j + 1) * TILE_M])
                ids = sorted(sid for sid in ready if _overlaps(self.scenes[sid][2], cell))
                if not ids:
                    continue
                z = float(np.median([np.median(np.asarray(self.scenes[sid][0].poses["t"])[:, 2]) for sid in ids]))
                tiles.append({"key": "%d_%d" % (i, j), "centre": [cx, cy, z], "radius": half_diagonal,
                              "version": hashlib.sha1(",".join(ids).encode()).hexdigest()[:12],
                              "ids": ids})
        return tiles

    def _cloud(self, scene_id, path):
        """A scene's built cloud in the city frame (kept for the next tiles)."""
        with self._lock:
            if scene_id in self._clouds:
                self._clouds.move_to_end(scene_id)
                return self._clouds[scene_id]
        cloud = read_aggregate(path)
        shift = self.scenes[scene_id][1]
        cloud = cloud.moved(np.append(shift, 0.0))
        with self._lock:
            self._clouds[scene_id] = cloud
            while len(self._clouds) > self.memory:
                self._clouds.popitem(last=False)
        return cloud

    def members(self, key):
        """(the built scenes that reach tile `key`, as {id: path}, the tile's version)."""
        i, j = (int(v) for v in key.split("_"))
        cell = np.array([i * TILE_M, j * TILE_M, (i + 1) * TILE_M, (j + 1) * TILE_M])
        ready = {sid: path for sid, path in self._ready().items() if _overlaps(self.scenes[sid][2], cell)}
        return ready, hashlib.sha1(",".join(sorted(ready)).encode()).hexdigest()[:12]

    def tile(self, key):
        """The merged cloud of one tile (city frame), or None when no built cloud reaches it."""
        i, j = (int(v) for v in key.split("_"))
        parts, voxel = [], None
        for scene_id, path in sorted(self.members(key)[0].items()):
            cloud = self._cloud(scene_id, path)
            if not len(cloud):
                continue
            voxel = float(cloud.stats.get("voxel_m", voxel or AGGREGATE_VOXEL_M))
            # The tile of a voxel is that of its grid cell, not of its centroid.
            corner = np.floor((cloud.xyz + self.grid_offset) / voxel) * voxel - self.grid_offset
            inside = (np.floor(corner[:, 0] / TILE_M) == i) & (np.floor(corner[:, 1] / TILE_M) == j)
            if inside.any():
                parts.append(cloud.select(inside))
        if not parts:
            return None
        return merge_aggregates(parts, voxel, self.grid_offset)


def _overlaps(bbox, cell):
    return bbox[0] < cell[2] and bbox[2] > cell[0] and bbox[1] < cell[3] and bbox[3] > cell[1]
