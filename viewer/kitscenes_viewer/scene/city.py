"""Every scene of one city: where they are, without opening their sensors.

A scene's city comes from the kitscenes scene table, else from where its
first pose lies (the nearest of the dataset's three cities). It is placed
through the origin of its pose frame, by the rule of
`kitscenes.aerial.placement_origin`: its own `maps/origin.json`, or, for the
test and map_change_test splits, which ship no maps but poses in their city's
map frame, the city's origin. The test-e2e splits' poses are relative to the
scene's first pose, so their scenes cannot be placed: the index lists them as
not placeable, with the reason.

Only `poses.txt`, `maps/origin.json` and `timestamp.reference.txt` are read
here. On a network mount every file operation is a round trip, so the walk
runs in a thread pool, never asks about a file it is about to read anyway,
and keeps what it read in a local cache (`IndexCache`): a restart only lists
the splits and reads the scenes it has not seen.
"""

import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .enu import SceneFrame
from .scene import NotPlaceable, open_scene, scene_origin, scene_poses, scene_reference_timestamps, track_enu

# A place this close to a city's map origin (`kitscenes.map.origin.CITY_ORIGINS`) is in that city.
CITY_RADIUS_KM = 30.0
# Scene ids end up in URLs; anything else in the data directory is ignored.
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
# Threads, because the walk waits on the network, not on Python.
DEFAULT_WORKERS = 32
# Bumped when what the index cache holds changes meaning.
INDEX_FORMAT = 2


def haversine_km(a, b):
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))


def city_of(lat, lon):
    """Lower-case name of the dataset's city nearest to (lat, lon), if within `CITY_RADIUS_KM`, else None."""
    from kitscenes.map.origin import CITY_ORIGINS
    km = {name.lower(): haversine_km((lat, lon), (o.lat, o.lon)) for name, o in CITY_ORIGINS.items()}
    best = min(km, key=km.get)
    return best if km[best] <= CITY_RADIUS_KM else None


def scene_city(scene_id, start):
    """The scene's city: from the kitscenes scene table, else the one `start` is in (`city_of`)."""
    from kitscenes.splits import scene_info
    info = scene_info(scene_id)
    if info is not None and info.city:
        return info.city.lower()
    return city_of(*start)


def _data_root(root):
    root = Path(root).expanduser().resolve()
    return root / "data" if (root / "data").is_dir() else root


class NotAScene(Exception):
    """A directory without `poses.txt`: not a scene, so look inside it."""


def _subdirs(directory):
    """Names of the directories in one directory, sorted.

    `scandir` knows which entries are directories from the listing itself,
    so this is one request however many entries there are.
    """
    dirs = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                try:
                    if entry.is_dir():
                        dirs.append(entry.name)
                except OSError:
                    continue
    except OSError:
        return []
    return sorted(dirs)


def scan(root, probe, workers=DEFAULT_WORKERS, max_depth=3):
    """Walk the tree in parallel: [(split, dir, probe result)].

    `probe(split, dir)` is called on every directory that could be a scene
    and raises `NotAScene` when there is no `poses.txt`, in which case the
    directory is listed instead (down to `max_depth`). Being a scene and
    reading it are one request, not a check and then a read.

    `root` may be `KITSCENES_ROOT`, its `data/`, one split, or one scene; a
    scene's split is its parent's name.
    """
    root = _data_root(root)
    found = []
    workers = max(1, int(workers))

    def visit(directory, split, level):
        if SCENE_ID_RE.match(directory.name):
            try:
                return "scene", probe(split, directory)
            except NotAScene:
                pass
        elif (directory / "poses.txt").is_file():
            return "skip", None    # a scene, but its name could not go in a URL
        if level > max_depth:
            return "skip", None
        return "list", _subdirs(directory)

    with ThreadPoolExecutor(workers, thread_name_prefix="ll2-city-scan") as pool:
        pending = {pool.submit(visit, root, root.parent.name, 1): (root, root.parent.name, 1)}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                directory, split, level = pending.pop(future)
                kind, value = future.result()
                if kind == "scene":
                    found.append((split, directory, value))
                elif kind == "list":
                    for name in value:
                        child = directory / name
                        pending[pool.submit(visit, child, directory.name, level + 1)] = (
                            child, directory.name, level + 1)
    found.sort(key=lambda item: str(item[1]))
    return found


def _why(exc):
    """An error for the log: its type and message (which names the file, unlike its repr)."""
    return "%s: %s" % (type(exc).__name__, exc)


def read_scene_files(scene_dir):
    """What the index needs from one scene: ("ok", origin, poses, timestamps).

    A scene the map cannot place (`scene_origin`) comes back as
    ("unplaced", reason). A read that fails for good (a missing or malformed
    file) comes back as ("error", message, True); one that may not fail next
    time (a network hiccup) as ("error", message, False), so the cache never
    keeps it.
    """
    scene = open_scene(scene_dir)
    try:
        poses = scene_poses(scene)
    except FileNotFoundError:
        raise NotAScene(scene_dir) from None
    except (ValueError, KeyError) as exc:
        return "error", _why(exc), True
    except OSError as exc:
        return "error", _why(exc), False
    try:
        return "ok", scene_origin(scene), poses, scene_reference_timestamps(scene)
    except NotPlaceable as exc:
        return "unplaced", str(exc)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return "error", _why(exc), True
    except OSError as exc:
        return "error", _why(exc), False


def same_entry(a, b):
    if a is None or b is None or a[0] != b[0]:
        return a is b
    if a[0] in ("error", "unplaced"):
        return a[1] == b[1]
    return (tuple(a[1]) == tuple(b[1])
            and all(np.array_equal(a[2][k], b[2][k]) for k in ("time", "t", "q"))
            and np.array_equal(a[3], b[3]))


class IndexCache:
    """What the index read from each scene, on local disk.

    Keyed by scene directory; only the dataset's listing is trusted to say
    which scenes exist, so new scenes are read and gone ones dropped at every
    start. What it cannot see is a file changed *inside* a scene it already
    knows, which is what `CityIndex.check_in_background` and `--rescan` are
    for. One `.npz`, no pickles.
    """

    def __init__(self, cache_dir, root):
        self.root = str(root)
        key = hashlib.sha1(self.root.encode("utf-8")).hexdigest()[:12]
        self.path = Path(cache_dir).expanduser() / ("city-index-%s-v%d.npz" % (key, INDEX_FORMAT))

    def load(self):
        """{dir: entry}, or {} when there is no usable cache."""
        try:
            with np.load(self.path, allow_pickle=False) as data:
                meta = json.loads(bytes(data["meta"]).decode("utf-8"))
                if meta.get("format") != INDEX_FORMAT or meta.get("root") != self.root:
                    return {}
                time_, t, q, ts = data["time"], data["t"], data["q"], data["ts"]
        except Exception:
            return {}   # none yet, or unreadable: read the dataset instead
        out, p, m = {}, 0, 0
        for item in meta["scenes"]:
            if "error" in item:
                out[item["dir"]] = ("error", item["error"], True)
                continue
            if "unplaced" in item:
                out[item["dir"]] = ("unplaced", item["unplaced"])
                continue
            n, k = item["poses"], item["times"]
            poses = {"time": time_[p:p + n], "t": t[p:p + n], "q": q[p:p + n]}
            out[item["dir"]] = ("ok", tuple(item["origin"]), poses, ts[m:m + k])
            p += n
            m += k
        return out

    def save(self, entries):
        """Write {dir: entry} atomically; errors that may pass are left out."""
        scenes, time_, t, q, ts = [], [], [], [], []
        for directory, entry in sorted(entries.items()):
            if entry[0] == "error":
                if entry[2]:
                    scenes.append({"dir": directory, "error": entry[1]})
                continue
            if entry[0] == "unplaced":
                scenes.append({"dir": directory, "unplaced": entry[1]})
                continue
            _, origin, poses, stamps = entry
            scenes.append({"dir": directory, "origin": list(origin),
                           "poses": len(poses["time"]), "times": len(stamps)})
            time_.append(poses["time"])
            t.append(poses["t"])
            q.append(poses["q"])
            ts.append(stamps)
        meta = json.dumps({"format": INDEX_FORMAT, "root": self.root, "scenes": scenes})
        cat = lambda parts, shape: (np.concatenate(parts) if parts  # noqa: E731
                                    else np.zeros(shape, np.float64))
        tmp = self.path.with_name("%s.%d.%d.part" % (self.path.name, os.getpid(),
                                                     threading.get_ident()))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as handle:
                np.savez(handle, meta=np.frombuffer(meta.encode("utf-8"), np.uint8),
                         time=cat(time_, (0,)), t=cat(t, (0, 3)), q=cat(q, (0, 4)),
                         ts=cat(ts, (0,)))
            tmp.replace(self.path)
            return True
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            return False


@dataclass
class CityScene:
    id: str
    split: str
    dir: Path
    origin: tuple
    poses: dict = field(repr=False)
    timestamps: np.ndarray = field(repr=False)
    frame_count: int = 0
    start: tuple = (0.0, 0.0)     # lat, lon of the first pose
    city: str = None

    @classmethod
    def build(cls, split, scene_dir, origin, poses, timestamps):
        """A scene from what `read_scene_files` read, checked."""
        frames = int(min(len(poses["time"]), len(timestamps)))
        # What would otherwise only fail later, while describing the whole
        # city -- and take every other scene down with it.
        if frames < 1:
            raise ValueError("no frames")
        if len(poses["time"]) > 1 and not np.all(np.diff(poses["time"]) > 0):
            raise ValueError("pose times are not strictly increasing")
        if not np.all(np.isfinite(poses["t"])) or not np.all(np.isfinite(timestamps[:frames])):
            raise ValueError("non-finite poses or timestamps")
        frame = SceneFrame(origin[0], origin[1], origin[0], origin[1])
        start = frame.to_lat_lon(poses["t"][0, 0], poses["t"][0, 1])
        return cls(id=scene_dir.name, split=split, dir=Path(scene_dir), origin=origin,
                   poses=poses, timestamps=timestamps[:frames], frame_count=frames,
                   start=(float(start[0]), float(start[1])), city=scene_city(scene_dir.name, start))

    def track(self, anchor, frames=None):
        """(positions (N,3), xyzw (N,4)) in viewer ENU about `anchor`."""
        frame = (frames or {}).get(self.origin)
        if frame is None:
            frame = SceneFrame(self.origin[0], self.origin[1], *anchor)
            if frames is not None:
                frames[self.origin] = frame
        return track_enu(self.poses, self.timestamps, frame)


class CityIndex:
    """The scenes of one city and their trajectories."""

    def __init__(self, root, city=None, log=print, workers=DEFAULT_WORKERS,
                 cache_dir=None, rescan=False):
        """`cache_dir` keeps what was read for the next start; `rescan` ignores it."""
        self.root = _data_root(root)
        self.log = log
        self.workers = workers
        self.cache = IndexCache(cache_dir, self.root) if cache_dir else None
        cached = self.cache.load() if (self.cache and not rescan) else {}
        found, self.read_count = self._scan(cached)
        self.cached_count = len(found) - self.read_count
        self._entries = {str(d): entry for _split, d, entry in found}
        if self.cache and (self.read_count or set(self._entries) != set(cached)):
            if not self.cache.save(self._entries):
                log("city: could not write the index cache %s" % self.cache.path)
        # unplaceable: (dir, reason) of the scenes `scene_origin` cannot place (test-e2e).
        everything, self.unreadable, self.unplaceable = [], [], []
        for split, scene_dir, entry in found:
            if entry[0] == "unplaced":
                self.unplaceable.append((scene_dir, entry[1]))
                continue
            try:
                if entry[0] == "error":
                    raise ValueError(entry[1])
                everything.append(CityScene.build(split, scene_dir, *entry[1:]))
            except (ValueError, KeyError) as exc:
                self.unreadable.append((scene_dir, entry[1] if entry[0] == "error" else repr(exc)))
        self.cities = {}
        for scene in everything:
            self.cities.setdefault(scene.city or "elsewhere", []).append(scene)
        if city:
            city = city.lower()
            if city not in self.cities:
                raise ValueError("no scenes in %r here; found %s" % (city, self.counts()))
        else:
            # Most scenes wins; the others are named in the log, not dropped
            # silently.
            named = {k: v for k, v in self.cities.items() if k != "elsewhere"}
            pool = named or self.cities
            if not pool:
                raise ValueError("no scenes under %s" % self.root)
            city = max(pool, key=lambda k: len(pool[k]))
        self.city = city
        self.scenes = sorted(self.cities[city], key=lambda s: (s.split, s.id))
        self.by_id = {s.id: s for s in self.scenes}

    def _scan(self, cached):
        """([(split, dir, entry)], how many scenes were read, not cached)."""
        read = []

        def probe(_split, scene_dir):
            entry = cached.get(str(scene_dir))
            if entry is not None:
                return entry
            entry = read_scene_files(scene_dir)
            read.append(1)
            return entry

        found = scan(self.root, probe, workers=self.workers)
        return found, len(read)

    def check_in_background(self):
        """Once, after starting from the cache: read every scene afresh.

        A changed file inside a known scene is the one thing the cached start
        cannot see. This finds it without delaying the start; the cache is
        refreshed for next time and the log says what changed. The running
        server keeps the index it started with. Returns the thread, or None
        when nothing came from the cache.
        """
        if not self.cache or not self.cached_count:
            return None

        def run():
            t0 = time.perf_counter()
            try:
                found, _n = self._scan({})
            except Exception as exc:  # a check must never take the server down
                self.log("city: checking the index cache failed (%r)" % exc)
                return
            fresh = {}
            for _split, d, entry in found:
                old = self._entries.get(str(d))
                # A hiccup now says nothing about the scene: keep what we had.
                transient = entry[0] == "error" and not entry[2]
                fresh[str(d)] = old if (transient and old is not None) else entry
            changed = sorted(k for k in set(fresh) | set(self._entries)
                             if not same_entry(fresh.get(k), self._entries.get(k)))
            if not changed:
                self.log("city: index cache checked against the dataset in %.0fs: up to date"
                         % (time.perf_counter() - t0))
                return
            self.cache.save(fresh)
            self.log("city: %d scenes changed since the index was cached (%s%s); the cache is "
                     "refreshed, restart to see them"
                     % (len(changed), ", ".join(Path(k).name for k in changed[:3]),
                        ", ..." if len(changed) > 3 else ""))

        thread = threading.Thread(target=run, name="ll2-city-index-check", daemon=True)
        thread.start()
        return thread

    def counts(self):
        return {k: len(v) for k, v in sorted(self.cities.items())}

    def describe_log(self):
        splits = {}
        for s in self.scenes:
            splits[s.split] = splits.get(s.split, 0) + 1
        parts = ["%s: %d scenes (%s)" % (self.city, len(self.scenes),
                                          ", ".join("%s %d" % kv for kv in sorted(splits.items())))]
        others = {k: v for k, v in self.counts().items() if k != self.city}
        if others:
            parts.append("other cities here: %s" % ", ".join("%s %d" % kv for kv in others.items()))
        if self.unplaceable:
            reasons = {}
            for _d, reason in self.unplaceable:
                reasons[reason] = reasons.get(reason, 0) + 1
            parts.append("%d scenes not placeable (skipped): %s" % (
                len(self.unplaceable), "; ".join("%s (%d)" % kv for kv in sorted(reasons.items()))))
        if self.unreadable:
            parts.append("%d scenes unreadable (skipped)" % len(self.unreadable))
        return "; ".join(parts)

    def default_anchor(self):
        """The city's projector origin: shared by its scenes, so it is stable."""
        origins = {}
        for s in self.scenes:
            origins[s.origin] = origins.get(s.origin, 0) + 1
        return max(origins, key=origins.get)

    def describe(self, anchor, base_url, aggregate_ready=None):
        """`/city.json`: every scene's trajectory in viewer ENU.

        Positions to the centimetre and quaternions to four decimals, which is
        what the source carries; at ~7 numbers a frame a city of a few
        hundred scenes is a few megabytes, sent once and gzipped.
        """
        frames = {}
        scenes = []
        for s in self.scenes:
            try:
                pos, quat = s.track(anchor, frames)
            except Exception as exc:  # one bad scene must not hide the city
                self.log("city: leaving out %s (%r)" % (s.id, exc))
                continue
            lo = pos.min(axis=0)
            hi = pos.max(axis=0)
            scenes.append({
                "id": s.id,
                "split": s.split,
                "frames": s.frame_count,
                "base": "%s/scene/%s" % (base_url, s.id),
                "pos": [round(float(v), 2) for v in pos.reshape(-1)],
                "quat": [round(float(v), 4) for v in quat.reshape(-1)],
                "bbox": [round(float(v), 1) for v in (lo[0], lo[1], hi[0], hi[1])],
                "aggregate": bool(aggregate_ready(s)) if aggregate_ready else False,
            })
        from kitscenes.instances import CLASS_NAMES, UNLABELLED, palette
        return {
            "city": self.city,
            # The Vistas table, so city clouds are coloured and named by class
            # before any one scene has been opened.
            "classes": {
                "palette": palette().tolist(),
                "names": {str(k): v for k, v in CLASS_NAMES.items()},
                "unlabelled": UNLABELLED,
            },
            "anchor": {"lat": anchor[0], "lon": anchor[1]},
            "base": base_url,
            "scenes": scenes,
        }
