"""The city index on a slow dataset: walked in parallel, cached locally.

The dataset usually sits on a network mount where every file operation is a
round trip, so the index reads in a thread pool and keeps what it read on
local disk. None of that may change *what* the index finds.
"""

import os
import tempfile
import unittest
from pathlib import Path


from kitscenes_viewer.scene.city import SCENE_ID_RE, CityIndex, CityScene, IndexCache, read_scene_files
from tests.test_city import FRANKFURT, KARLSRUHE, find_scenes, write_scene


def sequential_index(root, max_depth=3):
    """The index walked sequentially: the reference.

    One directory at a time, checking for `poses.txt` before reading it.
    Returns what a CityIndex exposes: scenes, unreadable dirs, not placeable dirs.
    """
    root = Path(root).expanduser().resolve()
    root = root / "data" if (root / "data").is_dir() else root
    found = []
    if (root / "poses.txt").is_file():
        found = [(root.parent.name, root)] if SCENE_ID_RE.match(root.name) else []
    else:
        def walk(directory, depth):
            for child in sorted(directory.iterdir()):
                if child.is_dir():
                    if (child / "poses.txt").is_file():
                        if SCENE_ID_RE.match(child.name):
                            found.append((directory.name, child))
                    elif depth < max_depth:
                        walk(child, depth + 1)
        walk(root, 1)
    scenes, unreadable, unplaceable = [], [], []
    for split, scene_dir in found:
        entry = read_scene_files(scene_dir)
        if entry[0] == "unplaced":
            unplaceable.append(scene_dir)
            continue
        try:
            if entry[0] == "error":
                raise ValueError(entry[1])
            scenes.append(CityScene.build(split, scene_dir, *entry[1:]))
        except (ValueError, KeyError):
            unreadable.append(scene_dir)
    return scenes, unreadable, unplaceable


def summary(scenes):
    """Everything about each scene that the city uses, comparable with ==."""
    return {s.id: (s.split, str(s.dir), s.origin, s.frame_count, s.start, s.city,
                   s.poses["time"].tobytes(), s.poses["t"].tobytes(), s.poses["q"].tobytes(),
                   s.timestamps.tobytes())
            for s in scenes}


def every_scene(index):
    return [s for group in index.cities.values() for s in group]


def build_tree(root):
    """A dataset with every case the walk has to get right."""
    write_scene(root, "train", "aaaa-1", FRANKFURT, start=(-500, -900))
    write_scene(root, "train", "kkkk-1", KARLSRUHE, frames=7)
    write_scene(root, "val", "bbbb-2", FRANKFURT, start=(200, 300))
    write_scene(root, "val", "llll-3", FRANKFURT, reference=False)   # no timestamp.reference.txt
    write_scene(root, "test", "cccc-3", FRANKFURT, start=(1000, -100), heading=0.4)
    write_scene(root, "val", "bad name", FRANKFURT)               # cannot go in a URL
    no_origin = write_scene(root, "val", "eeee-5", FRANKFURT)
    (no_origin / "maps" / "origin.json").unlink()
    garbled = write_scene(root, "val", "ffff-6", FRANKFURT)
    (garbled / "poses.txt").write_text("# header\n1.0 not a number 2 3 0 0 0 1\n")
    backwards = write_scene(root, "val", "gggg-7", FRANKFURT)
    lines = (backwards / "poses.txt").read_text().splitlines()
    (backwards / "poses.txt").write_text("\n".join([lines[0]] + lines[1:][::-1]))
    # A folder of folders inside a split: found one level down, not deeper.
    write_scene(root, "val/extra", "hhhh-8", FRANKFURT, start=(5, 5))
    write_scene(root, "val/deep/deeper", "iiii-9", FRANKFURT)
    # Sensor folders inside a scene are never walked.
    (root / "data" / "train" / "aaaa-1" / "lidar_top" / "jjjj-0").mkdir(parents=True)
    (root / "data" / "train" / "aaaa-1" / "lidar_top" / "jjjj-0" / "poses.txt").write_text("")


class ParallelWalkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "kitscenes"
        build_tree(self.root)
        self.cache = Path(self.tmp.name) / "cache"

    def tearDown(self):
        self.tmp.cleanup()

    def assertSameAsSequential(self, index, root):
        scenes, unreadable, unplaceable = sequential_index(root)
        self.assertEqual(summary(scenes), summary(every_scene(index)))
        self.assertEqual(sorted(map(str, unreadable)), sorted(str(d) for d, _ in index.unreadable))
        self.assertEqual(sorted(map(str, unplaceable)), sorted(str(d) for d, _ in index.unplaceable))

    def test_matches_the_sequential_walk_at_any_worker_count_and_from_the_cache(self):
        for root in (self.root, self.root / "data", self.root / "data" / "val",
                     self.root / "data" / "train" / "aaaa-1"):
            for workers in (1, 3, 32):
                with self.subTest(root=root.name, workers=workers):
                    index = CityIndex(root, log=lambda _m: None, workers=workers)
                    self.assertSameAsSequential(index, root)
            # Own cache per root: the tree's root and its data/ are one
            # dataset and rightly share one.
            cache = self.cache / root.name
            cold = CityIndex(root, log=lambda _m: None, cache_dir=cache)
            warm = CityIndex(root, log=lambda _m: None, cache_dir=cache)
            with self.subTest(root=root.name, cache="warm"):
                self.assertSameAsSequential(warm, root)
                self.assertEqual(0, warm.read_count)
                self.assertEqual(cold.read_count, warm.cached_count)
                self.assertEqual(cold.describe_log(), warm.describe_log())

    def test_what_the_tree_holds(self):
        """Guards the fixture itself: each case above really is in play."""
        index = CityIndex(self.root, log=lambda _m: None)
        self.assertEqual({"aaaa-1", "bbbb-2", "hhhh-8", "kkkk-1"},
                         {s.id for s in every_scene(index)})
        # cccc-3: the test split ships no maps and its city is unknown.
        self.assertEqual({"cccc-3"}, {d.name for d, _ in index.unplaceable})
        self.assertEqual({"eeee-5", "ffff-6", "gggg-7", "llll-3"}, {d.name for d, _ in index.unreadable})
        why = dict((d.name, w) for d, w in index.unreadable)
        self.assertIn("FileNotFoundError", why["llll-3"])
        self.assertIn("timestamp.reference.txt", why["llll-3"])
        self.assertEqual("extra", index.by_id["hhhh-8"].split)

    def test_find_scenes_agrees(self):
        scenes = find_scenes(self.root, workers=8)
        want, unreadable, unplaceable = sequential_index(self.root)
        # find_scenes only asks "is there a poses.txt", so it also lists the
        # scenes the index then cannot read or place.
        self.assertEqual(sorted([str(s.dir) for s in want] + [str(d) for d in unreadable + unplaceable]),
                         sorted(str(d) for _s, d in scenes))


class IndexCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "kitscenes"
        self.cache = Path(self.tmp.name) / "cache"
        self.a = write_scene(self.root, "val", "aaaa-1", FRANKFURT, start=(0, 0))
        self.b = write_scene(self.root, "val", "bbbb-2", FRANKFURT, start=(50, 0))
        self.logs = []

    def tearDown(self):
        self.tmp.cleanup()

    def index(self, **kw):
        return CityIndex(self.root, log=self.logs.append, cache_dir=self.cache, **kw)

    def test_a_restart_reads_only_new_scenes_and_forgets_gone_ones(self):
        self.assertEqual(2, self.index().read_count)
        self.assertEqual(0, self.index().read_count)
        write_scene(self.root, "val", "cccc-3", FRANKFURT, start=(90, 0))
        index = self.index()
        self.assertEqual((1, 2), (index.read_count, index.cached_count))
        self.assertIn("cccc-3", index.by_id)
        for p in sorted(self.b.rglob("*"), reverse=True):
            p.rmdir() if p.is_dir() else p.unlink()
        self.b.rmdir()
        index = self.index()
        self.assertEqual(["aaaa-1", "cccc-3"], sorted(index.by_id))
        self.assertEqual(2, len(IndexCache(self.cache, index.root).load()))

    def test_a_changed_file_is_found_by_the_check_and_by_rescan(self):
        self.index()
        # A known scene changes inside: the cached start cannot see it ...
        write_scene(self.root, "val", "aaaa-1-new", FRANKFURT, start=(7, 7))
        (self.a / "poses.txt").write_text(
            (self.root / "data" / "val" / "aaaa-1-new" / "poses.txt").read_text())
        stale = self.index()
        self.assertAlmostEqual(0.0, stale.by_id["aaaa-1"].poses["t"][0, 0])
        # ... the one check behind it does, and refreshes the cache for next time.
        stale.check_in_background().join(timeout=30)
        self.assertTrue(any("1 scenes changed" in m and "aaaa-1" in m for m in self.logs),
                        self.logs)
        self.assertAlmostEqual(7.0, self.index().by_id["aaaa-1"].poses["t"][0, 0])
        # --rescan sees it straight away.
        (self.a / "poses.txt").write_text((self.b / "poses.txt").read_text())
        self.assertAlmostEqual(50.0, self.index(rescan=True).by_id["aaaa-1"].poses["t"][0, 0])

    def test_the_check_says_when_all_is_well_and_only_runs_after_a_cached_start(self):
        self.assertIsNone(self.index().check_in_background())     # nothing came from the cache
        self.assertIsNone(CityIndex(self.root, log=self.logs.append).check_in_background())
        self.index().check_in_background().join(timeout=30)
        self.assertTrue(any("up to date" in m for m in self.logs), self.logs)

    @unittest.skipIf(os.name != "posix" or os.geteuid() == 0, "needs file permissions")
    def test_a_read_that_may_pass_next_time_is_not_cached(self):
        poses = self.a / "poses.txt"
        poses.chmod(0)
        try:
            index = self.index()
            self.assertEqual(["bbbb-2"], sorted(index.by_id))
            self.assertIn("PermissionError", index.unreadable[0][1])
        finally:
            poses.chmod(0o644)
        index = self.index()
        self.assertEqual((1, 1), (index.read_count, index.cached_count))
        self.assertEqual(["aaaa-1", "bbbb-2"], sorted(index.by_id))

    def test_a_missing_file_is_cached(self):
        (self.a / "maps" / "origin.json").unlink()
        self.assertEqual(1, len(self.index().unreadable))
        index = self.index()
        self.assertEqual((0, 1), (index.read_count, len(index.unreadable)))

    def test_a_broken_cache_is_ignored_and_rewritten(self):
        index = self.index()
        index.cache.path.write_bytes(b"not a zip")
        index = self.index()
        self.assertEqual(2, index.read_count)
        self.assertEqual(0, self.index().read_count)

    def test_the_dataset_is_never_written(self):
        def snapshot():
            return {str(p): (p.stat().st_mtime_ns, p.stat().st_size)
                    for p in [self.root] + sorted(self.root.rglob("*"))}
        before = snapshot()
        index = self.index()
        index.check_in_background().join(timeout=30) if index.cached_count else None
        self.index(rescan=True)
        self.index().check_in_background().join(timeout=30)
        self.assertEqual(before, snapshot())
        self.assertEqual([], [p for p in self.root.rglob("*.part")])


if __name__ == "__main__":
    unittest.main()
