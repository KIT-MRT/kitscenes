"""Aggregating a scene, and taking the traffic out of it (on kitscenes.mapping)."""

import importlib.util
import tempfile
import unittest

import numpy as np

from kitscenes_viewer.scene.aggregate import Aggregate, SceneAggregator, chunk_by_tile, merge_aggregates
from kitscenes_viewer.scene.scene import SceneSource
from tests.kitscenes_scene import ORIGIN, car_at, write_scene

HAS_TORCH = importlib.util.find_spec("torch") is not None

# A wall 6 m to the left of the road, from x 100 to 160 and 3 m high.
WALL = np.stack([
    np.linspace(100.0, 160.0, 400),
    np.full(400, 206.0),
    150.0 + np.tile(np.linspace(0.0, 3.0, 20), 20),
], axis=1)


def car_alongside(index):
    """A car 8 m ahead and 3 m to the right, moving with us: a different place every frame.

    0.8 m long at 1 m per frame, so no 0.3 m voxel (on a grid fixed in UTM, at any
    offset from this frame) can touch it in three consecutive frames.
    """
    car = np.random.default_rng(index).uniform(-0.4, 0.4, size=(300, 3))
    return car + car_at(index) + [8.0, -3.0, 0.8]


def wall_and_car(index):
    return np.vstack([WALL, car_alongside(index)])


class SceneCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def source(self, points_of=wall_and_car, **scene):
        return SceneSource(write_scene(self.tmp.name, points_of, **scene), anchor=ORIGIN)

    def build(self, source=None, **kw):
        if source is None:
            if not hasattr(self, "_default"):
                self._default = self.source()
            source = self._default
        options = dict(voxel=0.3, max_range_m=200.0, min_run=3, device="cpu", attributes="all",
                       log=lambda _m: None)
        options.update(kw)
        return SceneAggregator(source, **options).build()


@unittest.skipUnless(HAS_TORCH, "the aggregate is built with torch")
class AggregateTest(SceneCase):
    def test_the_moving_car_is_removed_and_the_wall_kept(self):
        agg = self.build()
        xyz, stats = agg.xyz, agg.stats
        near_wall = np.abs(xyz[:, 1] - 206.0) < 1.0
        on_car = xyz[:, 1] < 199.0
        self.assertGreater(near_wall.sum(), 100, "the static wall was thrown away")
        self.assertEqual(0, on_car.sum(), "%d moving-car points survived" % on_car.sum())
        self.assertEqual(12, stats["frames"])
        self.assertLess(stats["static_voxels"], stats["voxels"])

    def test_a_car_crossing_the_same_spot_three_times_is_still_not_static(self):
        """Counting distinct frames leaks on a long scene; a run does not."""
        def three_passes(index):
            # The same patch of road is occupied in frames 0, 5 and 10 by
            # three different cars -- three hits, never consecutive.
            if index not in (0, 5, 10):
                return WALL
            blob = np.random.default_rng(index).uniform(-0.4, 0.4, size=(200, 3)) + [120.0, 197.0, 150.8]
            return np.vstack([WALL, blob])

        xyz = self.build(self.source(three_passes)).xyz
        self.assertEqual(0, (xyz[:, 1] < 199.0).sum(),
                         "three separate passes were mistaken for a static object")

    def test_keeping_everything_keeps_the_car(self):
        xyz = self.build(min_run=1).xyz
        self.assertGreater((xyz[:, 1] < 199.0).sum(), 100, "min_run=1 should keep the traffic in")

    def test_downsamples_to_one_point_a_voxel(self):
        agg = self.build()
        xyz, stats = agg.xyz, agg.stats
        # Voxels lie on a grid fixed in UTM (kitscenes.mapping), offset from this frame.
        from kitscenes.map import MapOrigin
        from kitscenes.mapping import ANCHOR_STEP
        utm = MapOrigin(*ORIGIN).utm_offset
        shift = np.append(utm - np.round(utm / ANCHOR_STEP) * ANCHOR_STEP, 0.0)
        keys = {tuple(k) for k in np.floor((xyz + shift) / 0.3).astype(np.int64)}
        self.assertEqual(len(xyz), len(keys), "a voxel came back twice")
        self.assertEqual(stats["points"], len(xyz))

    def test_points_land_inside_their_voxel(self):
        xyz = self.build().xyz
        wall = xyz[np.abs(xyz[:, 1] - 206.0) < 1.0]
        # The wall is a plane at y = 206; centroids must sit on it, not drift.
        self.assertLess(np.abs(wall[:, 1] - 206.0).max(), 0.3)

    def test_range_limit_drops_far_returns(self):
        near = self.build(max_range_m=10.0).stats
        far = self.build(max_range_m=200.0).stats
        self.assertLess(near["points"], far["points"])

    def test_stats_describe_the_settings(self):
        stats = self.build(voxel=0.25, min_run=4).stats
        self.assertEqual(0.25, stats["voxel_m"])
        self.assertEqual(4, stats["min_run"])

    def test_progress_counts_frames(self):
        agg = SceneAggregator(self.source(), voxel=0.3, device="cpu", log=lambda _m: None)
        agg.build()
        self.assertEqual({"state": "ready", "done": 12, "total": 12}, {k: agg.progress[k] for k in ("state", "done", "total")})


@unittest.skipUnless(HAS_TORCH, "the aggregate is built with torch")
class AggregateCameraTest(SceneCase):
    """Each voxel keeps the class most of its points were given, and their mean camera colour."""

    def test_the_majority_class_wins_and_the_colour_is_the_cameras(self):
        # A Car in the last frame only. The camera sees the wall from 9.4 m ahead
        # on, so whatever it sees then it saw in every earlier frame as well.
        source = self.source(camera_rgb=(200, 40, 40), mask_class_of=lambda i: 58 if i == 11 else 48)
        agg = self.build(source, attributes="all")
        classes, rgb = agg.classes, agg.rgb
        seen = classes != 255  # what the front camera saw of the wall
        self.assertGreater(seen.sum(), 20)
        self.assertTrue((classes[seen] == 48).all(), "the one-frame class outvoted the one in every other frame")
        self.assertTrue((agg.class_votes[seen] > 0).all())
        np.testing.assert_allclose(rgb[seen].astype(int), np.tile([200, 40, 40], (seen.sum(), 1)), atol=8)

    def test_unlabelled_where_nothing_was_seen(self):
        source = self.source(camera_rgb=(0, 0, 0), mask_class_of=lambda i: None)
        classes = self.build(source, attributes="all").classes
        self.assertTrue((classes == 255).all())

    def test_a_scene_without_masks_or_cameras_still_aggregates(self):
        agg = self.build(attributes="all")
        self.assertGreater(len(agg), 0)
        self.assertIsNone(agg.classes)
        self.assertIsNone(agg.rgb)

    def test_geometry_only_and_colour_only(self):
        source = self.source(camera_rgb=(200, 40, 40), mask_class_of=lambda i: 48)
        geometry = self.build(source, attributes="geometry")
        self.assertIsNone(geometry.rgb)
        self.assertIsNone(geometry.classes)
        colour = self.build(source, attributes="colour")
        self.assertIsNotNone(colour.rgb)
        self.assertIsNone(colour.classes)
        np.testing.assert_array_equal(geometry.xyz, colour.xyz)
        self.assertEqual("geometry", geometry.stats["attributes"])


class ChunkTest(unittest.TestCase):
    def test_splits_by_tile_and_keeps_every_point(self):
        rng = np.random.default_rng(1)
        xyz = rng.uniform(-300, 300, size=(5000, 3))
        intensity = rng.random(5000).astype(np.float32)
        chunks = chunk_by_tile(xyz, intensity, tile_m=128.0)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(5000, sum(c["count"] for c in chunks))
        for chunk in chunks:
            spread = np.linalg.norm(chunk["xyz"] - np.array(chunk["centre"]), axis=1).max()
            self.assertLessEqual(spread, chunk["radius"] + 1e-6)

    def test_caps_the_size_of_one_chunk(self):
        xyz = np.zeros((1000, 3))
        chunks = chunk_by_tile(xyz, np.zeros(1000, np.float32), max_points=300)
        self.assertEqual(4, len(chunks))
        self.assertEqual(1000, sum(c["count"] for c in chunks))

    def test_empty(self):
        self.assertEqual([], chunk_by_tile(np.zeros((0, 3)), np.zeros(0, np.float32)))


if __name__ == "__main__":
    unittest.main()


class MergeTest(unittest.TestCase):
    """Clouds of several scenes on one voxel grid -> one point per voxel."""

    def test_weighted_by_the_points_behind_each_voxel(self):
        a = Aggregate(np.array([[0.05, 0.05, 0.05], [1.05, 0.0, 0.0]]), np.array([1.0, 0.2], np.float32),
                      np.array([1, 5], np.int32), np.array([48, 255], np.uint8),
                      np.array([[100, 0, 0], [90, 90, 90]], np.uint8), class_votes=np.array([1, 5], np.int32))
        b = Aggregate(np.array([[0.07, 0.05, 0.05]]), np.array([0.0], np.float32), np.array([3], np.int32),
                      np.array([58], np.uint8), np.array([[0, 0, 100]], np.uint8),
                      class_votes=np.array([3], np.int32))
        merged = merge_aggregates([a, b], 0.1)
        order = np.argsort(merged.xyz[:, 0])
        self.assertEqual(2, len(merged))
        np.testing.assert_allclose(merged.xyz[order[0]], [0.065, 0.05, 0.05])  # (1 * 0.05 + 3 * 0.07) / 4
        np.testing.assert_allclose(merged.intensity[order], [0.25, 0.2])
        self.assertEqual([4, 5], merged.count[order].tolist())
        self.assertEqual([58, 255], merged.classes[order].tolist())  # 3 Car points beat 1 Pole point
        np.testing.assert_array_equal(merged.rgb[order[0]], [25, 0, 75])  # count-weighted
        np.testing.assert_array_equal(merged.rgb[order[1]], [90, 90, 90])  # no camera saw it

    def test_votes_add_up_unlabelled_included(self):
        """Scenes vote with the points behind their class; "unlabelled" is a class like any other."""
        one = lambda cls, votes: Aggregate(np.zeros((1, 3)) + 0.05, np.ones(1, np.float32), np.array([10], np.int32),  # noqa: E731
                                           np.array([cls], np.uint8), class_votes=np.array([votes], np.int32))
        merged = merge_aggregates([one(19, 4), one(255, 6)], 0.1)
        self.assertEqual(([255], [6]), (merged.classes.tolist(), merged.class_votes.tolist()))
        merged = merge_aggregates([one(19, 4), one(19, 2), one(255, 5)], 0.1)
        self.assertEqual(([19], [6]), (merged.classes.tolist(), merged.class_votes.tolist()))
        self.assertEqual([255], merge_aggregates([one(19, 5), one(255, 5)], 0.1).classes.tolist())  # a tie

    def test_geometry_only_parts_merge_with_coloured_ones(self):
        a = Aggregate(np.zeros((1, 3)) + 0.05, np.ones(1, np.float32), np.ones(1, np.int32))
        b = Aggregate(np.zeros((1, 3)) + 0.05, np.ones(1, np.float32), np.ones(1, np.int32),
                      rgb=np.array([[10, 20, 30]], np.uint8))
        merged = merge_aggregates([a, b], 0.1)
        np.testing.assert_array_equal(merged.rgb, [[10, 20, 30]])
        self.assertIsNone(merged.classes)
        self.assertEqual(0, len(merge_aggregates([], 0.1)))


class AggregateCacheTest(SceneCase):
    """Building takes a while, so the result has to survive a restart."""

    def setUp(self):
        super().setUp()
        from kitscenes_viewer.scene.cache import FrameDiskCache
        self.disk = FrameDiskCache("stub", self.tmp.name + "/cache", log=lambda _m: None)
        self.agg = SceneAggregator(self.source(), disk_cache=self.disk, voxel=0.3, max_range_m=200.0,
                                   device="cpu", attributes="all", log=lambda _m: None)

    @unittest.skipUnless(HAS_TORCH, "the aggregate is built with torch")
    def test_round_trip(self):
        built = self.agg.build()
        self.assertIsNone(self.agg.load_cached(), "nothing should be cached yet")
        self.agg.save_cached(built)
        self.assertTrue(self.agg.cache_path().is_file(), "no file was written")
        loaded = self.agg.load_cached()
        self.assertIsNotNone(loaded, "the cache could not be read back")
        np.testing.assert_allclose(loaded.xyz, built.xyz, atol=1e-3)
        np.testing.assert_allclose(loaded.intensity, built.intensity, atol=1e-6)
        np.testing.assert_array_equal(loaded.count, built.count)
        self.assertEqual(built.stats["points"], int(loaded.stats["points"]))
        self.assertEqual(built.stats["min_run"], int(loaded.stats["min_run"]))

    def test_different_settings_use_different_files(self):
        other = SceneAggregator(self.agg.source, disk_cache=self.disk, voxel=0.5, max_range_m=200.0,
                                log=lambda _m: None)
        self.assertNotEqual(self.agg.cache_path(), other.cache_path())
        geometry = SceneAggregator(self.agg.source, disk_cache=self.disk, voxel=0.3, max_range_m=200.0,
                                   attributes="geometry", log=lambda _m: None)
        self.assertNotEqual(self.agg.cache_path(), geometry.cache_path())

    def test_no_cache_directory(self):
        agg = SceneAggregator(self.agg.source, disk_cache=None, log=lambda _m: None)
        self.assertIsNone(agg.cache_path())
        self.assertIsNone(agg.load_cached())
        agg.save_cached(Aggregate(np.zeros((1, 3)), np.zeros(1, np.float32), np.ones(1, np.int32)))  # no raise

    def test_a_corrupt_cache_is_ignored(self):
        self.agg.save_cached(Aggregate(np.zeros((1, 3)), np.zeros(1, np.float32), np.ones(1, np.int32)))
        self.agg.cache_path().write_bytes(b"not an npz")
        self.assertIsNone(self.agg.load_cached())


if __name__ == "__main__":
    unittest.main()
