"""Merged city tiles: the built clouds of every scene that reaches a tile, one point per voxel."""

import http.client
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kitscenes_viewer.scene.aggregate import SceneAggregator
from kitscenes_viewer.scene.cache import FrameDiskCache
from kitscenes_viewer.scene.city import CityIndex
from kitscenes_viewer.scene.points import decode_points
from tests.kitscenes_scene import ORIGIN, write_scene
from tests.test_aggregate import WALL

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "the clouds are built with torch")
class CityTilesTest(unittest.TestCase):
    """Two drives past the same wall: one wall in the merged tiles, with both drives' points."""

    def setUp(self):
        from kitscenes_viewer.scene.scene import SceneSource
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        for scene_id in ("drive-a", "drive-b"):
            write_scene(root, lambda i: WALL, scene_id=scene_id)
        self.cache = root / "cache"
        self.index = CityIndex(root, log=lambda _m: None)
        self.builders = {}
        for scene in self.index.scenes:
            source = SceneSource(scene.dir, anchor=ORIGIN)
            disk = FrameDiskCache(scene.id, self.cache, log=lambda _m: None)
            self.builders[scene.id] = SceneAggregator(source, disk_cache=disk, voxel=0.3, max_range_m=200.0,
                                                      device="cpu", attributes="geometry", log=lambda _m: None)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, scene_id):
        builder = self.builders[scene_id]
        aggregate = builder.build()
        builder.save_cached(aggregate)
        return aggregate

    def aggregate_file(self, scene):
        return self.builders[scene.id].cache_path()

    def tiles(self):
        from kitscenes_viewer.scene.tiles import CityTiles
        return CityTiles(self.index.scenes, self.aggregate_file, self.index.default_anchor(), reach_m=200.0,
                         log=lambda _m: None)

    def merged(self, tiles):
        """Every tile around the wall, merged, as one cloud."""
        from kitscenes_viewer.scene.aggregate import merge_aggregates
        near = tiles.near(130.0, 206.0, 100.0)
        return near, merge_aggregates([p for p in (tiles.tile(t["key"]) for t in near) if p is not None], 1.0)

    def test_two_drives_make_one_wall(self):
        alone = self.build("drive-a")
        near, _ = self.merged(self.tiles())
        one = [t["version"] for t in near]
        self.build("drive-b")
        tiles = self.tiles()
        near, _ = self.merged(tiles)
        self.assertNotEqual(one, [t["version"] for t in near], "a tile must change when one more scene is built")
        pieces = [p for p in (tiles.tile(t["key"]) for t in near) if p is not None]
        total = sum(len(p) for p in pieces)
        self.assertEqual(len(alone), total, "the second drive doubled the wall instead of merging into it")
        self.assertEqual(2 * alone.count.sum(), sum(p.count.sum() for p in pieces))
        xyz = np.concatenate([p.xyz for p in pieces])
        self.assertLess(np.abs(xyz[:, 1] - 206.0).max(), 0.3, "merged points left the wall")

    def test_tiles_need_a_built_cloud(self):
        self.assertEqual([], self.tiles().near(130.0, 206.0, 100.0))
        self.assertIsNone(self.tiles().tile("0_1"))

    def test_routes(self):
        from kitscenes_viewer.scene.http import CityHttpServer
        self.build("drive-a")
        server = CityHttpServer(self.index, open_scene=None, anchor=self.index.default_anchor(), port=0,
                                log=lambda _m: None, aggregate_file=self.aggregate_file, aggregate_range=200.0)
        server.start()
        try:
            def get(path):
                conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
                conn.request("GET", path)
                response = conn.getresponse()
                return response.status, response.read()

            # The city frame's origin is the viewer anchor here, so ENU is about the city frame.
            status, body = get("/city/tiles.json?x=130&y=206&r=100")
            self.assertEqual(200, status, body)
            tiles = json.loads(body)["tiles"]
            self.assertTrue(tiles)
            points = 0
            for t in tiles:
                status, blob = get("/city/tile/%s.bin" % t["key"])
                if status == 404:  # listed by extent, but no voxel of it falls in this tile
                    continue
                self.assertEqual(200, status, blob)
                points += len(decode_points(blob)[0])
            self.assertEqual(len(self.builders["drive-a"].load_cached()), points)
            self.assertEqual(400, get("/city/tiles.json?x=a&y=1")[0])
            self.assertEqual(404, get("/city/tile/99_99.bin")[0])
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
