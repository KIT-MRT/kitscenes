"""One scene without a map of its own: the test splits on their city's map, test-e2e in its own metres."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kitscenes_viewer.scene.enu import LocalFrame
from kitscenes_viewer.scene.scene import SceneSource
from tests.kitscenes_scene import car_at, write_scene

# Two lines near Karlsruhe's map origin (`kitscenes.map.origin.CITY_ORIGINS`).
CITY_OSM = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <node id="1" lat="49.0140000" lon="8.4170000"><tag k="ele" v="115.0"/></node>
  <node id="2" lat="49.0150000" lon="8.4180000"><tag k="ele" v="115.0"/></node>
  <node id="3" lat="49.0142000" lon="8.4174000"><tag k="ele" v="115.0"/></node>
  <node id="4" lat="49.0148000" lon="8.4176000"><tag k="ele" v="115.0"/></node>
  <way id="10"><nd ref="1"/><nd ref="2"/><tag k="type" v="line_thin"/><tag k="subtype" v="solid"/></way>
  <way id="11"><nd ref="3"/><nd ref="4"/><tag k="type" v="line_thin"/><tag k="subtype" v="dashed"/></way>
</osm>
"""


def lanelet2_or_skip():
    try:
        import lanelet2  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the venv
        raise unittest.SkipTest("lanelet2 not installed: %s" % exc)


def release_scene(root, split, scene_id):
    """A scene of a split that ships no maps: no `maps/` folder at all."""
    d = write_scene(root, lambda i: car_at(i) + [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]], frames=6,
                    split=split, scene_id=scene_id)
    (d / "maps" / "origin.json").unlink()
    (d / "maps").rmdir()
    return d


class SingleSceneMapTest(unittest.TestCase):
    def setUp(self):
        from kitscenes.map.origin import CITY_ORIGINS
        from kitscenes.splits import scene_table
        table = list(scene_table().values())

        def first(split):
            return next(i.scene_id for i in table if i.split == split and i.city == "Karlsruhe")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "kitscenes"
        self.test = release_scene(self.root, "test", first("test"))
        self.mc = release_scene(self.root, "map_change_test", first("map_change_test"))
        self.e2e = release_scene(self.root, "test-e2e", first("test-e2e"))
        city = CITY_ORIGINS["Karlsruhe"]
        self.origin = (city.lat, city.lon)

    def tearDown(self):
        self.tmp.cleanup()

    def city_map(self):
        path = self.root / "lanelet2_hd_maps" / "Karlsruhe.osm"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CITY_OSM)
        return path

    def test_a_test_scene_gets_its_citys_map(self):
        lanelet2_or_skip()
        path = self.city_map()
        for scene_dir in (self.test, self.mc):
            with self.subTest(split=scene_dir.parent.name):
                source = SceneSource(scene_dir)
                self.assertTrue(source.placed)
                self.assertEqual(self.origin, (source.origin_lat, source.origin_lon))
                self.assertEqual(path.resolve(), Path(source.map_path).resolve())
                self.assertIn("Karlsruhe city map", " ".join(source.notes))
                # JOSM's choice of anchor: the centre of the map's bbox.
                self.assertAlmostEqual(49.0145, source.anchor[0], places=6)
                self.assertAlmostEqual(8.4175, source.anchor[1], places=6)
                self.assertEqual({"way/10", "way/11"}, {f["id"] for f in source.map_features()})
                scene = source.describe("http://x")["scene"]
                self.assertEqual({"lat": source.anchor[0], "lon": source.anchor[1]}, scene["anchor"])

    def test_without_the_city_map_it_opens_without_a_map(self):
        source = SceneSource(self.test)
        self.assertTrue(source.placed)
        self.assertIsNone(source.map_path)
        self.assertEqual(self.origin, source.anchor)
        self.assertIn("lanelet2_hd_maps/Karlsruhe.osm: opened without a map", " ".join(source.notes))
        with self.assertRaises(FileNotFoundError):
            source.lanelet_map
        self.assertEqual(6, len(source.describe("http://x")["scene"]["ego"]["pos"]))

    def test_map_overrides_the_city_map(self):
        lanelet2_or_skip()
        self.city_map()
        own = Path(self.tmp.name) / "other.osm"
        own.write_text(CITY_OSM.replace('id="11"', 'id="12"'))
        source = SceneSource(self.test, map_path=own)
        self.assertEqual(own.resolve(), source.map_path)
        self.assertEqual([], source.notes)
        self.assertEqual({"way/10", "way/12"}, {f["id"] for f in source.map_features()})

    def test_a_test_e2e_scene_is_shown_in_its_own_metres(self):
        self.city_map()
        source = SceneSource(self.e2e, map_path=Path(self.tmp.name) / "unused.osm")
        self.assertFalse(source.placed)
        self.assertIsNone(source.anchor)
        self.assertIsNone(source.map_path)
        self.assertIsInstance(source.frame, LocalFrame)
        notes = " ".join(source.notes)
        self.assertIn("the test-e2e split's poses are relative to the scene's first pose", notes)
        self.assertIn("is not used", notes)
        scene = source.describe("http://x")["scene"]
        self.assertIsNone(scene["anchor"])
        # The poses as they are, not turned or scaled.
        np.testing.assert_allclose([car_at(i) for i in range(6)], scene["ego"]["pos"], atol=1e-4)
        xyz = source.to_enu(source.frame_points_scene(2)[0])
        np.testing.assert_allclose(car_at(2) + [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]], xyz, atol=0.01)
        source.set_anchor((49.0, 8.4))   # JOSM's anchor: nothing to place the scene with
        self.assertIsNone(source.anchor)


if __name__ == "__main__":
    unittest.main()
