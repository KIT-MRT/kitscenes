"""Map rules shared with the plugin: heights, one_way, arrows."""

import unittest

from kitscenes_viewer.scene.mapfeed import (fill_unknown_heights, is_two_way_for,
                               other_way_participants, parse_lanelet_bool, usable_ele)

FLT_MAX_SENTINEL = -340282349999999991754788743781432688640.0


class HeightTest(unittest.TestCase):
    def test_sentinel_is_not_a_height(self):
        # Real maps carry -FLT_MAX for "unknown".
        self.assertIsNone(usable_ele(FLT_MAX_SENTINEL))
        self.assertIsNone(usable_ele(float("nan")))
        self.assertIsNone(usable_ele(float("inf")))
        self.assertIsNone(usable_ele(1e6))
        self.assertIsNone(usable_ele("abc"))
        self.assertIsNone(usable_ele(None))
        self.assertEqual(144.6, usable_ele(144.6))
        self.assertEqual(-12.5, usable_ele("-12.5"))

    def test_interpolated_by_distance_between_known_heights(self):
        xs = [0.0, 10.0, 40.0]
        ys = [0.0, 0.0, 0.0]
        self.assertEqual([100.0, 101.0, 104.0],
                         fill_unknown_heights([100.0, None, 104.0], xs, ys))

    def test_copied_at_the_ends(self):
        xs = [0.0, 10.0, 20.0, 30.0]
        ys = [0.0] * 4
        self.assertEqual([163.5, 163.5, 164.0, 164.0],
                         fill_unknown_heights([None, 163.5, 164.0, None], xs, ys))

    def test_a_line_without_any_height_sits_at_zero(self):
        self.assertEqual([0.0, 0.0], fill_unknown_heights([None, None], [0.0, 5.0], [0.0, 0.0]))

    def test_known_heights_are_untouched(self):
        zs = [1.0, 2.0, 3.0]
        self.assertEqual(zs, fill_unknown_heights(zs, [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]))


class OneWayTest(unittest.TestCase):
    def test_lanelet2_bool_parsing(self):
        for value in ("1", "true", "yes"):
            self.assertIs(True, parse_lanelet_bool(value))
        for value in ("0", "false", "no"):
            self.assertIs(False, parse_lanelet_bool(value))
        self.assertIsNone(parse_lanelet_bool("maybe"))
        self.assertIsNone(parse_lanelet_bool(None))

    def test_one_way_decides_when_it_is_a_bool(self):
        self.assertFalse(is_two_way_for({"one_way": "yes"}))
        self.assertTrue(is_two_way_for({"one_way": "no"}))
        # an override cannot overrule an explicit one_way
        self.assertFalse(is_two_way_for({"one_way": "yes", "one_way:vehicle:car": "no"}))

    def test_overrides_apply_by_key_prefix(self):
        self.assertTrue(is_two_way_for({"one_way:vehicle": "no"}))
        self.assertTrue(is_two_way_for({"one_way:vehicle:car": "no"}))
        self.assertFalse(is_two_way_for({"one_way:vehicle:bus": "no"}))
        self.assertTrue(is_two_way_for({"one_way:bicycle": "no"}, "bicycle"))

    def test_without_any_tag_only_pedestrians_go_both_ways(self):
        self.assertFalse(is_two_way_for({}))
        self.assertTrue(is_two_way_for({}, "pedestrian"))

    def test_other_way_participants(self):
        self.assertEqual(["bicycle"], other_way_participants({"one_way:bicycle": "no"}))
        self.assertEqual([], other_way_participants({"one_way": "yes", "one_way:bicycle": "no"}))
        self.assertEqual([], other_way_participants({"one_way:vehicle": "no"}))


if __name__ == "__main__":
    unittest.main()


class StreamedTagsTest(unittest.TestCase):
    """What the viewer reads off a streamed way must be in the stream.

    The browser test drives a fake bridge that sends whatever tags it likes,
    which once hid that `height` never left this server: the Autoware light
    icon then always lay flat. So this goes through lanelet2's own loader.
    """

    OSM = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <node id="1" lat="49.0000000" lon="8.4000000"><tag k="ele" v="168.0"/></node>
  <node id="2" lat="49.0000000" lon="8.4000040"><tag k="ele" v="168.0"/></node>
  <way id="10">
    <nd ref="1"/><nd ref="2"/>
    <tag k="type" v="traffic_light"/>
    <tag k="subtype" v="red_yellow_green"/>
    <tag k="height" v="0.86"/>
    <tag k="original_id" v="8407"/>
  </way>
</osm>
"""

    def test_an_autoware_light_keeps_its_height(self):
        try:
            import lanelet2  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on the venv
            raise unittest.SkipTest("lanelet2 not installed: %s" % exc)
        import tempfile
        from pathlib import Path

        from kitscenes_viewer.scene.enu import SceneFrame
        from kitscenes_viewer.scene.mapfeed import MapConverter
        from kitscenes_viewer.scene.scene import MapFile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "light.osm"
            path.write_text(self.OSM)
            origin = (49.0, 8.4)
            lanelet_map = MapFile(path, origin).lanelet_map
            features = MapConverter(SceneFrame(*origin, *origin)).features(lanelet_map)
        light = next(f for f in features if f["id"] == "way/10")
        self.assertEqual({"type": "traffic_light", "subtype": "red_yellow_green",
                          "height": "0.86"}, light["tags"])
        self.assertEqual(6, len(light["pts"]), "two points: the light's bottom edge")
