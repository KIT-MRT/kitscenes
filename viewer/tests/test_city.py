"""A city of scenes: finding them, placing them, and serving them on one port."""

import gzip
import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from kitscenes_viewer.scene.city import CityIndex, NotAScene, city_of, scan
from kitscenes_viewer.scene.enu import SceneFrame, utm_epsg
from kitscenes_viewer.scene.scene import NotPlaceable, open_scene, scene_origin
from kitscenes_viewer.scene.http import CityHttpServer, SceneHttpServer
from tests.test_http import StubSource

FRANKFURT = (50.110423, 8.682138)   # the example scene's origin.json
KARLSRUHE = (49.0094, 8.4044)


def utm_offset(origin, lat, lon):
    """Scene metres of (lat, lon) in a scene whose projector origin is `origin`."""
    import pyproj
    fwd = pyproj.Transformer.from_crs("EPSG:4326", utm_epsg(*origin), always_xy=True)
    e0, n0 = fwd.transform(origin[1], origin[0])
    e, n = fwd.transform(lon, lat)
    return e - e0, n - n0


def find_scenes(root, workers=4):
    """Scene directories under `root`, as the index walks them: [(split, dir)]."""
    def probe(_split, directory):
        if not (directory / "poses.txt").is_file():
            raise NotAScene(directory)

    return [(split, directory) for split, directory, _ in scan(root, probe, workers=workers)]


def write_scene(root, split, scene_id, origin, start=(0.0, 0.0), frames=20, heading=0.0,
                reference=True):
    """A scene directory holding only what the city index reads (no `maps/` with `origin` None)."""
    d = Path(root) / "data" / split / scene_id
    d.mkdir(parents=True)
    if origin is not None:
        (d / "maps").mkdir()
        (d / "maps" / "origin.json").write_text(json.dumps(
            {"latitude": origin[0], "longitude": origin[1]}))
    qz, qw = np.sin(heading / 2), np.cos(heading / 2)
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    for i in range(frames):
        x = start[0] + np.cos(heading) * i
        y = start[1] + np.sin(heading) * i
        lines.append("%.3f %.4f %.4f 140.0 0 0 %.4f %.4f" % (100.0 + 0.1 * i, x, y, qz, qw))
    (d / "poses.txt").write_text("\n".join(lines) + "\n")
    if reference:
        (d / "timestamp.reference.txt").write_text(
            "\n".join("%.3f" % (100.0 + 0.1 * i) for i in range(frames)))
    return d


class FindScenesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_scene(self.root, "train", "aaaa-1", FRANKFURT, start=(-500, -900))
        write_scene(self.root, "val", "bbbb-2", FRANKFURT, start=(200, 300))
        write_scene(self.root, "test", "cccc-3", FRANKFURT, start=(1000, -100))
        write_scene(self.root, "train", "kkkk-1", KARLSRUHE)
        # Names that must never reach a URL, and a scene without an origin.
        write_scene(self.root, "val", "bad name", FRANKFURT)
        broken = write_scene(self.root, "val", "eeee-5", FRANKFURT)
        (broken / "maps" / "origin.json").unlink()

    def tearDown(self):
        self.tmp.cleanup()

    def test_any_level_of_the_tree_works_as_root(self):
        for root in (self.root, self.root / "data"):
            scenes = find_scenes(root)
            names = sorted(d.name for _split, d in scenes)
            self.assertEqual(["aaaa-1", "bbbb-2", "cccc-3", "eeee-5", "kkkk-1"], names)
        scenes = find_scenes(self.root / "data" / "train")
        self.assertEqual({"aaaa-1", "kkkk-1"}, {d.name for _s, d in scenes})
        scenes = find_scenes(self.root / "data" / "val" / "bbbb-2")
        self.assertEqual([("val", "bbbb-2")], [(s, d.name) for s, d in scenes])

    def test_the_split_is_the_parent_directory(self):
        scenes = find_scenes(self.root)
        self.assertEqual("test", dict((d.name, s) for s, d in scenes)["cccc-3"])

    def test_the_city_with_most_scenes_is_the_default(self):
        index = CityIndex(self.root, log=lambda _m: None)
        self.assertEqual("frankfurt", index.city)
        # The test split ships no maps and the scene table does not know
        # cccc-3, so it has no city whose origin it could be placed with.
        self.assertEqual(["aaaa-1", "bbbb-2"], sorted(s.id for s in index.scenes))
        self.assertEqual({"frankfurt": 2, "karlsruhe": 1}, index.counts())
        self.assertEqual([("cccc-3", "the test split ships no map origin and the scene's city is unknown")],
                         [(d.name, why) for d, why in index.unplaceable])
        self.assertEqual(["eeee-5"], [d.name for d, _ in index.unreadable])
        log = index.describe_log()
        self.assertIn("1 scenes not placeable (skipped)", log)
        self.assertIn("1 scenes unreadable", log)  # eeee-5: a val scene without its origin.json
        self.assertIn("karlsruhe 1", log)

    def test_another_city_by_name(self):
        index = CityIndex(self.root, city="Karlsruhe", log=lambda _m: None)
        self.assertEqual(["kkkk-1"], [s.id for s in index.scenes])
        with self.assertRaises(ValueError):
            CityIndex(self.root, city="sindelfingen", log=lambda _m: None)

    def test_city_of(self):
        self.assertEqual("frankfurt", city_of(50.10, 8.67))
        self.assertEqual("karlsruhe", city_of(49.0, 8.4))
        self.assertIsNone(city_of(52.5, 13.4))  # Berlin

    def test_the_scene_table_names_the_city(self):
        from kitscenes_viewer.scene.city import scene_city
        # A Karlsruhe scene of the scene table, wherever its trajectory starts.
        self.assertEqual("karlsruhe", scene_city("01883cf7-b15d-3336-a2dd-443a96e31d2c", FRANKFURT))
        self.assertEqual("frankfurt", scene_city("not-in-the-table", FRANKFURT))


class PlacementTest(unittest.TestCase):
    """Scenes without `maps/origin.json`, placed as `kitscenes.aerial.placement_origin` does."""

    def setUp(self):
        from kitscenes.map.origin import CITY_ORIGINS
        from kitscenes.splits import scene_table
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        table = list(scene_table().values())

        def first(split):
            return next(i.scene_id for i in table if i.split == split and i.city == "Karlsruhe")
        self.test_id, self.mc_id = first("test"), first("map_change_test")
        self.e2e_ids = first("test-e2e"), first("map_change_test-e2e")
        city = CITY_ORIGINS["Karlsruhe"]
        self.city_origin = (city.lat, city.lon)
        self.spot = (49.0100, 8.4100)
        # A scene with an origin of its own (a different one, as it may be),
        # and a test scene at the same place, in the city's map frame.
        self.own = (49.0000, 8.4000)
        write_scene(self.root, "train", "kkkk-1", self.own, start=utm_offset(self.own, *self.spot))
        write_scene(self.root, "test", self.test_id, None,
                    start=utm_offset(self.city_origin, *self.spot))
        write_scene(self.root, "map_change_test", self.mc_id, None, start=(300, 200))
        for split, sid in zip(("test-e2e", "map_change_test-e2e"), self.e2e_ids):
            write_scene(self.root, split, sid, None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_test_scenes_take_their_citys_origin(self):
        index = CityIndex(self.root, log=lambda _m: None)
        self.assertEqual("karlsruhe", index.city)
        self.assertEqual({"kkkk-1", self.test_id, self.mc_id}, set(index.by_id))
        self.assertEqual(self.city_origin, index.by_id[self.test_id].origin)
        self.assertEqual(self.city_origin, index.by_id[self.mc_id].origin)
        self.assertEqual(self.own, index.by_id["kkkk-1"].origin)
        anchor = index.default_anchor()
        a, _ = index.by_id["kkkk-1"].track(anchor)
        b, _ = index.by_id[self.test_id].track(anchor)
        self.assertLess(np.abs(a[0] - b[0]).max(), 1e-3, (a[0], b[0]))
        self.assertEqual(self.city_origin, scene_origin(open_scene(self.root / "data" / "test" / self.test_id)))

    def test_test_e2e_scenes_are_not_placeable_and_say_why(self):
        index = CityIndex(self.root, log=lambda _m: None)
        self.assertEqual([], index.unreadable)
        self.assertEqual(
            {(sid, "the %s split's poses are relative to the scene's first pose" % split)
             for split, sid in zip(("test-e2e", "map_change_test-e2e"), self.e2e_ids)},
            {(d.name, why) for d, why in index.unplaceable})
        self.assertIn("2 scenes not placeable (skipped): the map_change_test-e2e split's poses are "
                      "relative to the scene's first pose (1); the test-e2e split's poses",
                      index.describe_log())
        with self.assertRaisesRegex(NotPlaceable, "relative to the scene's first pose"):
            scene_origin(open_scene(self.root / "data" / "test-e2e" / self.e2e_ids[0]))

    def test_the_index_cache_keeps_both(self):
        cache = Path(self.tmp.name) / "cache"
        cold = CityIndex(self.root, log=lambda _m: None, cache_dir=cache)
        warm = CityIndex(self.root, log=lambda _m: None, cache_dir=cache)
        self.assertEqual((0, 5), (warm.read_count, warm.cached_count))
        self.assertEqual(cold.describe_log(), warm.describe_log())
        self.assertEqual(self.city_origin, warm.by_id[self.test_id].origin)
        self.assertEqual(sorted(map(str, cold.unplaceable)), sorted(map(str, warm.unplaceable)))


class TrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_place_is_one_point_whatever_the_scene_origin(self):
        """Scenes are placed through their own origin, so two origins agree."""
        spot = (50.1010, 8.6750)
        other_origin = (50.0900, 8.6600)
        write_scene(self.root, "val", "same-origin", FRANKFURT,
                    start=utm_offset(FRANKFURT, *spot))
        write_scene(self.root, "val", "other-origin", other_origin,
                    start=utm_offset(other_origin, *spot))
        index = CityIndex(self.root, log=lambda _m: None)
        anchor = index.default_anchor()
        a, _ = index.by_id["same-origin"].track(anchor)
        b, _ = index.by_id["other-origin"].track(anchor)
        self.assertLess(np.abs(a[0] - b[0]).max(), 1e-3, (a[0], b[0]))

    def test_describe_carries_every_frame_and_its_heading(self):
        write_scene(self.root, "val", "east", FRANKFURT, start=(100, 100), frames=12)
        write_scene(self.root, "val", "north", FRANKFURT, start=(300, 100), frames=8,
                    heading=np.pi / 2)
        index = CityIndex(self.root, log=lambda _m: None)
        city = index.describe(index.default_anchor(), "http://h:1")
        by_id = {s["id"]: s for s in city["scenes"]}
        self.assertEqual(12, by_id["east"]["frames"])
        self.assertEqual(12 * 3, len(by_id["east"]["pos"]))
        self.assertEqual(8 * 4, len(by_id["north"]["quat"]))
        self.assertEqual("http://h:1/scene/north", by_id["north"]["base"])
        # Driving north: the frames advance in +y, and the heading says so
        # (turned only by the meridian convergence, a fraction of a degree).
        pos = np.array(by_id["north"]["pos"]).reshape(-1, 3)
        self.assertGreater(pos[-1, 1] - pos[0, 1], 6.5)
        q = np.array(by_id["north"]["quat"][:4])
        yaw = np.degrees(2 * np.arctan2(q[2], q[3]))
        self.assertAlmostEqual(90.0, yaw, delta=1.0)
        east = np.array(by_id["east"]["pos"]).reshape(-1, 3)
        lo_x, lo_y, hi_x, hi_y = by_id["east"]["bbox"]
        self.assertAlmostEqual(east[:, 0].min(), lo_x, delta=0.06)
        self.assertAlmostEqual(east[:, 0].max(), hi_x, delta=0.06)
        self.assertAlmostEqual(east[:, 1].min(), lo_y, delta=0.06)

    def test_matches_the_scene_servers_own_track(self):
        """Picking a scene on the city map must land on the same line."""
        from kitscenes_viewer.scene.scene import scene_poses, scene_reference_timestamps, track_enu
        d = write_scene(self.root, "val", "s", FRANKFURT, start=(50, 60), frames=15, heading=0.3)
        index = CityIndex(self.root, log=lambda _m: None)
        anchor = (50.105, 8.677)
        pos, quat = index.by_id["s"].track(anchor)
        scene = open_scene(d)
        want_pos, want_quat = track_enu(scene_poses(scene), scene_reference_timestamps(scene),
                                        SceneFrame(*FRANKFURT, *anchor))
        np.testing.assert_allclose(pos, want_pos)
        np.testing.assert_allclose(quat, want_quat)


class StubSceneSource(StubSource):
    def __init__(self, scene_id):
        super().__init__(decode_seconds=0.2)
        self.scene_id = scene_id
        self.anchors = []

    def set_anchor(self, anchor):
        self.anchors.append(anchor)
        self.anchor = anchor


class CityServingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        for i, sid in enumerate(["s-1", "s-2", "s-3"]):
            write_scene(root, "val", sid, FRANKFURT, start=(100 * i, 0))
        self.index = CityIndex(root, log=lambda _m: None)
        self.opened = []
        self.sources = {}
        self.gates = []

        def open_scene(entry, base_url, gate, build_gate):
            self.opened.append(entry.id)
            self.gates.append((gate, build_gate))
            source = StubSceneSource(entry.id)
            self.sources[entry.id] = source
            return SceneHttpServer(source, log=lambda _m: None, listen=False,
                                   base_url=base_url, gate=gate, build_gate=build_gate)

        self.server = CityHttpServer(self.index, open_scene, self.index.default_anchor(),
                                     port=0, log=lambda _m: None, max_open=2,
                                     aggregate_cached=lambda s: s.id == "s-2")
        self.server.start()

    def tearDown(self):
        self.server.stop()
        self.tmp.cleanup()

    def _get(self, path, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=30)
        try:
            conn.request("GET", path, headers=headers or {})
            response = conn.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            conn.close()

    def test_city_json_plain_and_gzipped(self):
        status, body, _ = self._get("/city.json")
        self.assertEqual(200, status)
        city = json.loads(body)
        self.assertEqual("frankfurt", city["city"])
        self.assertEqual(["s-1", "s-2", "s-3"], [s["id"] for s in city["scenes"]])
        self.assertEqual([False, True, False], [s["aggregate"] for s in city["scenes"]])
        status, packed, headers = self._get("/city.json", {"Accept-Encoding": "gzip"})
        self.assertEqual("gzip", headers.get("Content-Encoding"))
        self.assertEqual(city, json.loads(gzip.decompress(packed)))
        self.assertEqual([], self.opened, "listing the city must not open any scene")

    def test_scene_routes_are_the_single_scene_routes(self):
        status, body, _ = self._get("/scene/s-3/healthz")
        self.assertEqual(200, status)
        self.assertEqual("s-3", json.loads(body)["scene"])
        status, body, _ = self._get("/scene/s-3/frame/2/points.bin")
        self.assertEqual(200, status)
        self.assertEqual(["s-3"], self.opened, "a scene is opened once and kept")
        self.assertEqual(self.index.default_anchor(), self.sources["s-3"].anchor)

    def test_unknown_and_malformed_scenes(self):
        self.assertEqual(404, self._get("/scene/nope/healthz")[0])
        self.assertEqual(404, self._get("/scene/../etc/healthz")[0])
        self.assertEqual(404, self._get("/frame/1/points.bin")[0], "no scene routes at the root")

    def test_least_recently_used_scenes_are_closed(self):
        for sid in ("s-1", "s-2", "s-1", "s-3"):
            self.assertEqual(200, self._get("/scene/%s/healthz" % sid)[0])
        status = json.loads(self._get("/city/status.json")[1])
        self.assertEqual({"s-1": True, "s-2": False, "s-3": True},
                         {k: v["open"] for k, v in status.items()})
        self.assertTrue(status["s-2"]["aggregate"])

    def test_every_scene_gets_the_same_two_gates(self):
        """One decode and one aggregate build at a time, city-wide (the gate
        itself is exercised in SharedGateTest)."""
        for sid in ("s-1", "s-2", "s-3"):
            self._get("/scene/%s/healthz" % sid)
        decode = {id(g) for g, _b in self.gates}
        build = {id(b) for _g, b in self.gates}
        self.assertEqual(1, len(decode))
        self.assertEqual(1, len(build))
        self.assertIsNot(self.gates[0][0], self.gates[0][1])


class SharedGateTest(unittest.TestCase):
    """The decode gate shared by two scenes really is one gate."""

    def test_two_scenes_never_decode_together(self):
        gate = threading.BoundedSemaphore(1)
        live = {"now": 0, "max": 0}
        lock = threading.Lock()

        class Counting(StubSceneSource):
            def frame_points_scene(self, index, lidars=None):
                with lock:
                    live["now"] += 1
                    live["max"] = max(live["max"], live["now"])
                try:
                    time.sleep(0.05)
                    return np.zeros((4, 3)), np.zeros(4, np.float32), np.zeros(4, np.uint8)
                finally:
                    with lock:
                        live["now"] -= 1

        servers = [SceneHttpServer(Counting("a%d" % i), log=lambda _m: None, listen=False,
                                   base_url="http://x/scene/a%d" % i, gate=gate)
                   for i in range(2)]
        threads = [threading.Thread(target=srv.points_blob, args=(i,))
                   for srv in servers for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(1, live["max"])


if __name__ == "__main__":
    unittest.main()
