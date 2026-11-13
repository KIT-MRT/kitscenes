"""Hardening of the scene server and the bridge: each test covers one failure mode."""

import http.client
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from kitscenes_viewer.scene.bridge import ViewerBridge, default_reply
from kitscenes_viewer.scene.cache import FrameDiskCache
from kitscenes_viewer.scene.city import CityIndex
from kitscenes_viewer.scene.http import SceneHttpServer
from kitscenes_viewer.scene.points import encode_points
from tests.test_city import FRANKFURT, write_scene
from tests.test_http import StubSource


class CountingAggregator:
    """Stands in for SceneAggregator: a slow build that counts itself."""

    def __init__(self):
        self.builds = 0
        self.progress = {"state": "idle"}
        self._lock = threading.Lock()

    def load_cached(self):
        return None

    def build(self):
        with self._lock:
            self.builds += 1
        time.sleep(0.3)
        xyz = np.zeros((4, 3))
        return xyz, np.zeros(4, np.float32), None, None, {"points": 4}

    def save_cached(self, *args):
        pass


class OneBuildTest(unittest.TestCase):
    def test_concurrent_asks_start_one_build(self):
        aggregator = CountingAggregator()
        server = SceneHttpServer(StubSource(), log=lambda _m: None, listen=False,
                                 base_url="http://x", aggregator=aggregator)
        threads = [threading.Thread(target=server.aggregate_meta) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for _ in range(50):
            if server.aggregate_meta().get("ready"):
                break
            time.sleep(0.05)
        self.assertEqual(1, aggregator.builds)

    def test_a_closed_scene_starts_no_build(self):
        aggregator = CountingAggregator()
        server = SceneHttpServer(StubSource(), log=lambda _m: None, listen=False,
                                 base_url="http://x", aggregator=aggregator)
        server.stop()
        self.assertIn("error", server.aggregate_meta())
        time.sleep(0.1)
        self.assertEqual(0, aggregator.builds)


class DiskCacheTest(unittest.TestCase):
    def test_trimming_leaves_other_files_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "someone-else"
            other.mkdir()
            (other / "photo-2024.bin").write_bytes(b"x" * 2000)
            disk = FrameDiskCache("stub", tmp, limit_gb=1e-6, log=lambda _m: None)
            for i in range(3):
                disk.put(i, b"y" * 600, "rgb")
            self.assertTrue((other / "photo-2024.bin").exists())

    def test_writers_do_not_share_a_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            disk = FrameDiskCache("stub", tmp, log=lambda _m: None)
            blobs = [encode_points(np.full((50, 3), float(i))) for i in range(6)]
            threads = [threading.Thread(target=disk.put, args=(7, b, "rgb")) for b in blobs]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertIn(disk.get(7, "rgb"), blobs, "an entry mixed from several writers")


class CityRobustnessTest(unittest.TestCase):
    def test_a_broken_scene_does_not_hide_the_city(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_scene(root, "val", "good", FRANKFURT, start=(0, 0))
            empty = write_scene(root, "val", "empty-times", FRANKFURT, start=(50, 0))
            (empty / "timestamp.reference.txt").write_text("")
            backwards = write_scene(root, "val", "backwards", FRANKFURT, start=(90, 0))
            lines = (backwards / "poses.txt").read_text().splitlines()
            (backwards / "poses.txt").write_text("\n".join([lines[0]] + lines[1:][::-1]))
            index = CityIndex(root, log=lambda _m: None)
            self.assertEqual(["good"], [s.id for s in index.scenes])
            self.assertEqual(2, len(index.unreadable))
            city = index.describe(index.default_anchor(), "http://x")
            self.assertEqual(["good"], [s["id"] for s in city["scenes"]])


class RequestChecksTest(unittest.TestCase):
    def setUp(self):
        self.server = SceneHttpServer(StubSource(decode_seconds=0), port=0,
                                      allow_origin="http://127.0.0.1:8765", log=lambda _m: None)
        self.server.start()

    def tearDown(self):
        self.server.stop()

    def _status(self, headers):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=10)
        try:
            conn.request("GET", "/healthz", headers=headers)
            return conn.getresponse().status
        finally:
            conn.close()

    def test_the_viewer_page_gets_through(self):
        self.assertEqual(200, self._status({}))
        self.assertEqual(200, self._status({"Origin": "http://127.0.0.1:8765",
                                            "Sec-Fetch-Site": "same-site"}))

    def test_other_sites_cannot_trigger_work(self):
        self.assertEqual(403, self._status({"Sec-Fetch-Site": "cross-site"}))
        self.assertEqual(403, self._status({"Origin": "https://attacker.example"}))

    def test_a_rebound_host_name_is_refused(self):
        self.assertEqual(403, self._status({"Host": "rebind.attacker.example:8767"}))


class BridgeCommandTest(unittest.TestCase):
    def test_malformed_commands_are_refused_not_fatal(self):
        self.assertFalse(default_reply({"type": "command", "ops": "zz"})["ok"])
        self.assertFalse(default_reply({"type": "command", "ops": ["zz"]})["ok"])
        bridge = ViewerBridge("127.0.0.1", 1, log=lambda _m: None)
        sent = []
        bridge.send = sent.append
        for line in (b"[1, 2]", b'"text"', b'{"type":"command","id":"x","ops":"zz"}', b"\xff"):
            bridge._handle(line)  # must not raise
        self.assertEqual(1, len(sent))
        self.assertFalse(sent[0]["ok"])


if __name__ == "__main__":
    unittest.main()
