"""Serving frames: one decode at a time, shared, and dropped when abandoned.

`ThreadingHTTPServer` gives every request its own thread, so without these
rules a few taps on the step key put several decodes on the machine at once
and every answer arrives later, including the frame the user is waiting for.
"""

import http.client
import socket
import tempfile
import threading
import time
import unittest

import numpy as np

from kitscenes_viewer.scene.http import GONE, SceneHttpServer, client_gone
from kitscenes_viewer.scene.points import decode_points


class StubSource:
    """Stands in for a SceneSource; counts and paces its decodes."""

    scene_id = "stub-scene"
    frame_count = 20
    cameras = ["camera_ring_front"]
    anchor = (49.0, 8.4)

    def __init__(self, decode_seconds=0.3):
        self.decode_seconds = decode_seconds
        self.decodes = []
        self.live = 0
        self.max_live = 0
        self._lock = threading.Lock()

    def frame_points_scene(self, index, lidars=None):
        with self._lock:
            self.live += 1
            self.max_live = max(self.max_live, self.live)
            self.decodes.append(index)
        try:
            time.sleep(self.decode_seconds)
            xyz = np.full((4, 3), float(index))
            return xyz, np.zeros(4, np.float32), np.zeros(4, np.uint8)
        finally:
            with self._lock:
                self.live -= 1

    def to_enu(self, xyz):
        return xyz

    def has_masks(self):
        return False

    def frame_rgb(self, index, xyz):
        return np.full((len(xyz), 3), 90, np.uint8)

    def image_bytes(self, camera, index):
        return None

    def describe(self, base_url):
        return {"scene": {"id": self.scene_id, "base": base_url}}


class ServingTest(unittest.TestCase):
    def setUp(self):
        self.source = StubSource()
        self.server = SceneHttpServer(self.source, port=0, log=lambda _m: None)
        self.server.start()

    def tearDown(self):
        self.server.stop()

    def _get(self, path, results=None, key=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=30)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            body = response.read()
            if results is not None:
                results[key] = (response.status, body)
            return response.status, body
        finally:
            conn.close()

    def test_serves_a_frame(self):
        status, body = self._get("/frame/3/points.bin")
        self.assertEqual(200, status)
        xyz, *_ = decode_points(body)
        self.assertEqual(4, len(xyz))
        np.testing.assert_allclose(xyz, 3.0)

    def test_only_one_decode_runs_at_a_time(self):
        threads = [threading.Thread(target=self._get, args=("/frame/%d/points.bin" % i,))
                   for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(1, self.source.max_live,
                         "%d decodes ran at once" % self.source.max_live)

    def test_the_same_frame_is_decoded_once_for_everyone(self):
        results = {}
        threads = [threading.Thread(target=self._get,
                                    args=("/frame/9/points.bin", results, i))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual([9], self.source.decodes, "the frame was decoded more than once")
        self.assertEqual(4, len(results))
        bodies = {body for _status, body in results.values()}
        self.assertEqual(1, len(bodies), "callers got different answers")

    def test_a_second_request_is_answered_from_memory(self):
        self._get("/frame/2/points.bin")
        self._get("/frame/2/points.bin")
        self.assertEqual([2], self.source.decodes)

    def test_queued_work_is_dropped_when_the_browser_leaves(self):
        slow = threading.Thread(target=self._get, args=("/frame/0/points.bin",))
        slow.start()
        time.sleep(0.05)  # let it take the decoder

        # Queue a request, then hang up like an aborted fetch does.
        sock = socket.create_connection(("127.0.0.1", self.server.port), timeout=5)
        sock.sendall(b"GET /frame/1/points.bin HTTP/1.1\r\nHost: x\r\n\r\n")
        time.sleep(0.05)
        sock.close()
        slow.join()
        time.sleep(0.6)
        self.assertNotIn(1, self.source.decodes,
                         "decoded a frame whose browser had already gone")

    def test_out_of_range(self):
        status, _ = self._get("/frame/999/points.bin")
        self.assertEqual(404, status)

    def test_healthz_and_scene_json(self):
        self.assertEqual(200, self._get("/healthz")[0])
        self.assertEqual(200, self._get("/scene.json")[0])
        self.assertEqual(404, self._get("/nope")[0])

    def test_a_vanishing_client_does_not_take_the_server_down(self):
        sock = socket.create_connection(("127.0.0.1", self.server.port), timeout=5)
        sock.sendall(b"GET /scene.json HTTP/1.1\r\nHost: x\r\n\r\n")
        sock.close()
        time.sleep(0.2)
        self.assertEqual(200, self._get("/healthz")[0], "the server stopped serving")


class ClientGoneTest(unittest.TestCase):
    def test_detects_a_closed_peer(self):
        a, b = socket.socketpair()
        try:
            self.assertFalse(client_gone(a))
            b.close()
            self.assertTrue(client_gone(a))
        finally:
            a.close()

    def test_pending_bytes_are_not_a_closed_peer(self):
        a, b = socket.socketpair()
        try:
            b.sendall(b"GET / HTTP/1.1\r\n")
            self.assertFalse(client_gone(a))
        finally:
            a.close()
            b.close()


class GoneSentinelTest(unittest.TestCase):
    def test_gone_is_distinct_from_an_empty_answer(self):
        self.assertIsNot(GONE, b"")
        self.assertIsNot(GONE, None)


class LabelledSource(StubSource):
    """A stub whose frames come with masks, so classes and instances appear."""

    def __init__(self):
        super().__init__(decode_seconds=0.0)
        self.label_calls = 0

    def has_masks(self):
        return True

    def frame_labels(self, index, xyz):
        self.label_calls += 1
        classes = np.full(len(xyz), 255, np.uint8)
        classes[:2] = 58
        instances = np.zeros(len(xyz), np.uint16)
        instances[0] = 1
        instances[1] = 1002  # the rear camera's numbering
        return classes, instances


class LabelsTest(unittest.TestCase):
    def setUp(self):
        self.source = LabelledSource()
        self.server = SceneHttpServer(self.source, port=0, log=lambda _m: None)
        self.server.start()

    def tearDown(self):
        self.server.stop()

    def _frame(self, index):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=30)
        try:
            conn.request("GET", "/frame/%d/points.bin" % index)
            return decode_points(conn.getresponse().read())
        finally:
            conn.close()

    def test_a_frame_carries_classes_and_instances(self):
        _xyz, _i, _s, _o, classes, _rgb, instances = self._frame(2)
        np.testing.assert_array_equal(classes[:2], [58, 58])
        np.testing.assert_array_equal(instances[:2], [1, 1002])
        self.assertEqual(0, instances[2], "an unlabelled point has no instance")

    def test_both_come_out_of_one_mask_decode(self):
        """Decoding the masks twice would double the cost of every frame."""
        self._frame(5)
        self.assertEqual(1, self.source.label_calls)

    def test_a_cached_frame_keeps_its_instances(self):
        """The disk cache holds classes and instances in separate entries."""
        from kitscenes_viewer.scene.cache import FrameDiskCache
        with tempfile.TemporaryDirectory() as tmp:
            source = LabelledSource()
            server = SceneHttpServer(source, port=0, log=lambda _m: None,
                                     disk_cache=FrameDiskCache("stub", tmp))
            first = server.points_blob(3)
            server.cache = type(server.cache)()
            second = server.points_blob(3)
            self.assertEqual(first, second)
            self.assertEqual(1, source.label_calls, "the masks were decoded again")
            self.assertEqual(2, len(decode_points(second)[6][:2].tolist()))
            np.testing.assert_array_equal(decode_points(second)[6][:2], [1, 1002])


class AggregateServingTest(unittest.TestCase):
    """The whole-scene cloud: progress while building, then chunks."""

    def setUp(self):
        import importlib.util
        import tempfile
        if importlib.util.find_spec("torch") is None:
            self.skipTest("the aggregate is built with torch")
        from kitscenes_viewer.scene.aggregate import SceneAggregator
        from kitscenes_viewer.scene.scene import SceneSource
        from tests.kitscenes_scene import ORIGIN, write_scene
        from tests.test_aggregate import wall_and_car
        self.tmp = tempfile.TemporaryDirectory()
        self.source = SceneSource(write_scene(self.tmp.name, wall_and_car), anchor=ORIGIN)
        aggregator = SceneAggregator(self.source, disk_cache=None, voxel=0.3,
                                     max_range_m=200.0, device="cpu", log=lambda _m: None)
        self.server = SceneHttpServer(self.source, port=0, log=lambda _m: None,
                                      aggregator=aggregator)
        self.server.start()

    def tearDown(self):
        self.server.stop()
        self.tmp.cleanup()

    def _get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=30)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def _wait_ready(self):
        import json
        for _ in range(200):
            status, body = self._get("/aggregate/meta.json")
            meta = json.loads(body)
            if status == 200 and meta.get("ready"):
                return meta
            self.assertIsNone(meta.get("error"), meta.get("error"))
            time.sleep(0.05)
        self.fail("the aggregate never finished")

    def test_reports_progress_then_serves_chunks(self):
        status, _ = self._get("/aggregate/meta.json")
        self.assertEqual(202, status, "the first ask should start the build")
        meta = self._wait_ready()
        self.assertGreater(len(meta["chunks"]), 0)
        self.assertGreater(meta["stats"]["points"], 0)
        total = 0
        for chunk in meta["chunks"]:
            status, body = self._get("/aggregate/%d.bin" % chunk["index"])
            self.assertEqual(200, status)
            xyz, _i, _s, _o, _c, _rgb, instances = decode_points(body)
            self.assertEqual(chunk["count"], len(xyz))
            self.assertIsNone(instances,
                              "the aggregate must not claim instances: nothing "
                              "numbers an object across frames")
            total += len(xyz)
        self.assertEqual(meta["stats"]["points"], total)

    def test_a_chunk_is_encoded_once_and_reused(self):
        meta = self._wait_ready()
        first = self._get("/aggregate/0.bin")[1]
        second = self._get("/aggregate/0.bin")[1]
        self.assertEqual(first, second)
        self.assertGreater(len(meta["chunks"]), 0)

    def test_unknown_chunk(self):
        self._wait_ready()
        self.assertEqual(404, self._get("/aggregate/999.bin")[0])

    def test_without_an_aggregator(self):
        import json
        server = SceneHttpServer(self.source, port=0, log=lambda _m: None)
        server.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
            conn.request("GET", "/aggregate/meta.json")
            response = conn.getresponse()
            meta = json.loads(response.read())
            conn.close()
            self.assertEqual(202, response.status)
            self.assertIn("aggregator", meta["error"])
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
