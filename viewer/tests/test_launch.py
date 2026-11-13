"""Starting the viewer: the command line, waiting for JOSM's map, picking the city."""

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import kitscenes_viewer.__main__ as launcher
from kitscenes_viewer.scene.__main__ import open_city_index, wait_for_viewer_anchor
from tests.test_city import FRANKFURT, KARLSRUHE, write_scene


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class WaitForMapTest(unittest.TestCase):
    def run_wait(self, answers, seconds):
        clock = FakeClock()
        calls = []

        def probe(host, port):
            calls.append(clock.t)
            return answers(clock.t)
        logs = []
        got = wait_for_viewer_anchor("h", 1, seconds, logs.append, probe=probe, sleep=clock.sleep, clock=clock)
        return got, calls, logs

    def test_a_map_already_there_is_taken_at_once(self):
        got, calls, logs = self.run_wait(lambda t: (49.0, 8.4), 60)
        self.assertEqual((49.0, 8.4), got)
        self.assertEqual(1, len(calls))
        self.assertEqual([], logs)

    def test_waits_for_josms_first_snapshot(self):
        got, calls, logs = self.run_wait(lambda t: (49.0, 8.4) if t >= 3 else None, 60)
        self.assertEqual((49.0, 8.4), got)
        self.assertLessEqual(calls[-1], 3.5)
        self.assertIn("waiting", logs[0])

    def test_gives_up_and_says_so(self):
        got, calls, logs = self.run_wait(lambda t: None, 5)
        self.assertIsNone(got)
        self.assertLessEqual(calls[-1], 5.5)
        self.assertIn("going on without it", logs[-1])

    def test_no_wait_unless_asked(self):
        got, calls, _ = self.run_wait(lambda t: None, 0)
        self.assertIsNone(got)
        self.assertEqual(1, len(calls))


class CityFromMapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for i in range(3):
            write_scene(self.root, "val", "ffff-%d" % i, FRANKFURT, start=(10 * i, 0))
        write_scene(self.root, "val", "kkkk-1", KARLSRUHE)

    def tearDown(self):
        self.tmp.cleanup()

    def index(self, anchor, city_name=None):
        args = argparse.Namespace(city=str(self.root), city_name=city_name, rescan=False)
        logs = []
        return open_city_index(args, anchor, None, logs.append), logs

    def test_without_a_map_the_city_with_most_scenes(self):
        self.assertEqual("frankfurt", self.index(None)[0].city)

    def test_josms_map_picks_the_city(self):
        self.assertEqual("karlsruhe", self.index((49.0143900, 8.4172200))[0].city)

    def test_a_map_elsewhere_falls_back_and_says_so(self):
        index, logs = self.index((48.7133, 9.0028))   # Sindelfingen: no scenes here
        self.assertEqual("frankfurt", index.city)
        self.assertIn("sindelfingen", logs[0])

    def test_a_named_city_wins_and_does_not_fall_back(self):
        self.assertEqual("frankfurt", self.index((49.01, 8.41), city_name="frankfurt")[0].city)
        with self.assertRaises(ValueError):
            self.index(None, city_name="sindelfingen")


class CommandLineTest(unittest.TestCase):
    """A wrong command line is an error before a server or a browser tab starts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scene = write_scene(self.root, "val", "ffff-0", FRANKFURT)

    def tearDown(self):
        self.tmp.cleanup()

    def refused(self, *argv):
        err = io.StringIO()
        with mock.patch.object(launcher, "start_viewer") as start, \
                mock.patch.object(launcher, "viewer_running", return_value=False) as running, \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as exit_:
            launcher.main([str(a) for a in argv])
        self.assertEqual(2, exit_.exception.code)
        start.assert_not_called()
        running.assert_not_called()
        return err.getvalue()

    def test_an_unknown_option(self):
        self.assertIn("--prewarm", self.refused(self.root, "--prewarm"))

    def test_a_missing_folder_or_map(self):
        self.assertIn("no such folder", self.refused(self.root / "nope"))
        self.assertIn("no such map file", self.refused(self.root, "--map", self.root / "nope.osm"))

    def test_city_options_on_one_scene(self):
        self.assertIn("--build-aggregates", self.refused(self.scene, "--build-aggregates"))
        self.assertIn("--city-name", self.refused(self.scene, "--city-name", "frankfurt"))

    def test_building_needs_the_cache(self):
        self.assertIn("--no-cache", self.refused(self.root, "--build-aggregates", "--no-cache"))

    def test_the_scene_server_gets_the_options(self):
        args = launcher.build_parser().parse_args(
            [str(self.root), "--port", "9000", "--data-port", "9005", "--city-name", "frankfurt", "--no-cache"])
        self.assertEqual(
            ["--city", str(self.root), "--viewer-host", "127.0.0.1", "--viewer-port", "9000", "--ingest-port", "9001",
             "--http-port", "9005", "--aggregate-attributes", "auto", "--city-name", "frankfurt", "--no-cache"],
            launcher.scene_arguments(args))
        args = launcher.build_parser().parse_args([str(self.scene)])
        self.assertEqual(["--scene", str(self.scene)], launcher.scene_arguments(args)[:2])


if __name__ == "__main__":
    unittest.main()
