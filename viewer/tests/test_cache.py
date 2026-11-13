"""The on-disk cache of per-frame colours and labels. It must never touch the dataset."""

import os
import tempfile
import time
import unittest
from pathlib import Path

from kitscenes_viewer.scene.cache import FORMAT_VERSION, FrameDiskCache, default_cache_dir


class DefaultDirTest(unittest.TestCase):
    def test_follows_xdg(self):
        old = os.environ.get("XDG_CACHE_HOME")
        try:
            os.environ["XDG_CACHE_HOME"] = "/somewhere/cache"
            self.assertEqual(Path("/somewhere/cache/kitscenes-viewer"),
                             default_cache_dir())
            os.environ.pop("XDG_CACHE_HOME")
            self.assertEqual(Path.home() / ".cache" / "kitscenes-viewer",
                             default_cache_dir())
        finally:
            if old is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = old


class FrameDiskCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = FrameDiskCache("scene-a", self.root, log=lambda _m: None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        self.cache.put(7, b"frame seven", "rgb")
        self.assertEqual(b"frame seven", self.cache.get(7, "rgb"))
        self.assertIsNone(self.cache.get(8, "rgb"))
        self.assertIsNone(self.cache.get(7, "class"))

    def test_lives_under_the_cache_root_not_the_dataset(self):
        self.cache.put(0, b"x", "rgb")
        written = list(self.root.rglob("*.bin"))
        self.assertEqual(1, len(written))
        self.assertTrue(str(written[0]).startswith(str(self.root)))
        self.assertIn("scene-a-v%d" % FORMAT_VERSION, str(written[0]))

    def test_a_new_format_ignores_old_files(self):
        self.cache.put(3, b"old", "rgb")
        other = FrameDiskCache("scene-a", self.root, log=lambda _m: None)
        other.dir = self.root / ("scene-a-v%d" % (FORMAT_VERSION + 1))
        other.dir.mkdir(parents=True, exist_ok=True)
        self.assertIsNone(other.get(3, "rgb"))

    def test_disabled_without_a_directory(self):
        off = FrameDiskCache("scene-a", None)
        off.put(1, b"ignored", "rgb")
        self.assertIsNone(off.get(1, "rgb"))
        self.assertFalse(off.enabled)
        self.assertEqual("cache off", off.describe())

    def test_unwritable_directory_turns_it_off_instead_of_failing(self):
        messages = []
        blocked = self.root / "blocked"
        blocked.mkdir()
        blocked.chmod(0o500)
        try:
            cache = FrameDiskCache("scene-b", blocked, log=messages.append)
            self.assertFalse(cache.enabled)
            self.assertTrue(any("not writable" in m for m in messages), messages)
            cache.put(1, b"x", "rgb")          # must not raise
            self.assertIsNone(cache.get(1, "rgb"))
        finally:
            blocked.chmod(0o700)

    def test_evicts_least_recently_used_over_the_limit(self):
        small = FrameDiskCache("scene-c", self.root, limit_gb=3000 / 1e9, log=lambda _m: None)
        for index in range(3):
            small.put(index, b"x" * 1000, "rgb")
            time.sleep(0.01)  # distinct atimes
        small.get(0, "rgb")   # frame 0 becomes the most recently used
        time.sleep(0.01)
        small.put(3, b"x" * 1000, "rgb")
        self.assertIsNotNone(small.get(0, "rgb"), "the recently used frame was evicted")
        self.assertIsNotNone(small.get(3, "rgb"), "the frame just written was evicted")
        kept = sum(1 for i in range(4) if small.get(i, "rgb") is not None)
        self.assertLessEqual(kept, 3, "nothing was evicted")

    def test_a_killed_write_leaves_no_half_entry(self):
        # put() writes to .part and renames, so only whole entries are visible.
        self.cache.put(5, b"abc", "rgb")
        self.assertEqual([], list(self.cache.dir.glob("*.part")))


if __name__ == "__main__":
    unittest.main()
