"""Disk cache for what is slow to derive from a frame.

Reading and deskewing a frame's points takes 0.1-0.3 s, so the points are
read from the dataset each time. Their camera colours and seamseg labels take
up to a second more per frame and are cached here, one small file per frame
and kind (`rgb`, `class`, `inst`), next to the scene's whole-scene cloud.

Nothing is ever written into the dataset. The cache lives under
`$XDG_CACHE_HOME/kitscenes-viewer` (`~/.cache/kitscenes-viewer`), or where
`--cache-dir` says, and is kept under a size limit by deleting the least
recently used entries.
"""

import json
import os
import threading
from pathlib import Path

# Bumped when the cached payload's meaning changes, so old files are ignored
# rather than silently misread.
FORMAT_VERSION = 1
DEFAULT_LIMIT_GB = 10.0


def default_cache_dir():
    base = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "kitscenes-viewer"


class FrameDiskCache:
    """Per-frame entries of one scene, by frame index and kind."""

    @staticmethod
    def scene_dir(root, scene_id):
        """Where one scene's cache lives, without creating it."""
        return Path(root).expanduser() / ("%s-v%d" % (scene_id, FORMAT_VERSION))

    def __init__(self, scene_id, directory=None, limit_gb=DEFAULT_LIMIT_GB, log=print):
        self.enabled = directory is not None
        self.log = log
        self.limit_bytes = int(max(0.0, limit_gb) * 1e9)
        self.dir = None
        self.hits = 0
        self.misses = 0
        if not self.enabled:
            return
        self.root = Path(directory).expanduser()   # what the caller asked for
        self.dir = self.scene_dir(self.root, scene_id)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "about.json").write_text(json.dumps(
                {"scene": scene_id, "format": FORMAT_VERSION,
                 "note": "kitscenes-viewer cache: point colours, labels and the whole-scene cloud; "
                         "safe to delete"},
                indent=2))
        except OSError as exc:
            self.log("scene: cache off, %s is not writable (%s)" % (self.dir, exc))
            self.enabled = False
            self.dir = None

    def _path(self, index, kind):
        return self.dir / ("%s-%06d.bin" % (kind, index))

    def get(self, index, kind):
        if not self.enabled:
            return None
        path = self._path(index, kind)
        try:
            blob = path.read_bytes()
        except OSError:
            self.misses += 1
            return None
        # Touch it so the eviction order is "least recently used", not oldest.
        try:
            os.utime(path, None)
        except OSError:
            pass
        self.hits += 1
        return blob

    def put(self, index, blob, kind):
        if not self.enabled:
            return
        path = self._path(index, kind)
        # Own temp name per writer: two requests can write the same entry
        # at once.
        tmp = path.with_name("%s.%d.%d.part" % (path.name, os.getpid(), threading.get_ident()))
        try:
            with open(tmp, "wb") as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(path)  # atomic, so a killed server leaves no half entry
        except OSError as exc:
            self.log("scene: could not cache %s %d (%s)" % (kind, index, exc))
            try:
                tmp.unlink()
            except OSError:
                pass
            return
        self._evict()

    def _evict(self):
        """Drop the least recently used entries once over the size limit.

        Only in this cache's own scene directories (`<scene>-v<n>` holding an
        about.json), never anything else that happens to share the root.
        """
        if not self.limit_bytes:
            return
        files = []
        for scene_dir in self.dir.parent.glob("*-v%d" % FORMAT_VERSION):
            if not (scene_dir / "about.json").is_file():
                continue
            for p in scene_dir.glob("*-[0-9]*.bin"):
                try:
                    st = p.stat()
                except OSError:
                    continue  # another writer removed or replaced it meanwhile
                files.append((st.st_atime, st.st_size, p))
        total = sum(size for _, size, _ in files)
        if total <= self.limit_bytes:
            return
        for _, size, path in sorted(files):
            try:
                path.unlink()
            except OSError:
                continue
            total -= size
            if total <= self.limit_bytes:
                break
        self.log("scene: cache trimmed to %.1f GB" % (total / 1e9))

    def size_bytes(self):
        if not self.enabled:
            return 0
        try:
            return sum(p.stat().st_size for p in self.dir.glob("*-[0-9]*.bin"))
        except OSError:
            return 0

    def describe(self):
        if not self.enabled:
            return "cache off"
        return "cache %s (%.1f GB used, limit %.0f GB)" % (
            self.dir, self.size_bytes() / 1e9, self.limit_bytes / 1e9)
