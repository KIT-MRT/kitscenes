"""crop_manifest_<cache_id>.json I/O (idempotent merge)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CropManifest:
    """In-memory view of a crop_manifest_<cache_id>.json file.

    Loaded from disk on construction (if present) and written back via
    save(). Tracks, per (scene_id, frame_idx), whether an aerial crop has
    already been produced and where it lives, so repeated prepare_aerials()
    calls can skip work idempotently instead of re-fetching/re-writing crops.
    """

    def __init__(self, path: Path, cache_id: str) -> None:
        """Load path if it exists, else start a fresh manifest for cache_id."""
        self._path = Path(path)
        self.cache_id = cache_id
        self.created_at = _now_iso()
        self.frames: dict[str, dict[str, dict]] = {}
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self._path.read_text())
        self.cache_id = data["cache_id"]
        self.created_at = data["created_at"]
        self.frames = data["frames"]

    def save(self) -> None:
        """Write the manifest to self._path, refreshing updated_at."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cache_id": self.cache_id,
            "created_at": self.created_at,
            "updated_at": _now_iso(),
            "frames": self.frames,
        }
        self._path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def is_complete(self, scene_id: str, frame_idx: int) -> bool:
        """True if a crop has already been recorded for this frame."""
        return str(frame_idx) in self.frames.get(scene_id, {})

    def record(self, scene_id: str, frame_idx: int, path: Path) -> None:
        """Record that a crop for this frame was written to path."""
        self.frames.setdefault(scene_id, {})[str(frame_idx)] = {"path": str(path)}
