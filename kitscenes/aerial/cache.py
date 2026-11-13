"""Reading a prepared crop cache: :class:`AerialCache`.

A crop cache lives in a writable folder (``cache_dir``, by default
``<dataset root>/aerial/``), next to or apart from the raw tiles::

    metadata/crop_manifest_<cache_id>.json         one entry per cropped frame
    crop_cache/<cache_id>/<split>/<area_id>/<scene_id>/<frame>.png

Crop paths in the manifest are relative to the folder, so it can be moved
or shared.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np

from kitscenes.aerial._frame_crops import FrameCrops
from kitscenes.aerial.config import AerialCacheConfig
from kitscenes.errors import AerialUnavailable, Availability

AERIAL_DIR = "aerial"
METADATA_DIR = "metadata"
CROP_CACHE_DIR = "crop_cache"
RAW_TILES_DIR = "raw_tiles"
_MANIFEST_PREFIX = "crop_manifest_"

NOT_PREPARED_MESSAGE = (
    "no aerial crops attached. Cut them on the fly from the raw tiles with ds.use_aerial(roi=(60, 30)), "
    "or attach a prepared crop cache with ds.use_aerial(cache_id, cache_dir=...)."
)


def aerial_root(dataset_root: Path | str) -> Path:
    """``<dataset root>/aerial``."""
    return Path(dataset_root) / AERIAL_DIR


def crop_manifest_path(cache_dir: Path | str, cache_id: str) -> Path:
    return Path(cache_dir) / METADATA_DIR / f"{_MANIFEST_PREFIX}{cache_id}.json"


def cache_ids(cache_dir: Path | str) -> list[str]:
    """IDs of the crop caches in *cache_dir* (e.g. ``<dataset root>/aerial``), sorted (empty if there is none)."""
    folder = Path(cache_dir) / METADATA_DIR
    if not folder.is_dir():
        return []
    return sorted(p.stem[len(_MANIFEST_PREFIX):] for p in folder.glob(f"{_MANIFEST_PREFIX}*.json"))


def load_crop(path: Path | str) -> np.ndarray:
    """A crop image as an (H, W, 3) uint8 RGB array."""
    from PIL import Image

    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


class AerialCache(FrameCrops):
    """One prepared crop cache, read-only.

    Made by :meth:`kitscenes.Dataset.prepare_aerials` or
    :meth:`kitscenes.Dataset.use_aerial`. ``cache[frame]`` reads the crop of
    a frame (:meth:`kitscenes.Frame.aerial` is the shortcut for the attached
    cache). The manifest is read on first use, once
    per process. Pickles to its folder, ID and config. For crops cut on the
    fly instead (no crop cache), see :class:`kitscenes.aerial.AerialCrops`.

    Args:
        cache_dir: The folder of the cache (holds ``metadata/`` and
            ``crop_cache/``; ``<dataset root>/aerial`` unless prepared elsewhere).
        cache_id: e.g. ``"dop20_res020_roi60x30_ego_v1"``.
        config: The parameters, if known (set when prepared in this process).

    Attributes:
        coverage: Set by ``prepare_aerials`` (an
            :class:`~kitscenes.aerial.AerialCoverage`: the frames without a
            crop and why), else None; also in ``metadata/coverage_<cache_id>.json``.
    """

    def __init__(self, cache_dir: Path | str, cache_id: str, config: Optional[AerialCacheConfig] = None) -> None:
        self._root = Path(cache_dir)
        self.cache_id = cache_id
        self.config = config
        self.coverage = None
        self._frames: Optional[dict] = None
        self._lock = threading.Lock()

    @classmethod
    def open(cls, cache_dir: Path | str, cache_id: Optional[str] = None) -> "AerialCache":
        """An existing cache in *cache_dir*; *cache_id* may be left out if the folder has exactly one.

        Raises:
            AerialUnavailable: ``NOT_PREPARED`` if there is no such cache.
            ValueError: If *cache_id* is None and the root has several.
        """
        available = cache_ids(cache_dir)
        if cache_id is None:
            if len(available) > 1:
                raise ValueError(f"{cache_dir} has several aerial caches, pick one: {available}")
            if not available:
                raise AerialUnavailable("aerial", None, Availability.NOT_PREPARED,
                                        f"no aerial cache in {cache_dir}. {NOT_PREPARED_MESSAGE}")
            cache_id = available[0]
        elif cache_id not in available:
            raise AerialUnavailable(
                "aerial", None, Availability.NOT_PREPARED,
                f"no aerial cache {cache_id!r} in {cache_dir} (found: {available or 'none'})",
            )
        return cls(cache_dir, cache_id)

    def __repr__(self) -> str:
        return f"AerialCache({str(self.root)!r}, {self.cache_id!r})"

    def __getstate__(self) -> dict:
        return {"root": self._root, "cache_id": self.cache_id, "config": self.config}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["root"], state["cache_id"], state["config"])

    @property
    def root(self) -> Path:
        """The folder of the cache (``<dataset root>/aerial`` unless prepared with ``cache_dir=``)."""
        return self._root

    @property
    def manifest_path(self) -> Path:
        return crop_manifest_path(self._root, self.cache_id)

    def _manifest_frames(self) -> dict:
        frames = self._frames
        if frames is None:
            from kitscenes.aerial._crop_manifest import CropManifest

            with self._lock:
                if self._frames is None:
                    if not self.manifest_path.is_file():
                        raise AerialUnavailable("aerial", None, Availability.NOT_PREPARED,
                                                f"no crop manifest at {self.manifest_path}")
                    self._frames = CropManifest(self.manifest_path, self.cache_id).frames
                frames = self._frames
        return frames

    @property
    def scene_ids(self) -> list[str]:
        """Scenes with at least one crop, sorted."""
        return sorted(s for s, frames in self._manifest_frames().items() if frames)

    def frame_indices(self, scene_id: str) -> np.ndarray:
        """Sorted frame indices of *scene_id* that have a crop (empty if none)."""
        return np.array(sorted(int(i) for i in self._manifest_frames().get(scene_id, {})), dtype=np.int64)

    def has_crop(self, scene_id: str, index: int) -> bool:
        return str(int(index)) in self._manifest_frames().get(scene_id, {})

    def crop_path(self, scene_id: str, index: int) -> Path:
        """File of the crop of frame *index* of *scene_id*.

        Raises:
            AerialUnavailable: ``DROPPED`` if the cache has none.
        """
        entry = self._manifest_frames().get(scene_id, {}).get(str(int(index)))
        if entry is None:
            raise AerialUnavailable(
                "aerial", int(index), Availability.DROPPED, f"no crop for scene {scene_id} in cache {self.cache_id!r}"
            )
        return self._resolve(entry["path"])

    def crop(self, scene_id: str, index: int) -> np.ndarray:
        """The crop as an (H, W, 3) uint8 RGB array; see :meth:`kitscenes.Frame.aerial`."""
        return load_crop(self.crop_path(scene_id, index))

    # What FrameCrops needs (cache[frame], Frame.aerial).
    def _has_crop_for(self, scene, index: int) -> bool:
        return self.has_crop(scene.id, index)

    def _crop_for(self, scene, index: int) -> np.ndarray:
        return self.crop(scene.id, index)

    def _resolve(self, stored: str) -> Path:
        path = Path(stored)
        if path.is_absolute():
            raise ValueError(f"{self.manifest_path}: crop path {stored!r} is not relative to the cache folder")
        return self.root / path
