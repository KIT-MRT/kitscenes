"""A prepared crop cache: :func:`prepare_aerials`.

For every scene of a dataset that has a map origin, the frame poses are
placed in EPSG:25832 and one ego-aligned crop per frame is cut from the raw
tiles (:class:`~kitscenes.aerial.AerialTiles`: the release tiles in
``<dataset root>/aerial`` or other tile folders, read only) with
:class:`~kitscenes.aerial.AerialCrops`, the function that cuts crops on the
fly, and written as a PNG into a writable cache folder. Scenes go one after
the other (with workers, one scene per task), frames in order, so the
decoded tiles are reused. Only what is missing is cut: a rerun over a
complete cache writes nothing.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

from kitscenes.aerial._crop_manifest import CropManifest
from kitscenes.aerial._poses import NO_SPLIT, ScenePoses, scene_poses
from kitscenes.aerial._progress import ProgressBar
from kitscenes.aerial._tile_io import DEFAULT_TILE_CACHE_MB
from kitscenes.aerial.cache import CROP_CACHE_DIR, METADATA_DIR, AerialCache, aerial_root
from kitscenes.aerial.crops import AerialCoverage, AerialCrops
from kitscenes.aerial.tiles import AerialTiles, PathLike
from kitscenes.errors import MapUnavailable
from kitscenes._layout import frame_stem

if TYPE_CHECKING:
    from kitscenes.catalog import Dataset

__all__ = ["NO_SPLIT", "ScenePoses", "prepare_aerials", "scene_poses"]

logger = logging.getLogger(__name__)


#: Default crop processes. Each holds up to ``tile_cache_mb`` of decoded tiles plus ~150 MiB
#: (interpreter, one crop window), ~400 MiB with the default cache.
DEFAULT_WORKERS = 8

#: Save the crop manifest every this many new crops, so a hard kill (e.g.
#: SIGKILL at a job time limit, which skips the final save) loses at most
#: this many records.
CROP_MANIFEST_SAVE_EVERY = 500


@dataclass
class _SceneTask:
    """The crops of one scene still to cut (picklable)."""

    scene_id: str
    area_id: Optional[str]
    frames: list[tuple[int, tuple[float, float], float, str, str]]  # index, ego_xy, yaw, out path, record path


_CROPS: Optional[AerialCrops] = None


def _init_worker(crops: AerialCrops) -> None:
    global _CROPS
    _CROPS = crops


def _cut(task: _SceneTask):
    """Cut the crops of *task* in frame order, yielding each frame index once its PNG is written."""
    from PIL import Image

    for index, ego_xy, yaw, out_path, _ in task.frames:
        crop = _CROPS.crop(ego_xy, yaw, area_id=task.area_id)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.name}.partial{os.getpid()}.png")
        Image.fromarray(crop).save(tmp)
        os.replace(tmp, out)
        yield index


def _cut_scene(task: _SceneTask) -> str:
    """A worker's task: all crops of one scene (module level, so it pickles for the pool)."""
    for _ in _cut(task):
        pass
    return task.scene_id


def prepare_aerials(
    dataset: "Dataset",
    roi: tuple[float, float],
    *,
    tiles: Union[AerialTiles, PathLike, Sequence[PathLike], None] = None,
    cache_dir: Optional[PathLike] = None,
    version: str = "v1",
    backend: str = "auto",
    verbose: bool = False,
    workers: Optional[int] = None,
    tile_cache_mb: float = DEFAULT_TILE_CACHE_MB,
) -> AerialCache:
    """Make sure every frame of *dataset* that the tiles cover has a crop in *cache_dir*; see
    :meth:`kitscenes.Dataset.prepare_aerials`.

    The returned cache has :attr:`AerialCache.coverage` set (also written to
    ``<cache_dir>/metadata/coverage_<cache_id>.json``).

    Raises:
        ValueError: If no scene of the dataset has a map origin and poses.
        FileNotFoundError: Without raw tiles.
    """
    roi_m = (float(roi[0]), float(roi[1]))
    # Resolved, so the relative manifest paths and the cache work from any working directory.
    cache_dir = Path(cache_dir).resolve() if cache_dir is not None else aerial_root(Path(dataset.root).resolve())

    # --- Step 0: frame poses in EPSG:25832, before anything is read or written
    placed: list[tuple] = []  # (scene, ScenePoses or None, reason)
    skipped: Counter = Counter()
    for scene in dataset:
        try:
            placed.append((scene, scene_poses(scene), None))
        except MapUnavailable as err:
            skipped[f"no map origin ({err.state.value})"] += 1
            placed.append((scene, None, f"no map origin ({err.state.value})"))
        except ValueError as err:
            skipped["no usable poses"] += 1
            logger.warning("Aerial: skipping %s", err)
            placed.append((scene, None, str(err)))
    if skipped:
        logger.info("Aerial: %d of %d scenes have no crops: %s", sum(skipped.values()), len(dataset), dict(skipped))
    if not any(poses is not None for _, poses, _ in placed):
        raise ValueError(
            f"No scene of {dataset!r} has a map origin and poses, so none can be placed on aerial imagery "
            f"({dict(skipped) or 'no scenes'}). The test splits ship no maps/origin.json."
        )

    # --- Step 1: the raw tiles (read only)
    tile_set = tiles if isinstance(tiles, AerialTiles) else AerialTiles(
        tiles if tiles is not None else dataset.root, cache_mb=tile_cache_mb)
    crops = AerialCrops(tile_set, roi_m, backend=backend, version=version)
    cache_id = crops.cache_id

    # --- Step 2: crops, merged into the crop manifest
    crop_manifest = CropManifest(cache_dir / METADATA_DIR / f"crop_manifest_{cache_id}.json", cache_id)
    coverage = {}
    tasks: list[_SceneTask] = []
    done_before = total = 0
    first_tile = {}
    for scene, poses, reason in placed:
        report = crops._coverage_of(scene, poses, reason)
        coverage[scene.id] = report
        if poses is None:
            continue
        split = scene.release_split or NO_SPLIT
        area_id = report.area_id
        covered = set(report.covered.tolist())
        frames = []
        for index, (e, n), yaw in zip(poses.indices.tolist(), poses.xy, poses.yaw):
            if index not in covered:
                continue
            total += 1
            if crop_manifest.is_complete(scene.id, index):
                done_before += 1
                continue
            rel = Path(CROP_CACHE_DIR, cache_id, split, area_id or "_none", scene.id, f"{frame_stem(index)}.png")
            frames.append((index, (float(e), float(n)), float(yaw), str(cache_dir / rel), rel.as_posix()))
        if frames:
            tasks.append(_SceneTask(scene.id, area_id, frames))
            first_tile[scene.id] = (tuple(poses.xy[0] // tile_set.tile_extent_m))
    report = AerialCoverage(cache_id, coverage)
    if report.num_uncovered or report.incomplete():
        incomplete = report.incomplete()
        logger.warning(
            "Aerial: %d frames of %d scenes need raw tiles that are not there and get no crop (%s)",
            report.num_uncovered, len(incomplete),
            ", ".join(f"{k} ({len(s.uncovered)})" for k, s in sorted(incomplete.items())[:10]),
        )
    # Neighbouring scenes one after the other, so the tile cache carries over between them.
    tasks.sort(key=lambda t: (t.area_id or "", first_tile[t.scene_id], t.scene_id))

    workers = workers if workers is not None else min(os.cpu_count() or 1, DEFAULT_WORKERS)
    workers = max(1, min(workers, len(tasks)))
    crops_since_save = 0

    def record(scene_id: str, index: int, record_path: str) -> None:
        nonlocal crops_since_save
        crop_manifest.record(scene_id, index, record_path)
        crops_since_save += 1
        if crops_since_save >= CROP_MANIFEST_SAVE_EVERY:
            crop_manifest.save()
            crops_since_save = 0

    with ProgressBar(total=total, initial=done_before, desc="aerial crops", disable=not verbose) as progress:
        try:
            if workers == 1:  # inline: tracebacks stay in this process
                _init_worker(crops)
                for task in tasks:
                    record_paths = {f[0]: f[4] for f in task.frames}
                    for index in _cut(task):
                        record(task.scene_id, index, record_paths[index])
                        progress.update(1)
            elif tasks:
                pending = iter(tasks)
                with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(crops,)) as pool:
                    in_flight = {}
                    for task in pending:
                        in_flight[pool.submit(_cut_scene, task)] = task
                        if len(in_flight) >= 2 * workers:
                            break
                    while in_flight:
                        finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                        for future in finished:
                            task = in_flight.pop(future)
                            future.result()  # a worker error propagates; the finally saves what is done
                            for index, _, _, _, record_path in task.frames:
                                record(task.scene_id, index, record_path)
                            progress.update(len(task.frames))
                            nxt = next(pending, None)
                            if nxt is not None:
                                in_flight[pool.submit(_cut_scene, nxt)] = nxt
        finally:
            if crops_since_save or not crop_manifest._path.exists():
                crop_manifest.save()
    report_path = cache_dir / METADATA_DIR / f"coverage_{cache_id}.json"
    text = report.to_json()
    if not report_path.is_file() or report_path.read_text() != text + "\n":
        report.to_json(report_path)

    cache = AerialCache(cache_dir, cache_id, crops.config)
    cache.coverage = report
    return cache
