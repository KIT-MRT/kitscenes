"""The tile store: raw aerial tiles converted once into memory-mappable ``.npy`` files.

Decoding a 2048 px JPEG tile takes ~50 ms (a PNG ~300 ms), and a crop
touches up to four tiles; read from a tile store, a crop memory-maps only
the window it needs and takes a few milliseconds, in any order.
:func:`build_tile_store` converts the tiles of any layout
(:class:`~kitscenes.aerial.AerialTiles`: the release zips or extracted
JPEGs, PNG downloads) into a writable folder with the same layout::

    <store>/ATTRIBUTION.md                                     copied (licences of the imagery)
    <store>/metadata/area_index.json, provider_metadata.json   merged from the sources
    <store>/metadata/raw_tile_manifest_<raw_cache_id>.json     tile_format "npy", one entry per tile
    <store>/raw_tiles/<raw_cache_id>/<area>/tile_<i>_<j>.npy   (S, S, 3) uint8, 12 MiB per 2048 px tile

so ``AerialTiles(store)`` reads it like any tile folder (~10 GB for the
three areas of the release).
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from kitscenes.aerial._tile_io import npy_is_valid
from kitscenes.aerial.tiles import (
    AREA_INDEX, ATTRIBUTION, MANIFEST_PREFIX, METADATA_DIR, PROVIDER_METADATA, RAW_TILES_DIR, AerialTiles, PathLike,
    TileLocation,
)

logger = logging.getLogger(__name__)

#: Saved every this many converted tiles, so a stopped run keeps (nearly) all it did.
SAVE_EVERY = 64
#: Default conversion processes. Each holds one encoded and one decoded tile (~15 MiB for 2048 px)
#: plus the interpreter (~100 MiB); more mostly adds write-back pressure.
DEFAULT_JOBS = 4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _decoder() -> str:
    import PIL
    from PIL import features

    turbo = features.version("libjpeg_turbo") if features.check("libjpeg_turbo") else None
    return f"Pillow {PIL.__version__}" + (f", libjpeg-turbo {turbo}" if turbo else "")


def _signature(tiles: AerialTiles, loc: TileLocation) -> list:
    """What identifies a source tile cheaply: a zip member's size and CRC-32, else a file's size and mtime."""
    if loc.member is not None:
        info = tiles._reader.zip(loc.path).infos[loc.member]
        return [info.file_size, info.CRC]
    st = os.stat(loc.path)
    return [st.st_size, st.st_mtime_ns]


_SOURCE: Optional[AerialTiles] = None


def _init_worker(source: AerialTiles) -> None:
    global _SOURCE
    _SOURCE = source


def _convert(area_id: str, i: int, j: int, out: str, size_px: int) -> Optional[str]:
    """Decode one tile and write it as ``.npy`` (to a temporary name, then renamed); an error message instead of raising."""
    from kitscenes.aerial._tile_io import decode_image

    source = _SOURCE
    try:
        loc = source.location(area_id, i, j)
        data = source.tile_bytes(area_id, i, j)
        if loc.member is not None:
            info = source._reader.zip(loc.path).infos[loc.member]
            if zlib.crc32(data) != info.CRC:
                return f"{loc.describe()}: CRC mismatch (a damaged zip: download it again with --verify)"
        pixels = np.load(io.BytesIO(data)) if loc.format == "npy" else decode_image(data)
    except Exception as err:  # noqa: BLE001 - reported per tile
        return f"{area_id} tile ({i}, {j}): {err}"
    if pixels.shape != (size_px, size_px, 3) or pixels.dtype != np.uint8:
        return f"{loc.describe()}: {pixels.shape} {pixels.dtype}, expected ({size_px}, {size_px}, 3) uint8"
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.partial{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.save(fh, np.ascontiguousarray(pixels))
        fh.flush()
        # Written through and dropped from the page cache: ~10 GB of dirty pages at once would put the
        # machine (or the cgroup of the job) under memory pressure. Crops map the pages again when needed.
        os.fsync(fh.fileno())
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    os.replace(tmp, path)
    return None


def build_tile_store(
    out_dir: PathLike,
    tiles: Union[AerialTiles, PathLike, Sequence[PathLike]],
    *,
    areas: Optional[Iterable[str]] = None,
    jobs: Optional[int] = None,
    verbose: bool = False,
) -> AerialTiles:
    """Convert raw aerial tiles into a tile store in *out_dir* (once; idempotent, resumable, parallel).

    Every tile of *tiles* (of *areas*) is decoded and written as an
    uncompressed ``.npy``; the metadata of the sources is merged into
    ``out_dir/metadata`` and their ``ATTRIBUTION.md`` copied. A rerun
    converts only what is missing or whose source changed (a zip member's
    CRC, a file's size and mtime); a run that stops keeps every tile it
    finished (written under a temporary name, then renamed). The sources
    are only read. At the end every tile listed by the sources is checked
    against the store (shape, dtype, file size).

    Args:
        out_dir: The writable store folder (not inside a source).
        tiles: The source: an :class:`~kitscenes.aerial.AerialTiles`, a
            dataset root, an aerial folder, or a list of them.
        areas: Only these areas (default: all).
        jobs: Conversion processes (default :data:`DEFAULT_JOBS`, 4; each needs
            ~130 MiB: one tile decoded at a time, no tile cache).
        verbose: A progress bar.

    Returns:
        The store as :class:`~kitscenes.aerial.AerialTiles`.

    Raises:
        ValueError: If tiles could not be converted (unreadable, damaged,
            wrong size; the others are kept and recorded), or *out_dir* is a source.
    """
    from kitscenes.aerial._progress import ProgressBar

    # No decoded-tile cache: every tile is read once (the workers get a copy without one too).
    source = AerialTiles(tiles.roots if isinstance(tiles, AerialTiles) else tiles,
                         raw_cache_id=tiles.raw_cache_id if isinstance(tiles, AerialTiles) else None, cache_mb=0)
    out = Path(out_dir).resolve()
    if out in source.roots:
        raise ValueError(f"The tile store {out} cannot be one of its sources")
    raw_cache_id = source.raw_cache_id
    size_px = source.tile_size_px
    shape = (size_px, size_px, 3)
    wanted = set(source.area_ids if areas is None else areas)
    unknown = sorted(wanted - set(source.area_ids))
    if unknown:
        raise ValueError(f"Unknown area(s) {unknown}; the tiles have {source.area_ids}")
    metadata = out / METADATA_DIR
    manifest_path = metadata / f"{MANIFEST_PREFIX}{raw_cache_id}.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError:
        manifest = None
    if manifest is not None and (manifest.get("effective_tile_size_px") != size_px
                                 or abs(float(manifest["resolution_m_per_px"]) - source.resolution_m) > 1e-9):
        raise ValueError(f"{out} holds a tile store of other tiles ({manifest_path.name}); use another folder")
    if manifest is None:
        manifest = {
            "raw_cache_id": raw_cache_id, "provider": source.provider, "resolution_m_per_px": source.resolution_m,
            "tile_size_px": source.requested_tile_size_px, "effective_tile_size_px": size_px,
            "tile_extent_m": source.tile_extent_m, "created_at": _now(), "areas": {},
        }
    manifest["tile_format"] = {
        "format": "npy", "dtype": "uint8", "shape": list(shape), "paths": "relative to the aerial root",
        "decoder": _decoder(), "sources": [str(r) for r in source.roots],
        "source_format": source.tile_format or {"format": "/".join(sorted(source.formats))},
    }
    areas_out = manifest["areas"]
    done: dict[str, dict[tuple[int, int], dict]] = {
        a: {(t["i"], t["j"]): t for t in e["tiles"]} for a, e in areas_out.items()}

    def save() -> None:
        for area_id, tiles_done in done.items():
            areas_out[area_id]["tiles"] = [tiles_done[k] for k in sorted(tiles_done)]
        manifest["updated_at"] = _now()
        metadata.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path.with_name(manifest_path.name + ".partial")
        tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        os.replace(tmp, manifest_path)

    todo = []
    kept = 0
    for area_id, (i, j), loc in source.iter_tiles(sorted(wanted)):
        grid = source.grid(area_id)
        entry = areas_out.setdefault(area_id, {"origin_xy": [grid.origin_e, grid.origin_n], "crs": "EPSG:25832",
                                               "tiles": []})
        if abs(entry["origin_xy"][0] - grid.origin_e) > 1e-6 or abs(entry["origin_xy"][1] - grid.origin_n) > 1e-6:
            raise ValueError(f"{out}: area {area_id} is on another tile grid than in the sources; use another folder")
        done.setdefault(area_id, {})
        rel = f"{RAW_TILES_DIR}/{raw_cache_id}/{area_id}/tile_{i}_{j}.npy"
        signature = _signature(source, loc)
        have = done[area_id].get((i, j))
        if have is not None and have.get("source") == signature and npy_is_valid(out / rel, shape):
            kept += 1
            continue
        done[area_id].pop((i, j), None)
        todo.append((area_id, i, j, rel, signature))

    # Temporary files of a run that was killed.
    for leftover in (out / RAW_TILES_DIR / raw_cache_id).glob("*/tile_*.npy.partial*"):
        leftover.unlink(missing_ok=True)
    # Metadata first: the store is a valid (if partial) tile folder from the start.
    metadata.mkdir(parents=True, exist_ok=True)
    index = {"strategy": "epsg25832_bucket", "bucket_size_m": source.bucket_size_m,
             "scene_to_area": {s: a for s, a in sorted(source.scene_to_area.items()) if a in wanted},
             "areas": {a: e for a, e in sorted(source.area_info.items()) if a in wanted}}
    (metadata / AREA_INDEX).write_text(json.dumps(index, indent=1, sort_keys=True))
    (metadata / PROVIDER_METADATA).write_text(json.dumps(source.provider_metadata, indent=1, sort_keys=True))
    attribution = source.attribution
    if attribution:
        shutil.copyfile(attribution[0], out / ATTRIBUTION)
    save()

    errors: list[str] = []
    converted = 0
    started = time.perf_counter()
    workers = max(1, min(jobs or min(os.cpu_count() or 1, DEFAULT_JOBS), len(todo) or 1))
    with ProgressBar(total=len(todo), desc="tile store", disable=not verbose) as progress:
        pool = None
        try:
            args = [(a, i, j, str(out / rel), size_px) for a, i, j, rel, _ in todo]
            if workers == 1:
                _init_worker(source)
                results = (_convert(*a) for a in args)
            else:
                pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(source,))
                results = pool.map(_convert, *zip(*args), chunksize=1) if args else iter(())
            for (area_id, i, j, rel, signature), error in zip(todo, results):
                if error:
                    errors.append(error)
                else:
                    done[area_id][(i, j)] = {"i": i, "j": j, "path": rel, "source": signature}
                    converted += 1
                    if converted % SAVE_EVERY == 0:
                        save()
                progress.update(1)
        finally:
            if pool is not None:
                pool.shutdown(cancel_futures=True)
            save()
    # Check the store against what the sources list.
    for area_id, (i, j), _ in source.iter_tiles(sorted(wanted)):
        rel = f"{RAW_TILES_DIR}/{raw_cache_id}/{area_id}/tile_{i}_{j}.npy"
        if (i, j) in done.get(area_id, {}) and not npy_is_valid(out / rel, shape):
            del done[area_id][(i, j)]
            errors.append(f"{rel}: missing or incomplete after conversion")
    if errors:
        save()
        raise ValueError(f"{len(errors)} tile(s) could not be converted (the others are in {out}): "
                         + "; ".join(errors[:5]) + (" ..." if len(errors) > 5 else ""))
    total = sum(len(t) for t in done.values())
    logger.info("Tile store %s: %d tiles (%d converted in %.0f s, %d kept), %.1f GB", out, total, converted,
                time.perf_counter() - started, kept, total * np.prod(shape) / 1e9)
    return AerialTiles(out, raw_cache_id=raw_cache_id)

