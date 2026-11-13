"""The local cache of parsed SD maps: one pickle per OSM file, in the user's cache folder.

The cache holds the parsed arrays (:class:`~kitscenes.sdmap._parse.OSMData`,
WGS84, no frame), keyed by the file's name, size and modification time and
the parser version. It is built on first use and never shipped: it is a copy
of OpenStreetMap data (ODbL), fine to keep locally, see the README.
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
import threading
from pathlib import Path
from typing import Optional

from kitscenes.sdmap._parse import PARSER_VERSION, OSMData, parse

logger = logging.getLogger(__name__)

#: Environment variable with the cache folder of kitscenes (the SD maps go to its ``sdmap/`` subfolder).
CACHE_ENV_VAR = "KITSCENES_CACHE"

_lock = threading.Lock()
_loaded: dict[tuple, OSMData] = {}  # per process: one parse per file


def default_cache_dir() -> Path:
    """``$KITSCENES_CACHE/sdmap``, else ``$XDG_CACHE_HOME/kitscenes/sdmap``, else ``~/.cache/kitscenes/sdmap``."""
    if os.environ.get(CACHE_ENV_VAR):
        return Path(os.environ[CACHE_ENV_VAR]) / "sdmap"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "kitscenes" / "sdmap"


def file_key(path: Path) -> tuple:
    st = path.stat()
    return (path.name, st.st_size, st.st_mtime_ns, PARSER_VERSION)


def cache_file(path: Path, cache_dir: Path) -> Path:
    name, size, mtime, version = file_key(path)
    return cache_dir / f"{Path(name).stem}-{size}-{mtime}-p{version}.pkl"


def load_osm(path: Path | str, *, cache_dir: Optional[Path | str] = None) -> OSMData:
    """The parsed arrays of *path*: from this process, from the cache folder, or parsed (and cached).

    Args:
        cache_dir: The cache folder (default :func:`default_cache_dir`).
    """
    path = Path(path)
    key = (str(path.resolve()), *file_key(path))
    with _lock:
        data = _loaded.get(key)
    if data is not None:
        return data
    folder = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    target = cache_file(path, folder)
    if target.is_file():
        try:
            with open(target, "rb") as fh:
                data = pickle.load(fh)
        except Exception as err:  # a broken or foreign cache file: parse again
            logger.warning("Ignoring unreadable SD-map cache %s (%s)", target, err)
            data = None
    if data is None:
        logger.info("Parsing %s (once; cached in %s)", path, folder)
        data = parse(path)
        _write(target, data)
    with _lock:
        _loaded[key] = data
    return data


def _write(target: Path, data: OSMData) -> None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=target.parent)
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, target)
    except OSError as err:
        logger.warning("Could not write the SD-map cache %s (%s); parsing again next time", target, err)


def clear_memory() -> None:
    """Forget the parsed files of this process (the cache folder is kept)."""
    with _lock:
        _loaded.clear()
