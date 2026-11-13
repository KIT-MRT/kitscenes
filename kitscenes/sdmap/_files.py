"""The SD-map files of a release root (``openstreetmap_sd_maps/``) and the snapshot names that select them.

Layout:

- ``<City>_<YYYY>-W<ww>.osm.pbf``: one snapshot per city and ISO recording week, taken at the Monday
  00:00:00Z of that week (the scene table names each scene's file,
  :attr:`kitscenes.splits.SceneInfo.sd_snapshot`). Snapshot name: the week, ``"2026-W06"``.
- ``<name>/<City>.osm`` (or ``.osm.pbf``): a named set, e.g. ``paper_2026-05-03/`` with the
  Overpass snapshots of 2026-05-03 that the paper used. Snapshot name: the folder name.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

#: The snapshot name of the 2026-05-03 Overpass files (``paper_2026-05-03/<City>.osm``): the SD maps of the paper
#: and of SDTagNet trainings made with them.
PAPER_SNAPSHOT = "paper_2026-05-03"
#: Snapshot names that are not a file: the latest weekly snapshot of the city, the scene's own one.
LATEST = "latest"
SCENE = "scene"

_WEEK = re.compile(r"^\d{4}-W\d{2}$")
_WEEKLY_FILE = re.compile(r"^(?P<city>[A-Za-z][A-Za-z-]*)_(?P<week>\d{4}-W\d{2})\.osm(?:\.pbf)?$")
_CITY_FILE = re.compile(r"^(?P<city>[A-Za-z][A-Za-z-]*)\.osm(?:\.pbf)?$")


def is_week(name: str) -> bool:
    return bool(_WEEK.match(name))


def sd_map_files(folder: Path) -> dict[str, dict[str, Path]]:
    """``{city: {snapshot name: path}}`` of an ``openstreetmap_sd_maps/`` folder (empty if it does not exist).

    Per city the weeks come first, oldest to latest, then the named sets by name.
    """
    found: dict[str, dict[str, Path]] = {}
    if not folder.is_dir():
        return {}
    for entry in sorted(folder.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            for f in sorted(entry.iterdir()):
                m = _CITY_FILE.match(f.name)
                if m and f.is_file():
                    found.setdefault(m["city"], {}).setdefault(entry.name, f)
            continue
        m = _WEEKLY_FILE.match(entry.name)
        if m:
            found.setdefault(m["city"], {})[m["week"]] = entry
    return {city: dict(sorted(snaps.items(), key=lambda kv: (not is_week(kv[0]), kv[0])))
            for city, snaps in sorted(found.items())}


def snapshot_name(snapshot: str, city: Optional[str] = None) -> str:
    """A snapshot as :func:`sd_map_files` names it, from what a user or the scene table gives.

    ``"2026-W06"``, ``"Karlsruhe_2026-W06"`` and ``"Karlsruhe_2026-W06.osm.pbf"`` give
    ``"2026-W06"``; ``"paper"`` gives :data:`PAPER_SNAPSHOT`; other names (``"latest"``, a folder
    name) stay as they are.

    Raises:
        ValueError: For a file name of another city than *city*.
    """
    name = str(snapshot).strip()
    if name.lower() == "paper":
        return PAPER_SNAPSHOT
    m = _WEEKLY_FILE.match(name) or _WEEKLY_FILE.match(name + ".osm")
    if m:
        if city is not None and m["city"].lower() != city.lower():
            raise ValueError(f"SD snapshot {snapshot!r} is not of {city}")
        return m["week"]
    if is_week(name.upper()):
        return name.upper()
    return name


def pick_snapshot(snapshots: dict[str, Path], snapshot: Optional[str], city: str) -> Optional[tuple[str, Path]]:
    """``(name, path)`` of *snapshot* among a city's *snapshots*, None if it is not there.

    None or ``"latest"`` is the latest week.
    """
    if snapshot is None or snapshot == LATEST:
        weeks = [n for n in snapshots if is_week(n)]
        return (weeks[-1], snapshots[weeks[-1]]) if weeks else None
    name = snapshot_name(snapshot, city)
    return (name, snapshots[name]) if name in snapshots else None
