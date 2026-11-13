"""The dataset splits: release folders, split definitions and the conventions of each release folder.

Scenes live in ``<root>/data/<release folder>/<scene_id>/``. The release
folders are the splits of the default split definition (:data:`DEFAULT_SPLIT`)
and are used unchanged by the download, the packer and
:class:`~kitscenes.catalog.Dataset`. A :class:`SplitDefinition` gives each
scene another split without moving files (``kitscenes.open(root,
split_definition=...)``); the conventions of a scene's data (maps, labels,
pose frame: :func:`split_info`) stay those of its release folder.

Every release folder exists twice: the normal one and a ``map_change_`` variant that
holds scenes driven where the map changed since it was made.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Literal, Mapping, Optional, Sequence, Union

__all__ = [
    "SPLITS", "RELEASE", "SplitInfo", "SceneInfo", "PoseFrame",
    "split_info", "known_split", "canonical_split_name", "scene_table", "scene_info",
    "find_scenes", "withheld_columns", "FILTER_COLUMNS",
    "DEFAULT_SPLIT", "SplitDefinition", "default_split",
]

PoseFrame = Literal["local_utm", "first_pose"]


@dataclass(frozen=True)
class SplitInfo:
    """Conventions of one split.

    Attributes:
        name: Folder name on disk, e.g. ``"test-e2e"`` or ``"map_change_train"``.
        base: The split without the ``map_change_`` prefix.
        family: ``"default"`` or ``"map_change"``.
        has_maps: Whether its scenes ship ``maps/`` (HD map and ``origin.json``).
        has_labels: Whether its scenes ship ``processed/`` (ground segmentation,
            instance masks).
        has_gnss: Whether its scenes ship ``gnss/`` and ``gnss_ins/`` (GNSS/INS
            messages, one Parquet file per topic, :mod:`kitscenes.messages`).
        pose_frame: Frame of ``poses.txt``: ``"local_utm"`` (local UTM around
            ``maps/origin.json``, the map's frame) or ``"first_pose"`` (relative
            to the first pose; its translation is zero).
        aliases: Other accepted spellings.
    """

    name: str
    base: str
    family: Literal["default", "map_change"]
    has_maps: bool
    has_labels: bool
    has_gnss: bool
    pose_frame: PoseFrame
    aliases: tuple[str, ...] = ()


_MAP_CHANGE_PREFIX = "map_change_"

# (base name, has_maps and labels, has_gnss, pose_frame, aliases), in the order
# scenes are listed. The release packer enforces these rules: this table is
# their one source.
_BASE_SPLITS: tuple[tuple[str, bool, bool, PoseFrame, tuple[str, ...]], ...] = (
    ("train", True, True, "local_utm", ()),
    ("val", True, True, "local_utm", ("validation",)),
    ("test", False, True, "local_utm", ()),
    ("test-e2e", False, False, "first_pose", ("test_e2e",)),
    ("overlap_train_val", True, True, "local_utm", ()),
)


def _build_table() -> dict[str, SplitInfo]:
    table: dict[str, SplitInfo] = {}
    for prefix, family in (("", "default"), (_MAP_CHANGE_PREFIX, "map_change")):
        for base, has_maps, has_gnss, pose_frame, aliases in _BASE_SPLITS:
            table[prefix + base] = SplitInfo(
                name=prefix + base,
                base=base,
                family=family,
                has_maps=has_maps,
                has_labels=has_maps,
                has_gnss=has_gnss,
                pose_frame=pose_frame,
                aliases=tuple(prefix + alias for alias in aliases),
            )
    return table


#: All splits by on-disk name.
SPLITS: dict[str, SplitInfo] = _build_table()

_BY_ALIAS: dict[str, str] = {
    alias: info.name for info in SPLITS.values() for alias in (info.name, *info.aliases)
}


def split_info(name: str) -> SplitInfo:
    """Look up a split by its folder name or an alias (``"test_e2e"``, ``"validation"``).

    Raises:
        ValueError: For an unknown name.
    """
    try:
        return SPLITS[_BY_ALIAS[name]]
    except KeyError:
        raise ValueError(f"Unknown split {name!r}. Known splits: {list(SPLITS)}") from None


def canonical_split_name(name: str) -> str:
    """Folder name of a split given by its name or an alias.

    Raises:
        ValueError: For an unknown name.
    """
    return split_info(name).name


def known_split(name: str) -> Optional[SplitInfo]:
    """Like :func:`split_info`, but ``None`` for an unknown name."""
    canonical = _BY_ALIAS.get(name)
    return SPLITS[canonical] if canonical is not None else None


def _split_name(name: str) -> str:
    """The folder name of a split given by its name or an alias (``test_e2e`` -> ``test-e2e``); another name as it is."""
    info = known_split(name)
    return info.name if info else name


def split_sort_key(name: Optional[str]) -> tuple[int, str]:
    """Order splits as in :data:`SPLITS`; unknown names after them, by name."""
    order = list(SPLITS)
    if name in SPLITS:
        return order.index(name), ""
    return len(order), name or ""




# ---------------------------------------------------------------------------
# Scene table of a release
# ---------------------------------------------------------------------------

#: The dataset release this package describes (its scene table, :func:`scene_table`).
RELEASE = "v2.0.0"

#: The values of the categorical columns of the scene table (:class:`SceneInfo`).
CITIES = ("Karlsruhe", "Frankfurt", "Sindelfingen")
LIDARS = ("velodyne", "hesai")
TIMES_OF_DAY = ("morning", "midday", "afternoon", "evening", "night")
LIGHT_CONDITIONS = ("day", "low_sun", "twilight", "night")
RIGS = ("velodyne_2025", "hesai_2026_before_incident", "hesai_2026_after_incident")
ROAD_TYPES = ("motorway", "arterial", "local", "service", "unmatched")
WEATHER = ("clear", "partly_cloudy", "overcast", "rain", "fog", "snow", "unknown")
ROAD_SURFACES = ("dry", "wet", "snow_ice", "unknown")
CONFIDENCES = ("high", "medium", "low")

City = Literal["Karlsruhe", "Frankfurt", "Sindelfingen"]
Lidar = Literal["velodyne", "hesai"]
TimeOfDay = Literal["morning", "midday", "afternoon", "evening", "night"]
Light = Literal["day", "low_sun", "twilight", "night"]
Rig = Literal["velodyne_2025", "hesai_2026_before_incident", "hesai_2026_after_incident"]
RoadType = Literal["motorway", "arterial", "local", "service", "unmatched"]
Weather = Literal["clear", "partly_cloudy", "overcast", "rain", "fog", "snow", "unknown"]
RoadSurface = Literal["dry", "wet", "snow_ice", "unknown"]
Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class SceneInfo:
    """One scene of a release: a row of the scene table shipped with the package.

    ``None`` means the value is not known for the scene, or withheld for its
    split (:func:`withheld_columns`: the test-e2e splits have no recording
    order and no road types). Only categories and rounded statistics: no time
    stamps, nothing finer than the week and the time of day about when a
    scene was recorded. In the test-e2e splits every column describes the
    shipped part of the scene (its first 4 s).

    Attributes:
        scene_id: Scene UUID.
        split: Its release folder (``data/<split>/``), which is its split in
            the default split definition (:data:`DEFAULT_SPLIT`).
        city: ``"Karlsruhe"``, ``"Frankfurt"`` or ``"Sindelfingen"``.
        lidar: ``"velodyne"`` (2025 recordings) or ``"hesai"`` (2026): the
            lidar generation, which changes the point density and labels.
        num_frames: Number of frames.
        recording_week: ISO week of the recording, ``"2026-W06"``.
        time_of_day: Local time (Europe/Berlin) at the middle of the scene:
            ``"morning"`` 05-11 h, ``"midday"`` 11-14, ``"afternoon"`` 14-18,
            ``"evening"`` 18-22, ``"night"`` 22-05.
        light: The sun's elevation at the middle of the scene: ``"day"``
            (>= 6°), ``"low_sun"`` (0 to 6°), ``"twilight"`` (-6 to 0°, civil
            twilight), ``"night"`` (< -6°).
        sun_below_horizon: The sun is below the horizon at the middle of the scene.
        dark: The sun is below -6° for the whole scene (a real night scene).
        recording_order: Position of the scene in recording order (by start
            time), 1..N over the scenes that have it, without gaps.
        drive: Anonymous ID (1..) of the drive (one continuous recording run)
            the scene was cut from.
        order_in_drive: Position of the scene within its drive, 1.. (no gaps).
        rig: The sensor rig and its calibration: ``"velodyne_2025"``,
            ``"hesai_2026_before_incident"`` or ``"hesai_2026_after_incident"``
            (the 2026 rig was disturbed and recalibrated on 2026-02-16).
        sd_snapshot: The OpenStreetMap snapshot of the scene's city and
            recording week, ``"Karlsruhe_2026-W06.osm.pbf"`` in
            ``openstreetmap_sd_maps/`` (:attr:`kitscenes.Scene.sd_map`).
        duration_s: Duration of the shipped poses in seconds.
        distance_m: Distance driven in metres (2D, along the poses).
        mean_speed_mps: :attr:`distance_m` / :attr:`duration_s`.
        max_speed_mps: Maximum speed over 1 s windows of the poses.
        road_motorway_share: Share of :attr:`distance_m` on motorways and
            trunk roads (and their links), by matching the poses to the
            scene's SD snapshot (the nearest drivable way within 15 m that runs
            in the driving direction).
        road_arterial_share: ... on primary and secondary roads.
        road_local_share: ... on tertiary, unclassified, residential roads and
            living streets.
        road_service_share: ... on service roads.
        road_unmatched_share: ... off the mapped roads. The five shares sum to 1.
        road_dominant: The road type with the largest share (:data:`ROAD_TYPES`).
        speed_limit_kph_median: Median OpenStreetMap speed limit along the
            driven distance in km/h; ``math.inf`` where the median is an
            unlimited motorway stretch, None where no road says.
        weather: ``"clear"``, ``"partly_cloudy"``, ``"overcast"``, ``"rain"``,
            ``"fog"``, ``"snow"`` or ``"unknown"`` (e.g. night without a
            visible sky), classified per stretch of about 10 minutes of a drive
            from four ring cameras.
        road_surface: ``"dry"``, ``"wet"``, ``"snow_ice"`` or ``"unknown"``.
        weather_confidence: How sure the weather and road surface are:
            ``"high"``, ``"medium"`` or ``"low"``.
    """

    scene_id: str
    split: str
    city: Optional[City]
    lidar: Optional[Lidar]
    num_frames: int
    recording_week: Optional[str] = None
    time_of_day: Optional[TimeOfDay] = None
    light: Optional[Light] = None
    sun_below_horizon: Optional[bool] = None
    dark: Optional[bool] = None
    recording_order: Optional[int] = None
    drive: Optional[int] = None
    order_in_drive: Optional[int] = None
    rig: Optional[Rig] = None
    sd_snapshot: Optional[str] = None
    duration_s: Optional[float] = None
    distance_m: Optional[float] = None
    mean_speed_mps: Optional[float] = None
    max_speed_mps: Optional[float] = None
    road_motorway_share: Optional[float] = None
    road_arterial_share: Optional[float] = None
    road_local_share: Optional[float] = None
    road_service_share: Optional[float] = None
    road_unmatched_share: Optional[float] = None
    road_dominant: Optional[RoadType] = None
    speed_limit_kph_median: Optional[float] = None
    weather: Optional[Weather] = None
    road_surface: Optional[RoadSurface] = None
    weather_confidence: Optional[Confidence] = None


def _parse_int(value: str) -> Optional[int]:
    return int(value) if value != "" else None


def _parse_float(value: str) -> Optional[float]:
    return float(value) if value != "" else None


def _parse_bool(value: str) -> Optional[bool]:
    if value == "":
        return None
    if value not in ("0", "1"):
        raise ValueError(f"expected 0 or 1, got {value!r}")
    return value == "1"


def _parse_speed(value: str) -> Optional[float]:
    if value == "":
        return None
    return math.inf if value == "none" else float(int(value))


def _parse_str(value: str) -> Optional[str]:
    return value or None


def _category(values: tuple[str, ...]):
    def parse(value: str) -> Optional[str]:
        if value == "":
            return None
        if value not in values:
            raise ValueError(f"expected one of {list(values)}, got {value!r}")
        return value
    return parse


#: Kind and parser of every column after ``scene_id`` and ``split``: ``"category"`` (strings, matched by
#: equality or membership), ``"bool"`` or ``"number"`` (equality, membership or a range), see :func:`find_scenes`.
_COLUMNS: dict[str, tuple[str, object, Optional[tuple[str, ...]]]] = {
    "city": ("category", _category(CITIES), CITIES),
    "lidar": ("category", _category(LIDARS), LIDARS),
    "num_frames": ("number", _parse_int, None),
    "recording_week": ("category", _parse_str, None),
    "time_of_day": ("category", _category(TIMES_OF_DAY), TIMES_OF_DAY),
    "light": ("category", _category(LIGHT_CONDITIONS), LIGHT_CONDITIONS),
    "sun_below_horizon": ("bool", _parse_bool, None),
    "dark": ("bool", _parse_bool, None),
    "recording_order": ("number", _parse_int, None),
    "drive": ("number", _parse_int, None),
    "order_in_drive": ("number", _parse_int, None),
    "rig": ("category", _category(RIGS), RIGS),
    "sd_snapshot": ("category", _parse_str, None),
    "duration_s": ("number", _parse_float, None),
    "distance_m": ("number", _parse_float, None),
    "mean_speed_mps": ("number", _parse_float, None),
    "max_speed_mps": ("number", _parse_float, None),
    "road_motorway_share": ("number", _parse_float, None),
    "road_arterial_share": ("number", _parse_float, None),
    "road_local_share": ("number", _parse_float, None),
    "road_service_share": ("number", _parse_float, None),
    "road_unmatched_share": ("number", _parse_float, None),
    "road_dominant": ("category", _category(ROAD_TYPES), ROAD_TYPES),
    "speed_limit_kph_median": ("number", _parse_speed, None),
    "weather": ("category", _category(WEATHER), WEATHER),
    "road_surface": ("category", _category(ROAD_SURFACES), ROAD_SURFACES),
    "weather_confidence": ("category", _category(CONFIDENCES), CONFIDENCES),
}

#: The columns of the scene table in file order (the fields of :class:`SceneInfo`).
_SCENE_COLUMNS: tuple[str, ...] = ("scene_id", "split", *_COLUMNS)

#: The columns :func:`find_scenes` and :meth:`kitscenes.Dataset.filter` select by, by kind.
FILTER_COLUMNS: dict[str, tuple[str, ...]] = {
    kind: tuple(name for name, (k, _, _) in _COLUMNS.items() if k == kind) for kind in ("category", "bool", "number")
}

_TABLE: Optional[dict[str, SceneInfo]] = None
_WITHHELD: Optional[dict[str, tuple[str, ...]]] = None


def _table_text(name: str) -> str:
    from importlib import resources

    return resources.files("kitscenes._release_tables").joinpath(RELEASE, name).read_text(encoding="utf-8")


def _parse_scene_row(row: dict) -> SceneInfo:
    """A :class:`SceneInfo` from one CSV row (strings; an empty string is None; missing columns are None).

    Raises:
        ValueError: For a value the column does not allow.
    """
    values = {}
    for name, (_, parse, _) in _COLUMNS.items():
        raw = row.get(name)
        try:
            values[name] = None if raw is None else parse(raw)  # type: ignore[operator]
        except ValueError as err:
            raise ValueError(f"Scene {row.get('scene_id')!r}, column {name}: {err}") from None
    return SceneInfo(scene_id=row["scene_id"], split=row["split"], **values)


def scene_table() -> dict[str, SceneInfo]:
    """All scenes of release :data:`RELEASE` by ID (shipped with the package), in table order."""
    global _TABLE
    if _TABLE is None:
        import csv

        rows = csv.DictReader(_table_text("scenes.csv").splitlines())
        _TABLE = {row["scene_id"]: _parse_scene_row(row) for row in rows}
    return _TABLE


def scene_info(scene_id: str) -> Optional[SceneInfo]:
    """The table entry of a scene, None if the release does not list it."""
    return scene_table().get(scene_id)


def withheld_columns(split: str) -> tuple[str, ...]:
    """The columns the scene table leaves empty on purpose for the scenes of *split* (a release folder).

    Release v2.0.0 withholds the recording order (``recording_order``,
    ``drive``, ``order_in_drive``) and the road types (``road_*_share``,
    ``road_dominant``, ``speed_limit_kph_median``) of the test-e2e splits.
    Their ``None`` there means "withheld", not "unknown".

    Raises:
        ValueError: For an unknown split.
    """
    global _WITHHELD
    name = canonical_split_name(split)
    if _WITHHELD is None:
        import json

        setting = json.loads(_table_text("withheld_columns.json"))["by_split"]
        _WITHHELD = {canonical_split_name(s): tuple(cols) for s, cols in setting.items()}
    return _WITHHELD.get(name, ())


def _known_value(column: str, value: object, known: Sequence[str]) -> str:
    """*value* as the scene table spells it (case-insensitive), or ValueError naming the known ones."""
    for name in known:
        if name.lower() == str(value).lower():
            return name
    raise ValueError(f"Unknown {column} {value!r}; the scene table has {sorted(known)}")


def _matcher(column: str, value: object, table: Sequence[SceneInfo]) -> Callable[[object], bool]:
    try:
        kind, _, vocabulary = _COLUMNS[column]
    except KeyError:
        raise TypeError(f"Unknown scene table column {column!r}; columns: {list(_COLUMNS)}") from None
    if kind == "category":
        values = [value] if isinstance(value, str) else list(value)  # type: ignore[call-overload]
        known = vocabulary or sorted({getattr(i, column) for i in table if getattr(i, column) is not None})
        wanted = {_known_value(column, v, known) for v in values}
        return lambda x: x in wanted
    if kind == "bool":
        if not isinstance(value, bool):
            raise TypeError(f"{column} takes True or False, got {value!r}")
        return lambda x: x is value
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{column}: a range is (low, high) with None for an open end, got {value!r}")
        lo, hi = value
        return lambda x: x is not None and (lo is None or x >= lo) and (hi is None or x <= hi)
    if isinstance(value, (list, set, frozenset, range)):
        wanted_numbers = set(value)
        return lambda x: x is not None and x in wanted_numbers
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return lambda x: x is not None and x == value
    raise TypeError(f"{column} takes a number, a list of numbers or a range (low, high), got {value!r}")


def find_scenes(**where: object) -> list[SceneInfo]:
    """The scenes of the scene table that match every criterion, in table order.

    Criteria are columns of :class:`SceneInfo` (:data:`FILTER_COLUMNS`); None means no criterion:

    - **categories** (``city``, ``lidar``, ``recording_week``, ``time_of_day``,
      ``light``, ``rig``, ``sd_snapshot``, ``road_dominant``, ``weather``,
      ``road_surface``, ``weather_confidence``): a value or a list of values
      (any of them), case-insensitive: ``weather=["rain", "overcast"]``;
    - **booleans** (``dark``, ``sun_below_horizon``): True or False;
    - **numbers** (``num_frames``, ``recording_order``, ``drive``,
      ``order_in_drive``, ``duration_s``, ``distance_m``, ``mean_speed_mps``,
      ``max_speed_mps``, ``road_*_share``, ``speed_limit_kph_median``): a
      value, a list of values, or a tuple ``(low, high)``, both ends included,
      None for an open end: ``max_speed_mps=(None, 15)``,
      ``road_motorway_share=(0.5, None)``.

    A scene whose value is None (unknown, or withheld for its split,
    :func:`withheld_columns`) never matches a criterion on that column. ::

        find_scenes(city="Karlsruhe", weather="rain", dark=False)
        find_scenes(time_of_day=["evening", "night"], max_speed_mps=(None, 15))

    Raises:
        TypeError: For an unknown column or a value of the wrong type.
        ValueError: For a category value the table does not have, or a malformed range.
    """
    table = list(scene_table().values())
    tests = [(column, _matcher(column, value, table)) for column, value in where.items() if value is not None]
    return [info for info in table if all(test(getattr(info, column)) for column, test in tests)]


# ---------------------------------------------------------------------------
# Split definitions
# ---------------------------------------------------------------------------

#: Name of the default split definition of release v2.0.0: the release folders. A geographic split: the test
#: regions are cut out of the city maps, so no test road is in a training map.
DEFAULT_SPLIT = "v2.0.0_default_geosplit"

_SPLIT_NAME = re.compile(r"[A-Za-z0-9_.-]+")
#: How many scene IDs a validation error shows per problem.
_SHOWN_IDS = 20
_DEFAULT: Optional["SplitDefinition"] = None


def _shown(ids: Sequence[str]) -> str:
    more = f" and {len(ids) - _SHOWN_IDS} more" if len(ids) > _SHOWN_IDS else ""
    return ", ".join(ids[:_SHOWN_IDS]) + more


def _definition_id(pairs: Iterable[tuple[str, str]]) -> str:
    text = "".join(f"{scene_id},{split}\n" for scene_id, split in sorted(pairs))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _default_id() -> str:
    return _definition_id((sid, info.split) for sid, info in scene_table().items())


@dataclass(frozen=True, eq=False)
class SplitDefinition:
    """A split definition: which scene of the release is in which split.

    The files stay in their release folders (``data/<release folder>/``); a
    definition only names the split of each scene. Use it with
    ``kitscenes.open(root, split_definition=...)``. Build one with
    :meth:`load` or from ``{scene_id: split}``; the constructor checks it. The
    benchmark uses the default, :data:`DEFAULT_SPLIT`; make another only
    with a clear scientific reason (docs/splits.md).

    Rules (:class:`ValueError` listing every problem):

    - every scene is a scene of the release (:func:`scene_table`, exact
      IDs), listed once; a definition may leave scenes out;
    - split names are ``[A-Za-z0-9_.-]+``;
    - **one split, one kind of scene**: the scenes of a split share their
      release folders' ``has_labels`` and ``pose_frame``. So a split is
      either labelled or label-free, and the test-e2e scenes stay among
      themselves.

    Attributes:
        name: The definition's name (:data:`DEFAULT_SPLIT` for the default).
        assignments: Scene ID -> split name, ordered by split (as in
            :attr:`splits`), then ID.
        splits: The split names, in the order they first appear (the
            default definition: the order of :data:`SPLITS`).
        id: 16 hex digits of the SHA-256 of the sorted lines
            ``scene_id,split``: the content only, not the name.
    """

    name: str
    assignments: Mapping[str, str]
    splits: tuple[str, ...] = ()
    id: str = field(init=False)

    def __post_init__(self) -> None:
        # _read_csv passes the rows as pairs, so that a scene listed twice is seen here
        pairs = list(self.assignments.items() if isinstance(self.assignments, Mapping) else self.assignments)
        order = list(dict.fromkeys(self.splits or [split for _, split in pairs]))
        order += [split for split in dict.fromkeys(split for _, split in pairs) if split not in order]
        problems = _problems(self.name, pairs, order)
        if problems:
            raise ValueError(f"Split definition {self.name}: " + "; ".join(problems))
        rank = {split: k for k, split in enumerate(order)}
        ordered = dict(sorted(pairs, key=lambda p: (rank[p[1]], p[0])))
        object.__setattr__(self, "assignments", MappingProxyType(ordered))
        object.__setattr__(self, "splits", tuple(s for s in order if s in set(ordered.values())))
        object.__setattr__(self, "id", _definition_id(pairs))
        if self.name == DEFAULT_SPLIT and self.id != _default_id():
            raise ValueError(f"The name {self.name!r} is the default split definition's; give yours another name.")

    def __repr__(self) -> str:
        return f"SplitDefinition(name={self.name!r}, id={self.id!r}, splits={list(self.splits)}, scenes={len(self)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SplitDefinition):
            return NotImplemented
        return (self.name, self.id) == (other.name, other.id)

    def __hash__(self) -> int:
        return hash((self.name, self.id))

    def __reduce__(self):
        return SplitDefinition, (self.name, dict(self.assignments), self.splits)

    def __len__(self) -> int:
        return len(self.assignments)

    def __contains__(self, scene_id: object) -> bool:
        return scene_id in self.assignments

    @classmethod
    def load(cls, name_or_path: Union[str, "os.PathLike[str]", "SplitDefinition"]) -> "SplitDefinition":
        """The definition :data:`DEFAULT_SPLIT`, or one from a CSV file.

        The file has the header ``scene_id,split`` and one row per scene;
        lines that start with ``#`` are comments. Its name is the file's
        stem, or what a ``# name: <name>`` line says. A
        :class:`SplitDefinition` is returned as it is.

        Raises:
            ValueError: For a file that breaks the rules (see :class:`SplitDefinition`)
                or a name that is neither the default nor a file.
        """
        if isinstance(name_or_path, SplitDefinition):
            return name_or_path
        if str(name_or_path) == DEFAULT_SPLIT:
            return default_split()
        path = Path(name_or_path)
        if not path.is_file():
            raise ValueError(f"No split definition {str(name_or_path)!r}: give {DEFAULT_SPLIT!r} or a CSV file "
                             "with the columns scene_id,split (kitscenes splits export OUT.csv writes the default).")
        return _read_csv(path)

    def to_csv(self, path: Union[str, "os.PathLike[str]"]) -> None:
        """Write the definition as :meth:`load` reads it: a ``# name:`` line, the header, one row per scene."""
        lines = [f"# name: {self.name}", "scene_id,split", *(f"{sid},{split}" for sid, split in self.assignments.items())]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def scene_ids(self, splits: Union[str, Sequence[str], None] = None) -> list[str]:
        """The scenes of *splits* (one name or several; None: all), in definition order.

        Raises:
            ValueError: For a split the definition does not have.
        """
        if splits is None:
            return list(self.assignments)
        names = [splits] if isinstance(splits, str) else list(splits)
        unknown = [s for s in names if s not in self.splits]
        if unknown:
            raise ValueError(f"Split definition {self.name}: no split(s) {unknown}; its splits: {list(self.splits)}")
        return [sid for sid, split in self.assignments.items() if split in names]

    def split_of(self, scene_id: str) -> Optional[str]:
        """The scene's split, None if the definition does not list it."""
        return self.assignments.get(scene_id)


def _problems(name: str, pairs: Sequence[tuple[str, str]], splits: Sequence[str]) -> list[str]:
    """Every rule the ``(scene_id, split)`` *pairs* of definition *name* break."""
    problems = []
    if not isinstance(name, str) or not _SPLIT_NAME.fullmatch(name):
        problems.append(f"the name {name!r} is not [A-Za-z0-9_.-]+")
    bad_names = [s for s in splits if not isinstance(s, str) or not _SPLIT_NAME.fullmatch(s)]
    if bad_names:
        problems.append(f"split names that are not [A-Za-z0-9_.-]+: {bad_names}")
    table = scene_table()
    unknown = [sid for sid, _ in pairs if sid not in table]
    if unknown:
        problems.append(f"{len(unknown)} scene(s) that are not scenes of release {RELEASE}: {_shown(unknown)}")
    counts = Counter(sid for sid, _ in pairs)
    twice = [sid for sid, n in counts.items() if n > 1]
    if twice:
        problems.append(f"{len(twice)} scene(s) listed more than once: {_shown(twice)}")
    kinds: dict[str, dict[tuple[bool, str], set[str]]] = {}
    for sid, split in pairs:
        if sid in table:
            info = SPLITS[table[sid].split]
            kinds.setdefault(split, {}).setdefault((info.has_labels, info.pose_frame), set()).add(sid)
    for split in splits:
        found = kinds.get(split, {})
        if len(found) > 1:
            parts = [f"{len(ids)} {_kind_name(*kind)} ({_shown(sorted(ids))})" for kind, ids in sorted(found.items())]
            problems.append(f"split {split} mixes kinds of scenes: {'; '.join(parts)}")
    if not pairs:
        problems.append("no scenes")
    return problems


def _kind_name(has_labels: bool, pose_frame: str) -> str:
    if has_labels:
        return "labelled scenes"
    return "label-free scenes" + (" with poses relative to the first pose" if pose_frame == "first_pose" else "")


def _read_csv(path: Path) -> SplitDefinition:
    import csv

    name, rows, problems = path.stem, [], []
    data = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            key, sep, value = stripped[1:].partition(":")
            if sep and key.strip() == "name":
                name = value.strip()
            continue
        if stripped:
            data.append(line)
    reader = csv.reader(data)
    header = [h.strip() for h in next(reader, [])]
    if header != ["scene_id", "split"]:
        raise ValueError(f"{path}: the header must be scene_id,split, got {','.join(header) or 'nothing'}")
    for k, row in enumerate(reader, 2):
        if len(row) != 2:
            problems.append(f"row {k}: expected scene_id,split, got {row}")
            continue
        rows.append((row[0].strip(), row[1].strip()))
    if problems:
        raise ValueError(f"{path}: split definition {name}: " + "; ".join(problems[:_SHOWN_IDS]))
    try:
        return SplitDefinition(name, rows)  # the constructor checks the rules
    except ValueError as err:
        raise ValueError(f"{path}: {err}") from None


def default_split() -> SplitDefinition:
    """The default split definition :data:`DEFAULT_SPLIT`: each scene in its release folder."""
    global _DEFAULT
    if _DEFAULT is None:
        table = scene_table()
        present = {info.split for info in table.values()}
        _DEFAULT = SplitDefinition(DEFAULT_SPLIT, {sid: info.split for sid, info in table.items()},
                                   tuple(s for s in SPLITS if s in present))
    return _DEFAULT
