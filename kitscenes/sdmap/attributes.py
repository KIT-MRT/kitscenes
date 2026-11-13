"""OpenStreetMap tags -> road attributes: class, direction, speed limit, lanes, turn lanes, access.

Every value comes with where it came from, because a third of the attributes
in OpenStreetMap are implicit:

- **Speed limits** (``maxspeed``, per direction ``maxspeed:forward`` /
  ``:backward``) are numbers in km/h (``"50"``), mph (``"30 mph"``),
  ``"none"`` (no limit: German Autobahn) or German implicit values
  (``"DE:urban"`` 50, ``"DE:rural"`` 100, ``"DE:zone30"`` 30,
  ``"DE:motorway"`` no limit, ``"DE:living_street"`` walking speed). Where
  ``maxspeed`` is missing, ``maxspeed:type`` / ``source:maxspeed`` and then
  ``zone:maxspeed`` are read, then a default by road class (urban: 50 km/h).
  The source is ``"tag"``, ``"implicit"`` (a ``DE:*`` value), ``"zone"``,
  ``"default"`` or None (unknown).
- **One-way:** ``oneway=yes|1|true`` (forward), ``-1|reverse`` (backward),
  ``no`` (both); a roundabout (``junction=roundabout|circular``) and a
  motorway (and its links) are one-way unless ``oneway=no``. No tag otherwise
  means two-way: that is OpenStreetMap's convention, not missing data.
- **Lanes** per direction: ``lanes:forward`` / ``:backward``, else ``lanes``
  of a one-way road, else half of ``lanes`` of a two-way road (source
  ``"tag_split"``), else a default by class (``"default"``).
- **Turn lanes** of a direction (``turn:lanes``, ``turn:lanes:forward`` /
  ``:backward``), left to right in the driving direction, each lane a set of
  ``"left"``, ``"through"``, ``"right"``, ``"slight_left"``, ...,
  ``"none"`` (empty).
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Mapping, Optional

#: km/h per mph.
MPH_IN_KPH = 1.609344


class RoadClass(str, Enum):
    """The ``highway`` class of a road; links (``*_link``) have their parent's class and ``SDRoad.is_link``."""

    MOTORWAY = "motorway"
    TRUNK = "trunk"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    UNCLASSIFIED = "unclassified"
    RESIDENTIAL = "residential"
    LIVING_STREET = "living_street"
    SERVICE = "service"
    ROAD = "road"
    BUSWAY = "busway"

    @property
    def rank(self) -> int:
        """0 for motorways up to 10 for busways: the importance order of the classes."""
        return list(RoadClass).index(self)


#: ``highway`` values that make an SD road (drivable by a car, maybe with an access restriction).
DRIVABLE_HIGHWAYS: frozenset[str] = frozenset(
    [c.value for c in RoadClass] + [f"{c}_link" for c in ("motorway", "trunk", "primary", "secondary", "tertiary")]
)

_YES = ("yes", "1", "true")
#: Access values a car may use with a reason (OSRM car profile: "restricted"; routable, not for through traffic).
RESTRICTED_ACCESS = frozenset(["private", "delivery", "destination", "customers", "permit", "residents", "unknown"])
#: Access values that close a road to cars (OSRM car profile's blacklist without the restricted ones).
NO_ACCESS = frozenset(["no", "agricultural", "forestry", "emergency", "psv", "taxi", "share_taxi", "minibus", "bus",
                       "foot", "emergency_vehicle", "restricted", "military", "official"])
#: The access levels (:func:`access`).
ACCESS_LEVELS = ("public", "restricted", "no")

#: Speed limits of the German implicit ``maxspeed`` values in km/h (``math.inf``: no limit).
IMPLICIT_SPEED_LIMITS_KPH: dict[str, float] = {
    "DE:urban": 50.0,
    "DE:rural": 100.0,
    "DE:motorway": math.inf,
    "DE:trunk": 100.0,
    "DE:zone30": 30.0,
    "DE:zone:30": 30.0,
    "DE:zone20": 20.0,
    "DE:zone:20": 20.0,
    "DE:zone10": 10.0,
    "DE:zone:10": 10.0,
    "DE:bicycle_road": 30.0,
    "DE:living_street": 7.0,
    "DE:walk": 7.0,
}

#: The speed limit by class where a road says nothing (the cities' urban default, 50 km/h), km/h or None.
DEFAULT_SPEED_LIMITS_KPH: dict[str, Optional[float]] = {
    "motorway": math.inf,
    "motorway_link": None,
    "trunk": None,
    "trunk_link": None,
    "living_street": 7.0,
}
URBAN_DEFAULT_KPH = 50.0

_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kmh|kph)?\s*$")


def road_class(highway: str) -> tuple[RoadClass, bool]:
    """``(class, is_link)`` of a drivable ``highway`` value.

    Raises:
        ValueError: For a value not in :data:`DRIVABLE_HIGHWAYS`.
    """
    if highway not in DRIVABLE_HIGHWAYS:
        raise ValueError(f"highway={highway!r} is not a drivable road class")
    if highway.endswith("_link"):
        return RoadClass(highway[: -len("_link")]), True
    return RoadClass(highway), False


def directions(tags: Mapping[str, str]) -> tuple[bool, bool]:
    """``(forward, backward)``: in which directions (along / against the node order) a car may drive the way."""
    oneway = tags.get("oneway", "")
    if oneway in _YES:
        return True, False
    if oneway in ("-1", "reverse"):
        return False, True
    if oneway in ("no", "alternating", "reversible"):
        return True, True
    implied = tags.get("junction") in ("roundabout", "circular") or tags.get("highway") in ("motorway", "motorway_link")
    return True, not implied


def access(tags: Mapping[str, str]) -> str:
    """Whether a car may use a way: ``"public"``, ``"restricted"`` (with a reason: ``private``,
    ``destination``, ``delivery``, ``customers``, ...) or ``"no"`` (``access=no``, ``agricultural``, a
    busway, ...). The most specific of ``motorcar``, ``motor_vehicle``, ``vehicle``, ``access`` that is
    set decides, as in OSRM's car profile."""
    if tags.get("highway") == "busway":
        value = tags.get("motorcar", tags.get("motor_vehicle", "no"))
    else:
        value = next((tags[k] for k in ("motorcar", "motor_vehicle", "vehicle", "access") if k in tags), "yes")
    value = value.split(";")[0].strip()
    if value in RESTRICTED_ACCESS:
        return "restricted"
    if value in NO_ACCESS:
        return "no"
    return "public"


def is_public(tags: Mapping[str, str]) -> bool:
    """Whether any car may use the way without a reason: :func:`access` is ``"public"``."""
    return access(tags) == "public"


def parse_speed(value: Optional[str]) -> tuple[Optional[float], Optional[str]]:
    """A ``maxspeed`` value -> ``(km/h or inf, "tag" | "implicit")``, ``(None, None)`` if it says no number.

    ``"none"`` is no limit (inf); ``"signals"`` / ``"variable"`` are unknown.
    The first number of a list (``"50;30"``) is used.
    """
    if value is None:
        return None, None
    value = value.strip().split(";")[0].strip()
    if value == "none":
        return math.inf, "tag"
    if value in IMPLICIT_SPEED_LIMITS_KPH:
        return IMPLICIT_SPEED_LIMITS_KPH[value], "implicit"
    if value.startswith("DE:"):
        m = re.search(r"(\d+)", value)
        return (float(m.group(1)), "implicit") if m else (None, None)
    m = _NUMBER.match(value)
    if not m:
        return None, None
    number = float(m.group(1))
    return number * (MPH_IN_KPH if m.group(2) == "mph" else 1.0), "tag"


def speed_limit(tags: Mapping[str, str], forward: bool = True) -> tuple[Optional[float], Optional[str]]:
    """The speed limit of a direction of a way in km/h, as tagged (inf: none), and its source; see the module docs.

    :class:`~kitscenes.sdmap.SDRoad` keeps it in m/s (``speed_limit_mps``, ``speed_limit_kph``).
    """
    direction = "forward" if forward else "backward"
    for key in (f"maxspeed:{direction}", "maxspeed"):
        value, source = parse_speed(tags.get(key))
        if value is not None:
            return value, source
    for key in ("maxspeed:type", "source:maxspeed"):
        value, _ = parse_speed(tags.get(key))
        if value is not None:
            return value, "implicit"
    zone = tags.get("zone:maxspeed") or tags.get("zone:traffic")
    if zone:
        value, _ = parse_speed(zone if zone in IMPLICIT_SPEED_LIMITS_KPH else zone.replace("DE:", "DE:zone"))
        if value is None:
            value, _ = parse_speed(zone.split(":")[-1])
        if value is not None:
            return value, "zone"
    highway = tags.get("highway", "")
    if highway in DEFAULT_SPEED_LIMITS_KPH:
        value = DEFAULT_SPEED_LIMITS_KPH[highway]
        return (value, "default") if value is not None else (None, None)
    if highway in DRIVABLE_HIGHWAYS:
        return URBAN_DEFAULT_KPH, "default"
    return None, None


def _int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        n = int(float(value.split(";")[0].strip()))
    except ValueError:
        return None
    return n if n > 0 else None


#: Lanes per direction where a road says nothing: one, two on a one-way motorway or trunk.
def default_lanes(highway: str, one_way: bool) -> int:
    if one_way and highway in ("motorway", "trunk"):
        return 2
    return 1


def lanes(tags: Mapping[str, str], forward: bool = True) -> tuple[int, str]:
    """Lanes of one direction of a way and the source (``"tag"``, ``"tag_split"``, ``"default"``)."""
    fwd, bwd = directions(tags)
    one_way = fwd != bwd
    direction = "forward" if forward else "backward"
    n = _int(tags.get(f"lanes:{direction}"))
    if n is not None:
        return n, "tag"
    total = _int(tags.get("lanes"))
    if total is not None:
        if one_way:
            return total, "tag"
        other = _int(tags.get(f"lanes:{'backward' if forward else 'forward'}"))
        if other is not None and total > other:
            return total - other, "tag"
        return max(1, total // 2), "tag_split"
    return default_lanes(tags.get("highway", ""), one_way), "default"


def turn_lanes(tags: Mapping[str, str], forward: bool = True) -> Optional[tuple[frozenset[str], ...]]:
    """The ``turn:lanes`` of one direction, left to right in the driving direction; None without the tag."""
    fwd, bwd = directions(tags)
    one_way = fwd != bwd
    direction = "forward" if forward else "backward"
    value = tags.get(f"turn:lanes:{direction}")
    if value is None and one_way:
        value = tags.get("turn:lanes")
    if value is None:
        return None
    return tuple(frozenset(t.strip() or "none" for t in lane.split(";")) for lane in value.split("|"))
