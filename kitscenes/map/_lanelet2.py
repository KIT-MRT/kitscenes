"""The only module that imports lanelet2: version check and feature probes."""

from __future__ import annotations

import warnings
from importlib import import_module
from types import ModuleType

from kitscenes.errors import MissingDependencyError  # noqa: F401  (re-exported by kitscenes.map)

#: Oldest supported lanelet2: the first PyPI release whose wheels work with numpy 2.
#: ``lanelet2.ml_converter`` additionally needs a build that ships it (see ``has_ml_converter``).
LANELET2_MIN_VERSION = "1.2.3"

#: The KITScenes lanelet2 build that ``kitscenes[map]`` pins (wheel URLs in ``pyproject.toml``, Linux x86_64,
#: CPython 3.9-3.13) and the ``"default"`` label recipe is made with: 1.2.3+kitscenes4 labels bicycle lane
#: markings (``type=bike_marking``) as ``BikeMarkingDashed`` / ``BikeMarkingSolid`` (kitscenes3: ``Dashed`` /
#: ``Solid`` / ``Unknown``) in the ``"default"`` and ``"road_border_merged"`` groupings.
LANELET2_BUILD = "1.2.3+kitscenes4"

#: The lanelet2 build of the v2.0.0 paper labels: label recipe ``"paper-v2.0.0"`` needs exactly this one.
LANELET2_PAPER_BUILD = "1.2.3+kitscenes3"


def _version_tuple(version: str) -> tuple[int, ...]:
    """``"1.2.3"``, ``"1.2.3.post1"``, ``"1.2.2+kitscenes"`` -> ``(1, 2, 3)``."""
    parts: list[int] = []
    for piece in version.split("+", 1)[0].split("."):
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
        if digits != piece:
            break
    return tuple(parts)


def _probe() -> tuple[bool, str]:
    try:
        import_module("lanelet2")
    except ImportError as err:
        return False, f"lanelet2 is not installed ({err})"
    try:
        from importlib.metadata import version

        found = version("lanelet2")
    except Exception:  # built from source without metadata
        return True, ""
    if _version_tuple(found) < _version_tuple(LANELET2_MIN_VERSION):
        return False, f"lanelet2 {found} is installed, but kitscenes needs lanelet2>={LANELET2_MIN_VERSION}"
    return True, ""


_STATE: tuple[bool, str] | None = None


def available() -> tuple[bool, str]:
    """``(usable, reason)`` for the installed lanelet2 (probed once)."""
    global _STATE
    if _STATE is None:
        _STATE = _probe()
    return _STATE


def require() -> ModuleType:
    """The lanelet2 module.

    Raises:
        MissingDependencyError: Without a supported lanelet2, with the install command.
    """
    ok, reason = available()
    if not ok:
        raise MissingDependencyError(
            f"The map API needs lanelet2>={LANELET2_MIN_VERSION}: {reason}. "
            'Install it with: pip install "kitscenes[map]"'
        )
    import lanelet2  # noqa: F401
    import lanelet2.matching  # noqa: F401  (submodules used by the map API)

    return lanelet2


def has_ml_converter() -> bool:
    """Whether the installed lanelet2 ships a working ``lanelet2.ml_converter``."""
    if not available()[0]:
        return False
    try:
        import_module("lanelet2.ml_converter")
    except ImportError:
        return False
    return True


def _installed_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("lanelet2")
    except Exception:  # built from source without metadata
        return None


def _kitscenes_build(version: str) -> int | None:
    """``"1.2.3+kitscenes2"`` -> 2; None for a version without a ``+kitscenesN`` label."""
    local = version.split("+", 1)[1] if "+" in version else ""
    for part in local.replace("_", ".").split("."):
        if part.startswith("kitscenes") and part[len("kitscenes"):].isdigit():
            return int(part[len("kitscenes"):])
    return None


#: The first KITScenes lanelet2 build whose ``MapData.processAll`` can be called again with another box
#: (1.2.3+kitscenes1 keeps an instance that was outside one box invalid for every later box).
REUSABLE_MAP_DATA_BUILD = "1.2.3+kitscenes2"


def has_reusable_map_data(version: str | None = None) -> bool:
    """Whether the installed converter's ``MapData`` can be processed for several boxes in turn.

    True from lanelet2 1.2.3+kitscenes2 on (and for any later base version);
    False for 1.2.3+kitscenes1, the PyPI builds and builds without version metadata.
    """
    if version is None:
        if not has_ml_converter():
            return False
        version = _installed_version()
        if version is None:
            return False
    base = _version_tuple(version)
    if base > _version_tuple(REUSABLE_MAP_DATA_BUILD):
        return True
    build = _kitscenes_build(version)
    return base == _version_tuple(REUSABLE_MAP_DATA_BUILD) and build is not None and build >= 2


def require_reusable_map_data(what: str) -> None:
    """Raise :class:`MissingDependencyError` unless :func:`has_reusable_map_data`."""
    if not has_reusable_map_data():
        found = _installed_version() or "a lanelet2 without version metadata"
        raise MissingDependencyError(
            f"{what} needs lanelet2 {REUSABLE_MAP_DATA_BUILD} or newer, whose ML converter can clip one "
            f"MapData to one box after another; the installed one is {found}, where an element outside one "
            "box stays invalid for every later box. Update with: pip install -U \"kitscenes[map]\""
        )


#: The lanelet2 routing-graph configuration kitscenes uses: a vehicle may change lanes across a bicycle lane
#: between two lanes when both bike-lane bounds may be crossed (dashed ``bike_marking`` or dashed lines).
#: Needs a lanelet2 with upstream PR #434 (``TrafficRules.canCrossBoundary``); PyPI lanelet2 1.2.3 lacks it.
ROUTING_GRAPH_CONFIG = {"allow_lane_change_across_bicycle_lane": "true"}

_WARNED_NO_SKIP_OVER: set[str] = set()


def has_bike_lane_lane_change() -> bool:
    """Whether the installed lanelet2 can link lanes across a bicycle lane (upstream PR #434)."""
    if not available()[0]:
        return False
    import lanelet2.traffic_rules

    return hasattr(lanelet2.traffic_rules.TrafficRules, "canCrossBoundary")


def warn_no_bike_lane_lane_change(where: str) -> None:
    """Warn once per *where* that lane changes across bicycle lanes are off with this lanelet2."""
    if where in _WARNED_NO_SKIP_OVER:
        return
    _WARNED_NO_SKIP_OVER.add(where)
    version = _installed_version()
    found = "lanelet2" if version is None else f"lanelet2 {version}"
    warnings.warn(
        f"{where}: no lane-change links across bicycle lanes with {found}; install kitscenes[map] "
        f"(lanelet2 {LANELET2_BUILD}).",
        UserWarning,
        stacklevel=_user_stacklevel(),
    )


def _user_stacklevel() -> int:
    """The ``stacklevel`` that attributes a warning to the first caller outside kitscenes (and functools)."""
    import functools
    import os
    import sys

    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
    frame, level = sys._getframe(2), 2  # level 1: the function that calls warnings.warn
    while frame is not None and (frame.f_code.co_filename.startswith(package)
                                 or frame.f_code.co_filename == functools.__file__):
        frame, level = frame.f_back, level + 1
    return level
