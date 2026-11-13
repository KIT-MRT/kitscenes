"""How map labels look: colours, line styles and icons, shared by the top-down view and the reprojection.

Keyed by the converter's type names as :class:`kitscenes.map.MapLabels` has
them (``"Dashed"``, ``"RoadBorder"``, ``"StopLine"``, ``"TSSpeedLimit"``, ...).
"""

from __future__ import annotations

import dataclasses
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from kitscenes.visualization import _deps

if TYPE_CHECKING:
    from kitscenes.map import SceneMap

#: Matplotlib colour of each line type (unknown types: purple).
LINE_COLOURS = {
    "RoadBorder": "green",
    "Dashed": "blue",
    "Solid": "blue",
    "SolidSolid": "darkblue",
    "SolidDashed": "royalblue",
    "DashedSolid": "cornflowerblue",
    "Virtual": "dimgrey",
    "Centerline": "darkred",
    "BikeCenterline": "orange",
    "Unknown": "darkgray",
    "CurbstoneHigh": "darkgreen",
    "CurbstoneLow": "limegreen",
    "Fence": "brown",
    "Building": "sandybrown",
    "Wall": "peru",
    "DrivableArea": "lightblue",
    "Divider": "gray",
    "BikeMarkingDashed": "darkorange",
    "BikeMarkingSolid": "orangered",
    "GuardRail": "saddlebrown",
    "PedestrianCrossing": "yellow",
    "ZebraCrossing": "gold",
}

BORDER_TYPES = ("RoadBorder", "CurbstoneHigh", "CurbstoneLow", "Fence", "GuardRail", "DrivableArea", "Building", "Wall")
DIVIDER_TYPES = ("Dashed", "Solid", "SolidSolid", "SolidDashed", "DashedSolid", "Virtual", "Divider",
                 "BikeMarkingDashed", "BikeMarkingSolid")
CENTERLINE_TYPES = ("Centerline", "BikeCenterline")
#: Closed outlines (crosswalks), drawn as their convex hull.
CROSSING_TYPES = ("ZebraCrossing", "PedestrianCrossing")
ARROW_TYPES = ("ArrowGoStraight", "ArrowTurnLeft", "ArrowTurnRight", "ArrowGoStraightOrLeft",
               "ArrowGoStraightOrRight", "ArrowTurnLeftOrRight")
#: Traffic elements that stand up (signs, lights); the others lie on the road.
UPRIGHT_PREFIXES = ("TS", "TL")

#: Legend entries (the converter's type names -> text), in legend order: the top-down view lists
#: those it drew, the video renderer (:mod:`kitscenes.visualization.gpu_video`) all of them.
LEGEND_NAMES = {
    "RoadBorder": "Road border", "CurbstoneHigh": "Curbstone high", "CurbstoneLow": "Curbstone low",
    "Fence": "Fence", "Building": "Building", "Wall": "Wall", "GuardRail": "Guard rail",
    "DrivableArea": "Drivable area", "Dashed": "Dashed", "Solid": "Solid", "SolidSolid": "Solid-solid",
    "SolidDashed": "Solid-dashed", "DashedSolid": "Dashed-solid", "Virtual": "Virtual", "Divider": "Divider",
    "Centerline": "Centerline", "BikeCenterline": "Bike centerline",
    "PedestrianCrossing": "Pedestrian crossing", "ZebraCrossing": "Zebra crossing", "StopLine": "Stop line",
}

#: Icon file per traffic element type (German StVO numbers), in ``icons/`` next to this module.
ICONS = {
    "TLCar": "traffic_light.png", "TLBike": "traffic_light_bikes.png",
    "TLPedestrian": "traffic_light_pedestrians.png", "TLMisc": "traffic_light_misc.png",
    "TSStop": "206.png", "TSYield": "205.png", "TSNoEntry": "267.png", "TSRightOfWay": "301.png",
    "TSPriorityRoad": "306.png", "TSOneWayStreet": "220-10.png", "TSRoundabout": "215.png",
    "TSSpeedLimit": "274-30.png", "TSPedestrianCrossing": "350-10.png",
    "TSTurnRight": "209.png", "TSTurnLeft": "209-10.png", "TSGoStraight": "209-30.png",
    "TSGoStraightOrRight": "214.png", "TSGoStraightOrLeft": "214-10.png", "TSTurnLeftOrRight": "214-30.png",
    "TSPassRight": "222.png", "TSPassLeft": "222-10.png",
    "ArrowGoStraight": "pf-g.png", "ArrowTurnLeft": "pf-l.png", "ArrowTurnRight": "pf-r.png",
    "ArrowGoStraightOrLeft": "pf-gl.png", "ArrowGoStraightOrRight": "pf-gr.png", "ArrowTurnLeftOrRight": "pf-lr.png",
    "BikeSymbol": "bike.png", "BusSymbol": "bus.png", "Symbol30": "30.png", "Symbol50": "50.png", "Symbol70": "70.png",
    "TSMisc": "traffic_sign.png", "Unknown": "traffic_sign.png",
}


@dataclasses.dataclass(frozen=True)
class LineStyle:
    """How a line type is drawn into a camera image (pixels at full size).

    ``dash`` 0 is solid; ``under`` is the width of a grey line drawn beneath
    (0: none), which tells solid and dashed markings apart on any road.
    """

    dash: int = 0
    gap: int = 0
    width: int = 5
    under: int = 0
    alpha: float = 0.9


LINE_STYLES = {
    "Dashed": LineStyle(40, 40, 2, 7),
    "Solid": LineStyle(0, 0, 2, 7),
    "SolidSolid": LineStyle(0, 0, 2, 7),
    "SolidDashed": LineStyle(0, 0, 5, 10),
    "DashedSolid": LineStyle(0, 0, 5, 10),
    "Centerline": LineStyle(10, 10, 2, 0, 0.7),
    "BikeCenterline": LineStyle(5, 5, 2, 0, 0.7),
    "Virtual": LineStyle(0, 0, 2, 0, 0.5),
}
DEFAULT_LINE_STYLE = LineStyle()


def colour(type_name: str) -> tuple[int, int, int]:
    """RGB (0-255) of a line type."""
    mcolors = _deps.matplotlib("colors")

    return tuple(int(round(c * 255)) for c in mcolors.to_rgb(LINE_COLOURS.get(type_name, "purple")))


def is_upright(te_type: str) -> bool:
    return te_type.startswith(UPRIGHT_PREFIXES)


def icon_dir() -> Path:
    """The traffic element icons: ``$KITSCENES_TE_ICON_PATH`` if it is a folder, else the ones shipped with kitscenes."""
    env = os.environ.get("KITSCENES_TE_ICON_PATH")
    if env and Path(env).is_dir():
        return Path(env)
    return Path(__file__).with_name("icons")


def icon_path(te_type: str) -> Optional[Path]:
    """The icon of a traffic element type, falling back to the generic light or sign; None if neither exists."""
    folder = icon_dir()
    for name in (ICONS.get(te_type, "traffic_sign.png"),
                 "traffic_light.png" if te_type.startswith("TL") else "traffic_sign.png"):
        if (folder / name).is_file():
            return folder / name
    return None


@lru_cache(maxsize=256)
def icon_rgba(te_type: str) -> Optional[np.ndarray]:
    """The icon as a read-only RGBA uint8 array, or None."""
    path = icon_path(te_type)
    if path is None:
        return None
    from PIL import Image

    with Image.open(path) as image:
        out = np.asarray(image.convert("RGBA"))
    out.setflags(write=False)
    return out


def car_image() -> Optional[Path]:
    """The ego car drawn in the top-down view."""
    path = Path(__file__).with_name("car.png")
    return path if path.is_file() else None


def unknown_sign_ids(scene_map: "SceneMap") -> frozenset[int]:
    """Traffic signs whose subtype is ``"unknown"``: drawn by neither view (no type to show)."""
    return frozenset(s.id for s in scene_map.traffic_signs() if s.subtype == "unknown")


def crossing_outline(points: np.ndarray) -> np.ndarray:
    """A crosswalk perimeter as its closed convex hull in x-y (the converter's ring can self-intersect)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return pts
    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = ConvexHull(pts[:, :2])
    except (QhullError, ValueError):
        return pts
    ring = pts[hull.vertices]
    return np.vstack([ring, ring[:1]])
