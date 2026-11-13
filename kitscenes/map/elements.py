"""Map elements as plain numpy data (no lanelet2 objects), so they pickle and outlive the map.

All points are (N, 3) float64 in the map frame (see :mod:`kitscenes.map.origin`);
an unknown height is NaN (the maps store -FLT_MAX for it).
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Mapping, Optional

import numpy as np

from kitscenes._repr import compact_repr


class LaneKind(str, Enum):
    """Lanelet ``subtype``."""

    ROAD = "road"
    BUS = "bus_lane"
    BICYCLE = "bicycle_lane"
    CROSSWALK = "crosswalk"
    EMERGENCY = "emergency_lane"
    OTHER = "other"

    @classmethod
    def of(cls, subtype: str) -> "LaneKind":
        try:
            return cls(subtype)
        except ValueError:
            return cls.OTHER


@dataclasses.dataclass(frozen=True, eq=False)
class MapLine:
    """A linestring: lane boundary, marking, curb, stop line, sign, light, pole, ...

    Attributes:
        id: OSM way ID.
        type, subtype: Lanelet2 tags (``"curbstone"``/``"high"``,
            ``"line_thin"``/``"dashed"``, ``"traffic_sign"``/``"de274-30"`` ...;
            ``""`` if untagged).
        points: (N, 3) map-frame points; z is NaN where the height is unknown.
        attributes: All tags.
    """

    __repr__ = compact_repr

    id: int
    type: str
    subtype: str
    points: np.ndarray
    attributes: Mapping[str, str] = dataclasses.field(default_factory=dict, repr=False)

    @property
    def has_height(self) -> bool:
        return bool(np.all(np.isfinite(self.points[:, 2])))


@dataclasses.dataclass(frozen=True, eq=False)
class Lane:
    """A lanelet.

    Attributes:
        id: Lanelet ID.
        kind: Its subtype.
        left, right: IDs of the boundary lines (see :meth:`SceneMap.line`).
        left_points, right_points, centerline: (N, 3) map-frame polylines in
            driving direction.
        speed_limit_mps: Speed limit from the traffic rules (tags, signs, the
            German defaults) in m/s; None where a vehicle may not drive.
            :attr:`speed_limit_kph` gives it in km/h.
        vehicle: Whether a vehicle may drive on it (the traffic rules).
        one_way: Whether a vehicle may only drive it in its direction.
        successors, predecessors: Lanes a vehicle can continue to / come from.
        left_neighbor, right_neighbor: Lanes a vehicle may change to. A
            bicycle lane between two lanes is skipped when both its bounds
            may be crossed (dashed markings), so the neighbour need not share
            a bound with this lane.
        regulatory_ids: Regulatory elements (lights, signs, right of way) of the lane.
        attributes: All tags.
    """

    __repr__ = compact_repr

    id: int
    kind: LaneKind
    left: int
    right: int
    left_points: np.ndarray
    right_points: np.ndarray
    centerline: np.ndarray
    speed_limit_mps: Optional[float]
    vehicle: bool
    one_way: bool
    successors: tuple[int, ...] = ()
    predecessors: tuple[int, ...] = ()
    left_neighbor: Optional[int] = None
    right_neighbor: Optional[int] = None
    regulatory_ids: tuple[int, ...] = ()
    attributes: Mapping[str, str] = dataclasses.field(default_factory=dict, repr=False)

    @property
    def speed_limit_kph(self) -> Optional[float]:
        """:attr:`speed_limit_mps` in km/h (None stays None)."""
        return None if self.speed_limit_mps is None else self.speed_limit_mps * 3.6

    @property
    def polygon(self) -> np.ndarray:
        """(N, 3) outline: left boundary forward, right boundary backward."""
        return np.concatenate([self.left_points, self.right_points[::-1]])


@dataclasses.dataclass(frozen=True, eq=False)
class StopLine:
    """A stop line, from ``type=stop_line`` lines and the ``ref_line`` of regulatory elements (deduplicated).

    Attributes:
        lane_ids: Lanes whose traffic light or right-of-way element refers to it.
    """

    __repr__ = compact_repr

    id: int
    points: np.ndarray
    lane_ids: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True, eq=False)
class TrafficSign(MapLine):
    """A ``traffic_sign`` line (subtype: the German sign code, or ``"unknown"``).

    Attributes:
        regulatory_ids: Regulatory elements referring to it (most signs have none).
        lane_ids: Lanes of those elements.
    """

    __repr__ = compact_repr

    regulatory_ids: tuple[int, ...] = ()
    lane_ids: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True, eq=False)
class TrafficLight(MapLine):
    """A ``traffic_light`` or ``traffic_light_pedestrians`` line.

    Attributes:
        pedestrian: Whether it is a pedestrian light.
        stop_line_ids: Stop lines of its regulatory elements.
        lane_ids: Lanes it controls.
    """

    __repr__ = compact_repr

    pedestrian: bool = False
    stop_line_ids: tuple[int, ...] = ()
    lane_ids: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True, eq=False)
class LaneMatch:
    """A lane a vehicle at a pose may be on, see :meth:`SceneMap.match`.

    Attributes:
        lane_id: The lane.
        reversed: True for a two-way lane driven against its definition order.
        distance: Distance of the pose to the lane in metres (0 inside).
        heading_error: Angle between the heading and the lane direction there, radians.
    """

    lane_id: int
    reversed: bool
    distance: float
    heading_error: float
