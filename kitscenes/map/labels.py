"""Map labels for machine learning, from ``lanelet2.ml_converter``.

The converter (a lanelet2 build that ships it, see the README) turns the map around a pose into instance labels: lane lines
chained across the splits of the annotation into compound lines (dividers,
road borders, centerlines, crosswalk outlines, drivable-space borders, ...),
traffic elements (stop lines, arrows, symbols, lights, signs) and the edges
between them. This module calls it and hands the result over as numpy
arrays; what a label is, is the converter's definition. A few options shape
the result for vector-map models (MapTR and its successors):

- ``classes``: which converter types make up which output class (and which
  are left out). Every class is a key of :attr:`MapLabels.lines`, empty
  where there is none, so the key set is the same for every pose.
- ``crosswalks="polygons"``: crosswalks as closed outlines clipped by the
  box (one per crossing) instead of the converter's perimeter lines, which
  the box cuts into open pieces.
- ``max_height_diff``: line labels on another level (a bridge above the
  vehicle, a road below it) are cut away.
- ``chaining="scene"``: chains built once per scene and clipped per pose,
  so the member ids of an element stay the same from frame to frame.

Besides the classes, every label set has the **lane segments** with their
successor graph (:attr:`MapLabels.lane_segments`, :attr:`MapLabels.lane_to_lane`),
for lane-graph models (TopoNet, LaneSegNet). The converter's centerlines are
paths through the region (MapTRv2's definition: overlapping, one per way
through it), so a successor graph between them would be empty.

Every line has a key ``(class, member ids, piece)`` (:meth:`MapLabels.line_keys`).
``kitscenes.MapLabelSpec.maptr()`` is the MapTR preset of these options.

Frames: labels are in the **ego frame** of the pose, the vehicle base frame
(x forward, y left, z up; its origin is about 2 m above the road). With
``dims=2`` they are 2D in the vehicle's heading frame (yaw only, no pitch or
roll), as the converter's 2D mode defines it. Lines are cut exactly to the
region in that frame's x-y (the converter cuts in its ground plane, which a
pitched or rolled base frame is not: a few centimetres on slopes).

Example::

    from kitscenes.map import MLConverter, Region, Spacing

    conv = MLConverter.for_scene(scene, grouping="m3tr", region=Region(ahead=60, behind=20, left=20, right=20))
    labels = conv(frame.pose)
    labels.lines["Dashed"]          # list of (N, 3) arrays, ego frame (an empty list where there are none)
    labels.line_ids["Dashed"]       # map element ids behind each of them
    labels.line_keys("Dashed")      # ("Dashed", ids, piece) per line
    MLConverter(scene.map, sampling=Spacing(1.0))(frame.pose)   # a point per metre instead of 20 per line
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Sequence, Union

import numpy as np

from kitscenes._repr import compact_repr
from kitscenes.map import _lanelet2, _vector
from kitscenes.map.roi import Circle, Region
from kitscenes.map.sampling import FixedPoints, Sampling, resample

if TYPE_CHECKING:
    from kitscenes.map.scene_map import SceneMap
    from kitscenes.samples import MapLabelSpec
    from kitscenes.scene import Scene
    from kitscenes.schema import EgoPose

#: Line type groupings of the converter: which line types are chained into one compound line, and its label.
GROUPINGS = ("default", "road_border_merged", "maptr", "m3tr")
_GROUPING_FUNCTIONS = {
    "default": "getDefaultLineStringTypeGrouping",
    "road_border_merged": "getRoadBorderMergedGrouping",
    "maptr": "getMapTRDefaultSimpleGrouping",
    "m3tr": "getM3TRDefaultGrouping",
}

#: The line types each grouping labels with (its representative types, in the converter's order): the keys of
#: :attr:`MapLabels.lines` without ``classes``.
GROUPING_CLASSES: dict[str, tuple[str, ...]] = {
    "default": ("RoadBorder", "Dashed", "Solid", "CurbstoneHigh", "CurbstoneLow", "Fence", "SolidSolid",
                "SolidDashed", "DashedSolid", "Virtual", "Centerline", "BikeCenterline", "Unknown", "DrivableArea",
                "Divider", "BikeMarkingDashed", "BikeMarkingSolid", "GuardRail", "PedestrianCrossing",
                "ZebraCrossing", "Building", "Wall"),
    "road_border_merged": ("RoadBorder", "Dashed", "Solid", "Unknown", "SolidSolid", "SolidDashed", "DashedSolid",
                           "Virtual", "Centerline", "BikeCenterline", "DrivableArea", "BikeMarkingDashed",
                           "BikeMarkingSolid", "PedestrianCrossing", "ZebraCrossing"),
    "maptr": ("RoadBorder", "Divider", "Virtual", "Centerline", "BikeCenterline", "Unknown", "DrivableArea",
              "PedestrianCrossing"),
    "m3tr": ("RoadBorder", "Dashed", "Solid", "Virtual", "Centerline", "BikeCenterline", "Unknown", "DrivableArea",
             "PedestrianCrossing"),
}

#: The traffic element types (the converter's default grouping keeps each apart): the keys of
#: :attr:`MapLabels.traffic_elements` without ``te_classes``.
TE_TYPES: tuple[str, ...] = (
    "TLCar", "TLBike", "TLPedestrian", "TLMisc", "TSMisc", "TSNoEntry", "TSTurnRight", "TSTurnLeft",
    "TSTurnLeftOrRight", "TSGoStraight", "TSGoStraightOrRight", "TSGoStraightOrLeft", "TSPassRight", "TSPassLeft",
    "TSOneWayStreet", "TSYield", "TSRightOfWay", "TSPriorityRoad", "TSStop", "TSCrossbuck", "TSRoundabout",
    "TSSpeedLimit", "TSPedestrianCrossing", "ArrowTurnRight", "ArrowTurnLeft", "ArrowTurnLeftOrRight",
    "ArrowGoStraight", "ArrowGoStraightOrRight", "ArrowGoStraightOrLeft", "BikeSymbol", "BusSymbol", "Symbol30",
    "Symbol50", "Symbol70", "StopLine", "Unknown",
)

#: The MapTR classes (MapTR's label order): lane markings, crosswalks and the drivable-space border
#: (``drivable_space_border`` lines, the closest match to the AV2 and nuScenes boundary).
MAPTR_CLASSES: dict[str, tuple[str, ...]] = {
    "divider": ("Divider",),
    "ped_crossing": ("PedestrianCrossing",),
    "boundary": ("DrivableArea",),
}
#: MapTRv2: the MapTR classes and lane centerlines.
MAPTRV2_CLASSES: dict[str, tuple[str, ...]] = {**MAPTR_CLASSES, "centerline": ("Centerline",)}

#: Converter types of crosswalk outlines (both from ``subtype=crosswalk`` lanelets).
CROSSING_TYPES = ("PedestrianCrossing", "ZebraCrossing")

Classes = Union[Mapping[str, Sequence[str]], Sequence[str]]


def require_ml_converter():
    """The ``lanelet2.ml_converter`` module.

    Raises:
        MissingDependencyError: If the installed lanelet2 does not ship it.
    """
    _lanelet2.require()
    if not _lanelet2.has_ml_converter():
        raise _lanelet2.MissingDependencyError(
            "Map labels need a lanelet2 build that ships lanelet2.ml_converter; the installed one "
            "does not (the PyPI wheels never do). pip install \"kitscenes[map]\" installs the KITScenes "
            f"build (lanelet2 {_lanelet2.LANELET2_BUILD}, Linux x86_64, Python 3.9-3.13, from "
            "https://github.com/KITScenes/Lanelet2/releases); see the kitscenes README, Setup"
        )
    import lanelet2.ml_converter as ml

    return ml


def normalize_classes(classes: Optional[Classes], available: Sequence[str], what: str = "classes"
                      ) -> dict[str, tuple[str, ...]]:
    """*classes* as ``{class: types}`` (None: one class per available type; a list: those types as classes).

    Raises:
        ValueError: For a type that is not available or is in two classes.
    """
    if classes is None:
        return {t: (t,) for t in available}
    if isinstance(classes, str):
        raise TypeError(f"{what} must be a mapping or a list of type names, not a string")
    out = {str(c): (str(c),) for c in classes} if not isinstance(classes, Mapping) else {
        str(name): ((types,) if isinstance(types, str) else tuple(str(t) for t in types))
        for name, types in classes.items()}
    seen: dict[str, str] = {}
    for name, types in out.items():
        for t in types:
            if t not in available:
                raise ValueError(f"{what}: {t!r} (class {name!r}) is not one of {tuple(available)}")
            if t in seen:
                raise ValueError(f"{what}: {t!r} is in two classes, {seen[t]!r} and {name!r}")
            seen[t] = name
    return out


@dataclasses.dataclass
class MapLabels:
    """The map labels around one pose.

    Attributes:
        lines: Lines by class (the converter's type names, e.g. ``"Dashed"``,
            ``"RoadBorder"``, ``"Centerline"``, or the names of the converter's
            ``classes``): lists of (N, dims) arrays in the ego frame. Every
            class is a key, with an empty list where there is none. A line
            the region cuts in two comes as two arrays.
        line_ids: The map element ids each line was chained from (lanelet
            ids for centerlines and for crosswalk polygons), index-aligned
            with :attr:`lines`.
        line_pieces: Which piece of its chain each line is: 0, 1, ... in
            the order along the chain (the region, and the height filter,
            can cut a chain into several pieces), index-aligned with
            :attr:`lines`. ``(class, line_ids, piece)`` is the line's key
            (:meth:`line_keys`), unique in a label set.
        traffic_elements: Traffic elements by class (e.g. ``"StopLine"``,
            ``"TSSpeedLimit"``): lists of (N, dims) arrays; every class is a key.
        traffic_element_ids: Map element id of each traffic element.
        untyped_sign_ids: Map ids of the traffic signs in these labels whose
            map subtype is ``"unknown"`` (no type to show; the plots leave
            them out by default).
        te_to_centerline: ``(te_class, te_index, centerline_index)``: the lane
            a traffic element applies to; an index into the lines of the
            class with the converter's centerlines (``lines["Centerline"]``
            unless ``classes`` renames it).
        te_to_te: ``(te_class, te_index, other_class, other_index)``, e.g. a
            traffic light to its stop line.
        lane_segments: Lane centerlines between branch points: a run of
            lanelets (of the converter's centerlines) where each has exactly
            one successor and that one exactly one predecessor, in the
            lanelets' direction (a bidirectional lanelet once, in its own
            direction), clipped by the region (and the height filter) like
            the lines; (N, dims) arrays.
        lane_segment_ids: The lanelet ids of each segment's whole run (also
            the parts outside the region, so the ids are the same in every
            frame), index-aligned with :attr:`lane_segments`.
        lane_segment_pieces: Which piece of its run each segment is (0, 1, ...
            along it), as :attr:`line_pieces`.
        lane_to_lane: ``(i, j)``: segment j follows segment i: j's first
            lanelet is a successor of i's last lanelet (from i's last piece
            to j's first), or i and j are consecutive pieces of one run (the
            region or the height filter cut it, so they need not touch).
        pose: The pose the labels belong to (map frame).
        region: The labelled region.
        dims: 3, or 2 for the converter's 2D mode.
        map_changed: The scene is a map_change scene: its map differs from
            the road it drove on, on purpose (for the whole map, not per
            element).
    """

    __repr__ = compact_repr

    lines: dict[str, list[np.ndarray]]
    line_ids: dict[str, list[tuple[int, ...]]]
    traffic_elements: dict[str, list[np.ndarray]]
    traffic_element_ids: dict[str, list[int]]
    te_to_centerline: list[tuple[str, int, int]]
    te_to_te: list[tuple[str, int, str, int]]
    pose: "EgoPose"
    region: Region
    dims: int
    map_changed: bool = False
    line_pieces: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    lane_segments: list[np.ndarray] = dataclasses.field(default_factory=list)
    lane_segment_ids: list[tuple[int, ...]] = dataclasses.field(default_factory=list)
    lane_segment_pieces: list[int] = dataclasses.field(default_factory=list)
    lane_to_lane: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    untyped_sign_ids: frozenset = frozenset()

    def __len__(self) -> int:
        return sum(len(v) for v in self.lines.values()) + sum(len(v) for v in self.traffic_elements.values())

    def line_keys(self, cls: str) -> list[tuple[str, tuple[int, ...], int]]:
        """``(class, member ids, piece)`` of each line of *cls*: the same element has the same key in every frame
        where the chain is the same (always with ``chaining="scene"``)."""
        pieces = self.line_pieces.get(cls) or [0] * len(self.lines[cls])
        return [(cls, ids, piece) for ids, piece in zip(self.line_ids[cls], pieces)]


#: The fields of :class:`kitscenes.MapLabelSpec` that are arguments of :meth:`MLConverter.for_scene` as they are.
_SPEC_ARGUMENTS = ("recipe", "grouping", "region", "dims", "sampling", "te_sampling", "ignore_elevation",
                   "crosswalks", "max_height_diff", "chaining")


class MLConverter:
    """Map labels around poses, from ``lanelet2.ml_converter`` on a :class:`SceneMap`.

    Args:
        scene_map: The map (a scene's, or a city map loaded in the scene's
            frame with ``scene.load_map``).
        grouping: Line type grouping, one of :data:`GROUPINGS` or a
            converter ``LineStringTypeGrouping``.
        region: Labelled region around the vehicle.
        dims: 3 (default) or 2.
        sampling: Points per line: :class:`FixedPoints` (default 20, evenly
            along the arc length), :class:`~kitscenes.map.sampling.Spacing`,
            :class:`~kitscenes.map.sampling.Adaptive` or ``None`` (the map's
            vertices). Always applied once to the converter's vertices.
        te_sampling: The same for traffic elements (default: 20 points).
        ignore_elevation: Flatten the map to z = 0 (the vehicle height is
            dropped with it, so the labels stay at the vehicle's level).
        map_changed: Passed on to :attr:`MapLabels.map_changed`
            (:meth:`for_scene` sets it from the split).
        classes: The line classes: ``{name: converter types}`` (e.g.
            :data:`MAPTR_CLASSES`), or a list of types to keep under their
            own names. Types in no class are left out. Default: every type of
            the grouping (:data:`GROUPING_CLASSES`).
        te_classes: The same for traffic elements (default: every type,
            :data:`TE_TYPES`; ``{}`` for none).
        crosswalks: ``"lines"`` (the converter's perimeter lines, cut open
            by the region) or ``"polygons"``: one closed outline per crossing
            (crosswalk lanelets that share a boundary edge are merged),
            clipped by the region and closed again (first point == last
            point), counter-clockwise; ids are the crosswalk lanelets.
            Outlines under 1 m² after clipping are dropped.
        max_height_diff: Cut away line labels more than this many metres
            above or below the road (bridges, underpasses); the road level
            follows *road_poses*. Traffic elements are kept. None: no filter.
        road_poses: The poses of the drive, for *max_height_diff*
            (:meth:`for_scene`: the scene's poses). None: the road plane
            under the vehicle at the labelled pose only.
        road_height_m: Height of the base frame above the road, for
            *max_height_diff* (:meth:`for_scene`: ``scene.road_height_m``;
            default :data:`kitscenes.constants.DEFAULT_ROAD_HEIGHT_M`).
        chaining: ``"frame"``: the converter chains the lines of the map
            around each pose anew, so a chain's members can change from frame
            to frame. A :class:`~kitscenes.map.Circle` (map frame): the chains
            are built once from the map in it and only clipped per pose, so
            an element keeps its member ids; ``for_scene(scene,
            chaining="scene")`` takes the circle around the scene's poses.
            Needs the KITScenes build of lanelet2 (``kitscenes[map]``).
        exact_cut: Cut lines, crosswalk polygons and lane segments exactly to
            the region in the label frame's x-y (pieces under 0.1 m and
            polygons under 1 m² dropped). False keeps the lanelet2 converter's
            own cut in its ground plane, which can reach a few cm past the
            region where the vehicle pitches or rolls, and the converter's order
            of the ``te_to_centerline`` edges (sorted with the exact cut): the
            labels of the ``"paper-v2.0.0"`` recipe (:meth:`for_scene` ``recipe=``).

    The converter objects are made on first use and not pickled, so an
    ``MLConverter`` can go to DataLoader workers.
    """

    def __init__(
        self,
        scene_map: "SceneMap",
        *,
        grouping: Union[str, object] = "default",
        region: Region = Region(),
        dims: int = 3,
        sampling: Sampling = FixedPoints(20),
        te_sampling: Sampling = FixedPoints(20),
        ignore_elevation: bool = False,
        map_changed: bool = False,
        classes: Optional[Classes] = None,
        te_classes: Optional[Classes] = None,
        crosswalks: str = "lines",
        max_height_diff: Optional[float] = None,
        chaining: Union[str, Circle] = "frame",
        exact_cut: bool = True,
        road_poses: Optional[Sequence["EgoPose"]] = None,
        road_height_m: Optional[float] = None,
    ) -> None:
        if dims not in (2, 3):
            raise ValueError("dims must be 2 or 3")
        if isinstance(grouping, str) and grouping not in GROUPINGS:
            raise ValueError(f"grouping must be one of {GROUPINGS} or a LineStringTypeGrouping")
        if crosswalks not in ("lines", "polygons"):
            raise ValueError(f"crosswalks must be 'lines' or 'polygons', got {crosswalks!r}")
        if max_height_diff is not None:
            if not max_height_diff > 0:
                raise ValueError("max_height_diff must be > 0 (or None)")
            if ignore_elevation:
                raise ValueError("max_height_diff needs the map's heights; it cannot be combined with ignore_elevation")
        if chaining == "scene":
            raise ValueError("chaining='scene' needs the scene's poses: MLConverter.for_scene(scene, chaining='scene'), "
                             "or chaining=Circle(...) for an area of the map")
        if not (chaining == "frame" or isinstance(chaining, Circle)):
            raise ValueError(f"chaining must be 'frame', 'scene' (for_scene) or a Circle, got {chaining!r}")
        self.scene_map = scene_map
        self.grouping = grouping
        self.region = region
        self.dims = dims
        self.sampling = sampling
        self.te_sampling = te_sampling
        self.ignore_elevation = ignore_elevation
        self.map_changed = map_changed
        self.classes = normalize_classes(classes, _grouping_types(grouping))
        self.te_classes = normalize_classes(te_classes, TE_TYPES, "te_classes")
        self.crosswalks = crosswalks
        self.max_height_diff = None if max_height_diff is None else float(max_height_diff)
        self.chaining = chaining
        self.exact_cut = bool(exact_cut)
        from kitscenes.constants import DEFAULT_ROAD_HEIGHT_M

        self.road_height_m = DEFAULT_ROAD_HEIGHT_M if road_height_m is None else float(road_height_m)
        self._road = None if road_poses is None else _RoadLevel.of(road_poses, self.road_height_m)
        if isinstance(chaining, Circle):
            _lanelet2.require_reusable_map_data("MLConverter(chaining=...), chains built once per scene,")
        self._interface = None
        self._chain_data = None
        self._crossings: Optional[list[_vector.Crossing]] = None
        self._runs: dict[int, tuple[int, ...]] = {}  # lanelet id -> its lane run (lane segments)
        self._run_lines: dict[tuple[int, ...], np.ndarray] = {}
        self._link_cache: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        self._untyped_signs: Optional[frozenset] = None  # map ids of the signs with subtype "unknown"

    @classmethod
    def for_scene(cls, scene: "Scene", spec: Optional["MapLabelSpec"] = None, **kwargs) -> "MLConverter":
        """A converter on ``scene.map``, flagged when the scene is a map_change scene.

        Args:
            scene: The scene.
            spec: A :class:`kitscenes.MapLabelSpec`: its fields are the
                converter's arguments. The labels stay in the base frame
                (:meth:`kitscenes.MapLabelSpec.labels` gives them in the spec's frame).
            **kwargs: Instead of *spec*, for one-off use: the arguments of
                :class:`MLConverter`, plus ``recipe``, a label recipe
                (:data:`kitscenes.map.LABEL_RECIPES`): ``"default"``, or
                ``"paper-v2.0.0"`` (``exact_cut=False``; needs lanelet2
                1.2.3+kitscenes3 and a scene opened with ``use_imu=False``).
                ``chaining="scene"`` builds the chains once from the map
                around all of the scene's poses (see *chaining*).

        Raises:
            ValueError: For a spec together with keywords, an unknown recipe,
                a scene the recipe refuses, or an ``exact_cut`` that
                contradicts a recipe other than ``"default"``.
            MissingDependencyError: If the recipe needs another lanelet2 build.
        """
        from kitscenes.map.recipes import check_label_recipe

        if spec is not None:
            if kwargs:
                raise ValueError(f"for_scene takes a MapLabelSpec or keywords, not both (got {sorted(kwargs)}); "
                                 "dataclasses.replace(spec, ...) changes a field of the spec")
            kwargs = {name: getattr(spec, name) for name in _SPEC_ARGUMENTS}
            for name in ("classes", "te_classes"):
                kwargs[name] = None if getattr(spec, name) is None else dict(getattr(spec, name))
        recipe = kwargs.pop("recipe", "default")
        settings = check_label_recipe(recipe, scene)
        exact = kwargs.setdefault("exact_cut", settings["exact_cut"])  # "default": exact_cut may be overridden
        if recipe != "default" and exact != settings["exact_cut"]:
            raise ValueError(f"label recipe {recipe!r} has exact_cut={settings['exact_cut']}, "
                             f"got exact_cut={kwargs['exact_cut']}")
        info = scene.split_info  # None for a scene outside any known split
        kwargs.setdefault("map_changed", info is not None and info.family == "map_change")
        if kwargs.get("max_height_diff") is not None and "road_poses" not in kwargs:
            kwargs["road_poses"] = scene._poses or None
        kwargs.setdefault("road_height_m", scene.road_height_m)
        if kwargs.get("chaining") == "scene":
            kwargs["chaining"] = scene_chain_area(scene, kwargs.get("region", Region()))
        return cls(scene.map, **kwargs)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_interface"] = state["_chain_data"] = state["_crossings"] = None
        state["_runs"], state["_run_lines"], state["_link_cache"] = {}, {}, {}
        return state

    # ------------------------------------------------------------------

    @property
    def _filter_heights(self) -> bool:
        return self.max_height_diff is not None

    def _config(self, ml, half_extents: tuple[float, float]):
        config = ml.MapDataInterface.Configuration()
        config.submapExtentLongitudinal, config.submapExtentLateral = half_extents
        config.ignoreMapElevation = self.ignore_elevation
        if hasattr(config, "allowLaneChangeAcrossBicycleLane"):
            config.allowLaneChangeAcrossBicycleLane = True  # lane-change edges across bike lanes, as in SceneMap
        else:
            _lanelet2.warn_no_bike_lane_lane_change("MLConverter")
        # The converter's own vertices: sampling is applied once, in Python (kitscenes.map.sampling.resample).
        config.resampleLanes, config.nPointsLanes = False, 0
        config.resampleTE, config.nPointsTE = False, 0
        grouping = self.grouping
        if isinstance(grouping, str):
            grouping = getattr(ml, _GROUPING_FUNCTIONS[grouping])()
        config.lineStringTypeGrouping = grouping
        config.teTypeGrouping = ml.getDefaultTETypeGrouping()
        return config

    def map_data(self, pose: "EgoPose"):
        """The converter's own ``MapData`` for *pose* (escape hatch; frame: the region's centre).

        With scene chaining it is one ``MapData`` for all poses, processed
        for this one (until the next call).

        Raises:
            RuntimeError: If the converter fails for this pose.
        """
        from lanelet2.core import BasicPoint2d, BasicPoint3d

        ml = require_ml_converter()
        centre, (yaw, pitch, roll), from2d = self._query(pose)
        if isinstance(self.chaining, Circle):
            if self._chain_data is None:
                self._chain_data = self._build_chain_data(ml)
            bbox = ml.getRotatedRect(BasicPoint3d(*(float(c) for c in centre)), *self.region.half_extents, yaw, from2d)
            self._chain_data.processAll(bbox, ml.ParametrizationType.LineString, False, 0, False, 0, pitch, roll)
            return self._chain_data
        if self._interface is None:
            self._interface = ml.MapDataInterface(self.scene_map.lanelet_map, self._config(ml, self.region.half_extents))
        if from2d:
            self._interface.setCurrPosAndExtractSubmap2d(BasicPoint2d(centre[0], centre[1]), yaw)
        else:
            self._interface.setCurrPosAndExtractSubmap(BasicPoint3d(*(float(c) for c in centre)), yaw, pitch, roll)
        return self._interface.mapData(True)

    def _build_chain_data(self, ml):
        """The unprocessed ``MapData`` of the map in the chaining circle (its chains)."""
        from lanelet2.core import BasicPoint3d

        area: Circle = self.chaining
        # The converter extracts the square of half side 1.1 * hypot(longitudinal, lateral) around a point.
        half = area.radius / (1.1 * np.sqrt(2.0))
        interface = ml.MapDataInterface(self.scene_map.lanelet_map, self._config(ml, (half, half)))
        interface.setCurrPosAndExtractSubmap(BasicPoint3d(area.center[0], area.center[1], 0.0), 0.0, 0.0, 0.0)
        return interface.mapData(False)

    def __call__(self, pose: "EgoPose") -> MapLabels:
        """The labels around *pose* (a map-frame pose, e.g. ``frame.pose``)."""
        ml = require_ml_converter()
        data = self.map_data(pose)
        points_2d = self.dims == 2 and not self._filter_heights
        tensor = data.getTensorInstanceData(points_2d, isinstance(self.chaining, Circle))
        to_ego = self._region_to_ego(pose)
        band = (-self.max_height_diff, self.max_height_diff) if self._filter_heights else None
        type_to_class = {t: name for name, types in self.classes.items() for t in types}

        # Lines: (class, ids, (compound order, arc position, sub-piece), points), and where each converter row went.
        found: list[tuple[str, tuple[int, ...], tuple, np.ndarray]] = []
        row_to_line: dict[tuple[str, int], int] = {}  # (converter type, row) -> index into found (first piece)
        for ls_type in ml.LineStringType.values.values():
            name = str(ls_type)
            cls = type_to_class.get(name)
            if cls is None or (self.crosswalks == "polygons" and name in CROSSING_TYPES):
                continue
            mats = tensor.compoundLineStringsOfType(ls_type)
            ids = [tuple(int(f.mapID) for f in tensor.pointMatrixCpdLineStrings(ls_type, i).features)
                   for i in range(len(mats))]
            positions = _piece_positions(tensor, ls_type, ids)
            for row, (m, row_ids) in enumerate(zip(mats, ids)):
                out, z = to_ego(m)
                pieces = [out] if band is None else [
                    p[:, :-1] for p in _vector.cut_to_band(np.column_stack([out, z]), z, *band)]
                pieces = [q for p in pieces for q in self._crop(p)]
                for sub, piece in enumerate(pieces):
                    if sub == 0:
                        row_to_line[(name, row)] = len(found)
                    found.append((cls, row_ids, (row if positions is None else positions[row][0],
                                                 0.0 if positions is None else positions[row][1], sub),
                                  resample(piece[:, :self.dims], self.sampling)))
        if self.crosswalks == "polygons":
            found += self._crossing_lines(pose, band, type_to_class, ml)

        lines: dict[str, list[np.ndarray]] = {name: [] for name in self.classes}
        line_ids: dict[str, list[tuple[int, ...]]] = {name: [] for name in self.classes}
        line_pieces: dict[str, list[int]] = {name: [] for name in self.classes}
        # Drop a line identical to an earlier one of its class (kept_as[k] is the line k stands for); pieces are
        # numbered among the kept lines.
        first: dict[tuple, int] = {}
        kept_as = [first.setdefault((cls, points.shape, points.tobytes()), k)
                   for k, (cls, _, _, points) in enumerate(found)]
        rank: dict[tuple[str, tuple[int, ...]], list[int]] = {}
        for k, (cls, ids, _, _) in enumerate(found):
            if kept_as[k] == k:
                rank.setdefault((cls, ids), []).append(k)
        piece_of = {}
        for members in rank.values():
            for piece, k in enumerate(sorted(members, key=lambda k: found[k][2])):
                piece_of[k] = piece
        index_in_class: list[int] = []
        position: dict[int, int] = {}  # index into found of a kept line -> its index in its class
        for k, (cls, ids, _, points) in enumerate(found):
            if kept_as[k] != k:
                index_in_class.append(position[kept_as[k]])
                continue
            position[k] = len(lines[cls])
            index_in_class.append(position[k])
            lines[cls].append(points)
            line_ids[cls].append(ids)
            line_pieces[cls].append(piece_of[k])

        traffic_elements: dict[str, list[np.ndarray]] = {name: [] for name in self.te_classes}
        traffic_element_ids: dict[str, list[int]] = {name: [] for name in self.te_classes}
        te_type_to_class = {t: name for name, types in self.te_classes.items() for t in types}
        te_row: dict[tuple[str, int], tuple[str, int]] = {}
        for te_type in ml.TEType.values.values():
            name = str(te_type)
            cls = te_type_to_class.get(name)
            if cls is None:
                continue
            mats = tensor.teInstancesOfType(te_type)
            if not mats:
                continue
            ids = _te_ids(data.validTEInstancesOfType(te_type), points_2d)
            if len(ids) != len(mats):  # the converter changed its ordering
                raise RuntimeError(f"cannot match {name} traffic elements to their map ids")
            for row, (m, map_id) in enumerate(zip(mats, ids)):
                te_row[(name, row)] = (cls, len(traffic_elements[cls]))
                traffic_elements[cls].append(resample(to_ego(m)[0][:, :self.dims], self.te_sampling))
                traffic_element_ids[cls].append(map_id)

        centerline_class = type_to_class.get("Centerline")
        te_to_centerline = []
        for t, i, j in tensor.teToCenterlineIndexEdges:
            te = te_row.get((str(t), int(i)))
            line = row_to_line.get(("Centerline", int(j)))
            if te is not None and line is not None and centerline_class is not None:
                te_to_centerline.append((te[0], te[1], index_in_class[line]))
        te_to_centerline = list(dict.fromkeys(te_to_centerline))  # unique: the edges of dropped lines repeat
        if self.exact_cut:  # in a fixed order; without the exact cut in the converter's order (the paper recipe)
            te_order = {name: i for i, name in enumerate(self.te_classes)}
            te_to_centerline.sort(key=lambda e: (te_order[e[0]], e[1], e[2]))
        te_to_te = []
        for a, i, b, j in tensor.teToTEIndexEdges:
            te_a, te_b = te_row.get((str(a), int(i))), te_row.get((str(b), int(j)))
            if te_a is not None and te_b is not None:
                te_to_te.append((te_a[0], te_a[1], te_b[0], te_b[1]))

        segments, segment_ids, segment_pieces, lane_to_lane = self._lane_segments(data, pose, band, ml)
        return MapLabels(
            lines=lines,
            line_ids=line_ids,
            line_pieces=line_pieces,
            traffic_elements=traffic_elements,
            traffic_element_ids=traffic_element_ids,
            te_to_centerline=te_to_centerline,
            te_to_te=te_to_te,
            pose=pose,
            region=self.region,
            dims=self.dims,
            map_changed=self.map_changed,
            lane_segments=segments,
            lane_segment_ids=segment_ids,
            lane_segment_pieces=segment_pieces,
            lane_to_lane=lane_to_lane,
            untyped_sign_ids=self._untyped({i for ids in traffic_element_ids.values() for i in ids}),
        )

    # ------------------------------------------------------------------

    def _untyped(self, ids: set) -> frozenset:
        """Those of the traffic element *ids* that are signs with subtype ``"unknown"``."""
        if not ids:
            return frozenset()
        if self._untyped_signs is None:
            self._untyped_signs = frozenset(s.id for s in self.scene_map.traffic_signs() if s.subtype == "unknown")
        return self._untyped_signs & ids

    def _crossing_lines(self, pose: "EgoPose", band, type_to_class: dict, ml) -> list:
        """Crosswalk polygons clipped by the region, as entries of ``found`` in :meth:`__call__`."""
        if self._crossings is None:
            zebra = {line.id for line in self.scene_map.lines(types=["zebra_marking"])}
            self._crossings = _vector.crossings(self.scene_map.lanes(kinds=["crosswalk"]), zebra)
        centre = self._query(pose)[0]
        half_x, half_y = self.region.half_extents
        reach = float(np.hypot(half_x, half_y))
        to_box, from_box = self._box_frame(pose)
        out = []
        for crossing in self._crossings:
            nearest = np.clip(centre[:2], crossing.lo, crossing.hi)
            if np.hypot(*(nearest - centre[:2])) > reach:
                continue
            clipped = _vector.clip_ring(to_box(crossing.ring), half_x, half_y)
            if len(clipped) < 3 or abs(_vector.signed_area(clipped)) < _vector.MIN_CROSSWALK_AREA:
                continue
            mapped, points = from_box(clipped)
            if band is not None:
                height = self._height_above_road(mapped, pose)
                if not band[0] <= float(height.mean()) <= band[1]:
                    continue
            points = self._crop(points, ring=True)
            if len(points) < 3 or abs(_vector.signed_area(points)) < _vector.MIN_CROSSWALK_AREA:
                continue
            kind = _crossing_type("ZebraCrossing" if crossing.zebra else "PedestrianCrossing", self.grouping, ml)
            cls = type_to_class.get(kind)
            if cls is None:
                continue
            out.append((cls, crossing.lane_ids, (np.inf, 0.0, 0),
                        resample(_vector.close_ring(points), self.sampling)))
        return out

    def _lane_segments(self, data, pose: "EgoPose", band, ml) -> tuple[list, list, list, list]:
        """``(lane_segments, lane_segment_ids, lane_segment_pieces, lane_to_lane)`` of :class:`MapLabels`.

        The lanelets come from every valid compound centerline instance (not the tensor rows, which leave out
        the lines that lanelet2 from 1.2.3+kitscenes3 on merges)."""
        kind = ml.LineStringType.Centerline
        lanelets = {int(f.mapID) for c in data.validCompoundLineStringsOfType(kind) for f in c.features}
        half_x, half_y = self.region.half_extents
        to_box, from_box = self._box_frame(pose)
        segments, ids, pieces = [], [], []
        span: dict[tuple[int, ...], tuple[int, int]] = {}  # run -> (its first segment, its last)
        lane_to_lane = []
        for run in sorted({self._run(i) for i in lanelets}):
            parts = []
            for clipped in _vector.clip_polyline(to_box(self._run_line(run)), half_x, half_y):
                mapped, points = from_box(clipped)
                if band is None:
                    parts += self._crop(points)
                else:
                    z = self._height_above_road(mapped, pose)
                    parts += [q for p in _vector.cut_to_band(np.column_stack([points, z]), z, *band)
                              for q in self._crop(p[:, :-1])]
            if not parts:
                continue
            span[run] = (len(segments), len(segments) + len(parts) - 1)
            for piece, points in enumerate(parts):
                if piece:
                    lane_to_lane.append((len(segments) - 1, len(segments)))
                segments.append(resample(points, self.sampling))
                ids.append(run)
                pieces.append(piece)
        for run, (_, last) in span.items():
            for successor in self._links(run[-1])[0]:
                nxt = self._runs.get(successor)
                if nxt in span and nxt[0] == successor:
                    lane_to_lane.append((last, span[nxt][0]))
        return segments, ids, pieces, sorted(lane_to_lane)

    def _run(self, lanelet: int) -> tuple[int, ...]:
        """The lane run through *lanelet*: as far as each lanelet has one successor and that one one predecessor."""
        run = self._runs.get(lanelet)
        if run is not None:
            return run
        def joined(a: int, b: int) -> bool:
            return self._links(a)[0] == (b,) and self._links(b)[1] == (a,)

        start, seen = lanelet, {lanelet}
        while len(self._links(start)[1]) == 1:
            prev = self._links(start)[1][0]
            if prev in seen or not joined(prev, start):
                break
            start = prev
            seen.add(start)
        members = [start]
        while len(self._links(members[-1])[0]) == 1:
            nxt = self._links(members[-1])[0][0]
            if nxt in members or not joined(members[-1], nxt):
                break
            members.append(nxt)
        run = tuple(members)
        for member in run:
            self._runs[member] = run
        return run

    def _links(self, lanelet: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """``(successors, predecessors)`` of *lanelet* in its own direction (no lane changes).

        A bidirectional lanelet's other direction is left out: the routing
        graph returns it inverted, which ``Lane.successors`` cannot tell apart.
        """
        links = self._link_cache.get(lanelet)
        if links is None:
            graph, ll = self.scene_map.routing_graph, self.scene_map.lanelet_map.laneletLayer[lanelet]
            links = (tuple(x.id for x in graph.following(ll) if not x.inverted()),
                     tuple(x.id for x in graph.previous(ll) if not x.inverted()))
            self._link_cache[lanelet] = links
        return links

    def _run_line(self, run: tuple[int, ...]) -> np.ndarray:
        """The centerline of a lane run (map frame, driving direction)."""
        line = self._run_lines.get(run)
        if line is None:
            parts = [self.scene_map.lane(i).centerline for i in run]
            line = np.concatenate([parts[0]] + [p[1:] if np.allclose(p[0], q[-1], atol=1e-3) else p
                                                for q, p in zip(parts, parts[1:])])
            self._run_lines[run] = line
        return line

    def _crop(self, points: np.ndarray, ring: bool = False):
        """Label-frame *points* cut exactly to the region in x-y: the pieces of a line, or a ring (open).

        Without *exact_cut*: *points* as they are (the converter's cut)."""
        if not self.exact_cut:
            return points if ring else [points]
        hx, hy = self.region.half_extents
        centre = np.zeros(points.shape[1])
        centre[:2] = self.region.centre
        local = points - centre
        if (np.abs(local[:, 0]) <= hx).all() and (np.abs(local[:, 1]) <= hy).all():
            return points if ring else [points]
        if ring:
            return _vector.clip_ring(local, hx, hy) + centre
        return [p + centre for p in _vector.clip_polyline(local, hx, hy)]

    def _box_frame(self, pose: "EgoPose") -> tuple[Callable, Callable]:
        """``(to_box, from_box)``: map-frame points (N, 3) into the frame of the converter's cut rectangle (its
        centre, turned by its yaw; z kept), and points of that frame back to ``(map frame, label frame)``."""
        centre, (yaw, _, _), _ = self._query(pose)
        c, s = np.cos(yaw), np.sin(yaw)
        R = _rotation(pose)
        origin = np.array([pose.xyz[0], pose.xyz[1], 0.0 if self.ignore_elevation else pose.xyz[2]])
        ch, sh = np.cos(pose.yaw), np.sin(pose.yaw)

        def to_box(points: np.ndarray) -> np.ndarray:
            d = np.asarray(points, dtype=np.float64)[:, :2] - centre[:2]
            z = 0.0 if self.ignore_elevation else np.asarray(points, dtype=np.float64)[:, 2]
            return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], np.broadcast_to(z, len(d))])

        def from_box(local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mapped = np.column_stack([centre[0] + c * local[:, 0] - s * local[:, 1],
                                      centre[1] + s * local[:, 0] + c * local[:, 1], local[:, 2]])
            if self.dims == 3:
                return mapped, (mapped - origin) @ R
            dm = mapped[:, :2] - pose.xyz[:2]
            return mapped, np.column_stack([ch * dm[:, 0] + sh * dm[:, 1], -sh * dm[:, 0] + ch * dm[:, 1]])

        return to_box, from_box

    def _query(self, pose: "EgoPose"):
        """``(centre in the map frame, (yaw, pitch, roll), from2d)`` the converter is asked for."""
        if self.dims == 2:
            yaw, pitch, roll = _heading(pose)
            from2d = not self._filter_heights  # the 3D box at the heading keeps z for the height filter
        else:
            yaw, pitch, roll = _converter_angles(pose)
            from2d = False
        dx, dy = self.region.centre
        c, s = np.cos(yaw), np.sin(yaw)
        x, y, z = np.asarray(pose.xyz, dtype=np.float64)
        if self.ignore_elevation:
            z = 0.0
        # The converter's region is a rectangle in the map's ground plane, turned by the yaw only.
        return np.array([x + c * dx - s * dy, y + s * dx + c * dy, z]), (yaw, pitch, roll), from2d

    def _region_to_ego(self, pose: "EgoPose") -> Callable[[np.ndarray], tuple[np.ndarray, Optional[np.ndarray]]]:
        """Converter output (frame at the region's centre) -> ``(ego frame points, height above the road or None)``.

        The height is only computed with *max_height_diff*.
        """
        dx, dy = self.region.centre
        R, t = _rotation(pose), np.asarray(pose.xyz, dtype=np.float64)
        if self.dims == 3:
            if dx == 0 and dy == 0:
                offset = np.zeros(3)
            else:
                # centre - pose = Rz(yaw) (dx, dy, 0), yaw as the converter got it; into the ego frame with R^T.
                yaw = _converter_angles(pose)[0]
                c, s = np.cos(yaw), np.sin(yaw)
                offset = R.T @ np.array([c * dx - s * dy, s * dx + c * dy, 0.0])

            def to_ego(m):
                ego = np.asarray(m, dtype=np.float64) + offset
                return ego, self._height_above_road(ego @ R.T + t, pose) if self._filter_heights else None

            return to_ego
        if not self._filter_heights:
            offset2 = np.array([dx, dy])
            return lambda m: (np.asarray(m, dtype=np.float64) + offset2, None)
        # 2D with heights: the converter's 3D output at the heading (no pitch or roll), x-y as in its 2D mode.
        c, s = np.cos(pose.yaw), np.sin(pose.yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        offset3 = np.array([dx, dy, 0.0])

        def to_heading(m):
            h = np.asarray(m, dtype=np.float64) + offset3
            return h, self._height_above_road(h @ Rz.T + t, pose)

        return to_heading

    def _height_above_road(self, points: np.ndarray, pose: "EgoPose") -> np.ndarray:
        """Height (m) of map-frame *points* (N, 3) above the road plane under the nearest road pose."""
        road = self._road if self._road is not None else _RoadLevel.of([pose], self.road_height_m)
        return road.height_above(points)


@dataclasses.dataclass
class _RoadLevel:
    """The road along a drive: the road plane under each pose (*road_height_m* below its base frame)."""

    ground: np.ndarray  # (K, 3) map-frame point on the road under each pose
    normal: np.ndarray  # (K, 3) the base frame's z axis at each pose
    _tree: object = None

    @classmethod
    def of(cls, poses: Sequence["EgoPose"], road_height_m: float) -> "_RoadLevel":
        from scipy.spatial.transform import Rotation

        xyz = np.array([p.xyz for p in poses], dtype=np.float64).reshape(-1, 3)
        quat = np.array([p.quat for p in poses], dtype=np.float64).reshape(-1, 4)
        keep = np.isfinite(xyz).all(axis=1) & np.isfinite(quat).all(axis=1)
        if not keep.any():
            raise ValueError("road_poses has no finite pose")
        R = Rotation.from_quat(quat[keep]).as_matrix()
        return cls(xyz[keep] - road_height_m * R[:, :, 2], R[:, :, 2])

    def __getstate__(self) -> dict:
        return {"ground": self.ground, "normal": self.normal, "_tree": None}

    def height_above(self, points: np.ndarray) -> np.ndarray:
        if len(self.ground) == 1:
            k = np.zeros(len(points), dtype=np.int64)
        else:
            if self._tree is None:
                from scipy.spatial import cKDTree

                self._tree = cKDTree(self.ground)
            k = self._tree.query(points)[1]
        return np.einsum("ij,ij->i", points - self.ground[k], self.normal[k])


def scene_chain_area(scene: "Scene", region: Region = Region()) -> Circle:
    """The map area whose chains cover every pose of *scene* (``chaining="scene"``).

    A circle around the region centres of the scene's poses, wide enough
    that it holds the map the converter would take around each of them.

    Raises:
        ValueError: If the scene has no poses.
    """
    poses = scene._poses
    if not poses:
        raise ValueError(f"scene {scene.id} has no poses to chain its map around")
    xy = np.array([p.xyz[:2] for p in poses], dtype=np.float64)
    yaw = np.array([p.yaw for p in poses], dtype=np.float64)
    dx, dy = region.centre
    centres = xy + np.column_stack([np.cos(yaw) * dx - np.sin(yaw) * dy, np.sin(yaw) * dx + np.cos(yaw) * dy])
    middle = (centres.min(axis=0) + centres.max(axis=0)) / 2
    per_pose = 1.1 * float(np.hypot(*region.half_extents)) * np.sqrt(2.0)  # half diagonal of one pose's map square
    radius = float(np.linalg.norm(centres - middle, axis=1).max()) + per_pose + 1.0
    return Circle((float(middle[0]), float(middle[1])), radius)


def _grouping_types(grouping) -> tuple[str, ...]:
    """The types a grouping labels with (:data:`GROUPING_CLASSES`, or a converter grouping's representatives)."""
    if isinstance(grouping, str):
        return GROUPING_CLASSES[grouping]
    return tuple(dict.fromkeys(str(pair.representative) for pair in grouping))


def _crossing_type(kind: str, grouping, ml) -> str:
    """The class a crosswalk of converter type *kind* gets in *grouping*."""
    if isinstance(grouping, str):
        return "PedestrianCrossing" if grouping in ("maptr", "m3tr") else kind
    return str(ml.getLineStringTypeRepresentative(getattr(ml.LineStringType, kind), grouping))


def _piece_positions(tensor, ls_type, ids: list[tuple[int, ...]]) -> Optional[list[tuple[int, float]]]:
    """``(compound, arc position on its uncut chain)`` of each row, for chains the region cut into pieces.

    Rows of one compound are consecutive. None where no compound has more
    than one piece (its rows keep the converter's order).
    """
    n = len(ids)
    if n < 2 or all(ids[i] != ids[i + 1] for i in range(n - 1)):
        return None
    out: list[tuple[int, float]] = []
    row = 0
    while row < n:
        end = row + 1
        while end < n and ids[end] == ids[row]:
            end += 1
        if end - row == 1:
            out.append((row, 0.0))
        else:
            compound = tensor.pointMatrixCpdLineStrings(ls_type, row)
            raw = np.array([[p.x, p.y] for p in compound.rawInstance])
            cuts = compound.cutInstance
            if len(cuts) == end - row and len(raw) >= 2:
                out += [(row, _vector.arc_position(raw, np.array([cut[0].x, cut[0].y]))) for cut in cuts]
            else:  # two compounds with the same members next to each other: keep the converter's order
                out += [(k, 0.0) for k in range(row, end)]
        row = end
    return out


def _rotation(pose: "EgoPose") -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(np.asarray(pose.quat, dtype=np.float64)).as_matrix()


def _te_ids(instances: dict, points_in_2d: bool) -> list[int]:
    """Map ids in the converter's tensor order: by id, one entry per piece of a cut element."""
    ids: list[int] = []
    for map_id in sorted(instances):
        pieces = len(instances[map_id].computeInstanceVectors(True, points_in_2d))
        ids += [int(map_id)] * pieces
    return ids


def _heading(pose: "EgoPose") -> tuple[float, float, float]:
    """``(yaw, 0, 0)``: the heading, for the 2D mode."""
    return pose.yaw, 0.0, 0.0


def _converter_angles(pose: "EgoPose") -> tuple[float, float, float]:
    """``(yaw, pitch, roll)`` of the intrinsic x-y-z decomposition: the converter applies
    Rz(-yaw) Ry(-pitch) Rx(-roll), the inverse of Rx Ry Rz."""
    from scipy.spatial.transform import Rotation

    roll, pitch, yaw = Rotation.from_quat(np.asarray(pose.quat, dtype=np.float64)).as_euler("XYZ")
    return float(yaw), float(pitch), float(roll)
