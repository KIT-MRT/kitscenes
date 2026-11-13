"""HD maps (Lanelet2) as typed numpy elements.

Importing this package does not import lanelet2; loading a map does
(``pip install "kitscenes[map]"``). Frames, units and conventions:

- **map frame**: local UTM around ``maps/origin.json`` (:class:`MapOrigin`);
  the frame of ``poses.txt`` in scenes with maps. Metres; z is the height
  above the ellipsoid, NaN where unknown.
- **ego frame**: the vehicle base frame at a pose (``MapPatch.in_ego``);
  its origin is about 2 m above the road.
- Speeds in m/s, angles in radians.

Example::

    from kitscenes.map import Circle, MLConverter, Region

    smap = scene.map
    pose = frame.pose
    region = Region(ahead=60, behind=20, left=20, right=20)   # around the vehicle
    lanes = smap.lanes(region.around(pose))                    # a Box in the map frame
    patch = smap.query(Circle.around(pose, 50.0)).in_ego(pose)
    matches = smap.match(pose)

    labels = MLConverter.for_scene(scene, region=region)(pose)   # ML labels (lanelet2.ml_converter), ego frame
    maptr = kitscenes.MapLabelSpec.maptr().converter(scene)(pose)  # the MapTR classes: divider, ped_crossing, boundary
"""

from kitscenes.map._lanelet2 import (
    LANELET2_BUILD, LANELET2_MIN_VERSION, LANELET2_PAPER_BUILD, MissingDependencyError, has_ml_converter,
)
from kitscenes.map.labels import (
    GROUPING_CLASSES, GROUPINGS, MAPTR_CLASSES, MAPTRV2_CLASSES, TE_TYPES, MapLabels, MLConverter,
)
from kitscenes.map.recipes import LABEL_RECIPES, check_label_recipe
from kitscenes.map.sampling import Adaptive, FixedPoints, Spacing
from kitscenes.map.elements import Lane, LaneKind, LaneMatch, MapLine, StopLine, TrafficLight, TrafficSign
from kitscenes.map.origin import MapOrigin
from kitscenes.map.projection import project_to_camera
from kitscenes.map.roi import Box, Circle, Region
from kitscenes.map.scene_map import MapPatch, MapProblemWarning, SceneMap

__all__ = [
    "SceneMap",
    "MapPatch",
    "MapProblemWarning",
    "MapOrigin",
    "Lane",
    "LaneKind",
    "LaneMatch",
    "MapLine",
    "StopLine",
    "TrafficSign",
    "TrafficLight",
    "Box",
    "Circle",
    "project_to_camera",
    "MLConverter",
    "MapLabels",
    "Region",
    "GROUPINGS",
    "GROUPING_CLASSES",
    "TE_TYPES",
    "MAPTR_CLASSES",
    "LABEL_RECIPES",
    "check_label_recipe",
    "FixedPoints",
    "Spacing",
    "Adaptive",
    "has_ml_converter",
    "MissingDependencyError",
    "LANELET2_BUILD",
    "LANELET2_PAPER_BUILD",
]
