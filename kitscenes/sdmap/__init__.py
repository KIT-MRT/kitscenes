"""SD (standard-definition) maps from OpenStreetMap: raw elements, a road graph, matching, routing, navigation.

Separate from the HD map (:mod:`kitscenes.map`, Lanelet2): every SD class has
the ``SD`` prefix (:class:`SDMap`, :class:`SDRoad`, :class:`SDRoute`, ...).
The data are the OpenStreetMap snapshots of the release (© OpenStreetMap
contributors, ODbL 1.0): one per city and recording week,
``openstreetmap_sd_maps/<City>_<YYYY>-W<ww>.osm.pbf`` (each scene uses its
own, :attr:`kitscenes.splits.SceneInfo.sd_snapshot`), and the 2026-05-03 files
of the paper, ``openstreetmap_sd_maps/paper_2026-05-03/<City>.osm``; :meth:`kitscenes.Dataset.sd_maps` lists them.

Frames, units and conventions:

- **map frame**: the local UTM frame around the map's origin, the frame of
  ``poses.txt`` (every scene of a city shares its city's origin, so one
  :class:`SDMap` per snapshot file serves all scenes that use it, test and
  map_change_test included). OpenStreetMap is 2D: roads and routes are (N, 2).
- **ego frame**: the vehicle's base frame at a pose (``in_ego(pose)``);
  SD elements are put on the road, ``road_height_m`` below the base origin.
- Metres, m/s, radians (left / counter-clockwise positive); ``speed_limit_kph`` gives km/h.
- An empty result (no elements, no road, no match) is a normal answer, e.g.
  outside the snapshot's area (:meth:`SDMap.covers`), never an exception.

Example::

    import kitscenes
    from kitscenes.map import Region
    from kitscenes.navigation import KITSCENES, NAVSIM

    smap = scene.sd_map                                    # its week's snapshot; or ds.sd_map("Karlsruhe", snapshot="2026-W06")
    pose = frame.pose
    elements = smap.elements(Region(60, 60, 30, 30).around(pose)).in_ego(pose)   # raw OSM, all tags
    road = smap.attributes_at(pose)                        # SDRoad: speed_limit_mps (_kph), lanes, road_class, ...
    route = smap.route(scene.track.pose(0), scene.track.pose(-1))   # SDRoute (turn restrictions on)
    route.command(pose, NAVSIM), route.next_maneuver(pose), route.target_points(pose)
    driven = scene.sd_route                                # the driven route (HMM matching), cached
    frame.navigation(KITSCENES)                            # command + target points of this frame

OpenStreetMap data is under the ODbL: anything derived from it that you
publish (graphs, crops with tags, per-frame attributes) must be under the
ODbL too; results produced with it (models, figures, papers) need only the
attribution :data:`ATTRIBUTION`. The parsed cache stays on your machine.
"""

from kitscenes.sdmap.attributes import RoadClass
from kitscenes.sdmap.elements import (
    SDElements, SDJunction, SDMatch, SDRelation, SDRoad, SDTrack, SDTurnRestriction,
)
from kitscenes.sdmap.route import Direction, SDManeuver, SDRoute
from kitscenes.sdmap.sd_map import ATTRIBUTION, SDMap

__all__ = [
    "SDMap",
    "SDElements",
    "SDRelation",
    "SDRoad",
    "SDJunction",
    "SDTurnRestriction",
    "SDMatch",
    "SDTrack",
    "SDRoute",
    "SDManeuver",
    "Direction",
    "RoadClass",
    "ATTRIBUTION",
]
