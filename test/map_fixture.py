"""A synthetic lanelet2 map with the cases the release maps have, and poses on it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kitscenes.map import MapOrigin
from kitscenes.schema import EgoPose
from release_fixture import make_scene, write_poses

ORIGIN = MapOrigin(49.0144, 8.4172)
FLT_MAX = "-3.40282e+38"
NORTH, SOUTH = np.pi / 2, -np.pi / 2


def write_osm(path: Path) -> Path:
    """Local-frame layout (metres, x east, y north):

    - 100: road northbound, x 0..3.5, y 0..40 (urban default limit); 104 continues it to y 60
    - 101: road southbound, x 3.5..7, speed_limit=30
    - 102: bicycle lane northbound, x -1.5..0
    - 103: crosswalk across all at y 20..23
    - stop line 8 at y 18 under two traffic-light elements (200 on 100, 201 on 101)
    - light 9 (with height), sign 10 de274-30 without height, sign 11 'unknown'
    """
    nodes: dict[int, tuple[float, float, str]] = {}
    ways: list[tuple[int, list[int], dict[str, str]]] = []
    next_node = [1000]

    def node(x: float, y: float, ele: str = "100.0") -> int:
        next_node[0] += 1
        nodes[next_node[0]] = (x, y, ele)
        return next_node[0]

    def way(wid: int, pts: list[tuple[float, float]], tags: dict[str, str], ele: str = "100.0") -> list[int]:
        ids = [node(x, y, ele) for x, y in pts]
        ways.append((wid, ids, tags))
        return ids

    line = {"type": "line_thin", "subtype": "solid"}
    a = way(1, [(0, 0), (0, 40)], line)
    b = way(2, [(3.5, 0), (3.5, 40)], {"type": "line_thin", "subtype": "dashed"})
    ways.append((4, b[::-1], {"type": "line_thin", "subtype": "dashed"}))  # same nodes, southbound
    way(3, [(7, 40), (7, 0)], {"type": "curbstone", "subtype": "high", "drivable_space_border": "true"})
    way(5, [(-1.5, 0), (-1.5, 40)], {"type": "curbstone", "subtype": "high"})
    way(6, [(-1.5, 23), (7, 23)], {"type": "zebra_marking"})
    way(7, [(-1.5, 20), (7, 20)], {"type": "zebra_marking"})
    way(8, [(0, 18), (3.5, 18)], {"type": "stop_line"})
    way(9, [(8, 19), (8, 19.3)], {"type": "traffic_light", "subtype": "red_yellow_green"}, ele="105.0")
    way(10, [(8.5, 10), (8.5, 10.6)], {"type": "traffic_sign", "subtype": "de274-30"}, ele=FLT_MAX)
    way(11, [(-3, 30), (-3, 30.6)], {"type": "traffic_sign", "subtype": "unknown"}, ele="104.0")
    # 104 continues 100: share the end nodes of ways 1 and 2
    ways.append((12, [a[-1], node(0, 60)], line))
    ways.append((13, [b[-1], node(3.5, 60)], line))

    latlon = {nid: ORIGIN.to_latlon(np.array([[x, y]]))[0] for nid, (x, y, _) in nodes.items()}
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6">']
    for nid, (x, y, ele) in nodes.items():
        lat, lon = latlon[nid]
        out.append(f'  <node id="{nid}" lat="{lat:.10f}" lon="{lon:.10f}"><tag k="ele" v="{ele}"/></node>')
    for wid, ids, tags in ways:
        out.append(f'  <way id="{wid}">' + "".join(f'<nd ref="{i}"/>' for i in ids)
                   + "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items()) + "</way>")

    def lanelet(rid, left, right, tags, regs=()):
        members = f'<member type="way" ref="{left}" role="left"/><member type="way" ref="{right}" role="right"/>'
        members += "".join(f'<member type="relation" ref="{r}" role="regulatory_element"/>' for r in regs)
        tags = {"type": "lanelet", "location": "urban", **tags}
        out.append(f'  <relation id="{rid}">{members}' + "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items()) + "</relation>")

    for rid in (200, 201):
        out.append(f'  <relation id="{rid}"><member type="way" ref="9" role="refers"/><member type="way" ref="8" role="ref_line"/>'
                   '<tag k="type" v="regulatory_element"/><tag k="subtype" v="traffic_light"/></relation>')
    lanelet(100, 1, 2, {"subtype": "road", "one_way": "yes"}, regs=(200,))
    lanelet(101, 3, 4, {"subtype": "road", "one_way": "yes", "speed_limit": "30"}, regs=(201,))
    lanelet(102, 5, 1, {"subtype": "bicycle_lane", "one_way": "yes"})
    lanelet(103, 6, 7, {"subtype": "crosswalk"})
    lanelet(104, 12, 13, {"subtype": "road", "one_way": "yes"})
    out.append("</osm>")
    path.write_text("\n".join(out))
    return path


def write_bike_lane_osm(path: Path) -> Path:
    """Northbound road 300 (x 0..3.5) | bicycle lane 301 (x 3.5..5) | road 302 (x 5..8.5), y 0..40.

    Both bike-lane bounds are dashed bike markings, so a vehicle may cross the bike lane.
    """
    nodes, ways, rels = [], [], []
    for i, x in enumerate((0.0, 3.5, 5.0, 8.5)):
        for j, y in enumerate((0.0, 40.0)):
            lat, lon = ORIGIN.to_latlon(np.array([[x, y]]))[0]
            nodes.append(f'  <node id="{10 + 10 * j + i}" lat="{lat:.10f}" lon="{lon:.10f}"><tag k="ele" v="100.0"/></node>')
    tags = {1: ("road_border", None), 2: ("bike_marking", "dashed"), 3: ("bike_marking", "dashed"), 4: ("road_border", None)}
    for wid, (typ, sub) in tags.items():
        tag = f'<tag k="type" v="{typ}"/>' + (f'<tag k="subtype" v="{sub}"/>' if sub else "")
        ways.append(f'  <way id="{wid}"><nd ref="{9 + wid}"/><nd ref="{19 + wid}"/>{tag}</way>')
    for rid, left, right, subtype in ((300, 1, 2, "road"), (301, 2, 3, "bicycle_lane"), (302, 3, 4, "road")):
        # lanelet2 lanelets have the left bound on the left: northbound, so left = west = smaller x
        rels.append(f'  <relation id="{rid}"><member type="way" ref="{left}" role="left"/>'
                    f'<member type="way" ref="{right}" role="right"/><tag k="type" v="lanelet"/>'
                    f'<tag k="subtype" v="{subtype}"/><tag k="location" v="urban"/><tag k="one_way" v="yes"/></relation>')
    path.write_text("\n".join(['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6">', *nodes, *ways, *rels, "</osm>"]))
    return path


def ego_pose(x: float, y: float, yaw: float, pitch: float = 0.0, roll: float = 0.0, z: float = 102.0) -> EgoPose:
    """An ego pose at (x, y, z) with the given yaw, pitch and roll in radians."""
    from scipy.spatial.transform import Rotation

    return EgoPose(0, np.array([x, y, z]), Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat())


def map_root(tmp_path: Path) -> Path:
    """A root with a train scene on the release fixture's map and a test scene without a map."""
    train = make_scene(tmp_path / "data" / "train" / "with_map", n=3, maps=True)
    write_poses(train, 3, speed=1.0, origin=(1.75, 2.0, 2.0))  # on the fixture map's road, 2 m above it
    make_scene(tmp_path / "data" / "test" / "no_map", n=2)
    return tmp_path
