"""A small synthetic OpenStreetMap file for the SD-map tests, laid out in metres around Karlsruhe's city origin.

Layout (map frame, x east, y north; a few hundred metres each)::

    crossroads at (300, 200), traffic signals          via-way gadget at x ~ 2000
                                                       (no_right_turn A -> via V -> B)
          6 (300,400) ---14--- 7 (500,400)                  103 --B-- 104 --R-- 105
          |                    ^ 13 one-way north            |V      /G
          12 (zone 30)         |                       101 -A- 102 -A2- 106
          |                    |
    1 ---10 (primary)--- 2 --- 3 ------------ 4        dual carriageway at x ~ 1000
    (-100,200)           |     |  (500,200)            (two one-ways 10 m apart, merged junction)
                         15    11
                         (private service, dead end)
                               5 (300,0), a dead end

- way 10: ``primary``, ``maxspeed=50``, ``lanes=4``, ``turn:lanes:forward``, its name double-encoded like the release files
- relation 90: ``no_left_turn`` from 10 via node 3 to 12 (eastbound 10 may not turn north)
- relation 190: ``no_right_turn`` from A via way V to B
- relation 91: a bus route over 10 and 12 with node 3 as a stop
- node 40: a tagged crossing; way 16: a building; way 17: a footway; nodes 50, 51 span the area
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from xml.sax.saxutils import quoteattr

from kitscenes.map.origin import CITY_ORIGINS, MapOrigin

ORIGIN: MapOrigin = CITY_ORIGINS["Karlsruhe"]
OSM_BASE = "2026-05-03T15:30:44Z"
#: The name of way 10 as the release files store such text (UTF-8 read as cp1252, encoded again).
MOJIBAKE_NAME = "Kaiserstraße".encode("utf-8").decode("cp1252")

NODES: dict[int, tuple[float, float, dict]] = {
    1: (-100, 200, {}), 2: (100, 200, {}), 3: (300, 200, {"highway": "traffic_signals"}), 4: (500, 200, {}),
    5: (300, 0, {}), 6: (300, 400, {}), 7: (500, 400, {}), 8: (100, 100, {}),
    20: (150, 250, {}), 21: (170, 250, {}), 22: (170, 270, {}), 23: (150, 270, {}),
    30: (0, 215, {}), 31: (200, 215, {}),
    40: (200, 195, {"highway": "crossing", "crossing": "traffic_signals"}),
    41: (250, 250, {"amenity": "bench"}),
    # untagged nodes of no way: the corners of the "download box" (SDMap.covers)
    50: (-200, -150, {}), 51: (2300, 450, {}),
    # dual carriageway
    201: (900, 0, {}), 202: (1000, 0, {}), 203: (1100, 0, {}),
    204: (1100, 10, {}), 205: (1000, 10, {}), 206: (900, 10, {}),
    207: (1000, -100, {}), 208: (1000, 110, {}),
    # via-way gadget
    101: (1900, 0, {}), 102: (2000, 0, {}), 103: (2000, 50, {}), 104: (2100, 50, {}), 105: (2200, 50, {}),
    106: (2100, 0, {}), 107: (2150, 25, {}),
}

WAYS: list[tuple[int, list[int], dict]] = [
    (10, [1, 2, 3, 4], {"highway": "primary", "name": MOJIBAKE_NAME, "maxspeed": "50", "lanes": "4",
                        "turn:lanes:forward": "left|through|through;right", "surface": "asphalt"}),
    (11, [5, 3], {"highway": "residential", "source:maxspeed": "DE:zone30"}),
    (12, [3, 6], {"highway": "residential", "zone:maxspeed": "DE:30"}),
    (13, [4, 7], {"highway": "secondary", "oneway": "yes", "maxspeed": "70", "lanes": "2"}),
    (14, [6, 7], {"highway": "tertiary", "maxspeed": "DE:urban"}),
    (15, [2, 8], {"highway": "service", "access": "private"}),
    (16, [20, 21, 22, 23, 20], {"building": "yes"}),
    (17, [30, 31], {"highway": "footway"}),
    (210, [201, 202, 203], {"highway": "primary", "oneway": "yes"}),
    (211, [204, 205, 206], {"highway": "primary", "oneway": "yes"}),
    (212, [207, 202, 205, 208], {"highway": "secondary"}),
    (110, [101, 102], {"highway": "residential"}),
    (111, [102, 103], {"highway": "residential"}),
    (112, [103, 104], {"highway": "residential"}),
    (113, [104, 105], {"highway": "residential"}),
    (114, [102, 106], {"highway": "residential"}),
    (115, [106, 107, 104], {"highway": "residential"}),
]

RELATIONS: list[tuple[int, list[tuple[str, int, str]], dict]] = [
    (90, [("way", 10, "from"), ("node", 3, "via"), ("way", 12, "to")],
     {"type": "restriction", "restriction": "no_left_turn"}),
    (91, [("way", 10, ""), ("way", 12, ""), ("node", 3, "stop")], {"type": "route", "route": "bus", "ref": "62"}),
    (190, [("way", 110, "from"), ("way", 111, "via"), ("way", 112, "to")],
     {"type": "restriction", "restriction": "no_right_turn"}),
]


def write_osm(path: Path, origin: MapOrigin = ORIGIN, *, extra_nodes: Optional[dict] = None,
              correct_utf8: bool = False, osm_base: Optional[str] = OSM_BASE) -> Path:
    """The fixture as OpenStreetMap XML (as Overpass writes it, without contributor metadata).

    *correct_utf8* writes the name of way 10 correctly (as the weekly snapshots have it) instead of
    double-encoded (as the 2026-05-03 files); *osm_base* None leaves out the ``<meta>``.
    """
    nodes = dict(NODES)
    nodes.update(extra_nodes or {})
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="kitscenes test">']
    if osm_base is not None:
        lines.append(f'<meta osm_base="{osm_base}"/>')
    for node_id, (x, y, tags) in nodes.items():
        lat, lon = origin.to_latlon([[x, y]])[0]
        attrs = f'id="{node_id}" lat="{lat:.9f}" lon="{lon:.9f}" version="1"'
        if not tags:
            lines.append(f"  <node {attrs}/>")
            continue
        lines.append(f"  <node {attrs}>")
        lines += [f"    <tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in tags.items()]
        lines.append("  </node>")
    for way_id, refs, tags in WAYS:
        if correct_utf8 and "name" in tags:
            tags = {**tags, "name": tags["name"].encode("cp1252").decode("utf-8")}
        lines.append(f'  <way id="{way_id}" version="1">')
        lines += [f'    <nd ref="{r}"/>' for r in refs]
        lines += [f"    <tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in tags.items()]
        lines.append("  </way>")
    for rel_id, members, tags in RELATIONS:
        lines.append(f'  <relation id="{rel_id}" version="1">')
        lines += [f'    <member type="{t}" ref="{r}" role="{role}"/>' for t, r, role in members]
        lines += [f"    <tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in tags.items()]
        lines.append("  </relation>")
    lines.append("</osm>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
