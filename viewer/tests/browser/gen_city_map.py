#!/usr/bin/env python3
"""A synthetic city-size Lanelet2 map for perf_viewer.mjs, about Karlsruhe's size.

A street grid every 100 m. Each 100 m street segment carries five ways
(two borders, two lane edges, a dashed centre line) and two lanelets; the
borders vary between high and low curbs, guard rails, walls and fences, so
every raised shape is in it. Gently rolling heights, so 3D matters.
Deterministic:   python3 gen_city_map.py out.osm [--grid 120]
"""
import argparse
import math

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--grid", type=int, default=120, help="intersections per side (120: ~144k ways)")
args = p.parse_args()

LAT0, LON0 = 49.0, 8.4
R = 6378137.0
SPACING = 100.0
NODES_PER_WAY = 4
BORDERS = [("curbstone", "high"), ("curbstone", "low"), ("guard_rail", None), ("curbstone", "high"),
           ("wall", None), ("curbstone", "high"), ("fence", None)]

def latlon(x, y):
    return (LAT0 + math.degrees(y / R), LON0 + math.degrees(x / (R * math.cos(math.radians(LAT0)))))

def height(x, y):
    return 110.0 + 6.0 * math.sin(x / 700.0) * math.cos(y / 900.0)

nid = 0
wid = 0
rid = 0
nodes, ways, rels = [], [], []

def add_way(points, tags):
    global nid, wid
    refs = []
    for x, y in points:
        nid += 1
        lat, lon = latlon(x, y)
        # One element per line, as JOSM writes it (perf_viewer.mjs reads lines).
        nodes.append('  <node id="%d" lat="%.9f" lon="%.9f" version="1">\n    <tag k="ele" v="%.3f"/>\n  </node>'
                     % (nid, lat, lon, height(x, y)))
        refs.append(nid)
    wid += 1
    body = "".join('\n    <nd ref="%d"/>' % r for r in refs) + "".join(
        '\n    <tag k="%s" v="%s"/>' % kv for kv in tags.items())
    ways.append('  <way id="%d" version="1">%s\n  </way>' % (wid, body))
    return wid

def add_lanelet(left, right):
    global rid
    rid += 1
    rels.append('  <relation id="%d" version="1">\n    <member type="way" ref="%d" role="left"/>\n'
                '    <member type="way" ref="%d" role="right"/>\n    <tag k="type" v="lanelet"/>\n'
                '    <tag k="subtype" v="road"/>\n  </relation>' % (rid, left, right))

def street(x0, y0, dx, dy, k):
    """One 100 m street segment from (x0, y0) along (dx, dy)."""
    nx, ny = -dy, dx  # left
    def off(d):
        return [(x0 + dx * SPACING * i / (NODES_PER_WAY - 1) + nx * d,
                 y0 + dy * SPACING * i / (NODES_PER_WAY - 1) + ny * d) for i in range(NODES_PER_WAY)]
    t, st = BORDERS[k % len(BORDERS)]
    border = {"type": t}
    if st:
        border["subtype"] = st
    add_way(off(4.2), border)
    left_edge = add_way(off(3.5), {"type": "line_thick", "subtype": "solid"})
    centre = add_way(off(0.0), {"type": "line_thin", "subtype": "dashed"})
    right_edge = add_way(off(-3.5), {"type": "line_thick", "subtype": "solid"})
    add_way(off(-4.2), border)
    add_lanelet(left_edge, centre)
    add_lanelet(centre, right_edge)

k = 0
half = args.grid * SPACING / 2
for i in range(args.grid):
    for j in range(args.grid - 1):
        street(-half + i * SPACING, -half + j * SPACING + 6, 0, 1, k)
        street(-half + j * SPACING + 6, -half + i * SPACING, 1, 0, k + 3)
        k += 1

with open(args.out, "w") as f:
    f.write("<?xml version='1.0' encoding='UTF-8'?>\n<osm version='0.6' generator='gen_city_map.py'>\n")
    f.write("\n".join(nodes) + "\n" + "\n".join(ways) + "\n" + "\n".join(rels) + "\n</osm>\n")
print("%d nodes, %d ways, %d lanelets -> %s" % (len(nodes), len(ways), len(rels), args.out))
