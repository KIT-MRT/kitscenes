"""A Lanelet2 map -> the viewer's protocol v2 features.

The wire format is the one the JOSM plugin streams, so the browser cannot
tell whether a map came from JOSM or from here. The geometry comes from
lanelet2 itself: `leftBound` / `rightBound` already in driving direction and
its own `centerline`, so the arrow sits where lanelet2 says it does.

A height beyond +-100 km is no height: these maps carry `-FLT_MAX` for
"unknown" on traffic signs and poles that were never given an elevation.
"""

from .enu import round_coord

MAX_ABS_ELE_M = 100_000.0
ARROW_AT = 0.35  # fraction along the centerline
ONE_WAY = "one_way"
ARROW_PARTICIPANT = "vehicle:car"
# Tags worth sending: the browser styles by `type`/`subtype` and shows icons.
# "color" and "sidewalk_entry" are what the viewer's line styles read, and
# "height" makes a 2-point Autoware traffic light or sign stand upright
# (`autowareLightQuad`). The JOSM plugin streams the same keys.
KEEP_TAGS = ("type", "subtype", "participant:bicycle", "location", "region", "color",
             "sidewalk_entry", "height")


def usable_ele(z):
    """The point's height, or None when it is the 'no height' sentinel."""
    try:
        z = float(z)
    except (TypeError, ValueError):
        return None
    if z != z or z in (float("inf"), float("-inf")) or abs(z) > MAX_ABS_ELE_M:
        return None
    return z


def fill_unknown_heights(zs, xs, ys):
    """Heights for the points that have none.

    Interpolated by distance between the known heights around them, copied at
    the ends, 0.0 when the line has no height at all. Display only -- nothing
    is written back to the map.
    """
    n = len(zs)
    out = list(zs)
    known = [z is not None for z in zs]
    if not any(known):
        return [0.0] * n

    def run(a, b):
        d = 0.0
        for k in range(a, b):
            d += ((xs[k + 1] - xs[k]) ** 2 + (ys[k + 1] - ys[k]) ** 2) ** 0.5
        return d

    prev = -1
    i = 0
    while i < n:
        if known[i]:
            prev = i
            i += 1
            continue
        nxt = i
        while nxt < n and not known[nxt]:
            nxt += 1
        for j in range(i, nxt):
            if prev < 0:
                out[j] = out[nxt]
            elif nxt >= n:
                out[j] = out[prev]
            else:
                d0, d1 = run(prev, j), run(j, nxt)
                z0, z1 = out[prev], out[nxt]
                out[j] = (z0 + z1) / 2 if d0 + d1 < 1e-9 else z0 + (z1 - z0) * d0 / (d0 + d1)
        i = nxt
    return out


def parse_lanelet_bool(v):
    """lanelet2's `Attribute::asBool`: 1/0, then true/yes, false/no."""
    if v is None:
        return None
    v = str(v).strip()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    return None


def is_two_way_for(tags, participant=ARROW_PARTICIPANT):
    """`GenericTrafficRules::isOneWay` negated, as the plugin ports it."""
    explicit = parse_lanelet_bool(tags.get(ONE_WAY))
    if explicit is not None:
        return not explicit
    if any(k.startswith(ONE_WAY) for k in tags):
        full = "%s:%s" % (ONE_WAY, participant)
        key = next((k for k in sorted(tags) if full.startswith(k)), None)
        return not (parse_lanelet_bool(tags.get(key)) if key else True)
    return participant == "pedestrian"


def other_way_participants(tags):
    car = is_two_way_for(tags)
    prefix = ONE_WAY + ":"
    return sorted(k[len(prefix):] for k in tags
                  if k.startswith(prefix) and is_two_way_for(tags, k[len(prefix):]) != car)


def _attrs(obj):
    try:
        return {str(k): str(v) for k, v in obj.attributes.items()}
    except Exception:  # some lanelet2 bindings expose only iteration over keys
        return {str(k): str(obj.attributes[k]) for k in obj.attributes}


def _point_dist(px, py, ax, ay, bx, by):
    """Distance from a point to the segment a-b."""
    ex, ey = bx - ax, by - ay
    len2 = ex * ex + ey * ey
    t = 0.0 if len2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * ex + (py - ay) * ey) / len2))
    return ((ax + ex * t - px) ** 2 + (ay + ey * t - py) ** 2) ** 0.5


def _dist_to_polyline(px, py, pts):
    best = float("inf")
    for i in range(len(pts) - 1):
        best = min(best, _point_dist(px, py, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]))
    return best


class MapConverter:
    """Turns one `LaneletMap` into viewer features in a given ENU frame."""

    def __init__(self, frame):
        self.frame = frame  # SceneFrame: scene metres -> viewer ENU

    def _line_points(self, linestring):
        """ENU xyz and point ids of a linestring, unknown heights filled."""
        xs, ys, zs, ids = [], [], [], []
        for p in linestring:
            x, y = self.frame.apply(p.x, p.y)
            xs.append(x)
            ys.append(y)
            zs.append(usable_ele(p.z))
            ids.append(int(p.id))
        return xs, ys, fill_unknown_heights(zs, xs, ys), ids

    def line_feature(self, linestring):
        xs, ys, zs, ids = self._line_points(linestring)
        if len(xs) < 2:
            return None
        tags = {k: v for k, v in _attrs(linestring).items() if k in KEEP_TAGS}
        pts = []
        for x, y, z in zip(xs, ys, zs):
            pts += [round_coord(x), round_coord(y), round_coord(z)]
        return {"id": "way/%d" % linestring.id, "kind": "line", "tags": tags,
                "pts": pts, "nodes": ids}

    def lanelet_feature(self, lanelet):
        left, right = lanelet.leftBound, lanelet.rightBound
        if len(left) < 2 or len(right) < 2:
            return None
        tags = {k: v for k, v in _attrs(lanelet).items()
                if k in KEEP_TAGS or k.startswith(ONE_WAY)}
        arrow = self._arrow(lanelet)
        rep = arrow[:3] if arrow else self._mid(left, right)
        feat = {
            "id": "relation/%d" % lanelet.id,
            "kind": "lanelet",
            "tags": tags,
            "pts": rep,
            "left": "way/%d" % left.id,
            "right": "way/%d" % right.id,
            "lrev": bool(left.inverted()),
            "rrev": bool(right.inverted()),
            "two": is_two_way_for(tags),
        }
        owx = other_way_participants(tags)
        if owx:
            feat["owx"] = owx
        if arrow:
            feat["arrow"] = arrow
        return feat

    def _mid(self, left, right):
        a, b = left[0], right[0]
        x, y = self.frame.apply((a.x + b.x) / 2, (a.y + b.y) / 2)
        za, zb = usable_ele(a.z), usable_ele(b.z)
        z = [v for v in (za, zb) if v is not None]
        return [round_coord(x), round_coord(y), round_coord(sum(z) / len(z) if z else 0.0)]

    def _arrow(self, lanelet):
        """Pose at ARROW_AT of lanelet2's own centerline: x, y, z, dx, dy, dz, width."""
        cx, cy, cz, _ = self._line_points(lanelet.centerline)
        if len(cx) < 2:
            return None
        cum = [0.0]
        for i in range(1, len(cx)):
            cum.append(cum[-1] + ((cx[i] - cx[i - 1]) ** 2 + (cy[i] - cy[i - 1]) ** 2) ** 0.5)
        total = cum[-1]
        if total < 1e-6:
            return None
        target = total * ARROW_AT
        k = 0
        # Skip zero-length segments: a centerline may repeat a vertex.
        while k < len(cx) - 2 and (cum[k + 1] < target or cum[k + 1] - cum[k] <= 1e-9):
            k += 1
        seg = cum[k + 1] - cum[k]
        if seg <= 1e-9:
            return None
        t = max(0.0, min(1.0, (target - cum[k]) / seg))
        x = cx[k] + (cx[k + 1] - cx[k]) * t
        y = cy[k] + (cy[k + 1] - cy[k]) * t
        z = cz[k] + (cz[k + 1] - cz[k]) * t
        dx = (cx[k + 1] - cx[k]) / seg
        dy = (cy[k + 1] - cy[k]) / seg
        dz = (cz[k + 1] - cz[k]) / seg
        lx, ly, _, _ = self._line_points(lanelet.leftBound)
        rx, ry, _, _ = self._line_points(lanelet.rightBound)
        width = _dist_to_polyline(x, y, list(zip(lx, ly))) + _dist_to_polyline(x, y, list(zip(rx, ry)))
        return [round_coord(v) for v in (x, y, z, dx, dy, dz, width)]

    def features(self, lanelet_map):
        """Every linestring as a line, every lanelet after its bounds."""
        out = []
        for ls in lanelet_map.lineStringLayer:
            feat = self.line_feature(ls)
            if feat:
                out.append(feat)
        for ll in lanelet_map.laneletLayer:
            feat = self.lanelet_feature(ll)
            if feat:
                out.append(feat)
        return out
