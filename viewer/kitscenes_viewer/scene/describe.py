"""Answer the viewer's read-only `describe` op from the loaded Lanelet2 map.

The viewer's Selection tab asks JOSM for every tag of the selected objects
and the relations they belong to (the map stream carries only type and
subtype). With the scene server's own read-only map there is no JOSM, so the
scene server answers in the same shape, from the lanelet2 map it sent:

  {"id": "way/12", "tags": {...}, "nodes": 4,
   "memberships": [{"id": "relation/7", "role": "left", "tags": {...}}]}

Ids are the ones the map feed uses (`mapfeed.MapConverter`): line strings
are ways, points are nodes, lanelets (and regulatory elements, areas) are
relations. Memberships: the lanelets a line bounds (role left / right, via
`findUsages`), the areas it outlines (outer / inner) and the regulatory
elements that reference it, in the role their parameters give. Regulatory
elements are indexed from their own parameters once per map rather than
looked up with `findUsages`, whose overloads on that layer differ between
binding versions (a missing one silently lost every traffic light).

A traffic light belongs to a *lanelet* (its `regulatory_element` member), not
to the lane's bounds, so for a line the regulatory elements of the lanelets
it bounds come along too, marked `via` that lanelet. The other way round, a
line or point that a regulatory element names (the light itself, its stop
line, a sign) lists the lanelets that use that regulatory element, marked
`via` it, and a regulatory element lists them as its own memberships. A
relation (a lanelet, a regulatory element) also lists its `members`.

Everything is looked up defensively: the lanelet2 bindings differ a little
between versions in which layer offers which `findUsages`, and a missing
lookup must cost a membership, never the answer.
"""

from .mapfeed import _attrs

MAX_DESCRIBE = 500  # objects described per request, the JOSM plugin's limit too


def _get(layer, pid):
    try:
        exists = getattr(layer, "exists", None)
        if exists is not None and not exists(pid):
            return None
        return layer[pid]
    except (KeyError, IndexError, RuntimeError, TypeError, ValueError):
        return None


def _usages(layer, prim):
    try:
        return list(layer.findUsages(prim))
    except Exception:  # noqa: BLE001 -- an overload this binding lacks
        return []


def _id(obj):
    try:
        return int(obj.id)
    except (AttributeError, TypeError, ValueError):
        return None


def _sorted(tags):
    return {k: tags[k] for k in sorted(tags)}


def _relation_tags(obj, default_type):
    tags = dict(_attrs(obj))
    tags.setdefault("type", default_type)
    return _sorted(tags)


def _parameter_roles(regelem, pid):
    """The roles in which a regulatory element's parameters name `pid`."""
    roles = []
    try:
        params = regelem.parameters
        items = params.items() if hasattr(params, "items") else [(k, params[k]) for k in params]
    except Exception:  # noqa: BLE001
        return roles
    for role, values in items:
        try:
            if any(_id(v) == pid for v in values):
                roles.append(str(role))
        except TypeError:
            if _id(values) == pid:
                roles.append(str(role))
    return roles


def _line_memberships(lanelet_map, line, pid):
    out = []
    for ll in _usages(lanelet_map.laneletLayer, line):
        tags = _relation_tags(ll, "lanelet")
        bounds_it = False
        for role, bound in (("left", "leftBound"), ("right", "rightBound")):
            if _id(getattr(ll, bound, None)) == pid:
                out.append(("relation/%d" % ll.id, role, tags))
                bounds_it = True
        # The lane's traffic lights, signs and rules hang off the lanelet.
        if bounds_it:
            for re_ in _lanelet_regelems(ll):
                out.append(("relation/%d" % re_.id, "regulatory_element",
                            _relation_tags(re_, "regulatory_element"), "relation/%d" % ll.id))
    area_layer = getattr(lanelet_map, "areaLayer", None)
    for area in _usages(area_layer, line) if area_layer is not None else []:
        tags = _relation_tags(area, "multipolygon")
        outer = getattr(area, "outerBound", []) or []
        inner = [ls for ring in (getattr(area, "innerBounds", []) or []) for ls in ring]
        if any(_id(ls) == pid for ls in outer):
            out.append(("relation/%d" % area.id, "outer", tags))
        if any(_id(ls) == pid for ls in inner):
            out.append(("relation/%d" % area.id, "inner", tags))
    out += _regelem_memberships(lanelet_map, line, pid)
    return out


_REGELEM_INDEX = {}  # id(lanelet map) -> {primitive id: [(regelem, role)]}


def _parameter_items(regelem):
    try:
        params = regelem.parameters
        return list(params.items()) if hasattr(params, "items") else [(k, params[k]) for k in params]
    except Exception:  # noqa: BLE001
        return []


def _values(values):
    try:
        return list(values)
    except TypeError:
        return [values]


def _regelem_index(lanelet_map):
    """Every primitive a regulatory element names, by id (lanelet2 ids are map-wide unique)."""
    key = id(lanelet_map)
    index = _REGELEM_INDEX.get(key)
    if index is not None:
        return index
    index = {}
    layer = getattr(lanelet_map, "regulatoryElementLayer", None)
    try:
        regelems = list(layer) if layer is not None else []
    except Exception:  # noqa: BLE001 -- a layer that does not iterate: no index
        regelems = []
    for re_ in regelems:
        for role, values in _parameter_items(re_):
            for v in _values(values):
                vid = _id(v)
                if vid is not None:
                    index.setdefault(vid, []).append((re_, str(role)))
    _REGELEM_INDEX[key] = index
    return index


_USERS_INDEX = {}  # id(lanelet map) -> {regulatory element id: [lanelet or area]}


def _regelem_users(lanelet_map):
    """The lanelets and areas that use each regulatory element, by its id.

    Built from the lanelets' own `regulatoryElements` once per map, like
    `_regelem_index`, rather than trusting a `findUsages` overload.
    """
    key = id(lanelet_map)
    users = _USERS_INDEX.get(key)
    if users is not None:
        return users
    users = {}
    for name in ("laneletLayer", "areaLayer"):
        layer = getattr(lanelet_map, name, None)
        try:
            items = list(layer) if layer is not None else []
        except Exception:  # noqa: BLE001 -- a layer that does not iterate
            items = []
        for item in items:
            for re_ in _lanelet_regelems(item):
                rid = _id(re_)
                if rid is not None:
                    users.setdefault(rid, []).append(item)
    _USERS_INDEX[key] = users
    return users


def _users_of(lanelet_map, regelem_id, via=None):
    """Memberships in the lanelets / areas using a regulatory element."""
    out = []
    for user in _regelem_users(lanelet_map).get(regelem_id, []):
        default = "lanelet" if hasattr(user, "leftBound") else "multipolygon"
        entry = ("relation/%d" % user.id, "regulatory_element", _relation_tags(user, default))
        out.append(entry + ((via,) if via else ()))
    return out


def _with_users(lanelet_map, memberships):
    """Direct regulatory element memberships plus the lanes using each, `via` it."""
    out = list(memberships)
    for m in memberships:
        if len(m) == 3 and m[2].get("type") == "regulatory_element":
            rid = int(m[0].partition("/")[2])
            for entry in _users_of(lanelet_map, rid, via=m[0]):
                if entry not in out:
                    out.append(entry)
    return out


def _regelem_memberships(lanelet_map, prim, pid):
    out = []
    seen = set()
    for re_, role in _regelem_index(lanelet_map).get(pid, []):
        if (re_.id, role) in seen:
            continue
        seen.add((re_.id, role))
        out.append(("relation/%d" % re_.id, role, _relation_tags(re_, "regulatory_element")))
    # findUsages as well, where the binding has it (a regulatory element that
    # names the primitive some other way than through its parameters).
    layer = getattr(lanelet_map, "regulatoryElementLayer", None)
    for re_ in _usages(layer, prim) if layer is not None else []:
        if any(seen_id == re_.id for seen_id, _ in seen):
            continue
        roles = _parameter_roles(re_, pid) or [""]
        for role in roles:
            seen.add((re_.id, role))
            out.append(("relation/%d" % re_.id, role, _relation_tags(re_, "regulatory_element")))
    return out


def _lanelet_regelems(lanelet):
    try:
        return list(getattr(lanelet, "regulatoryElements", []) or [])
    except TypeError:
        return []


def _member_token(obj):
    """How the viewer names a parameter / member: node, way or relation."""
    oid = _id(obj)
    if oid is None:
        return None
    if hasattr(obj, "leftBound") or hasattr(obj, "outerBound") or hasattr(obj, "parameters"):
        return "relation/%d" % oid
    try:
        len(obj)
        return "way/%d" % oid
    except TypeError:
        return "node/%d" % oid


def _member_tags(obj, token):
    if token.startswith("relation/"):
        default = "lanelet" if hasattr(obj, "leftBound") else (
            "multipolygon" if hasattr(obj, "outerBound") else "regulatory_element")
        return _relation_tags(obj, default)
    return _sorted(_attrs(obj))


def _relation_members(prim):
    """A lanelet's bounds and regulatory elements, or a regulatory element's parameters."""
    members = []
    if hasattr(prim, "leftBound"):
        for role, attr in (("left", "leftBound"), ("right", "rightBound")):
            bound = getattr(prim, attr, None)
            token = _member_token(bound) if bound is not None else None
            if token:
                members.append((token, role, _member_tags(bound, token)))
        for re_ in _lanelet_regelems(prim):
            members.append(("relation/%d" % re_.id, "regulatory_element", _relation_tags(re_, "regulatory_element")))
    else:
        for role, values in _parameter_items(prim):
            for v in _values(values):
                token = _member_token(v)
                if token:
                    members.append((token, str(role), _member_tags(v, token)))
    return members


def _find_relation(lanelet_map, pid):
    for name, default_type in (("laneletLayer", "lanelet"), ("regulatoryElementLayer", "regulatory_element"),
                               ("areaLayer", "multipolygon")):
        layer = getattr(lanelet_map, name, None)
        obj = _get(layer, pid) if layer is not None else None
        if obj is not None:
            return obj, default_type
    return None, None


def describe(lanelet_map, ids):
    """One entry per id (at most MAX_DESCRIBE, duplicates once), in order."""
    details = []
    for token in list(dict.fromkeys(str(i) for i in ids))[:MAX_DESCRIBE]:
        kind, _, num = token.partition("/")
        try:
            pid = int(num)
        except ValueError:
            details.append({"id": token, "missing": True, "tags": {}, "memberships": []})
            continue
        entry = {"id": token, "tags": {}, "memberships": []}
        memberships = []
        members = None
        if kind == "way":
            prim = _get(lanelet_map.lineStringLayer, pid)
            if prim is not None:
                entry["tags"] = _sorted(_attrs(prim))
                try:
                    entry["nodes"] = len(prim)
                except TypeError:
                    pass
                memberships = _with_users(lanelet_map, _line_memberships(lanelet_map, prim, pid))
        elif kind == "node":
            prim = _get(lanelet_map.pointLayer, pid)
            if prim is not None:
                entry["tags"] = _sorted(_attrs(prim))
                memberships = _with_users(lanelet_map, _regelem_memberships(lanelet_map, prim, pid))
        elif kind == "relation":
            prim, default_type = _find_relation(lanelet_map, pid)
            if prim is not None:
                entry["tags"] = _relation_tags(prim, default_type)
                memberships = _regelem_memberships(lanelet_map, prim, pid)
                if default_type == "regulatory_element":
                    memberships += _users_of(lanelet_map, pid)
                members = _relation_members(prim)
        else:
            prim = None
        if prim is None:
            entry["missing"] = True
        memberships.sort(key=lambda m: (len(m) > 3, m[2].get("type", ""), m[0], m[1]))
        entry["memberships"] = [dict({"id": m[0], "role": m[1], "tags": m[2]}, **({"via": m[3]} if len(m) > 3 else {}))
                                for m in memberships]
        if members is not None:
            entry["members"] = [{"id": i, "role": r, "tags": t} for i, r, t in members]
        details.append(entry)
    return details


def make_command_handler(map_owner, log=print):
    """`on_command` for the bridge: answers describe-only commands.

    `map_owner()` returns the lanelet2 map this server sent, None when there
    is no map, and False when another bridge (JOSM) streams the map. With
    False nothing is answered at all, edits included: the browser takes the
    first `command_result` with a matching id, and that must be JOSM's.
    Describe ops are answered from the map; everything else falls to the
    bridge's default (accept camera/selection traffic, refuse edits).
    """
    def handle(command):
        state = map_owner()
        if state is False:
            return False
        ops = command.get("ops")
        if not isinstance(ops, list) or not ops:
            return None
        if not all(isinstance(op, dict) and op.get("op") == "describe" for op in ops):
            return None
        if state is None:
            return {"ok": False, "message": "The scene server has no map loaded"}
        ids = [i for op in ops for i in (op.get("ids") or []) if isinstance(i, (str, int))]
        details = describe(state, ids)
        return {"ok": True, "message": "Described %d object(s) from the scene server's map" % len(details),
                "details": details}
    return handle
