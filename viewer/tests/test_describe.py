"""The scene server's answer to the viewer's `describe` op.

A fake map with the shape of lanelet2's Python objects (layers with
`exists`, item access and `findUsages`; primitives with `id` and
`attributes`; lanelets with bounds; regulatory elements with parameters).
Checked against the real bindings by hand; this pins the logic.
"""

import unittest

from kitscenes_viewer.scene.bridge import ViewerBridge
from kitscenes_viewer.scene.describe import MAX_DESCRIBE, describe, make_command_handler


class Prim:
    def __init__(self, pid, attrs=None, points=()):
        self.id = pid
        self.attributes = dict(attrs or {})
        self._points = list(points)

    def __len__(self):
        return len(self._points)

    def __iter__(self):
        return iter(self._points)


class Lanelet(Prim):
    def __init__(self, pid, attrs, left, right, regelems=()):
        super().__init__(pid, attrs)
        self.leftBound = left
        self.rightBound = right
        self.regulatoryElements = list(regelems)


class RegElem(Prim):
    def __init__(self, pid, attrs, parameters):
        super().__init__(pid, attrs)
        self.parameters = parameters


class Layer:
    def __init__(self, items, usages=None):
        self._items = {p.id: p for p in items}
        self._usages = usages or (lambda prim: [])

    def exists(self, pid):
        return pid in self._items

    def __getitem__(self, pid):
        return self._items[pid]

    def findUsages(self, prim):
        return self._usages(prim)

    def __iter__(self):
        return iter(self._items.values())


def fake_map():
    p1 = Prim(101, {"ele": "110.5"})
    p2 = Prim(102)
    left = Prim(11, {"type": "line_thin", "subtype": "dashed", "file_origin": "/maps/a.osm"}, [p1, p2])
    right = Prim(12, {"type": "curbstone", "subtype": "high"}, [p2, p1])
    sign = Prim(13, {"type": "traffic_sign", "subtype": "de205"}, [p1, p2])
    light = Prim(15, {"type": "traffic_light", "subtype": "red_yellow_green"}, [p1, p2])
    stop = Prim(16, {"type": "stop_line"}, [p1, p2])
    tl20 = RegElem(20, {"subtype": "traffic_light"}, {"refers": [light], "ref_line": [stop]})
    ll7 = Lanelet(7, {"subtype": "road", "one_way": "yes"}, left, right, [tl20])
    # Line 11 is the right bound of lanelet 8 as well: two memberships.
    ll8 = Lanelet(8, {"subtype": "road"}, Prim(14), left)
    re9 = RegElem(9, {"type": "regulatory_element", "subtype": "right_of_way"},
                  {"refers": [sign], "ref_line": [left]})
    re10 = RegElem(10, {"subtype": "traffic_sign"}, {"refers": [p1]})
    lanelets = [ll7, ll8]
    regelems = [re9, re10, tl20]

    def lanelet_usages(prim):
        return [ll for ll in lanelets if prim.id in (ll.leftBound.id, ll.rightBound.id)]

    def regelem_usages(prim):
        return [r for r in regelems if any(v.id == prim.id for vs in r.parameters.values() for v in vs)]

    class Map:
        pass

    m = Map()
    m.pointLayer = Layer([p1, p2])
    m.lineStringLayer = Layer([left, right, sign, light, stop])
    m.laneletLayer = Layer(lanelets, lanelet_usages)
    m.regulatoryElementLayer = Layer(regelems, regelem_usages)
    return m


class DescribeTest(unittest.TestCase):
    def setUp(self):
        self.map = fake_map()

    def test_a_line_gets_every_attribute_and_its_relations(self):
        [d] = describe(self.map, ["way/11"])
        self.assertEqual({"file_origin": "/maps/a.osm", "subtype": "dashed", "type": "line_thin"}, d["tags"])
        self.assertEqual(["file_origin", "subtype", "type"], list(d["tags"]))
        self.assertEqual(2, d["nodes"])
        self.assertNotIn("missing", d)
        got = [(m["id"], m["role"], m["tags"].get("type"), m.get("via")) for m in d["memberships"]]
        self.assertEqual([("relation/7", "left", "lanelet", None), ("relation/8", "right", "lanelet", None),
                          ("relation/9", "ref_line", "regulatory_element", None),
                          # The lane's traffic light hangs off lanelet 7, not off the line.
                          ("relation/20", "regulatory_element", "regulatory_element", "relation/7")], got)
        # A lanelet's own attributes, with the relation type the stream shows.
        self.assertEqual({"one_way": "yes", "subtype": "road", "type": "lanelet"}, d["memberships"][0]["tags"])

    def test_a_sign_line_is_referred_to_by_its_regulatory_element(self):
        [d] = describe(self.map, ["way/13"])
        self.assertEqual([("relation/9", "refers")], [(m["id"], m["role"]) for m in d["memberships"]])

    def test_a_point_gets_its_attributes_and_regulatory_elements(self):
        [d] = describe(self.map, ["node/101"])
        self.assertEqual({"ele": "110.5"}, d["tags"])
        self.assertNotIn("nodes", d)
        self.assertEqual([("relation/10", "refers", "regulatory_element")],
                         [(m["id"], m["role"], m["tags"]["type"]) for m in d["memberships"]])

    def test_a_relation_id_finds_the_lanelet(self):
        [d] = describe(self.map, ["relation/7"])
        self.assertEqual("lanelet", d["tags"]["type"])
        self.assertEqual("road", d["tags"]["subtype"])

    def test_missing_malformed_duplicate_and_capped(self):
        out = describe(self.map, ["way/999", "way/x", "bogus/1", "way/11", "way/11"])
        self.assertEqual(["way/999", "way/x", "bogus/1", "way/11"], [d["id"] for d in out])
        self.assertTrue(out[0]["missing"] and out[1]["missing"] and out[2]["missing"])
        self.assertEqual({}, out[0]["tags"])
        many = describe(self.map, ["node/%d" % i for i in range(MAX_DESCRIBE + 20)])
        self.assertEqual(MAX_DESCRIBE, len(many))

    def test_regulatory_elements_do_not_need_find_usages(self):
        # Some bindings lack findUsages(LineString) on this layer: the
        # parameters index still finds the traffic light and its stop line.
        def broken(prim):
            raise TypeError("No registered converter")
        self.map.regulatoryElementLayer._usages = broken
        [light] = describe(self.map, ["way/15"])
        self.assertEqual([("relation/20", "refers")],
                         [(m["id"], m["role"]) for m in light["memberships"] if "via" not in m])
        [stop] = describe(self.map, ["way/16"])
        self.assertEqual([("relation/20", "ref_line")],
                         [(m["id"], m["role"]) for m in stop["memberships"] if "via" not in m])
        [d] = describe(self.map, ["way/11"])
        self.assertIn("relation/9", [m["id"] for m in d["memberships"]])

    def test_a_broken_lanelet_lookup_costs_those_memberships_not_the_answer(self):
        def broken(prim):
            raise TypeError("No registered converter")
        self.map.laneletLayer._usages = broken
        [d] = describe(self.map, ["way/11"])
        self.assertEqual(["relation/9"], [m["id"] for m in d["memberships"]])

    def test_a_lanelet_lists_its_bounds_and_regulatory_elements(self):
        [d] = describe(self.map, ["relation/7"])
        self.assertEqual([("way/11", "left", "line_thin"), ("way/12", "right", "curbstone"),
                          ("relation/20", "regulatory_element", "regulatory_element")],
                         [(m["id"], m["role"], m["tags"].get("type")) for m in d["members"]])
        self.assertEqual("traffic_light", d["members"][2]["tags"]["subtype"])

    def test_a_traffic_light_lists_its_lights_and_stop_line(self):
        [d] = describe(self.map, ["relation/20"])
        self.assertEqual("regulatory_element", d["tags"]["type"])
        self.assertEqual([("way/15", "refers"), ("way/16", "ref_line")], [(m["id"], m["role"]) for m in d["members"]])


    def test_a_traffic_light_lists_the_lanes_it_controls(self):
        """The light way -> its regulatory element -> the lanelets using that."""
        for token, role in (("way/15", "refers"), ("way/16", "ref_line")):
            [d] = describe(self.map, [token])
            got = [(m["id"], m["role"], m["tags"].get("type"), m.get("via")) for m in d["memberships"]]
            self.assertEqual([("relation/20", role, "regulatory_element", None),
                              ("relation/7", "regulatory_element", "lanelet", "relation/20")], got, token)
            self.assertEqual("road", d["memberships"][1]["tags"]["subtype"])

    def test_a_regulatory_element_lists_the_lanes_using_it(self):
        [d] = describe(self.map, ["relation/20"])
        self.assertEqual([("relation/7", "regulatory_element", "lanelet")],
                         [(m["id"], m["role"], m["tags"]["type"]) for m in d["memberships"]])
        self.assertNotIn("via", d["memberships"][0])
        # A rule no lanelet uses has nothing to list.
        [unused] = describe(self.map, ["relation/9"])
        self.assertEqual([], unused["memberships"])

    def test_the_lanes_need_no_find_usages_either(self):
        def broken(prim):
            raise TypeError("No registered converter")
        self.map.laneletLayer._usages = broken
        self.map.regulatoryElementLayer._usages = broken
        [light] = describe(self.map, ["way/15"])
        self.assertIn(("relation/7", "relation/20"),
                      [(m["id"], m.get("via")) for m in light["memberships"]])


class HandlerTest(unittest.TestCase):
    def test_answers_describe_from_its_own_map(self):
        handle = make_command_handler(lambda: fake_map())
        reply = handle({"type": "command", "id": "c1", "ops": [{"op": "describe", "ids": ["way/12"]}]})
        self.assertTrue(reply["ok"])
        self.assertEqual("curbstone", reply["details"][0]["tags"]["type"])

    def test_stays_silent_when_josm_owns_the_map(self):
        handle = make_command_handler(lambda: False)
        self.assertIs(False, handle({"ops": [{"op": "describe", "ids": ["way/1"]}]}))

    def test_stays_silent_on_edits_when_josm_owns_the_map(self):
        """A read-only refusal from here beat JOSM's success on an edit it applied."""
        handle = make_command_handler(lambda: False)
        for ops in ([{"op": "set_tag", "id": "way/1", "key": "subtype", "value": "dashed"}],
                    [{"op": "move_node", "id": "node/1", "z": 1}], [{"op": "undo"}],
                    [{"op": "delete_selection", "ids": ["way/1"]}], [{"op": "select", "ids": []}], "junk"):
            self.assertIs(False, handle({"type": "command", "id": "c1", "ops": ops}), ops)

    def test_its_own_map_still_refuses_edits(self):
        sent = []
        bridge = ViewerBridge(on_command=make_command_handler(lambda: fake_map()), log=lambda m: None)
        bridge.send = sent.append
        bridge._handle(b'{"type":"command","id":"e1","ops":[{"op":"set_tag","id":"way/11","key":"k","value":"v"}]}')
        self.assertEqual([("e1", False)], [(m["id"], m["ok"]) for m in sent])
        self.assertIn("read-only", sent[0]["message"])

    def test_says_so_without_a_map(self):
        handle = make_command_handler(lambda: None)
        reply = handle({"ops": [{"op": "describe", "ids": ["way/1"]}]})
        self.assertFalse(reply["ok"])

    def test_anything_else_goes_to_the_default(self):
        handle = make_command_handler(lambda: fake_map())
        self.assertIsNone(handle({"ops": [{"op": "move_node", "id": "node/1", "z": 1}]}))
        self.assertIsNone(handle({"ops": [{"op": "describe", "ids": []}, {"op": "select", "ids": []}]}))

    def test_the_bridge_sends_the_answer_or_nothing(self):
        sent = []
        bridge = ViewerBridge(on_command=make_command_handler(lambda: fake_map()), log=lambda m: None)
        bridge.send = sent.append
        bridge._handle(b'{"type":"command","id":"c9","ops":[{"op":"describe","ids":["way/11"]}]}')
        self.assertEqual(1, len(sent))
        self.assertEqual(("command_result", "c9", True), (sent[0]["type"], sent[0]["id"], sent[0]["ok"]))
        self.assertEqual(4, len(sent[0]["details"][0]["memberships"]))
        silent = ViewerBridge(on_command=make_command_handler(lambda: False), log=lambda m: None)
        silent.send = sent.append
        silent._handle(b'{"type":"command","id":"c10","ops":[{"op":"describe","ids":["way/11"]}]}')
        silent._handle(b'{"type":"command","id":"c11","ops":[{"op":"set_tag","id":"way/11","key":"k","value":"v"}]}')
        self.assertEqual(1, len(sent), "JOSM owns the map: no reply from the scene server, not even to an edit")


if __name__ == "__main__":
    unittest.main()
