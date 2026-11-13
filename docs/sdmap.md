# SD maps (OpenStreetMap) and navigation: `kitscenes.sdmap`, `kitscenes.navigation`

The OpenStreetMap layer of the API: road graph, map matching, routing, junctions and driving commands.

**In short**

- The release ships OpenStreetMap snapshots in `openstreetmap_sd_maps/`: one per city and recording week, and the 2026-05-03 snapshots used in the paper. They come with every download and are under the ODbL 1.0, not the dataset licence.
- `scene.sd_map` is the scene's own weekly snapshot (`scene.info.sd_snapshot`). `kitscenes.open(..., sd_snapshot="paper_2026-05-03")` uses the paper's snapshots for every scene.
- `kitscenes.sdmap` is separate from the HD map (`kitscenes.map`): every SD class has the `SD` prefix. The test-e2e splits cannot be placed on the SD map.
- Driving commands follow a `CommandRule`: `NAVSIM`, `NUSCENES` or `KITSCENES` (junction-aware). Experimental: the benchmark's navigation input is not final; the test-e2e scenes will ship their commands.
- What you publish that is derived from the OSM files falls under the ODbL; results (models, figures, papers) need only the attribution.

```python
smap = scene.sd_map                                    # parsed once, then cached
route = smap.route(scene[0].pose, scene[len(scene) - 1].pose)   # an SDRoute with its maneuvers
```

## Overview

```python
ds.sd_maps()                                       # {"Karlsruhe": {"2025-W33": path, "2026-W06": ..., "2026-W12": ..., "paper_2026-05-03": ...}, ...}
ds.sd_map("Karlsruhe")                             # the city's latest week; snapshot="2026-W06", "Karlsruhe_2026-W06.osm.pbf" or "paper"
ds.sd_map("Karlsruhe").snapshot                    # "2026-03-16T00:00:00Z": the data state (PBF header)
kitscenes.open(root, sd_snapshot="paper_2026-05-03")   # every scene.sd_map (sd_route, navigation, SDMapSpec) on the paper's files
```

```python
from kitscenes.map import Region
from kitscenes.navigation import KITSCENES, NAVSIM, NUSCENES, driving_commands

smap = scene.sd_map                                # its recording week's snapshot; parsed once, cached in ~/.cache/kitscenes/sdmap
pose = frame.pose
box = Region(60, 60, 30, 30).around(pose)          # a Box in the map frame, as for the HD map
raw = smap.elements(box).in_ego(pose, road_height_m=scene.road_height_m)
raw.way_points, raw.way_tags, raw.relations       # every OSM element in the box, all tags (an empty box is an empty result)
road = smap.attributes_at(pose)                    # SDRoad: speed_limit_mps / _kph (+ _source), lanes, turn_lanes, road_class, one_way, access, tags
smap.match(pose), smap.junctions(box), smap.restrictions(box)   # heading-aware snapping, merged junctions, turn restrictions
route = smap.route(scene[0].pose, scene[len(scene) - 1].pose)            # SDRoute, turn restrictions on; weight="time" for travel time
route.maneuvers, route.next_maneuver(pose), route.target_points(pose)   # junction-aware maneuvers, CARLA-style target points
driven = scene.sd_route                            # the driven route (HMM map matching of the poses), computed once per scene
commands = driving_commands(scene, KITSCENES)      # (F,) int8 LEFT 0, STRAIGHT 1, RIGHT 2, UNKNOWN 3; also NAVSIM, NUSCENES
nav = frame.navigation(NAVSIM)                     # command + one goal point (the end of the driven SD route; goal="path": the exact last pose); targets="route": points ahead (leak the future path)
spec = kitscenes.SampleSpec(sd_map=kitscenes.SDMapSpec(), navigation=kitscenes.NavigationSpec(KITSCENES))
```

An SD route as HD lanes (needs `kitscenes[map]`):

```python
lane_ids = scene.sd_route.to_lanelets(scene.map)   # lanelet IDs along the SD route, None if the HD map has no such route
if lane_ids is not None:
    centerlines = [scene.map.lane(i).centerline for i in lane_ids]   # (N, 3) each, map frame
```

The HD route joins the lanes at the route's ends and may use a neighbouring lane, so it lies within a few metres of the poses.

- **Frames:** the map frame of the scene's poses, 2D (OpenStreetMap has no heights); `in_ego(pose, road_height_m=...)` puts SD elements on the road below the base frame. One `SDMap` per file, shared by every scene that uses it; the test splits are placed with their city's origin.
- **Files:** the weekly snapshots are PBF without contributor metadata. Parsing takes 5–20 s per file once; the parsed arrays are cached (`~/.cache/kitscenes/sdmap`, `$KITSCENES_CACHE`). Details: [Data and licence](#data-and-licence).
- **Attributes come with their source:** `speed_limit_source` is `"tag"`, `"implicit"` (`DE:urban`, `DE:zone30`, ...), `"zone"` or `"default"`; `lanes_source` `"tag"`, `"tag_split"` or `"default"`; no `oneway` tag means two-way.
- **Commands** (`kitscenes.navigation`) are a `CommandRule`, always a parameter: see [Command rules](#command-rules). `source="path"` uses the recorded path, `source="sd_route"` the driven SD route.
- **Routing:** A* on the road graph, deterministic, with turn restrictions (via nodes and via ways) and the access and U-turn rules of OSRM's car profile. `SDMap.to_networkx()` gives the road graph as a `networkx.DiGraph` (needs `kitscenes[sdmap]`).
- **ODbL:** anything derived from the OSM files that you publish (graphs, crops with tags, per-frame attributes) is a derivative database under the ODbL; results (models, figures, papers) need only the attribution `kitscenes.sdmap.ATTRIBUTION`. The local caches are not published.

## Data and licence

- **Files** in `openstreetmap_sd_maps/` (WGS84, contributor metadata removed); `ds.sd_maps()` lists them as
  `{city: {snapshot: path}}`:
  - **weekly snapshots** `<City>_<YYYY>-W<ww>.osm.pbf`, one per city and ISO recording week, the OpenStreetMap
    state at the Monday 00:00:00 UTC of the week (cut from the Geofabrik full-history extracts with osmium, the same tag
    selection as the Overpass query of the 2026-05-03 files).
    PBF without metadata (no version, timestamp, changeset, user); the snapshot time is the header option
    `osmosis_replication_timestamp` (`SDMap.snapshot`). Frankfurt 2025-W34, Karlsruhe 2025-W33, 2026-W06 and
    2026-W12, Sindelfingen 2026-W08; each covers every scene of its city with at least 2.9 km margin.
    `scene.sd_map` loads the scene's own one (`scene.info.sd_snapshot`), `ds.sd_map(city)` the latest week,
    `ds.sd_map(city, snapshot="2026-W06")` a given one;
  - **the paper's snapshots** `paper_2026-05-03/<City>.osm` (Overpass XML, data state 2026-05-03). Snapshot name `"paper_2026-05-03"`:
    `ds.sd_map(city, snapshot="paper")`, or `kitscenes.open(root, sd_snapshot="paper_2026-05-03")` for every
    `scene.sd_map` (and so `sd_route`, navigation and `SDMapSpec` samples), e.g. to reproduce a
    training made with them.
    They come with every download (~425 MB).
- **ODbL 1.0, © OpenStreetMap contributors**, not the dataset licence (`openstreetmap_sd_maps/ATTRIBUTION.md`).
  - What the API builds on your machine (the parsed cache in `~/.cache/kitscenes/sdmap` or
    `$KITSCENES_CACHE/sdmap`) is not published, so it carries no obligation.
  - Anything derived from the files that you **publish** (graphs, crops with tags, per-frame SD attributes,
    label files with OSM elements) is a derivative database: publish it under the ODbL and
    offer it (or how to recreate it).
  - Results (trained models, figures, metrics, papers) are produced works: they need the attribution
    `kitscenes.sdmap.ATTRIBUTION` ("© OpenStreetMap contributors, ODbL 1.0").
  - Nothing OSM-derived is precomputed or shipped with the dataset; the code is Apache-2.0 and its
    dependencies are permissive (pyosmium BSD-2, networkx BSD-3).
- **Coverage:** the snapshots are bounding-box cuts (Overpass for 2026-05-03, the same selection with osmium
  for the weekly ones). Inside the box they are complete; outside it only the ways that cross its border reach
  out. `SDMap.covers(xy, margin_m)` and `SDMap.coverage_latlon` estimate the box from the nodes that belong to
  no way and no relation. Seven train scenes lie partly or fully outside the 2026-05-03 boxes; the weekly snapshots cover
  them, see [Scenes outside the snapshot](#scenes-outside-the-snapshot).
- **Text encoding:** the 2026-05-03 files store every non-ASCII tag value double-encoded (UTF-8 read as
  cp1252 and encoded again: `StraÃŸe`); they are kept unchanged so that the paper's
  results stay reproducible. `SDMap` repairs it on load (more than 97 % of the non-ASCII strings of every city
  come back exactly; the rest had bytes that cp1252 does not define). The weekly snapshots
  are correct UTF-8. `repair_encoding="auto"` (the default) repairs a file only if most of its non-ASCII
  strings are double encoded: the 2026-05-03 files, not the weekly ones.
  `SDMap.encoding_repaired` tells what was done; `repair_encoding=False` gives the strings as stored, `True` always repairs.

## API

```text
kitscenes.sdmap
  SDMap(osm_path, origin, *, city, cache_dir, junction_merge_m=20, repair_encoding="auto")
    .elements(roi, *, ways="all"|"highway"|"roads"|"none", nodes="tagged"|"all"|"none", clip=True) -> SDElements
    .relations() -> list[SDRelation]   (every relation of the file)
    .roads(roi=None, classes=None, *, public_only, include_links) -> list[SDRoad];  .road(id)
    .junctions(roi=None) -> list[SDJunction];  .junction(id)
    .restrictions(roi=None) -> list[SDTurnRestriction]
    .match(pose, *, max_distance=10, max_heading_error=pi/4) -> list[SDMatch];  .attributes_at(pose) -> SDRoad | None
    .match_trajectory(xy, yaw=None, *, sigma_m=4, max_distance=20, step_m=5, beta_m=5) -> SDTrack
    .route(start, goal, *, via=(), restrictions=True, weight="length"|"time") -> SDRoute | None
    .route_along(xy, yaw=None) -> SDRoute | None
    .covers(xy, margin_m=0) -> (N,) bool;  .coverage_latlon;  .snapshot (ISO time);  .encoding_repaired;  .to_networkx(roi=None)
  SDElements(node_ids, node_points, node_tags, way_ids, way_points, way_tags, relations, frame)
    .in_ego(pose, *, road_height_m) -> SDElements;  .resampled(n);  .empty
  SDRelation(id, tags, member_types, member_ids, member_roles, member_index)
  SDRoad(id, way_id, forward, from_node, to_node, points, length_m, road_class, is_link, one_way, public, access,
         speed_limit_mps (.speed_limit_kph), speed_limit_source, lanes, lanes_source, turn_lanes, junction, bridge, tunnel, layer,
         name, ref, successors, predecessors, reverse_id, tags)
  SDJunction(id, node_ids, xy, control, incoming, outgoing, internal);  SDTurnRestriction(id, kind, ..., road_sequences)
  SDMatch(road_id, s_m, distance_m, heading_error, xy);  SDTrack(road_ids, s_m, distance_m, matched, routes, route_index)
  SDRoute(road_ids, start_s_m, end_s_m, points, s_m, length_m, engine, sd_map)
    .maneuvers -> (SDManeuver, ...);  .next_maneuver(pose) -> (SDManeuver, distance_m) | None
    .command(pose, rule);  .target_points(pose, *, max_distance_m=50, spacing_m=None, at_maneuvers=True)
    .project(pose);  .point_at(s);  .heading_at(s);  .in_ego(pose);  .speed_limits();  .free_run_m
    .to_lanelets(scene_map) -> list[int] | None   (an HD lane route along it, needs kitscenes[map])
  SDManeuver(kind, direction, angle_rad, s_m, end_s_m, xy, junction_id, options, control, roundabout_exit,
             road_in, road_out, lanes_valid, straightest);  Direction (OSRM's bins);  RoadClass
kitscenes.navigation
  Command (LEFT 0, STRAIGHT 1, RIGHT 2, UNKNOWN 3; .onehot("navsim" | "vad"))
  CommandRule(name, kind="lateral"|"future_pose"|"maneuver", lookahead_m, horizon_s, threshold_m, ...)
  NAVSIM, NUSCENES, KITSCENES, PRESETS
  driving_commands(scene, rule, *, source="auto"|"path"|"sd_route", route=None) -> (F,) int8
  Frame.navigation(rule, *, source, targets="goal"|"route", goal="sd_route"|"path", ...) -> Navigation
    targets="goal" (default): one point, the end of the scene's route (the same for every frame), no route shape;
      goal="sd_route" (default, every rule): the end of the driven SD route on the road centre line, a route-level
      destination; goal="path": the exact last recorded pose (lane-level);
    targets="route": route points / polyline / next maneuver ahead, which reveal the future path (analysis only)
Dataset.sd_map(city, *, snapshot=None|"latest"|"2026-W06"|"Karlsruhe_2026-W06.osm.pbf"|"paper_2026-05-03")
Dataset.sd_maps() -> {city: {snapshot: path}};  kitscenes.open(root, sd_snapshot="scene"|"latest"|"paper_2026-05-03"|week)
Scene.sd_map (the scene's snapshot), Scene.sd_route, Scene.load_sd_map(path), availability("sd_map")
SampleSpec(sd_map=SDMapSpec(region, points, classes), navigation=NavigationSpec(rule, source, targets="goal", goal="sd_route", num_targets=1, ...))
```

Conventions: the map frame of the scene's poses (one `SDMap` per file, shared by all scenes that use it and
`ds.sd_map`), 2D; `in_ego` puts SD elements on the road (`road_height_m` below the base origin; pass
`scene.road_height_m`). Every SD class has the `SD` prefix; nothing is shared with the HD map's classes
(`kitscenes.map`) except the regions (`Box`, `Circle`, `Region`) and `MapOrigin`. Empty answers (no
elements, no match, no route) are normal results, e.g. outside the snapshot; only scenes that cannot be
placed at all raise (`MapUnavailable`: `WITHHELD_BY_SPLIT` in test-e2e, `NOT_DOWNLOADED` without the file).

### How it works

- **Parsing:** with pyosmium (a dependency of `kitscenes`) into arrays (string table, CSR tags), cached as one
  pickle per file, keyed by name, size, modification time and parser version. A file takes 5–20 s to parse
  once and well under a second from the cache; graph and junctions are built on first use in 2–5 s.
- **Road graph:** drivable `highway` classes (motorway ... residential, links, living_street, service, road,
  busway), cut where they meet; a road is one drivable direction of a piece (38 k to 83 k roads per weekly snapshot).
  One-way rules, access levels (`public` / `restricted` / `no`) and speed, lane and turn-lane attributes as in
  `kitscenes.sdmap.attributes`, every value with its source.
- **Turn restrictions** (`restriction`, `restriction:motorcar`; `except=motorcar` skipped; via node or via
  ways) become road sequences. Where the from or to way passes through the via node (OSM wants it to end
  there), only the road pairs whose turn angle fits the kind are kept (a `no_left_turn` does not also ban the
  opposite direction's right turn).
- **Junctions:** graph nodes with three or more neighbours, merged across internal pieces shorter than
  `junction_merge_m` (20 m), so a turn across a dual carriageway is one junction and one maneuver. Control
  from `highway=traffic_signals|stop|give_way` on the nodes and within 30 m before them on the incoming roads.
- **Matching:** `match` snaps to road pieces within `max_distance` of the OSM centre line (a vehicle in its
  lane is 1-5 m beside it) in the heading's direction. `match_trajectory` is an HMM (Newson & Krumm) over the
  poses every 5 m, Viterbi, transitions by driving vs straight distance, U-turns at graph nodes allowed for
  30 m; `scene.sd_route` is its longest piece over the scene's poses. With the weekly snapshots the driven
  route is matched for every scene with poses in the city frame.
- **Routing:** A* on the road graph with turn restrictions (via-way ones as road
  sequences in the search state), OSRM's car-profile rules for access (restricted roads only where needed,
  closed ones only to end on) and U-turns at junctions (20 s), deterministic tie-breaking. The start and goal
  are snapped heading-aware (best 3 candidates each). A route from the first to the last pose is found for
  more than 99 % of the scenes and takes 10–70 ms.
- **Maneuvers:** at every junction the route passes where it had a choice (two or more exits, U-turns not
  counted), the turn angle between the route's direction over 20 m before and 20 m after the junction,
  binned like OSRM (straight up to 20°, slight to 40°, turn to 120°, sharp to 170°, U-turn); kinds turn,
  continue, fork, roundabout (one maneuver per passage, with the exit number), on/off ramp, merge, u_turn.

## Command rules

Experimental: the benchmark's navigation input is not final; the test-e2e scenes will ship their commands.

| Preset | Kind | Definition | Needs |
|---|---|---|---|
| `NAVSIM` | lateral | point 20 m ahead along the path, lateral offset from the projection onto the path in the heading frame; ≥ 2 m LEFT, ≤ -2 m RIGHT; UNKNOWN if the path ends sooner | poses (`source="path"`) or the SD route |
| `NUSCENES` | future_pose | recorded pose 3 s ahead in the vehicle frame, y ≥ 2 m LEFT, ≤ -2 m RIGHT; UNKNOWN past the scene end | poses |
| `KITSCENES` | maneuver | next decision maneuver of the SD route within 40 m (or the one the vehicle is in): LEFT/RIGHT from 35°, from 20° onto a branch while another exit is straighter, forks by side, U-turn LEFT; else STRAIGHT; UNKNOWN only if the route ends within 40 m and the road past its end reaches a choice within the 40 m | the SD map |

`CommandRule` is a frozen dataclass, so a benchmark can fix its own rule (name and version go with any file
made from it). The rules cannot be computed on test-e2e (relative poses, 4 s shipped).

- `NAVSIM` on the SD route gives the command of `NAVSIM` on the recorded path for 92 % of the frames; the rest sit at the ±2 m threshold.
- `KITSCENES` and `NAVSIM` agree for 83 % of the frames, and on the side of a turn for 95 %. They differ by design: `NAVSIM` turns on any lateral offset of 2 m at 20 m (long bends, lane offsets, roundabouts), `KITSCENES` only at a junction with a choice, and 40 m ahead.
- UNKNOWN: about 16 % of the frames for `NAVSIM` (the path ends within 20 m), 7 % for `KITSCENES` (it looks past the route's end along the road until the next choice).

## Scenes outside the snapshot

With the 2026-05-03 files, three Sindelfingen train scenes (`a18b7d47`, `dbdaea96`, `b3d0df28`) leave the
snapshot's bounding box (lat 48.6447-48.7302, lon 8.9286-9.0488) and have frames without any SD data. Four
Frankfurt train scenes (`0c535163`, `1133194d`, `4d601ea2`, `efbc5ca2`) are partly or fully outside their box, where
only the ways that cross its border remain. Karlsruhe, val and the test splits are inside.

The weekly snapshots cover every scene with at least 2.9 km margin.

Outside a snapshot the API returns empty results (`SDElements.empty`, `scene.sd_route` None, commands
UNKNOWN), no error; `SDMap.covers(xy, margin_m)` tells which poses are inside.
