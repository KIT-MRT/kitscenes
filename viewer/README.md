# kitscenes-viewer — KITScenes in the browser

A 3D viewer for the [KITScenes](https://github.com/KIT-MRT/kitscenes) dataset:
the Lanelet2 HD map, the lidar point clouds, the ring camera images and the
trajectories of one scene or of every scene in a city, in one view, read
straight from the release.

```bash
pip install -e . -e viewer                 # in a clone of the repository
pip install -e "viewer[clouds]"            # + torch, for whole-scene clouds and city tiles

kitscenes-viewer /data/kitscenes/data/train/<scene-uuid>   # one scene
kitscenes-viewer /data/kitscenes                           # the city with most scenes
kitscenes-viewer /data/kitscenes --city-name frankfurt --map /data/kitscenes/lanelet2_hd_maps/Frankfurt.osm
kitscenes view /data/kitscenes                             # the same, through the kitscenes CLI
```

A folder holding `poses.txt` is one scene; any other folder is searched for
scenes and shown as a city. A browser tab opens at <http://127.0.0.1:8765/>.

Both servers listen on this machine only. With the dataset on another
machine, run the viewer there with `--no-browser`, forward the two ports and
open the same address locally:

```bash
ssh -L 8765:127.0.0.1:8765 -L 8767:127.0.0.1:8767 <host>
```

## Options

| option | meaning |
|---|---|
| `path` | a scene folder or a dataset root (`KITSCENES_ROOT`, its `data/`, or one split) |
| `--city-name NAME` | dataset root: `karlsruhe`, `frankfurt` or `sindelfingen` (default: the city with the most scenes) |
| `--map PATH` | the Lanelet2 map to show instead of the scene's own, e.g. a city-wide map |
| `--cache-dir PATH` | where the cache lives (default `~/.cache/kitscenes-viewer`) |
| `--no-cache` | cache nothing on disk |
| `--aggregate-attributes` | what a whole-scene cloud carries: `geometry`, `colour`, `all` (colour and classes); default `auto`: `all` with a GPU, `geometry` without |
| `--build-aggregates` | dataset root: build every scene's whole-scene cloud into the cache, then exit |
| `--port N` | the page's port (default 8765); the viewer also uses the next one |
| `--data-port N` | the scene data's port (default 8767) |
| `--no-browser` | do not open a browser tab |
| `--tile-proxy-host HOST` | fetch background map tiles from this host through the viewer, for tile servers that do not allow cross-origin use; repeatable |

## One scene

The scene's controls appear by themselves: a timeline under the view, a
camera bar over it, and the *Sensor data* tab in the side panel. `?` lists
every key, `Ctrl+K` finds any command by name.

| | |
|---|---|
| timeline `◀` `▶|` / `,` `.` | step frames; the slider scrubs |
| timeline `▶` | play |
| timeline *Car* / `K` | camera behind the car at this frame |
| timeline *Follow* / `Shift+K` | keep the camera where it is relative to the car while the frames change |
| *This frame* / `P` | show or hide the single frame's cloud |
| Colour | intensity, height, sensor, class, rgb, instance, with a legend |
| *Whole scene* / `Shift+P` | the whole scene as one static cloud (hides the single frame) |
| *Tint this frame over it* / `U` | with the whole scene shown, `P` brings the frame back over it; `U` paints it one bright colour |
| camera bar: a ring button / `Shift+L` | put the view exactly where that ring camera was; the lit button or `Esc` lets go and leaves the camera where it is |
| `[` `]` | walk the six ring cameras |
| camera bar: image button / `O`, slider | blend the recorded image over the render |

**The map** is the scene's own `maps/map.osm`. Scenes of the test and
map_change_test splits ship none: they are shown on their city's HD map,
`<root>/lanelet2_hd_maps/<City>.osm`, when the dataset has it. The poses of
the test-e2e splits are relative to the scene's first pose, so such a scene is
shown in its own metres, without a map. `--map PATH` puts any other scene on
that map instead; the log says which map a scene got.

The map is read-only. A click selects a line or a lane, and the *Selection*
tab shows its attributes and the lanelets and regulatory elements that use it.

### Colour modes

Three colour modes come from the cameras:

- **class** paints each point with the Mapillary Vistas colour of the
  [seamseg](https://github.com/mapillary/seamseg) mask it falls into
  (`processed/seamseg_instances/`). Only the front and rear cameras have
  masks, so about 4 % of a frame gets a class; the rest stays "unlabelled"
  grey.
- **rgb** samples the colour from the six ring camera images, so about 90 %
  of a frame gets a real colour.
- **instance** gives every detected object its own colour. The numbers are per
  frame and not tracked between frames, so this is a single-frame mode.

A camera colours and labels every point that lands in its image: a point
behind something nearer takes the colour and class of what is in front of it.

### Filters and opacity

| | |
|---|---|
| class chips | one per class present, commonest first; click to hide |
| `Hide` / `Only` | what a ticked class means: hidden, or the only classes drawn |
| `Vehicles` | preset: ticks cars, trucks, buses, bikes, motorcycles, trailers, caravans and their riders |
| `Above ground` | drop everything higher than *n* m over the ground under it (the ground follows a slope) |
| `Intensity` | drop everything below a reflectivity floor; markings and signs survive it |
| opacity | fade the points so map lines, nodes and icons behind them stay readable |

### The whole scene as one cloud

`Shift+P` stacks every frame into one cloud (`kitscenes.mapping.VoxelMap`,
needs `torch`: `viewer[clouds]`). Points are binned to 0.15 m voxels, one
point per voxel, within 60 m of the car. A voxel is kept when it was seen in
3 consecutive frames: a static surface is hit frame after frame, a passing
vehicle occupies a voxel for an instant, so the traffic drops out without any
labels. Wind-blown foliage drops out the same way. Each voxel keeps the mean
camera colour and the class most of its points voted for.

Building a 190-frame scene, 0.15 m voxels:

| | |
|---|---|
| GPU, colours and classes | 5–9 s |
| 4 CPU cores, geometry only | 42–55 s |
| 4 CPU cores, colours and classes | 90–126 s |

The cloud is cached, so later runs load it at once.

## A whole city

```bash
# every scene of a city (all splits) on the city-wide map
kitscenes-viewer $KITSCENES_ROOT --city-name frankfurt --map $KITSCENES_ROOT/lanelet2_hd_maps/Frankfurt.osm

# once, ahead of time: build every scene's static cloud into the cache
kitscenes-viewer $KITSCENES_ROOT --build-aggregates
```

Each scene's city comes from the kitscenes scene table; a scene the table
does not list goes to the nearest of the three cities, from where its
trajectory starts. A scene is placed with the origin of its pose frame
(`kitscenes.aerial.placement_origin`): its `maps/origin.json`, or, for the
test and map_change_test splits, its city's origin. The scenes of the test-e2e
splits cannot be placed and are left out; the log lists them with the reason.

The index of a city (three small files per scene) is kept in the cache
directory. New and removed scenes are noticed at every start. A file changed
inside a known scene is found by a check that runs in the background after
the start; the log names the scene, and the next start shows the change.

| | |
|---|---|
| tracks | every scene's ego trajectory: a line, a dot per frame, an arrow per frame once close enough. One colour per scene |
| hover a track | scene id, split and frame under the pointer |
| click a dot, arrow or line | that scene becomes the open scene, at that frame; everything under [One scene](#one-scene) then works. `K` goes to the car |
| the city in the breadcrumb | back to the city alone |
| the scene list | every scene, nearest the camera first, filtered by id or split; a click opens it and goes to its car |
| `Load 300 m` | the static clouds of every scene within the radius of the camera, merged into tiles; clouds not built yet are built now, one at a time |
| `Auto` | keep loading around the camera as it moves, but only clouds that are already built |
| Radius | how far around the camera; clouds beyond 1.3× of it are dropped again |
| `Unload`, *Trajectories* | drop every city cloud; hide the trajectories |

**Merged tiles.** One street driven by five scenes is one cloud, not five
stacked copies: the built clouds of every scene that reaches a 128 m tile are
merged voxel by voxel (all scenes share one voxel grid). A tile is fetched
again when one more of its scenes is built. Merging needs the cache; with
`--no-cache` each scene's own cloud is shown, stacked.

The trajectories are the pose of the base frame, at `lidar_top`, about
2.05 m above the road.

## Cache

**The dataset is never written to.** The cache lives in
`~/.cache/kitscenes-viewer` (`$XDG_CACHE_HOME/kitscenes-viewer`, or
`--cache-dir`) and is safe to delete at any time. It holds:

- per frame shown, the camera colours and seamseg labels of its points
  (about 4.5 MB a frame; they take 0.35–1.1 s and 0.1–0.3 s to compute).
  These are trimmed to 10 GB, least recently used first;
- per scene, its whole-scene cloud (`aggregate-*.npz`, about 31 MB). These
  are never trimmed: a city of 300 scenes is about 9 GB;
- the index of each city.

A frame's points are not cached: reading and deskewing them takes 0.1–0.3 s.
The server decodes one frame at a time and the page asks for one frame at a
time, the newest wanted one, so holding the step key jumps to where you stop.

## How it fits together

`kitscenes-viewer` starts two processes. `server.py` (standard library only)
serves the page and passes messages on to every open tab.
`python -m kitscenes_viewer.scene` reads the dataset through `kitscenes`,
sends the map and the scene description to `server.py`, and serves points and
images to the page on the data port. When a viewer already answers on the
port, `kitscenes-viewer` sends the scenes into that one.

The two halves by hand (`--help` lists the options of each):

```bash
python -m kitscenes_viewer.server --no-browser
python -m kitscenes_viewer.scene --scene /path/to/<scene-uuid>
```

**Coordinates.** The scene and its map live in lanelet2's `UtmProjector`
frame; the page draws in a local ENU about an anchor, which is turned against
it by the meridian convergence (about 0.25°) and scaled slightly differently.
Map and points go through one exact chain into it
(`kitscenes_viewer/scene/enu.py`).

## Tests

```bash
python -m pytest viewer/tests                 # from the repository root; the whole-scene cloud tests need torch
node viewer/tests/browser/e2e_scene.mjs       # browser checks: Node 22 and Chrome, see tests/browser/README.md
```
