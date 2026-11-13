# 3D viewer browser checks

End-to-end checks for the browser half of the live 3D viewer
(`kitscenes_viewer/server.py` and `static/`). They run the real `server.py`,
play the JOSM bridge on the ingest socket (sending scenes, recording the
commands the page posts back), and drive the page in headless Chrome over the
DevTools Protocol.

Not part of the pytest suite: they need Chrome and a display-less GL
(SwiftShader), which CI does not guarantee.

## Run

```bash
node viewer/tests/browser/e2e_viewer.mjs                 # camera + picking; Node 22+, no npm install
node viewer/tests/browser/e2e_edit.mjs                   # selection, moves, JOSM sync, refusals
node viewer/tests/browser/e2e_controls.mjs               # keys for camera and selection
node viewer/tests/browser/e2e_lanelets.mjs               # lanelet surfaces, arrows, surface picking
node viewer/tests/browser/e2e_heights.mjs                # typed heights, snapping
node viewer/tests/browser/e2e_ui.mjs                     # side panel, palette, help, settings, compass, narrow window
node viewer/tests/browser/e2e_linestyles.mjs             # styled lines, read off screen pixels
node viewer/tests/browser/e2e_viewer.mjs --shots /tmp/v3d  # also save screenshots
CHROME=/path/to/chrome node viewer/tests/browser/e2e_viewer.mjs
# As root (containers), Chrome needs --no-sandbox: point CHROME at a wrapper
#   printf '#!/bin/sh\nexec /path/to/chrome --no-sandbox "$@"\n' > chrome.sh
```

Exit code 0 means every check passed.

Performance on a real map (streams the whole `.osm` as one snapshot and
reports load time, draw calls and frame time):

```bash
node viewer/tests/browser/perf_viewer.mjs path/to/lanelet2_map.osm
LL2_VIEWER_DIR=/path/to/other/viewer3d node viewer/tests/browser/perf_viewer.mjs map.osm  # compare builds
python3 viewer/tests/browser/gen_city_map.py /tmp/city.osm   # a synthetic map the size of Karlsruhe
node viewer/tests/browser/perf_viewer.mjs /tmp/city.osm --lines simple   # or flat / styled (default)
```

## How the checks see the page

The page is opened with `?test=1`, which makes `app.js` expose
`window.__ll2test`: the renderer's own projection of a world point to a page
pixel, the exact camera state, the last orbit pivot and the selection. Use it
instead of parsing the Debug panel, whose one-decimal readout is ~10 px off at close
range.

The fake bridge answers every command that carries an `id` with an accepting
`command_result`; set `session.reply = (cmd) => ({ ok: false, message })` to
simulate JOSM refusing.

`window.__ll2ui` reports the interface itself (panel, tab, open dialog,
status hints, running jobs) and `window.__ll2commands` the command list.

The 3D view is the box between the bars and the side panel, not the window:
take sizes from the canvas (`getBoundingClientRect`), never from
`innerWidth` / `innerHeight`.

`session.reveal(point)` pans a world point out from under the panels
before a click; clicking where a panel sits hits the panel, not the map.

Ports are picked free per run, so a viewer already running on 8765/8766 does
not interfere.
