// Lanelet2 live 3D viewer (browser side): wiring and the render loop.
//
// The modules under js/ do the work:
//   store.js      streamed map as plain data (features, tiles, node index)
//   render/*      batched lines, lanelet surfaces + arrows, icons, overlays
//   camera.js     heading/pitch camera, orbit, walk, framing
//   picking.js    screen-space picking over the store
//   nav.js        mouse navigation;  keys.js  keyboard, FPS look
//   edit.js       selection and the move gizmo;  net.js  SSE in, commands out
//   josmview.js   JOSM view follow / drive;  hud.js  readouts and toasts
//   imagery.js    "Street view" button (Mapillary / Google / Apple)
//   ui/*          the frame: side panel, menus, dialogs, command palette,
//                 compass, status bar (shell.js holds the layout state)
//
// Coordinates arrive as local ENU metres relative to an anchor lat/lon, so the
// scene is centred near the origin. Z is up.
import * as THREE from "three";
import { PROFILE, TEST_HOOKS, now, plog } from "./js/util.js";
// The shell goes first: it fills in the icons and restores the panel, so the
// viewport has its final size before the renderer is sized to it.
import { installShell, isStandalone, observeViewport, setHints, setCrumbs, setTabAvailable, showTab } from "./js/ui/shell.js";
import { renderer, scene, mapRoot } from "./js/scene.js";
import { camera, view, syncProjection, setCameraView } from "./js/camera.js";
import { store, nodeToken } from "./js/store.js";
import { lineLayer } from "./js/render/lines.js";
import { setRibbonResolution } from "./js/render/ribbons.js";
import { setScreenLineResolution } from "./js/render/screenlines.js";
import { iconLayer } from "./js/render/icons.js";
import { laneletLayer } from "./js/render/lanelets.js";
import { updateScreenSizedMarkers } from "./js/render/overlays.js";
import { updateHighlight, highlightStats } from "./js/render/highlight.js";
import { updateHover, hoverItem } from "./js/hover.js";
import { pointLayer } from "./js/render/points.js";
import { installScene, onSceneMessage, sceneState } from "./js/sensors/scene.js";
import { installCity, onCityMessage, update as updateCity, cityState } from "./js/sensors/city.js";
import { installBasemap, update as updateBasemap } from "./js/basemap.js";
import { rigActive, rig, update as updateRig } from "./js/sensors/camrig.js";
import { selection } from "./js/selection.js";
import { edit, gizmoBusy, transform, clickTarget, editBlockedReason, addViewClickFilter, setEscapeGuard } from "./js/edit.js";
import { nav, navBusy, installNav } from "./js/nav.js";
import { installKeys, applyHeldCamera, isPointerLocked } from "./js/keys.js";
import { installJosmView, maybeSyncJosmView } from "./js/josmview.js";
import { connect, onCommandResult } from "./js/net.js";
import { installImagery } from "./js/imagery.js";
import { frameAll } from "./js/framing.js";
import { setConn, refreshCounts, refreshCameraHud, setPerfLine, setRenderLine } from "./js/hud.js";
import { installCommands, setConnectionInfo } from "./js/ui/commands.js";
import { installSettings } from "./js/ui/settings.js";
import { installNavGizmo, updateNavGizmo } from "./js/ui/navgizmo.js";
import { installInspector } from "./js/ui/inspector.js";
import { installLineStyleUi } from "./js/ui/linestyleui.js";
import { ortho } from "./js/camera.js";

installShell();
// Under the camera lock picking assumes a projection the view does not have,
// and Esc releases the lock before it deselects anything.
addViewClickFilter(() => rigActive());
setEscapeGuard(() => rigActive());

// edit.js created TransformControls at import, so these listeners come after
// its pointerdown (see nav.js).
// The camera lock drives the camera from the recorded pose, so navigation
// and the hold-to-move keys have nothing to say while it is on.
installNav({ gizmoBusy: () => gizmoBusy() || rigActive(), pointerLocked: isPointerLocked });
installKeys();
installJosmView({ busy: () => gizmoBusy() || navBusy() });
installImagery();
installScene();
installCity();
installBasemap();
installCommands();
installSettings();
installNavGizmo();
installInspector();
installLineStyleUi();
if (PROFILE) setTabAvailable("debug", true);
pointLayer.attach(mapRoot);

// The breadcrumb: city.js names the city, scene.js the scene.
const crumbs = { city: null, scene: null, sceneTitle: "" };
window.addEventListener("ll2:crumbs", (e) => {
  Object.assign(crumbs, e.detail);
  setCrumbs({ ...crumbs, city: cityState.scenes.length || cityState.info ? crumbs.city : null });
});
document.getElementById("crumbScene")?.addEventListener("click", () => showTab("scene"));

// The connection popover: what the viewer is connected to and where the map
// comes from.
let connLive = false;
let connText = "connecting...";
setConnectionInfo(() => {
  const rows = [["Viewer server", connLive ? "connected" : connText, connLive ? "good" : "warnText"]];
  const n = store.featureCount();
  const src = store.mapSource;
  rows.push(["Map", !n ? "none yet"
    : src === "scene_server" ? `${n} features from the scene server (view only)`
    : `${n} features from JOSM`, src === "scene_server" ? "warnText" : ""]);
  if (store.anchor) rows.push(["Anchor", `${store.anchor.lat.toFixed(6)}, ${store.anchor.lon.toFixed(6)}`]);
  if (n && !isStandalone()) rows.push(["Editing", editBlockedReason() ? "not possible: read-only map" : "possible (Edit, or E)"]);
  if (cityState.info) rows.push(["City", `${cityState.name || "city"} · ${cityState.scenes.length} scenes · ${cityState.info.base}`]);
  if (sceneState.info) rows.push(["Scene", `${sceneState.info.id} · ${sceneState.info.base}`]);
  const jv = document.getElementById("josmView");
  if (jv && jv.textContent && jv.textContent !== "idle") rows.push(["JOSM view", jv.textContent]);
  return rows;
});

// Status-bar hints follow what the mouse and keys do right now.
function currentHints() {
  if (rig.on) {
    return [["[ ]", "switch camera"], ["O", "image"], ["Esc", "release"], [", .", "frames"]];
  }
  if (edit.on) {
    return [["Click", "select"], ["Shift", "add"], ["Ctrl+drag", "box"], ["G R H", "tools"],
            ["Z", "type height"], ["Del", "delete"], ["Esc", "deselect"]];
  }
  const base = ortho.on
    ? [["Left-drag", "pan"], ["Right-drag", "turn"], ["Wheel", "zoom"], ["B", "north up"], ["V", "3D"]]
    : [["Click", "inspect"], ["Left-drag", "orbit"], ["Right-drag", "look"], ["Middle", "pan"], ["Wheel", "zoom"],
       ["WASD", "walk"]];
  if (cityState.scenes.length && !sceneState.info) base.push(["Click a track", "open that scene"]);
  else if (sceneState.info) base.push([", .", "frames"], ["K", "car"]);
  base.push(["Ctrl K", "commands"], ["?", "all keys"]);
  return base;
}

// A clicked button must not keep keyboard focus, or Enter clicks it again
// instead of reaching the map. (Space is prevented as a move key anyway.)
document.addEventListener("click", (e) => {
  const b = e.target && e.target.closest ? e.target.closest("button") : null;
  if (b) b.blur();
}, true);
// The same for sliders, checkboxes and drop-downs once the mouse is done
// with them: otherwise WASD stays with the control until the map is clicked.
// Keyboard users who tabbed there keep the focus (no pointer involved).
let pointerOnControl = null;
document.addEventListener("pointerdown", (e) => {
  const el = e.target;
  pointerOnControl = el && el.tagName === "INPUT" && (el.type === "range" || el.type === "checkbox") ? el : null;
}, true);
document.addEventListener("pointerup", () => {
  const el = pointerOnControl;
  pointerOnControl = null;
  if (el) setTimeout(() => el.blur(), 0);
}, true);
document.addEventListener("change", (e) => {
  if (e.target && e.target.tagName === "SELECT") e.target.blur();
}, true);

// Only auto-frame the first time data appears; later snapshots (e.g. live
// edits streamed from JOSM) must not yank the camera the user has set. Framed
// right away rather than on the next frame, so a view chosen in between (B)
// is not overridden.
store.on("snapshot", ({ wasEmpty }) => { if (wasEmpty) frameAll(); });
store.on("applied", (msg) => {
  if (msg.type === "command_result") onCommandResult(msg);
  // A reloaded tab gets the scene announcement folded into its bootstrap
  // snapshot, the same way JOSM's selection arrives.
  else if (msg.type === "scene") onSceneMessage(msg);
  else if (msg.type === "city") onCityMessage(msg);
  else if (msg.type === "snapshot") {
    if (msg.city) onCityMessage({ city: msg.city });
    if (msg.scene) onSceneMessage({ scene: msg.scene });
  }
});

// Who the page works for. Started by `kitscenes-viewer`, the server serves the
// page as the KITScenes viewer (body.standalone) and it stays that for the
// session. Otherwise it is the JOSM plugin's "Lanelet2 3D" view, with Edit
// mode and the JOSM controls (.josmOnly); those are hidden while the map is
// the scene server's read-only copy, or scenes came with no map at all.
function setStandalone(standalone) {
  document.body.classList.toggle("standalone", standalone);
  document.title = standalone ? "KITScenes viewer" : "Lanelet2 3D Viewer";
  const brand = document.getElementById("brandName");
  if (brand) brand.textContent = standalone ? "KITScenes viewer" : "Lanelet2 3D";
  const menu = document.getElementById("viewMenuBtn");
  if (menu) menu.dataset.tip = standalone ? "Camera views" : "Camera views and JOSM sync";
}
if (isStandalone()) {
  setStandalone(true);
} else {
  store.on("applied", () => {
    const standalone = store.mapSource === "scene_server"
      || (!store.mapSource && !!(cityState.info || sceneState.info));
    if (standalone !== isStandalone()) setStandalone(standalone);
  });
}

connect((msg, meta) => {
  const t0 = PROFILE ? now() : 0;
  const info = store.applyMessage(msg);
  if (!info.applied) return; // stale / out-of-order patch
  refreshCounts(store.featureCount(), msg.seq, store.anchor);
  if (PROFILE) {
    const parseMs = meta ? meta.parseMs : 0;
    const line = `${msg.type} ops=${info.nOps} feats=${store.featureCount()} tiles=${store.tiles.size}` +
      ` nodes=${store.nodeIndexOn ? store.nodes.size : "-"} bytes=${meta ? meta.bytes : 0}` +
      ` parse=${parseMs.toFixed(1)} apply=${(now() - t0).toFixed(1)}ms`;
    plog(line);
    setPerfLine(line);
  }
}, (live, text) => {
  connLive = !!live;
  connText = text;
  setConn(live, text);
});

// --- render loop -----------------------------------------------------------
let fpsFrames = 0, fpsT0 = now(), renderMsEma = 0;
const _vp = new THREE.Vector4();
let lastTick = now();
function tick() {
  requestAnimationFrame(tick);
  const tNow = now();
  const dt = Math.min(0.05, (tNow - lastTick) / 1000);
  lastTick = tNow;
  if (!rigActive()) applyHeldCamera(dt);
  maybeSyncJosmView();
  // Styled lines and the highlight are sized in pixels of the current
  // viewport (the camera lock letterboxes it).
  renderer.getViewport(_vp);
  setRibbonResolution(_vp.z, _vp.w);
  setScreenLineResolution(_vp.z, _vp.w);
  lineLayer.update();
  lineLayer.updateDetail(camera, _vp.w);
  laneletLayer.update();
  updateHover(gizmoBusy() || navBusy() || isPointerLocked() || rigActive());
  updateHighlight();
  iconLayer.update(camera);
  updateScreenSizedMarkers();
  updateCity();
  // Ortho sizes its frustum from the camera height, which held keys and the
  // wheel change between frames.
  syncProjection();
  updateBasemap();
  refreshCameraHud(nav);
  updateNavGizmo();
  setHints(currentHints());
  const tr = PROFILE ? now() : 0;
  renderer.render(scene, camera);
  if (PROFILE) {
    renderMsEma = renderMsEma * 0.9 + (now() - tr) * 0.1;
    fpsFrames++;
    const t = now();
    if (t - fpsT0 >= 1000) {
      const fps = (fpsFrames * 1000) / (t - fpsT0);
      fpsFrames = 0;
      fpsT0 = t;
      const info = renderer.info.render;
      setRenderLine(`${fps.toFixed(0)} fps | ${renderMsEma.toFixed(2)}ms | ` +
        `${info.calls} draws | ${store.featureCount()} feats | ${lineLayer.chunks.size} chunks`);
    }
  }
}
tick();

// The 3D view is the box between the bars and the side panel, so it resizes
// when the panel opens or closes as well as with the window.
observeViewport((w, h) => {
  renderer.setSize(w, h);
  if (rigActive()) {
    updateRig(); // re-letterboxes and restores the intrinsic projection
    return;
  }
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
});

if (TEST_HOOKS) {
  const _tp = new THREE.Vector3();
  window.__ll2test = {
    // Page pixel of a world point, as the renderer draws it; [x, y, ndcZ].
    project(x, y, z) {
      camera.updateMatrixWorld();
      _tp.set(x, y, z).project(camera);
      const r = renderer.domElement.getBoundingClientRect();
      return [r.left + ((_tp.x + 1) / 2) * r.width, r.top + ((1 - _tp.y) / 2) * r.height, _tp.z];
    },
    camera: () => {
      // `dir` is the real look direction, which under the camera lock does not
      // follow from yaw/pitch (the lock writes the matrix directly).
      const dir = new THREE.Vector3();
      camera.getWorldDirection(dir);
      return { pos: camera.position.toArray(), yaw: view.yaw, pitch: view.pitch,
               fov: camera.fov, dir: dir.toArray() };
    },
    pivot: () => ({ point: nav.lastPivot ? nav.lastPivot.toArray() : null, kind: nav.lastPivotKind }),
    // The one selected node as "node/<id>", else null (older checks use this).
    selected: () => {
      const one = selection.single();
      return one && one.type === "node" ? nodeToken(one.id) : null;
    },
    selection: () => ({ nodes: Array.from(selection.nodes).map(nodeToken), ways: Array.from(selection.ways),
                        relations: Array.from(selection.relations) }),
    // What a left click at page pixel (x, y) would select, as { type, id } or null.
    pick: (x, y) => clickTarget(x, y, false),
    // What a click at the pointer would select: "node/<id>", "way/<id>" or null.
    hover: () => {
      const h = hoverItem();
      return h ? (h.type === "node" ? nodeToken(h.id) : h.id) : null;
    },
    // What the selection highlight draws: { segments, dots, triangles }.
    highlight: () => { updateHighlight(); return highlightStats(); },
    edit: () => ({ on: edit.on, tool: edit.tool, heightOnly: edit.heightOnly, keysMove: edit.keysMove, snap: edit.snap }),
    // The gizmo handle under the pointer (null for none) and the handles it has.
    gizmo: () => ({ axis: transform.axis, dragging: transform.dragging, mode: transform.mode,
      at: transform.object && transform.visible ? transform.object.position.toArray() : null,
      handles: transform._gizmo.picker[transform.mode].children.map((c) => c.name) }),
    // Put the camera exactly here (ENU metres, radians).
    setView: (pos, yaw, pitch) => setCameraView(new THREE.Vector3(pos[0], pos[1], pos[2]), yaw, pitch),
    // A drag in progress, without the gizmo: moves nodes locally only.
    moveLocal: (moves) => store.moveNodes(moves),
    node: (id) => {
      const rec = store.node(id);
      return rec ? [rec.x, rec.y, rec.z] : null;
    },
    toast: () => {
      const el = document.getElementById("toast");
      return el && el.classList.contains("show") ? { text: el.textContent, kind: el.className } : null;
    },
    scene: () => ({
      frame: sceneState.frame,
      frames: sceneState.info ? sceneState.info.frames : 0,
      points: pointLayer.stats(),
    }),
    icons: () => iconLayer.stats(),
    iconMesh: (id) => iconLayer.meshOf(id),
    lanelets: () => ({
      count: laneletLayer.lanelets.size,
      surfaceTiles: laneletLayer.chunks.size,
      oneWayArrows: laneletLayer.arrowMeshes[0] ? laneletLayer.arrowMeshes[0].count : 0,
      twoWayArrows: laneletLayer.arrowMeshes[1] ? laneletLayer.arrowMeshes[1].count : 0,
      overrideArrows: Array.from(laneletLayer.lanelets.values()).filter((f) => f.lanelet.owx.length).length,
      visible: laneletLayer.visible,
    }),
    stats: () => ({
      features: store.featureCount(),
      tiles: store.tiles.size,
      chunks: lineLayer.chunks.size,
      drawCalls: renderer.info.render.calls,
    }),
  };
}
