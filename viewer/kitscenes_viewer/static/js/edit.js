// 3D editing: select nodes and ways, move / rotate them, delete, undo.
//
// Moves happen locally while dragging (every way using a node follows), then
// go to JOSM as one command = one undo step. JOSM applies it and echoes the
// authoritative geometry back through the normal patch path; if it refuses
// (layer hidden, node gone), the viewer puts the nodes back and says why.
//
// Selecting works in View mode too (to inspect tags, also on a read-only
// map); everything that changes the map needs Edit mode.
//
//   click          select the node under the cursor, else the way. In View
//                  mode: the way, else the lanelet under the cursor (nodes
//                  only with the "View clicks select nodes" setting)
//   shift+click    add / remove it
//   ctrl+drag      box select (shift: add)
//   middle-click   cycle through everything under the cursor (coincident nodes)
//   G / R / H      move, rotate about the vertical, height only
//   T              keys move the camera or the selection (WASD / Space / C,
//                  alt+left/right turn); one key hold = one undo step
//   I              interpolate heights along the selected way (between its ends,
//                  or between 2+ selected nodes of it)
//   Z              type a height for the selection: 112.35 sets it, +0.2 / -0.2
//                  shift it, =-1.5 sets a negative height; Enter applies
//   M              snap gizmo moves (height to the nearest node / lanelet
//                  surface, a single node onto another); ctrl inverts while dragging
//   Del            delete through JOSM's Delete action;  ctrl+Z / ctrl+Y undo / redo
import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { Emitter, round3, clamp } from "./util.js";
import { canvas, scene } from "./scene.js";
import { camera, view } from "./camera.js";
import { store, nodeToken } from "./store.js";
import { lineLayer } from "./render/lines.js";
import { setHighlightVisible } from "./render/highlight.js";
import { selection } from "./selection.js";
import {
  pickItem, nodesUnderCursor, featuresUnderCursor, laneletsUnderCursor, screenProjector, projectToPage,
} from "./picking.js";
import { sendCommand } from "./net.js";
import { snapHeight, snapToNode, showSnap, hideSnap } from "./snap.js";
import { setEditHud, setSelectionCard, toast } from "./hud.js";

const GIZMO_CLICK_NODE_PX = 4; // a click on a gizmo handle selects only a node dot this close
const CLICK_SLOP_PX = 5;
const MOVE_EPS_M = 0.0005;     // below this a coordinate counts as unchanged

export const editEvents = new Emitter(); // "mode" (on)
export const edit = { on: false, tool: "translate", heightOnly: false, keysMove: "camera", snap: false };

const KEY_MOVE_IDLE_MS = 250;     // a key-driven move is committed this long after the last key
const KEY_TURN_RAD_S = 0.5;       // alt+left/right turn rate (×3 with shift)

// The gizmo moves this invisible pivot at the selection's centre; the
// selected nodes follow it.
const pivot = new THREE.Object3D();
scene.add(pivot);
export const transform = new TransformControls(camera, canvas);
transform.setSize(0.9);
scene.add(transform);
transform.addEventListener("mouseDown", beginGizmoDrag);
transform.addEventListener("objectChange", onGizmoChange);
transform.addEventListener("mouseUp", endGizmoDrag);

export const gizmoBusy = () => transform.dragging;

const _axis = new THREE.Vector3();

// Each gesture stamps its nodes with a generation. A refusal from JOSM
// reverts only nodes no later gesture has touched since, so a slow answer to
// an older move cannot undo a newer one.
let genCounter = 0;
const nodeGen = new Map();
function touch(ids) {
  const g = ++genCounter;
  for (const id of ids) nodeGen.set(id, g);
  return g;
}
let drag = null;                 // { start, orig: Map<id, [x, y, z]>, moved }

// --- mode, tool, gizmo ----------------------------------------------------------------

/** Why editing is not possible right now, or null when it is. */
export function editBlockedReason() {
  if (store.mapSource === "scene_server") {
    return "This map comes from the scene server and is read-only.";
  }
  return null;
}

/** What the Selection panel describes: the selection, in either mode. */
export function inspectedTokens() {
  return selection.allTokens();
}

// View mode is for looking: a click picks the way (or the lanelet), not the
// node on it, unless the setting says otherwise.
const viewPick = { nodes: false };

/** Whether a click in View mode selects nodes as well (Settings). */
export function setViewSelectsNodes(on) {
  viewPick.nodes = !!on;
}

// A click in View mode may belong to something else first (a trajectory in
// city mode) or to nothing at all (the camera lock): filters say so.
const clickFilters = [];

/** `fn(x, y)` returns true when a View-mode click at page pixel (x, y) is not a selection. */
export function addViewClickFilter(fn) {
  clickFilters.push(fn);
}

// The camera lock's Esc releases the lock first; it is not a deselect.
let escapeBelongsElsewhere = () => false;
export function setEscapeGuard(fn) {
  escapeBelongsElsewhere = fn;
}

// The selection and the hover show in both modes.
setHighlightVisible(true);

export function setEditMode(on) {
  if (on && !edit.on) {
    const why = editBlockedReason();
    if (why) {
      toast(why, "warn", 5000);
      return;
    }
  }
  edit.on = !!on;
  if (edit.on) store.ensureNodeIndex();
  lineLayer.setDotsVisible(edit.on);
  const bar = document.getElementById("editBar");
  if (bar) bar.hidden = !edit.on;
  const card = document.getElementById("selCard");
  if (card) card.hidden = !edit.on;
  const btn = document.getElementById("editBtn");
  if (btn) btn.classList.toggle("on", edit.on);
  const viewBtn = document.getElementById("viewModeBtn");
  if (viewBtn) viewBtn.classList.toggle("on", !edit.on);
  refreshGizmo();
  editEvents.emit("mode", edit.on);
}

export function setTool(tool) {
  edit.tool = tool;
  if (tool === "rotate") edit.heightOnly = false;
  refreshGizmo();
}

/** Whether WASD / Space / C / alt+arrows (and the pads) move the camera or the selection. */
export function setKeysMove(target) {
  edit.keysMove = target;
  refreshGizmo();
}

/** True when movement keys should move the selection rather than the camera. */
export function keysMoveSelection() {
  return edit.on && edit.keysMove === "selection" && selection.hasEditable() && !drag;
}

export function toggleSnap() {
  edit.snap = !edit.snap;
  refreshGizmo();
}

export function toggleHeightOnly() {
  edit.heightOnly = !edit.heightOnly;
  if (edit.heightOnly) edit.tool = "translate";
  refreshGizmo();
}

function refreshGizmo() {
  const show = edit.on && selection.hasEditable();
  if (show && !drag) placePivot();
  if (show) {
    const rotate = edit.tool === "rotate";
    transform.setMode(rotate ? "rotate" : "translate");
    // Rotation is about the vertical only; height-only shows just the Z arrow.
    const zOnly = rotate || edit.heightOnly;
    transform.showX = !zOnly;
    transform.showY = !zOnly;
    transform.showZ = true;
    if (transform.object !== pivot) transform.attach(pivot);
  } else if (transform.object) {
    transform.detach();
  }
  for (const b of document.querySelectorAll("#editBar [data-edit]")) {
    const k = b.getAttribute("data-edit");
    if (k === "keys") {
      const label = b.querySelector(".lbl");
      if (label) label.textContent = edit.keysMove === "selection" ? "sel" : "cam";
      b.dataset.tip = `Keys move the ${edit.keysMove}. WASD, Space / C and Alt+arrows move the camera `
        + "or the selection (one key hold = one undo step)";
      b.classList.toggle("on", edit.keysMove === "selection");
      continue;
    }
    if (k === "snap") {
      b.classList.toggle("on", edit.snap);
      continue;
    }
    b.classList.toggle("on", (k === "height" && edit.heightOnly) || (k === edit.tool && !edit.heightOnly));
  }
  refreshEditHud();
}

function refreshEditHud() {
  if (!edit.on) {
    setEditHud("off");
    return;
  }
  const parts = [];
  if (selection.nodes.size) parts.push(`${selection.nodes.size} node${selection.nodes.size > 1 ? "s" : ""}`);
  if (selection.ways.size) parts.push(`${selection.ways.size} way${selection.ways.size > 1 ? "s" : ""}`);
  const rels = selection.relations.size;
  if (rels) parts.push(`${rels} relation${rels > 1 ? "s" : ""} (not editable here)`);
  const one = selection.single();
  const what = one ? (one.type === "node" ? nodeToken(one.id) : one.id) : parts.join(", ");
  const tool = edit.heightOnly ? "height only" : edit.tool === "rotate" ? "rotate" : "move";
  const keys = edit.keysMove === "selection" ? " · keys move selection" : "";
  const snap = edit.snap ? " · snap" : "";
  setEditHud(`ON · ${what || "nothing selected"} · ${tool}${keys}${snap}`);
  setSelectionCard(`${what || "nothing selected"} · ${tool}${snap}`);
  refreshZField();
}

/** Centre of the nodes a move would act on (mean position). */
function placePivot() {
  let n = 0, x = 0, y = 0, z = 0;
  for (const id of selection.effectiveNodeIds()) {
    const rec = store.node(id);
    if (!rec) continue;
    x += rec.x; y += rec.y; z += rec.z; n++;
  }
  if (n) pivot.position.set(x / n, y / n, z / n);
  pivot.quaternion.identity();
  pivot.updateMatrixWorld();
}

selection.on("changed", () => refreshGizmo());
store.on("applied", (msg) => {
  if (msg.type === "selection" || (msg.type === "snapshot" && msg.selection)) {
    const sel = msg.type === "selection" ? msg : msg.selection;
    selection.applyFromJosm(sel);
  }
  if (!drag) refreshGizmo();
});
store.on("nodes-moved", () => { if (!drag && edit.on && selection.hasEditable()) placePivot(); });

// --- gizmo drag ---------------------------------------------------------------------

function beginGizmoDrag() {
  const orig = new Map();
  for (const id of selection.effectiveNodeIds()) {
    const rec = store.node(id);
    if (rec) orig.set(id, [rec.x, rec.y, rec.z]);
  }
  pivot.quaternion.identity();
  drag = { start: pivot.position.clone(), orig, moved: false, exclude: new Set(orig.keys()), gen: touch(orig.keys()) };
}

// Ctrl while dragging inverts the snap toggle (read from the pointer events,
// so it follows the key mid-drag).
let ctrlDown = false;
canvas.addEventListener("pointermove", (e) => { ctrlDown = e.ctrlKey || e.metaKey; });

/**
 * Snap the translation `d` of the current drag: heights for a vertical move,
 * a single node onto another for a free one. The gizmo follows the snap.
 */
function snapTranslation(d) {
  const vertical = edit.heightOnly || transform.axis === "Z";
  const o = drag.start;
  if (vertical) {
    const x = o.x + d.x, y = o.y + d.y, z = o.z + d.z;
    const hit = snapHeight(x, y, z, drag.exclude);
    if (!hit) return hideSnap();
    d.z = hit.z - o.z;
    showSnap(x, y, hit.z);
  } else if (drag.orig.size === 1) {
    const p = drag.orig.values().next().value;
    const hit = snapToNode([p[0] + d.x, p[1] + d.y, p[2] + d.z], drag.exclude);
    if (!hit) return hideSnap();
    d.set(hit.x - p[0], hit.y - p[1], hit.z - p[2]);
    showSnap(hit.x, hit.y, hit.z);
  } else {
    return hideSnap();
  }
  pivot.position.copy(o).add(d);
}

// Drag in progress: move the nodes locally so connected ways follow live.
function onGizmoChange() {
  if (!drag) return;
  const moves = [];
  if (transform.mode === "rotate") {
    // Only the heading change counts: how the X axis turned in the ground
    // plane. The Z ring is the only handle: with X/Y hidden, TransformControls
    // also hides its view-axis ring ("E") and trackball ("XYZE").
    _axis.set(1, 0, 0).applyQuaternion(pivot.quaternion);
    const a = Math.atan2(_axis.y, _axis.x);
    const c = Math.cos(a), s = Math.sin(a);
    const o = drag.start;
    for (const [id, p] of drag.orig) {
      const dx = p[0] - o.x, dy = p[1] - o.y;
      moves.push({ id, x: o.x + c * dx - s * dy, y: o.y + s * dx + c * dy, z: p[2] });
    }
  } else {
    const d = pivot.position.clone().sub(drag.start);
    if (edit.heightOnly) d.x = d.y = 0;
    if (edit.snap !== ctrlDown) snapTranslation(d);
    else hideSnap();
    for (const [id, p] of drag.orig) moves.push({ id, x: p[0] + d.x, y: p[1] + d.y, z: p[2] + d.z });
  }
  drag.moved = true;
  store.moveNodes(moves);
}

function endGizmoDrag() {
  const d = drag;
  drag = null;
  hideSnap();
  pivot.quaternion.identity();
  if (!d || !d.moved) return;
  commitMoves(d.orig, d.gen);
  placePivot();
}

/**
 * Send the nodes' new positions to JOSM as one command. Only changed
 * components go out: a height-only move carries no x/y (lat/lon stay exact),
 * a level move no z (no `ele` added). Reverts locally if JOSM refuses, but
 * only nodes that no later gesture (generation > `gen`) has touched.
 */
export function commitMoves(orig, gen = touch(orig.keys())) {
  const ops = [];
  const revert = [];
  for (const [id, o] of orig) {
    const rec = store.node(id);
    if (!rec) continue;
    const xy = Math.abs(rec.x - o[0]) > MOVE_EPS_M || Math.abs(rec.y - o[1]) > MOVE_EPS_M;
    const z = Math.abs(rec.z - o[2]) > MOVE_EPS_M;
    if (!xy && !z) continue;
    const op = { op: "move_node", id: nodeToken(id) };
    if (xy) {
      op.x = round3(rec.x);
      op.y = round3(rec.y);
    }
    if (z) op.z = round3(rec.z);
    ops.push(op);
    revert.push({ id, x: o[0], y: o[1], z: o[2] });
  }
  const settle = () => {
    for (const id of orig.keys()) if (nodeGen.get(id) === gen) nodeGen.delete(id);
  };
  if (!ops.length) {
    settle();
    return Promise.resolve(null);
  }
  return sendCommand(ops, { awaitResult: true }).then((info) => {
    const refused = !info.delivered
      ? (info.error ? `could not reach the viewer server (${info.error})` : "JOSM is not connected")
      : info.result && info.result.ok === false ? info.result.message : null;
    if (refused) {
      store.moveNodes(revert.filter((m) => nodeGen.get(m.id) === gen));
      toast(`Move not applied: ${refused}`, "error");
    } else if (info.result && info.result.warning) {
      toast(info.result.warning, "warn", 8000);
    }
    settle();
    return info;
  });
}

// --- keyboard moves -----------------------------------------------------------------
// Held keys move / turn the selection like walking the world: right/forward
// along the camera heading, up/down, turn about the vertical through the
// selection's centre. Speed follows the camera distance, so close up the
// selection creeps (precise), far away it travels. The whole hold is one
// command, sent once no key has moved it for KEY_MOVE_IDLE_MS.
let keyMove = null; // { orig: Map, cx, cy, ox, oy, oz, yaw, timer }

/**
 * Apply one frame of held keys. right/forward/up/turn are -1..1 intents,
 * `fast` is shift. Height-only ignores everything but `up`.
 */
export function nudgeSelection({ right = 0, forward = 0, up = 0, turn = 0 }, dt, fast) {
  if (!keysMoveSelection()) return;
  if (edit.heightOnly) {
    right = forward = turn = 0;
  }
  if (!right && !forward && !up && !turn) return;
  if (!keyMove) {
    const orig = new Map();
    let cx = 0, cy = 0;
    for (const id of selection.effectiveNodeIds()) {
      const rec = store.node(id);
      if (!rec) continue;
      orig.set(id, [rec.x, rec.y, rec.z]);
      cx += rec.x;
      cy += rec.y;
    }
    if (!orig.size) return;
    keyMove = { orig, cx: cx / orig.size, cy: cy / orig.size, ox: 0, oy: 0, oz: 0, yaw: 0, timer: null, gen: touch(orig.keys()) };
  }
  const km = keyMove;
  const speed = clamp(0.25 * camera.position.distanceTo(pivot.position), 0.05, 20) * (fast ? 5 : 1);
  const s = Math.sin(view.yaw), c = Math.cos(view.yaw);
  // right = (c, s), forward = (-s, c), as camera.walk
  km.ox += (c * right - s * forward) * speed * dt;
  km.oy += (s * right + c * forward) * speed * dt;
  km.oz += up * speed * dt;
  km.yaw += turn * KEY_TURN_RAD_S * (fast ? 3 : 1) * dt;
  const cr = Math.cos(km.yaw), sr = Math.sin(km.yaw);
  const moves = [];
  for (const [id, p] of km.orig) {
    let x = p[0], y = p[1];
    if (km.yaw !== 0) { // unturned moves stay exact (no centre round trip)
      const dx = x - km.cx, dy = y - km.cy;
      x = km.cx + cr * dx - sr * dy;
      y = km.cy + sr * dx + cr * dy;
    }
    moves.push({ id, x: x + km.ox, y: y + km.oy, z: p[2] + km.oz });
  }
  store.moveNodes(moves);
  if (km.timer !== null) clearTimeout(km.timer);
  km.timer = setTimeout(finishKeyMove, KEY_MOVE_IDLE_MS);
}

function finishKeyMove() {
  const km = keyMove;
  keyMove = null;
  if (!km) return;
  commitMoves(km.orig, km.gen);
  placePivot();
}

// --- numeric height --------------------------------------------------------------------
// The edit bar's Z field shows the selection's height (mean, range in the
// tooltip) and takes a typed one: "112.35" sets every selected node to it,
// "+0.2" / "-0.2" shift them, "=-1.5" sets a negative height. Enter applies
// (one undo step), Esc drops the edit, up/down arrows step 1 cm (shift 10 cm).
const zField = document.getElementById("zField");
let zDirty = false;

function selectionHeights() {
  const zs = [];
  for (const id of selection.effectiveNodeIds()) {
    const rec = store.node(id);
    if (rec) zs.push(rec.z);
  }
  return zs;
}

const fmtZ = (v) => String(round3(v));

export function refreshZField() {
  if (!zField || (document.activeElement === zField && zDirty)) return;
  const zs = edit.on ? selectionHeights() : [];
  zField.disabled = zs.length === 0;
  if (!zs.length) {
    zField.value = "";
    zField.title = "Select nodes or ways to type their height (Z)";
    return;
  }
  const min = Math.min(...zs), max = Math.max(...zs);
  zField.value = fmtZ(zs.reduce((a, b) => a + b, 0) / zs.length);
  zField.title = (min === max ? `${zs.length} node(s) at ${fmtZ(min)} m` : `mean of ${zs.length} nodes, ${fmtZ(min)} … ${fmtZ(max)} m`) +
    ". Type 112.35 to set, +0.2 / -0.2 to shift, =-1.5 for a negative height; Enter applies.";
}

/** Parse the field: { set } for an absolute height, { shift } for an offset, or null. */
export function parseHeightInput(text) {
  const t = String(text).trim().replace(",", ".");
  if (!t) return null;
  const num = (v) => (/^[+-]?(\d+\.?\d*|\.\d+)$/.test(v) ? Number(v) : NaN);
  if (t.startsWith("=")) {
    const v = num(t.slice(1).trim());
    return Number.isFinite(v) ? { set: v } : null;
  }
  const v = num(t);
  if (!Number.isFinite(v)) return null;
  return t[0] === "+" || t[0] === "-" ? { shift: v } : { set: v };
}

export function applyHeightInput(text) {
  const parsed = parseHeightInput(text);
  if (!parsed) {
    toast("Height: type 112.35 to set, +0.2 / -0.2 to shift, =-1.5 for a negative height", "warn", 5000);
    return false;
  }
  const orig = new Map();
  const moves = [];
  for (const id of selection.effectiveNodeIds()) {
    const rec = store.node(id);
    if (!rec) continue;
    orig.set(id, [rec.x, rec.y, rec.z]);
    moves.push({ id, x: rec.x, y: rec.y, z: parsed.set !== undefined ? parsed.set : rec.z + parsed.shift });
  }
  if (!moves.length) return false;
  store.moveNodes(moves);
  commitMoves(orig);
  placePivot();
  return true;
}

if (zField) {
  zField.addEventListener("input", () => { zDirty = true; });
  zField.addEventListener("focus", () => { zDirty = false; zField.select(); });
  zField.addEventListener("blur", () => { zDirty = false; refreshZField(); });
  zField.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (zDirty) applyHeightInput(zField.value);
      zDirty = false;
      zField.blur();
    } else if (e.key === "Escape") {
      e.preventDefault();
      zDirty = false;
      zField.blur();
    } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      e.preventDefault();
      const cur = Number(zField.value.replace(",", "."));
      if (!Number.isFinite(cur)) return;
      const step = (e.shiftKey ? 0.1 : 0.01) * (e.key === "ArrowUp" ? 1 : -1);
      zField.value = fmtZ(cur + step);
      zDirty = true;
    }
  });
}
selection.on("changed", () => refreshZField());
store.on("nodes-moved", () => refreshZField());

// --- JOSM actions --------------------------------------------------------------------

export function deleteSelection() {
  if (!selection.hasEditable()) {
    toast("Nothing selected to delete", "warn", 2000);
    return;
  }
  sendCommand([{ op: "delete_selection", ids: selection.tokens() }], { awaitResult: true })
    .then((info) => reportAction("Delete", info));
}

export function undo(redo = false) {
  sendCommand([{ op: redo ? "redo" : "undo" }], { awaitResult: true })
    .then((info) => reportAction(redo ? "Redo" : "Undo", info));
}

function reportAction(what, info, successToast = false) {
  const r = info.result;
  if (!info.delivered) toast(`${what}: JOSM is not connected`, "error");
  else if (r && r.ok === false) toast(`${what}: ${r.message}`, "warn");
  else if (r && r.warning) toast(r.warning, "warn", 8000);
  else if (r && successToast) toast(r.message, "info", 2500);
}

/**
 * The way to interpolate along and its anchors, from the selection: one way
 * (plus any of its nodes that are selected), or 2+ nodes that share a way.
 */
function interpolationTarget() {
  let way = null;
  if (selection.ways.size === 1) {
    way = selection.ways.values().next().value;
  } else if (selection.ways.size === 0 && selection.nodes.size >= 2) {
    const counts = new Map();
    for (const id of selection.nodes) {
      const rec = store.node(id);
      const seen = new Set();
      for (const { f } of rec ? rec.refs : []) {
        if (seen.has(f.id)) continue;
        seen.add(f.id);
        counts.set(f.id, (counts.get(f.id) || 0) + 1);
      }
    }
    for (const [fid, n] of counts) if (n === selection.nodes.size) way = fid;
  }
  if (!way) return null;
  const f = store.features.get(way);
  const onWay = f ? Array.from(selection.nodes).filter((id) => f.nodes.includes(id)) : [];
  return { way, anchors: onWay.map(nodeToken) };
}

/** Set heights along the selected way linearly between its anchors (JOSM computes). */
export function interpolateSelection() {
  const target = interpolationTarget();
  if (!target) {
    toast("Interpolate: select one way (optionally with 2+ of its nodes as anchors), or 2+ nodes of one way", "warn", 5000);
    return;
  }
  sendCommand([{ op: "interpolate_height", way: target.way, anchors: target.anchors }], { awaitResult: true })
    .then((info) => reportAction("Interpolate", info, true));
}

// --- pointer: click, box, cycle -------------------------------------------------------
// Registered right after TransformControls, whose own pointerdown therefore
// runs first and sets `dragging` when the press hits a handle. Its pointerup
// clears that again before ours runs, so remember it at press time. Box
// select stops the event so the navigation listeners (added later) never
// start an orbit for it.
let down = null;   // { x, y, onGizmo, button }
let box = null;    // { x0, y0, x1, y1, add, pointerId }
let cycle = null;  // { x, y, camKey, items, idx }
const boxEl = document.getElementById("boxSel");

canvas.addEventListener("pointerdown", (e) => {
  down = { x: e.clientX, y: e.clientY, onGizmo: transform.dragging, button: e.button };
  if (edit.on && e.button === 0 && (e.ctrlKey || e.metaKey) && !transform.dragging) {
    e.stopImmediatePropagation();
    box = { x0: e.clientX, y0: e.clientY, x1: e.clientX, y1: e.clientY, add: e.shiftKey, pointerId: e.pointerId };
    canvas.setPointerCapture(e.pointerId);
    drawBox();
  }
});

canvas.addEventListener("pointermove", (e) => {
  if (!box || e.pointerId !== box.pointerId) return;
  box.x1 = e.clientX;
  box.y1 = e.clientY;
  drawBox();
});

canvas.addEventListener("pointerup", (e) => {
  if (box && e.pointerId === box.pointerId) {
    const b = box;
    box = null;
    drawBox();
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    finishBox(b);
    return;
  }
  if (!down || transform.dragging) return;
  if (Math.abs(e.clientX - down.x) > CLICK_SLOP_PX || Math.abs(e.clientY - down.y) > CLICK_SLOP_PX) return;
  if (!edit.on) {
    // View mode: a click selects to inspect, unless it belongs elsewhere.
    if (document.pointerLockElement === canvas) return;
    if (e.button !== 0 && e.button !== 1) return;
    if (clickFilters.some((fn) => fn(e.clientX, e.clientY))) return;
  }
  if (e.button === 0) clickSelect(e.clientX, e.clientY, e.shiftKey, down.onGizmo);
  else if (e.button === 1) cycleAt(e.clientX, e.clientY);
});

function applyPick(item, add) {
  cycle = null;
  if (add) selection.toggle(item);
  else selection.set([item]);
}

/**
 * What a left click at (x, y) selects, or null. A click on a gizmo handle
 * keeps the selection rather than selecting whichever node lies near the
 * arrow; only a click squarely on a node dot counts there.
 */
export function clickTarget(x, y, onGizmo) {
  if (onGizmo) return pickItem(x, y, { nodePx: GIZMO_CLICK_NODE_PX, nodeOnly: true });
  return edit.on ? pickItem(x, y) : pickItem(x, y, { nodes: viewPick.nodes, lanelets: true });
}

function clickSelect(x, y, add, onGizmo) {
  const item = clickTarget(x, y, onGizmo);
  if (item) {
    applyPick(item, add);
  } else if (!add && !onGizmo) {
    cycle = null;
    selection.clear();
  }
}

const camKey = () => `${camera.position.toArray().join()},${view.yaw},${view.pitch}`;

/**
 * Middle-click: select the next of everything under the cursor, nodes first
 * (front to back), then ways. Repeated clicks on the same spot walk the list,
 * which is the only way to reach a node that sits exactly on another. View
 * mode follows its click: nodes only with the setting, lanelets last.
 */
function cycleAt(x, y) {
  if (cycle && Math.hypot(x - cycle.x, y - cycle.y) <= 3 && cycle.camKey === camKey()) {
    cycle.idx = (cycle.idx + 1) % cycle.items.length;
  } else {
    const view = !edit.on;
    const items = [
      ...(!view || viewPick.nodes ? nodesUnderCursor(x, y).map((n) => ({ type: "node", id: n.id })) : []),
      ...featuresUnderCursor(x, y).map((f) => ({ type: "way", id: f.id })),
      ...(view ? laneletsUnderCursor(x, y).map((id) => ({ type: "relation", id })) : []),
    ];
    if (!items.length) {
      cycle = null;
      return;
    }
    const cur = selection.single();
    const k = cur ? items.findIndex((it) => it.type === cur.type && it.id === cur.id) : -1;
    cycle = { x, y, camKey: camKey(), items, idx: k >= 0 ? (k + 1) % items.length : 0 };
  }
  const item = cycle.items[cycle.idx];
  selection.set([item]);
  const label = item.type === "node" ? nodeToken(item.id) : item.id;
  toast(`${cycle.idx + 1} of ${cycle.items.length} under the cursor: ${label}`, "info", 1500);
}

function drawBox() {
  if (!boxEl) return;
  if (!box) {
    boxEl.style.display = "none";
    return;
  }
  boxEl.style.display = "block";
  boxEl.style.left = `${Math.min(box.x0, box.x1)}px`;
  boxEl.style.top = `${Math.min(box.y0, box.y1)}px`;
  boxEl.style.width = `${Math.abs(box.x1 - box.x0)}px`;
  boxEl.style.height = `${Math.abs(box.y1 - box.y0)}px`;
}

/** Nodes inside the box, and ways entirely inside it (like JOSM's lasso). */
function finishBox(b) {
  const x0 = Math.min(b.x0, b.x1), x1 = Math.max(b.x0, b.x1);
  const y0 = Math.min(b.y0, b.y1), y1 = Math.max(b.y0, b.y1);
  if (x1 - x0 < 3 && y1 - y0 < 3) return;
  const pr = screenProjector();
  const inside = (p, i) => {
    const s = projectToPage(pr, p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    return s !== null && s[0] >= x0 && s[0] <= x1 && s[1] >= y0 && s[1] <= y1;
  };
  const items = [];
  const seen = new Set();
  for (const f of store.features.values()) {
    if (f.kind !== "line") continue;
    const n = Math.min(f.nodes.length, f.pos.length / 3);
    let all = n > 0;
    for (let i = 0; i < n; i++) {
      if (inside(f.pos, i)) {
        if (!seen.has(f.nodes[i])) {
          seen.add(f.nodes[i]);
          items.push({ type: "node", id: f.nodes[i] });
        }
      } else {
        all = false;
      }
    }
    if (all) items.push({ type: "way", id: f.id });
  }
  cycle = null;
  if (b.add) selection.add(items);
  else selection.set(items);
  toast(`Box: ${seen.size} node(s)`, "info", 1500);
}

// --- keys and toolbar ------------------------------------------------------------------

/** Edit keys; returns true when the event was consumed. */
export function handleEditKey(e) {
  const k = e.key;
  const ctrl = e.ctrlKey || e.metaKey;
  if (ctrl && (k === "z" || k === "Z")) {
    e.preventDefault();
    undo(e.shiftKey);
    return true;
  }
  if (ctrl && (k === "y" || k === "Y")) {
    e.preventDefault();
    undo(true);
    return true;
  }
  // Esc deselects in View mode too (in Edit mode below, with the tools).
  if (!edit.on && k === "Escape" && !ctrl && !selection.isEmpty() && !escapeBelongsElsewhere()) {
    cycle = null;
    selection.clear();
    return true;
  }
  if (!edit.on || ctrl || e.altKey) return false;
  if (k === "Delete" || k === "Backspace") {
    e.preventDefault();
    deleteSelection();
    return true;
  }
  if (k === "Escape") {
    cycle = null;
    selection.clear();
    return true;
  }
  if (e.repeat) return false;
  if (k === "g" || k === "G") { setTool("translate"); return true; }
  if (k === "r" || k === "R") { setTool("rotate"); return true; }
  if (k === "h" || k === "H") { toggleHeightOnly(); return true; }
  if (k === "i" || k === "I") { interpolateSelection(); return true; }
  if (k === "m" || k === "M") { toggleSnap(); return true; }
  if ((k === "z" || k === "Z") && zField && !zField.disabled) {
    e.preventDefault();
    zField.focus();
    return true;
  }
  if (k === "t" || k === "T") {
    setKeysMove(edit.keysMove === "camera" ? "selection" : "camera");
    toast(`Keys move the ${edit.keysMove}`, "info", 1500);
    return true;
  }
  return false;
}

for (const b of document.querySelectorAll("#editBar [data-edit]")) {
  b.addEventListener("click", () => {
    const k = b.getAttribute("data-edit");
    if (k === "translate" || k === "rotate") setTool(k);
    else if (k === "height") toggleHeightOnly();
    else if (k === "interpolate") interpolateSelection();
    else if (k === "delete") deleteSelection();
    else if (k === "undo") undo(false);
    else if (k === "redo") undo(true);
    else if (k === "keys") setKeysMove(edit.keysMove === "camera" ? "selection" : "camera");
    else if (k === "snap") toggleSnap();
  });
}
const editBtn = document.getElementById("editBtn");
if (editBtn) editBtn.addEventListener("click", () => setEditMode(!edit.on));
const viewModeBtn = document.getElementById("viewModeBtn");
if (viewModeBtn) viewModeBtn.addEventListener("click", () => setEditMode(false));

// A read-only map arriving (the scene server's copy) ends editing, and the
// switch says why it is off.
store.on("snapshot", () => {
  const why = editBlockedReason();
  if (why && edit.on) setEditMode(false);
  if (editBtn) {
    editBtn.classList.toggle("blocked", !!why);
    editBtn.dataset.tip = why || "Select and move nodes and ways";
  }
});
