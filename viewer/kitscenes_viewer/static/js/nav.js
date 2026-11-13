// Mouse navigation. Replaces OrbitControls, whose pivot was a fixed target
// that framing put at the centre of the whole map and that walking / looking
// kept at the same distance, so every orbit swung around a point hundreds of
// metres away.
//
//   left-drag              orbit around the map point under the cursor at
//                          press time; over empty sky, turn in place
//   right-drag             turn in place (FPV look)
//   middle / shift+left    pan; the grabbed point stays under the cursor
//   wheel                  move toward / away from the point under the cursor
import * as THREE from "three";
import { now } from "./util.js";
import { canvas } from "./scene.js";
import { camera, view, camRefs, orbitCamera, look, pixelsToMetres, ortho } from "./camera.js";
import { pickMapPoint, cursorRay } from "./picking.js";
import { pivotMarker, zoomMarker } from "./render/overlays.js";

const NAV_DRAG_THRESHOLD_PX = 3;          // below this a press is a click (selection)
const ORBIT_RAD_PER_HEIGHT = 2 * Math.PI; // full-height drag = one turn, as OrbitControls
export const LOOK_RAD_PER_PX = 0.0025;    // right-drag and captured-mouse look
const WHEEL_STEP = 0.85;                  // distance factor per wheel notch
const DOLLY_MIN_M = 0.05;
const ZOOM_MARKER_MS = 700;     // how long the zoom target stays marked after the wheel stops
const _panRight = new THREE.Vector3();
const _panUp = new THREE.Vector3();
const _dolly = new THREE.Vector3();

export const nav = {
  pointerId: null,  // pointer that owns the current drag gesture
  mode: null,       // "orbit" | "look" | "pan"
  moving: false,    // drag threshold passed
  downX: 0, downY: 0, lastX: 0, lastY: 0,
  pivot: null,      // orbit pivot of the current drag; null turns in place
  panDepth: 0,      // view depth of the grabbed point while panning
  lastPivot: null,  // for the HUD
  lastPivotKind: "",
  zoomTarget: null, // last wheel target (THREE.Vector3) or null over sky
};
let zoomMarkerTimer = null;

/** True while a mouse gesture owns the camera. */
export const navBusy = () => nav.pointerId !== null;

/**
 * Register the listeners. Call after TransformControls exists: it decides
 * `dragging` inside its own pointerdown, so only a later listener sees that a
 * press belongs to the gizmo. `gizmoBusy()` and `pointerLocked()` gate input.
 */
export function installNav({ gizmoBusy, pointerLocked }) {
  canvas.style.touchAction = "none";
  canvas.addEventListener("contextmenu", (e) => e.preventDefault()); // right-drag looks

  canvas.addEventListener("pointerdown", (e) => {
    if (nav.pointerId !== null || pointerLocked() || gizmoBusy()) return;
    let mode = null;
    // In the orthographic map view a left-drag pans, as in any map; turning
    // it is the right-drag.
    if (e.button === 0) mode = e.shiftKey || ortho.on ? "pan" : "orbit";
    else if (e.button === 1) mode = "pan";
    else if (e.button === 2) mode = "look";
    if (!mode) return;
    if (e.button === 1) e.preventDefault(); // no middle-click autoscroll
    nav.pointerId = e.pointerId;
    nav.mode = mode;
    nav.moving = false;
    nav.downX = nav.lastX = e.clientX;
    nav.downY = nav.lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (e.pointerId !== nav.pointerId) return;
    if (!nav.moving) {
      if (Math.hypot(e.clientX - nav.downX, e.clientY - nav.downY) < NAV_DRAG_THRESHOLD_PX) return;
      beginDrag();
    }
    const dx = e.clientX - nav.lastX;
    const dy = e.clientY - nav.lastY;
    nav.lastX = e.clientX;
    nav.lastY = e.clientY;
    drag(dx, dy);
  });

  canvas.addEventListener("pointerup", endNav);
  canvas.addEventListener("pointercancel", endNav);
  canvas.addEventListener("lostpointercapture", endNav);
  canvas.addEventListener("wheel", (e) => onWheel(e, gizmoBusy, pointerLocked), { passive: false });
}

// Pick lazily once the press turns into a drag, so plain clicks stay cheap.
// Nothing has moved since the press, so picking at the press position is exact.
function beginDrag() {
  nav.moving = true;
  if (nav.mode === "look") return;
  const hit = pickMapPoint(nav.downX, nav.downY);
  if (hit) camRefs.refDist = Math.max(DOLLY_MIN_M, camera.position.distanceTo(hit.point));
  if (nav.mode === "orbit") {
    nav.pivot = hit ? hit.point : null;
    nav.lastPivot = nav.pivot;
    nav.lastPivotKind = hit ? hit.kind : "none";
    pivotMarker.visible = !!hit;
    if (hit) pivotMarker.position.copy(hit.point);
  } else {
    nav.panDepth = hit ? Math.max(DOLLY_MIN_M, hit.depth) : camRefs.refDist;
  }
}

function drag(dx, dy) {
  if (nav.mode === "pan") {
    // Screen-parallel move that keeps the grabbed depth under the cursor.
    const m = pixelsToMetres(1, nav.panDepth);
    _panRight.set(1, 0, 0).applyQuaternion(camera.quaternion);
    _panUp.set(0, 1, 0).applyQuaternion(camera.quaternion);
    camera.position.addScaledVector(_panRight, -dx * m).addScaledVector(_panUp, dy * m);
    camera.updateMatrixWorld();
  } else if (nav.mode === "orbit" && nav.pivot) {
    // Drag right swings the camera left around the pivot (the scene turns with
    // the cursor); drag down lifts it to look down more.
    const k = ORBIT_RAD_PER_HEIGHT / (canvas.clientHeight || 1);
    orbitCamera(nav.pivot, -dx * k, -dy * k);
  } else {
    look(dx * LOOK_RAD_PER_PX, -dy * LOOK_RAD_PER_PX);
  }
}

export function endNav(e) {
  if (nav.pointerId === null || (e && e.pointerId !== nav.pointerId)) return;
  const id = nav.pointerId;
  nav.pointerId = null;
  nav.mode = null;
  nav.moving = false;
  nav.pivot = null;
  pivotMarker.visible = false;
  if (canvas.hasPointerCapture(id)) canvas.releasePointerCapture(id);
}

// Dolly along the cursor ray toward the point under it. Moving along that ray
// keeps the point on the same pixel, so while the cursor and camera rest the
// previous pick stays valid and is reused rather than re-walking every line.
let wheelPick = null; // { x, y, t, point, pos, yaw, pitch }

function onWheel(e, gizmoBusy, pointerLocked) {
  e.preventDefault();
  if (gizmoBusy() || pointerLocked()) return;
  let notches = e.deltaMode === 1 ? e.deltaY / 3 : e.deltaMode === 2 ? e.deltaY : e.deltaY / 100;
  notches = Math.max(-5, Math.min(5, notches));
  if (!notches) return;
  const scale = Math.pow(WHEEL_STEP, -notches); // < 1 moves closer
  const t = now();
  const reuse = wheelPick && t - wheelPick.t < 400
    && Math.hypot(e.clientX - wheelPick.x, e.clientY - wheelPick.y) <= 2
    && camera.position.distanceToSquared(wheelPick.pos) < 1e-10
    && view.yaw === wheelPick.yaw && view.pitch === wheelPick.pitch;
  if (!reuse) {
    const hit = pickMapPoint(e.clientX, e.clientY);
    wheelPick = { x: e.clientX, y: e.clientY, point: hit ? hit.point : null, pos: new THREE.Vector3() };
  }
  // Mark the target: the wheel zooms toward the point under the cursor, not
  // toward the orbit pivot, and without a marker that is easy to misread.
  nav.zoomTarget = wheelPick.point;
  zoomMarker.visible = !!wheelPick.point;
  if (wheelPick.point) zoomMarker.position.copy(wheelPick.point);
  if (zoomMarkerTimer !== null) clearTimeout(zoomMarkerTimer);
  zoomMarkerTimer = setTimeout(() => { zoomMarker.visible = false; zoomMarkerTimer = null; }, ZOOM_MARKER_MS);
  if (wheelPick.point) {
    const dist = camera.position.distanceTo(wheelPick.point);
    const next = Math.max(DOLLY_MIN_M, dist * scale);
    _dolly.subVectors(camera.position, wheelPick.point).setLength(next);
    camera.position.copy(wheelPick.point).add(_dolly);
    camRefs.refDist = next;
  } else {
    // Over sky: step along the cursor ray, sized like the last real target.
    const ray = cursorRay(e.clientX, e.clientY);
    camera.position.addScaledVector(ray.direction, camRefs.refDist * (1 - scale));
  }
  camera.updateMatrixWorld();
  wheelPick.t = t;
  wheelPick.pos.copy(camera.position);
  wheelPick.yaw = view.yaw;
  wheelPick.pitch = view.pitch;
}
