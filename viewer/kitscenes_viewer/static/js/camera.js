// Camera state and every way of moving it (orbit, look, walk, framing).
//
// The camera has no look-at target. It is a position plus a heading
// (`view.yaw`: rotation about world Z, 0 = looking north along +Y,
// counter-clockwise positive) and a pitch (0 = level, -90° = straight down).
// World Z is always screen-up, so there is no roll, and a straight-down BEV is
// an ordinary state with a defined north rather than a lookAt singularity.
//
// Orthographic mode keeps the same camera object and the same meaning of its
// height: the frustum is sized to show exactly what the perspective camera
// would show at the ground from there. So zooming is still moving down, and
// everything that sizes things by distance keeps working; only the parallax
// goes. The camera then flags itself orthographic, which is what three's
// Raycaster and TransformControls branch on. Pitch is pinned straight down:
// this is a map view, turned about the vertical only.
import * as THREE from "three";
import { renderer } from "./scene.js";

export const CAM_NEAR_M = 0.02;
export const CAM_FAR_M = 500000;
const HALF_PI = Math.PI / 2;
const DEFAULT_BEV_HEIGHT_M = 100;
const _worldUp = new THREE.Vector3(0, 0, 1);
const _camX = new THREE.Vector3(1, 0, 0);
const _qYaw = new THREE.Quaternion();
const _qPitch = new THREE.Quaternion();
const _qOrbit = new THREE.Quaternion();
const _orbitAxis = new THREE.Vector3();
const _orbitOffset = new THREE.Vector3();
const _lookDir = new THREE.Vector3();

export const camera = new THREE.PerspectiveCamera(
  55, window.innerWidth / window.innerHeight, CAM_NEAR_M, CAM_FAR_M);
camera.up.copy(_worldUp); // Z is up (ENU)

export const view = { yaw: 0, pitch: 0 };

/**
 * Scale hints shared by navigation and framing: the distance of the last map
 * pick (sizes moves over empty sky) and the map height near the cursor (the
 * ground-plane fallback of picking).
 */
export const camRefs = { refDist: 20, groundZ: 0 };

/** Orthographic mode; `groundZ` is the height the scale is measured from. */
export const ortho = { on: false, groundZ: 0 };
// Depth range around the camera; the near plane is *behind* it, so geometry
// above a low camera (a tall building while zoomed in) is still drawn.
const ORTHO_NEAR_M = -10000;
const ORTHO_FAR_M = 100000;

const perspectiveProjection = camera.updateProjectionMatrix.bind(camera);
// Every caller (resize, the camera lock's release) gets the right projection
// for the current mode.
camera.updateProjectionMatrix = function updateProjectionMatrix() {
  if (!ortho.on) {
    perspectiveProjection();
    return;
  }
  const top = orthoHalfHeight();
  const right = top * camera.aspect;
  camera.left = -right;
  camera.right = right;
  camera.top = top;
  camera.bottom = -top;
  camera.projectionMatrix.makeOrthographic(-right, right, top, -top, ORTHO_NEAR_M, ORTHO_FAR_M);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
};

/** Half the visible ground height in ortho: what the perspective shows from this high. */
function orthoHalfHeight() {
  const h = Math.max(0.5, camera.position.z - ortho.groundZ);
  return h * Math.tan((camera.fov * Math.PI) / 360);
}

/** In ortho the frustum follows the camera height; call before projecting. */
export function syncProjection() {
  if (ortho.on) camera.updateProjectionMatrix();
}

/**
 * Switch between perspective and a straight-down orthographic view. Going
 * ortho looks straight down, north up, at what the camera was looking at,
 * from as far as it was; going back keeps that view, now with perspective.
 */
export function setOrtho(on) {
  const next = !!on;
  if (next === ortho.on) return ortho.on;
  if (next) {
    camera.getWorldDirection(_lookDir);
    const g = camRefs.groundZ;
    let target = null;
    if (_lookDir.z < -1e-3) {
      const t = (g - camera.position.z) / _lookDir.z;
      if (t > 0 && t < 5000) target = camera.position.clone().addScaledVector(_lookDir, t);
    }
    if (!target) target = new THREE.Vector3(camera.position.x, camera.position.y, g);
    const dist = Math.max(5, camera.position.distanceTo(target));
    ortho.groundZ = target.z;
    ortho.on = true;
    camera.isPerspectiveCamera = false;
    camera.isOrthographicCamera = true;
    setCameraView(new THREE.Vector3(target.x, target.y, target.z + dist), 0, -HALF_PI);
  } else {
    ortho.on = false;
    camera.isPerspectiveCamera = true;
    camera.isOrthographicCamera = false;
  }
  camera.updateProjectionMatrix();
  window.dispatchEvent(new CustomEvent("ll2:ortho", { detail: { on: ortho.on } }));
  return ortho.on;
}

/** Ground metres per screen pixel at the view centre. */
export function metresPerPixel() {
  const h = renderer.domElement.clientHeight || window.innerHeight || 1;
  if (ortho.on) return (2 * orthoHalfHeight()) / h;
  return pixelsToMetres(1, Math.max(0.5, camera.position.z - camRefs.groundZ));
}

export function clampPitch(p) {
  return Math.max(-HALF_PI, Math.min(HALF_PI, p));
}

// Rebuild the camera orientation from `view`: tip the default -Z look
// direction up to the horizon (+90° about X), then turn it about world Z.
function applyCameraRotation() {
  _qYaw.setFromAxisAngle(_worldUp, view.yaw);
  _qPitch.setFromAxisAngle(_camX, HALF_PI + view.pitch);
  camera.quaternion.multiplyQuaternions(_qYaw, _qPitch);
  camera.updateMatrixWorld();
}

export function setCameraView(position, yaw, pitch) {
  if (position) camera.position.copy(position);
  view.yaw = Math.atan2(Math.sin(yaw), Math.cos(yaw));
  view.pitch = ortho.on ? -HALF_PI : clampPitch(pitch);
  applyCameraRotation();
  syncProjection();
}

/** Move the camera without turning it. */
export function translateCamera(dx, dy, dz) {
  camera.position.x += dx;
  camera.position.y += dy;
  camera.position.z += dz;
  camera.updateMatrixWorld();
  syncProjection();
}

// Aim at `target` from the current position. Straight up or down has no
// heading, so that keeps `yawIfVertical` (default: the current heading).
export function lookAtPoint(target, yawIfVertical) {
  _lookDir.subVectors(target, camera.position);
  const horiz = Math.hypot(_lookDir.x, _lookDir.y);
  const yaw = horiz > 1e-9
    ? Math.atan2(-_lookDir.x, _lookDir.y)
    : (yawIfVertical === undefined ? view.yaw : yawIfVertical);
  setCameraView(null, yaw, Math.atan2(_lookDir.z, horiz));
}

// Rotate the camera about `pivot`: heading about the vertical through the
// pivot, pitch about the camera's horizontal right axis through it. Position
// and orientation turn by the same rotation, so the pivot stays on the same
// pixel and nothing snaps to centre it.
export function orbitCamera(pivot, dYaw, dPitch) {
  // A map view turns about the vertical only.
  const pitch = ortho.on ? view.pitch : clampPitch(view.pitch + dPitch);
  _orbitOffset.subVectors(camera.position, pivot);
  _orbitAxis.set(Math.cos(view.yaw), Math.sin(view.yaw), 0); // camera right
  _qOrbit.setFromAxisAngle(_orbitAxis, pitch - view.pitch);
  _orbitOffset.applyQuaternion(_qOrbit);
  _qOrbit.setFromAxisAngle(_worldUp, dYaw);
  _orbitOffset.applyQuaternion(_qOrbit);
  setCameraView(_orbitOffset.add(pivot), view.yaw + dYaw, pitch);
}

// FPS look: turn in place. Positive yaw turns right, positive pitch looks up.
export function look(yawRad, pitchRad) {
  setCameraView(null, view.yaw - yawRad, view.pitch + pitchRad);
}

// Walk level along the heading, so forward is screen-up even in a straight-down
// BEV. dx = right, dy = forward, in units of `metres`.
export function walk(dx, dy, metres) {
  const s = Math.sin(view.yaw);
  const c = Math.cos(view.yaw);
  // right = (c, s), forward = (-s, c)
  translateCamera((c * dx - s * dy) * metres, (s * dx + c * dy) * metres, 0);
}

export function elevate(dz, metres) {
  translateCamera(0, 0, dz * metres);
}

/** Compass heading in degrees, clockwise from north. */
export function headingDeg() {
  return (((-view.yaw * 180) / Math.PI) % 360 + 360) % 360;
}

// Metres spanned by `px` screen pixels at `depth` metres in front of the camera.
export function pixelsToMetres(px, depth) {
  const h = renderer.domElement.clientHeight || window.innerHeight || 1;
  if (ortho.on) return (px * 2 * orthoHalfHeight()) / h; // the same at every depth
  return (px * 2 * depth * Math.tan((camera.fov * Math.PI) / 360)) / h;
}

// --- framing -------------------------------------------------------------------
// Each returns the numbers the frame-debug panel shows.

/** Oblique view of `box` (F). */
export function frameBox(box) {
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 5) * 0.5;
  const dist = (radius / Math.tan((camera.fov * Math.PI) / 360)) * 1.6;
  if (ortho.on) {
    // No oblique view in a map: straight down over the middle, heading kept.
    const camPos = new THREE.Vector3(center.x, center.y, ortho.groundZ + dist);
    setCameraView(camPos, view.yaw, -HALF_PI);
    camRefs.groundZ = center.z;
    camRefs.refDist = dist;
    camera.updateProjectionMatrix();
    return { radius, dist, camPos, target: center };
  }
  const camPos = new THREE.Vector3(center.x + dist * 0.4, center.y - dist * 0.8, center.z + dist * 0.6);
  camera.position.copy(camPos);
  lookAtPoint(center);
  camRefs.groundZ = center.z;
  camRefs.refDist = camPos.distanceTo(center);
  return { radius, dist, camPos, target: center };
}

/** Straight-down view of `box`, north up (B). */
export function bevBox(box) {
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, 5) * 0.5;
  const dist = (radius / Math.tan((camera.fov * Math.PI) / 360)) * 1.6;
  // In ortho the scale is measured from ortho.groundZ, not the box.
  const camPos = new THREE.Vector3(center.x, center.y, (ortho.on ? ortho.groundZ : center.z) + dist);
  setCameraView(camPos, 0, -HALF_PI);
  camRefs.groundZ = center.z;
  camRefs.refDist = dist;
  return { radius, dist, camPos, target: center };
}

/** Fixed sanity view when framing fails or data is far from the origin. */
export function defaultBev() {
  // In ortho the height is the scale, measured from ortho.groundZ.
  setCameraView(new THREE.Vector3(0, 0, (ortho.on ? ortho.groundZ : 0) + DEFAULT_BEV_HEIGHT_M), 0, -HALF_PI);
  camRefs.refDist = DEFAULT_BEV_HEIGHT_M;
  return DEFAULT_BEV_HEIGHT_M;
}

camera.position.set(40, -60, 50);
lookAtPoint(new THREE.Vector3(0, 0, 0));
