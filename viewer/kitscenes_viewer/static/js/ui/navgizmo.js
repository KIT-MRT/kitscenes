// The compass in the corner of the view, with the view shortcuts next to it.
// The rose turns with the camera; a click turns the view to face north,
// about the map point in the middle of the view so that point stays put.
import * as THREE from "three";
import { camera, view, orbitCamera, camRefs } from "../camera.js";
import { canvas } from "../scene.js";
import { pickMapPoint } from "../picking.js";
import { rigActive } from "../sensors/camrig.js";

const _dir = new THREE.Vector3();
let rose = null;
let shown = NaN;

/** Compass heading of the real look direction, degrees clockwise from north. */
export function lookHeadingDeg() {
  camera.getWorldDirection(_dir);
  if (Math.hypot(_dir.x, _dir.y) < 1e-6) {
    // Straight down: the screen's up direction is what the heading means.
    _dir.set(0, 1, 0).applyQuaternion(camera.quaternion);
  }
  return ((Math.atan2(_dir.x, _dir.y) * 180) / Math.PI + 360) % 360;
}

/** Turn the view to face north, keeping the middle of the view in place. */
export function faceNorth() {
  if (rigActive()) return false;
  const r = canvas.getBoundingClientRect();
  const hit = pickMapPoint(r.left + r.width / 2, r.top + r.height / 2);
  const pivot = hit ? hit.point.clone()
    : new THREE.Vector3(camera.position.x, camera.position.y, camRefs.groundZ);
  let turn = -view.yaw;
  turn = Math.atan2(Math.sin(turn), Math.cos(turn));
  orbitCamera(pivot, turn, 0);
  return true;
}

export function installNavGizmo() {
  rose = document.getElementById("compassRose");
  document.getElementById("compassBtn")?.addEventListener("click", () => faceNorth());
  document.getElementById("northBtn")?.addEventListener("click", () => faceNorth());
}

/** Per frame: turn the rose; cheap when nothing changed. */
export function updateNavGizmo() {
  if (!rose) return;
  const h = lookHeadingDeg();
  if (Math.abs(h - shown) < 0.2) return;
  shown = h;
  rose.style.transform = `rotate(${(-h).toFixed(1)}deg)`;
}
