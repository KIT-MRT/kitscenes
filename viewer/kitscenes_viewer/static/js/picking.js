// Screen-space picking over the feature store.
//
// Tolerances are in screen pixels. three's Line/Points raycast thresholds are
// world metres, which grabs the wrong vertex up close (nodes 20 cm apart) and
// misses everything far away. Tile and feature bounding spheres prune the
// search, so a click on a city map only walks the segments near the cursor.
import * as THREE from "three";
import { store } from "./store.js";
import { canvas } from "./scene.js";
import { camera, camRefs, ortho, syncProjection } from "./camera.js";
import { laneletLayer } from "./render/lanelets.js";

export const PICK_RADIUS_PX = 10;    // a line / node this close counts as under the cursor
export const PICK_TIE_PX = 1.5;      // candidates this close on screen: nearer one wins
const HEIGHT_PROBE_PX = 160;         // nearest line within this sets the local ground height
const GROUND_PICK_MAX_M = 5000;      // ground hits further out count as sky

const raycaster = new THREE.Raycaster();
const _ndc = new THREE.Vector2();
const _segA = new Float64Array(6); // world x, y, z, view x, view y, depth
const _segB = new Float64Array(6);
const _tmp = new Float64Array(6);
const _segHit = { px: 0, depth: 0, x: 0, y: 0, z: 0 };

export function screenProjector() {
  camera.updateMatrixWorld();
  syncProjection();
  const rect = canvas.getBoundingClientRect();
  return {
    e: camera.matrixWorldInverse.elements,
    f: rect.height / 2 / Math.tan((camera.fov * Math.PI) / 360), // pixels per unit at depth 1
    // Orthographic: pixels per metre, the same at every depth.
    ortho: ortho.on,
    k: ortho.on ? rect.height / 2 / camera.top : 0,
    // In ortho the frustum reaches behind the camera (see camera.js), so
    // geometry above a low camera is drawn and must be pickable too.
    near: ortho.on ? -1e4 : camera.near,
    cx: rect.left + rect.width / 2,
    cy: rect.top + rect.height / 2,
  };
}

/** Page x / y of a view-space point at depth d. */
const screenX = (pr, vx, d) => pr.cx + (pr.ortho ? pr.k * vx : (pr.f * vx) / d);
const screenY = (pr, vy, d) => pr.cy - (pr.ortho ? pr.k * vy : (pr.f * vy) / d);

export function toViewSpace(e, x, y, z, out) {
  out[0] = x; out[1] = y; out[2] = z;
  out[3] = e[0] * x + e[4] * y + e[8] * z + e[12];
  out[4] = e[1] * x + e[5] * y + e[9] * z + e[13];
  out[5] = -(e[2] * x + e[6] * y + e[10] * z + e[14]); // the camera looks down -Z
}

/** Page pixel [x, y] of a world point, or null behind the camera. */
export function projectToPage(pr, x, y, z) {
  toViewSpace(pr.e, x, y, z, _tmp);
  const d = _tmp[5];
  if (d < pr.near) return null;
  return [screenX(pr, _tmp[3], d), screenY(pr, _tmp[4], d), d];
}

// Conservative test: can any point of sphere {x,y,z,r} land within `radiusPx`
// of the cursor? Angles from the eye: every direction within radiusPx of the
// cursor lies within alpha of the cursor ray (sin alpha <= radiusPx / f, as
// the image plane is at distance >= f), the sphere covers the directions
// within beta = asin(r / distance) of its centre, and the two cones meet only
// if the angle between their axes is at most alpha + beta. Unlike a
// screen-space bound this stays tight for big spheres around the camera
// (street level, where half the tiles reach past the near plane).
function sphereMayBeNear(s, pr, mx, my, radiusPx) {
  toViewSpace(pr.e, s.x, s.y, s.z, _tmp);
  const vx = _tmp[3], vy = _tmp[4], d = _tmp[5]; // centre in view space, d along the view axis
  if (pr.ortho) {
    // No perspective: the sphere is a disc of radius r * k pixels.
    return Math.hypot(screenX(pr, vx, d) - mx, screenY(pr, vy, d) - my) <= radiusPx + s.r * pr.k;
  }
  const len = Math.hypot(vx, vy, d);
  if (len <= s.r) return true; // the camera is inside
  const ax = (mx - pr.cx) / pr.f;
  const ay = (pr.cy - my) / pr.f; // cursor ray (ax, ay, 1) in the same frame
  const cosG = (vx * ax + vy * ay + d) / (len * Math.hypot(ax, ay, 1));
  const sinA = Math.min(1, radiusPx / pr.f);
  const sinB = s.r / len;
  const cosAB = Math.sqrt(1 - sinA * sinA) * Math.sqrt(1 - sinB * sinB) - sinA * sinB; // cos(alpha + beta)
  return cosG >= cosAB;
}

/** Line features whose tile and own sphere may come within `radiusPx`. */
function* candidateFeatures(pr, mx, my, radiusPx) {
  for (const tile of store.tiles.values()) {
    if (!sphereMayBeNear(store.tileSphere(tile), pr, mx, my, radiusPx)) continue;
    for (const f of tile.features) {
      if (f.kind !== "line" || f.pos.length < 6) continue;
      if (!sphereMayBeNear(f.bs, pr, mx, my, radiusPx)) continue;
      yield f;
    }
  }
}

// Closest approach of the cursor to segment A-B on screen, clipped at the near
// plane. Fills _segHit with the pixel distance and the 3D point on the segment
// under that screen position (perspective-correct).
function segmentUnderCursor(pr, mx, my, A, B) {
  const near = pr.near;
  const da = A[5];
  const db = B[5];
  if (da < near && db < near) return false;
  const t0 = da < near ? (near - da) / (db - da) : 0;
  const t1 = db < near ? (near - da) / (db - da) : 1;
  const d0 = da + (db - da) * t0;
  const d1 = da + (db - da) * t1;
  const sx0 = screenX(pr, A[3] + (B[3] - A[3]) * t0, d0);
  const sy0 = screenY(pr, A[4] + (B[4] - A[4]) * t0, d0);
  const sx1 = screenX(pr, A[3] + (B[3] - A[3]) * t1, d1);
  const sy1 = screenY(pr, A[4] + (B[4] - A[4]) * t1, d1);
  const ex = sx1 - sx0;
  const ey = sy1 - sy0;
  const len2 = ex * ex + ey * ey;
  let s = len2 > 1e-12 ? ((mx - sx0) * ex + (my - sy0) * ey) / len2 : 0;
  s = s < 0 ? 0 : s > 1 ? 1 : s;
  // Screen fraction s -> fraction u of the clipped 3D segment, then back to A-B
  // (linear in ortho, where depth does not divide).
  const u = pr.ortho ? s : (s * d0) / ((1 - s) * d1 + s * d0);
  const t = t0 + (t1 - t0) * u;
  _segHit.px = Math.hypot(sx0 + ex * s - mx, sy0 + ey * s - my);
  _segHit.depth = d0 + (d1 - d0) * u;
  _segHit.x = A[0] + (B[0] - A[0]) * t;
  _segHit.y = A[1] + (B[1] - A[1]) * t;
  _segHit.z = A[2] + (B[2] - A[2]) * t;
  return true;
}

function isBetterPick(px, depth, best) {
  if (!best) return true;
  if (px < best.px - PICK_TIE_PX) return true;
  return px <= best.px + PICK_TIE_PX && depth < best.depth;
}

/**
 * Walk the segments of every candidate line, calling visit(f, segIndex) with
 * _segHit filled for segments within `radiusPx`.
 */
function forEachSegmentNear(pr, mx, my, radiusPx, visit) {
  for (const f of candidateFeatures(pr, mx, my, radiusPx)) {
    const p = f.pos;
    let A = _segA;
    let B = _segB;
    const n = p.length / 3;
    for (let i = 0; i < n; i++) {
      toViewSpace(pr.e, p[i * 3], p[i * 3 + 1], p[i * 3 + 2], B);
      if (i > 0 && segmentUnderCursor(pr, mx, my, A, B) && _segHit.px <= radiusPx) visit(f, i - 1);
      const prev = A;
      A = B;
      B = prev;
    }
  }
}

export function cursorRay(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  _ndc.set(
    ((clientX - rect.left) / rect.width) * 2 - 1,
    -((clientY - rect.top) / rect.height) * 2 + 1,
  );
  camera.updateMatrixWorld();
  syncProjection();
  raycaster.setFromCamera(_ndc, camera);
  if (ortho.on) {
    // three starts an orthographic ray on the near plane, which sits behind
    // the camera here; start it level with the camera instead, so distances
    // along it mean what they mean in perspective.
    const back = _rayTmp.subVectors(raycaster.ray.origin, camera.position).dot(raycaster.ray.direction);
    raycaster.ray.origin.addScaledVector(raycaster.ray.direction, -back);
  }
  return raycaster.ray;
}
const _rayTmp = new THREE.Vector3();

// The map point under the cursor: the nearest line within PICK_RADIUS_PX;
// else the lanelet surface under it; else the cursor ray on a level plane at
// the height of the nearest line (the map height near the cursor); else null
// (sky). Returns { point, kind: "line" | "surface" | "ground", featureId, depth }.
export function pickMapPoint(clientX, clientY) {
  const pr = screenProjector();
  let hit = null;
  let probe = null;
  const keep = (best, f) => {
    const out = best || {};
    out.px = _segHit.px;
    out.depth = _segHit.depth;
    out.x = _segHit.x;
    out.y = _segHit.y;
    out.z = _segHit.z;
    out.featureId = f.id;
    return out;
  };
  forEachSegmentNear(pr, clientX, clientY, HEIGHT_PROBE_PX, (f) => {
    const { px, depth } = _segHit;
    if (isBetterPick(px, depth, probe)) probe = keep(probe, f);
    if (px <= PICK_RADIUS_PX && isBetterPick(px, depth, hit)) hit = keep(hit, f);
  });
  if (hit) {
    camRefs.groundZ = hit.z;
    return {
      point: new THREE.Vector3(hit.x, hit.y, hit.z),
      kind: "line", featureId: hit.featureId, depth: hit.depth,
    };
  }
  if (probe) camRefs.groundZ = probe.z;
  const ray = cursorRay(clientX, clientY);
  const surfaces = laneletLayer.surfaceMeshes();
  if (surfaces.length) {
    const hits = raycaster.intersectObjects(surfaces, false);
    if (hits.length) {
      const point = hits[0].point.clone();
      camRefs.groundZ = point.z;
      toViewSpace(pr.e, point.x, point.y, point.z, _tmp);
      return { point, kind: "surface", featureId: null, depth: _tmp[5] };
    }
  }
  if (Math.abs(ray.direction.z) < 1e-9) return null;
  const t = (camRefs.groundZ - ray.origin.z) / ray.direction.z;
  if (!(t > camera.near) || t > GROUND_PICK_MAX_M) return null;
  const point = ray.at(t, new THREE.Vector3());
  toViewSpace(pr.e, point.x, point.y, point.z, _tmp);
  return { point, kind: "ground", featureId: null, depth: _tmp[5] };
}

/**
 * Every node within `radiusPx` of the cursor, nearest on screen first; near
 * ties (PICK_TIE_PX) go to the one closer to the camera. Coincident nodes all
 * come back, which is what middle-click cycling walks through.
 */
export function nodesUnderCursor(clientX, clientY, radiusPx = PICK_RADIUS_PX) {
  const pr = screenProjector();
  const byId = new Map();
  for (const f of candidateFeatures(pr, clientX, clientY, radiusPx)) {
    const p = f.pos;
    const n = Math.min(f.nodes.length, p.length / 3);
    for (let i = 0; i < n; i++) {
      toViewSpace(pr.e, p[i * 3], p[i * 3 + 1], p[i * 3 + 2], _tmp);
      const d = _tmp[5];
      if (d < pr.near) continue;
      const px = Math.hypot(screenX(pr, _tmp[3], d) - clientX, screenY(pr, _tmp[4], d) - clientY);
      if (px > radiusPx) continue;
      const id = f.nodes[i];
      const prev = byId.get(id);
      if (!prev || px < prev.px) byId.set(id, { id, px, depth: d });
    }
  }
  return sortCandidates(Array.from(byId.values()));
}

/** Every line within `radiusPx` of the cursor, ordered like nodesUnderCursor. */
export function featuresUnderCursor(clientX, clientY, radiusPx = PICK_RADIUS_PX) {
  const pr = screenProjector();
  const byId = new Map();
  forEachSegmentNear(pr, clientX, clientY, radiusPx, (f) => {
    const prev = byId.get(f.id);
    if (!prev || _segHit.px < prev.px) byId.set(f.id, { id: f.id, px: _segHit.px, depth: _segHit.depth });
  });
  return sortCandidates(Array.from(byId.values()));
}

// Screen distance in PICK_TIE_PX bands, then depth: coincident or nearly
// coincident candidates order front to back. (A pairwise "within tie distance"
// comparator is not transitive, which Array.sort does not tolerate.)
function sortCandidates(list) {
  return list.sort((a, b) => {
    const ba = Math.floor(a.px / PICK_TIE_PX);
    const bb = Math.floor(b.px / PICK_TIE_PX);
    return ba !== bb ? ba - bb : a.depth - b.depth;
  });
}

/** Node id nearest the cursor within `radiusPx`, or null. */
export function pickNode(clientX, clientY, radiusPx = PICK_RADIUS_PX) {
  const list = nodesUnderCursor(clientX, clientY, radiusPx);
  return list.length ? list[0].id : null;
}

/**
 * Lanelet feature ids whose surface the cursor ray meets, front to back (the
 * surfaces have no feature of their own to pick; each triangle knows its
 * lanelet). Empty while the surfaces are hidden.
 */
export function laneletsUnderCursor(clientX, clientY) {
  const surfaces = laneletLayer.surfaceMeshes();
  if (!surfaces.length) return [];
  cursorRay(clientX, clientY);
  const out = [];
  for (const h of raycaster.intersectObjects(surfaces, false)) {
    const ids = h.object.userData.faceLanelets;
    const id = ids && h.faceIndex != null ? ids[h.faceIndex] : null;
    if (id && !out.includes(id)) out.push(id);
  }
  return out;
}

/**
 * What a click at the cursor selects: the nearest node within `nodePx`, else
 * the nearest line, else (with `lanelets`) the lanelet under it, else null.
 * `nodeOnly` looks for a node only; `nodes: false` never picks one.
 */
export function pickItem(clientX, clientY,
  { nodePx = PICK_RADIUS_PX, nodeOnly = false, nodes = true, lanelets = false } = {}) {
  if (nodes || nodeOnly) {
    const id = pickNode(clientX, clientY, nodePx);
    if (id !== null) return { type: "node", id };
  }
  if (nodeOnly) return null;
  const ways = featuresUnderCursor(clientX, clientY);
  if (ways.length) return { type: "way", id: ways[0].id };
  if (!lanelets) return null;
  const ll = laneletsUnderCursor(clientX, clientY);
  return ll.length ? { type: "relation", id: ll[0] } : null;
}
