// Traffic-element icons (signs, lights, arrows, symbols) as textured quads.
// One group per feature; groups further than ICON_MAX_DIST_M from the camera
// are hidden, which keeps draw calls bounded on city-size maps (the icons are
// unreadable at that distance anyway).
import * as THREE from "three";
import { store } from "../store.js";
import { mapRoot } from "../scene.js";
import { iconSpec, isFacedIconType } from "./style.js";

const ICON_LIFT_M = 0.03;
const ICON_MIN_EXTENT_M = 0.08;
const ICON_BACK_COLOR = 0x808080; // 50% grey multiply on the reverse face
const ICON_MAX_DIST_M = 400;

// How solid the icons are. Point clouds sit behind and through them, so the
// user can fade them without losing the map. Materials are made per mesh and
// asynchronously (after the texture loads), so new ones read this too.
let iconOpacity = 1;

const textureLoader = new THREE.TextureLoader();
// url -> { tex, ready, pending: [{onLoad, onError}] }
const textureCache = new Map();
const failedIcons = new Set();

function loadIconTexture(file, onLoad, onError) {
  const url = "/style_images/" + encodeURIComponent(file);
  if (failedIcons.has(url)) {
    if (onError) onError();
    return;
  }
  const cached = textureCache.get(url);
  if (cached) {
    if (cached.ready) onLoad(cached.tex);
    else cached.pending.push({ onLoad, onError });
    return;
  }
  const entry = { tex: null, ready: false, pending: [{ onLoad, onError }] };
  textureCache.set(url, entry);
  textureLoader.load(url, (tex) => {
    if (tex.colorSpace !== undefined) tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    entry.tex = tex;
    entry.ready = true;
    for (const p of entry.pending) p.onLoad(tex);
    entry.pending = [];
  }, undefined, () => {
    failedIcons.add(url);
    textureCache.delete(url);
    for (const p of entry.pending) {
      if (p.onError) p.onError();
    }
    entry.pending = [];
  });
}

function textureAspect(tex) {
  const img = tex && tex.image;
  if (img && img.width && img.height) return img.width / img.height;
  return 1;
}

function uniquePolyPoints(pos) {
  const out = [];
  for (let i = 0; i < pos.length; i += 3) {
    const v = new THREE.Vector3(pos[i], pos[i + 1], pos[i + 2]);
    if (out.length && out[out.length - 1].distanceToSquared(v) < 1e-8) continue;
    out.push(v);
  }
  if (out.length >= 2 && out[0].distanceToSquared(out[out.length - 1]) < 1e-8) {
    out.pop();
  }
  return out;
}

// Fit an oriented rectangle to the way points. Image +Y follows world-up
// projected into the plane (or the start→end chord on the ground). Size is
// the actual extent of the polyline, so a 0.6 m sign stays 0.6 m.
function fitIconFrame(pos, aspect) {
  const verts = uniquePolyPoints(pos);
  if (verts.length < 2) return null;

  const center = new THREE.Vector3();
  for (let i = 0; i < verts.length; i++) center.add(verts[i]);
  center.multiplyScalar(1 / verts.length);

  const normal = new THREE.Vector3();
  if (verts.length >= 3) {
    for (let i = 0; i < verts.length; i++) {
      const a = verts[i].clone().sub(center);
      const b = verts[(i + 1) % verts.length].clone().sub(center);
      normal.add(a.cross(b));
    }
    if (normal.lengthSq() < 1e-10) normal.set(0, 0, 1);
    else normal.normalize();
  } else {
    const chord = verts[1].clone().sub(verts[0]);
    const horiz = Math.hypot(chord.x, chord.y);
    if (Math.abs(chord.z) > horiz * 0.5) {
      // Mostly vertical 2-point way: face sideways so the icon stands up.
      normal.set(-chord.y, chord.x, 0);
      if (normal.lengthSq() < 1e-10) normal.set(1, 0, 0);
      normal.normalize();
    } else {
      normal.set(0, 0, 1);
    }
  }

  const yAxis = new THREE.Vector3(0, 0, 1).addScaledVector(normal, -normal.z);
  if (yAxis.lengthSq() < 1e-8) {
    yAxis.copy(verts[verts.length - 1]).sub(verts[0]);
    if (yAxis.lengthSq() < 1e-8) yAxis.set(0, 1, 0);
  }
  yAxis.normalize();
  const xAxis = new THREE.Vector3().crossVectors(yAxis, normal);
  if (xAxis.lengthSq() < 1e-10) xAxis.set(1, 0, 0);
  else xAxis.normalize();
  yAxis.crossVectors(normal, xAxis).normalize();

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const tmp = new THREE.Vector3();
  for (let i = 0; i < verts.length; i++) {
    tmp.copy(verts[i]).sub(center);
    const x = tmp.dot(xAxis);
    const y = tmp.dot(yAxis);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  let width = maxX - minX;
  let height = maxY - minY;
  const ar = (aspect > 0.05 && aspect < 20) ? aspect : 1;
  // A 2-point way has no width: fill the missing axis from the texture aspect.
  if (width < ICON_MIN_EXTENT_M && height >= ICON_MIN_EXTENT_M) {
    width = height * ar;
  } else if (height < ICON_MIN_EXTENT_M && width >= ICON_MIN_EXTENT_M) {
    height = width / ar;
  } else if (width < ICON_MIN_EXTENT_M && height < ICON_MIN_EXTENT_M) {
    height = 0.5;
    width = height * ar;
  }

  center.addScaledVector(normal, ICON_LIFT_M);
  return { center, xAxis, yAxis, normal, width, height };
}

// Lanelet2 sign/light quad, facing the "camera" on the front:
//   P1 bottom-left, P2 top-left, P3 top-right, P4 bottom-right.
// Front normal is right × up = (P4-P1) × (P2-P1). UVs put the icon upright
// on that face; two triangles are wound CCW from the front (P1-P4-P3, P1-P3-P2).
function signQuadGeometry(p1, p2, p3, p4) {
  const geom = new THREE.BufferGeometry();
  const positions = new Float32Array([
    p1.x, p1.y, p1.z, p4.x, p4.y, p4.z, p3.x, p3.y, p3.z,
    p1.x, p1.y, p1.z, p3.x, p3.y, p3.z, p2.x, p2.y, p2.z,
  ]);
  const uvs = new Float32Array([
    0, 0, 1, 0, 1, 1,
    0, 0, 1, 1, 0, 1,
  ]);
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geom.computeVertexNormals();
  return geom;
}

function addFacedSignIcon(group, p1, p2, p3, p4, tex) {
  const right = new THREE.Vector3().subVectors(p4, p1);
  const up = new THREE.Vector3().subVectors(p2, p1);
  const front = new THREE.Vector3().crossVectors(right, up);
  if (front.lengthSq() < 1e-12) return;
  front.normalize();
  const lift = front.multiplyScalar(ICON_LIFT_M);
  const geom = signQuadGeometry(p1.clone().add(lift), p2.clone().add(lift), p3.clone().add(lift), p4.clone().add(lift));
  const frontMat = new THREE.MeshBasicMaterial({
    map: tex, transparent: true, opacity: iconOpacity, side: THREE.FrontSide, depthWrite: false,
  });
  const backMat = new THREE.MeshBasicMaterial({
    map: tex, color: ICON_BACK_COLOR, transparent: true, opacity: iconOpacity,
    side: THREE.BackSide, depthWrite: false,
  });
  const frontMesh = new THREE.Mesh(geom, frontMat);
  const backMesh = new THREE.Mesh(geom.clone(), backMat);
  frontMesh.renderOrder = 20;
  backMesh.renderOrder = 19;
  group.add(frontMesh);
  group.add(backMesh);
}

/**
 * Autoware's traffic light: a line along the light's bottom edge, from its
 * bottom-left to its bottom-right corner as the approaching vehicle sees it,
 * and a `height` tag for its vertical size (not its position -- that is the
 * nodes' ele). See autoware_lanelet2_extension's lanelet2_format_extension.md.
 * Traffic signs drawn the same way are read the same way.
 * Returns the quad [bottom-left, top-left, top-right, bottom-right] for such a
 * way, else null so the ordinary path draws it.
 */
export function autowareLightQuad(feature) {
  const tags = feature.tags || {};
  const type = tags.type || "";
  if (type.indexOf("traffic_light") !== 0 && type !== "traffic_sign") return null;
  const m = /^\s*(\d+(?:\.\d+)?|\.\d+)\s*(?:m)?\s*$/.exec(String(tags.height ?? ""));
  const height = m ? Number(m[1]) : NaN;
  if (!(height > 0 && height < 20)) return null;
  const verts = uniquePolyPoints(feature.pos || []);
  if (verts.length !== 2) return null;
  const [left, right] = verts;
  const up = new THREE.Vector3(0, 0, height);
  return [left, left.clone().add(up), right.clone().add(up), right];
}

function buildIcon(group, feature, spec) {
  loadIconTexture(spec.file, (tex) => {
    if (group.userData.disposed) return;
    const pos = feature.pos;
    if (pos.length < 3) return;
    const tags = feature.tags || {};
    const autoware = autowareLightQuad(feature);
    if (autoware) {
      addFacedSignIcon(group, autoware[0], autoware[1], autoware[2], autoware[3], tex);
      return;
    }
    if (isFacedIconType(tags.type)) {
      const verts = uniquePolyPoints(pos);
      if (verts.length >= 4) {
        addFacedSignIcon(group, verts[0], verts[1], verts[2], verts[3], tex);
        return;
      }
    }
    const frame = fitIconFrame(pos, textureAspect(tex));
    if (!frame) return;
    const geom = new THREE.PlaneGeometry(frame.width, frame.height);
    const mat = new THREE.MeshBasicMaterial({
      map: tex, transparent: true, opacity: iconOpacity, side: THREE.DoubleSide, depthWrite: false,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.copy(frame.center);
    const basis = new THREE.Matrix4();
    basis.makeBasis(frame.xAxis, frame.yAxis, frame.normal);
    mesh.quaternion.setFromRotationMatrix(basis);
    mesh.renderOrder = 20;
    group.add(mesh);
  });
}

function disposeGroup(group) {
  group.userData.disposed = true;
  group.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    // Textures are shared through the cache; only materials are per mesh.
    if (child.material) child.material.dispose();
  });
}

class IconLayer {
  constructor() {
    this.visible = true;
    this.groups = new Map(); // feature id -> THREE.Group
    store.on("upsert", (f, old) => {
      if (old) this._remove(old.id);
      const spec = iconSpec(f);
      if (!spec) return;
      const group = new THREE.Group();
      group.userData.center = new THREE.Vector3(f.bs.x, f.bs.y, f.bs.z);
      buildIcon(group, f, spec);
      mapRoot.add(group);
      this.groups.set(f.id, group);
    });
    store.on("remove", (f) => this._remove(f.id));
    store.on("clear", () => {
      for (const id of Array.from(this.groups.keys())) this._remove(id);
    });
  }

  _remove(id) {
    const g = this.groups.get(id);
    if (!g) return;
    disposeGroup(g);
    mapRoot.remove(g);
    this.groups.delete(id);
  }

  /** Per frame: hide icons too far away to read. */
  update(camera) {
    if (!this.visible) {
      for (const g of this.groups.values()) g.visible = false;
      return;
    }
    const max2 = ICON_MAX_DIST_M * ICON_MAX_DIST_M;
    const cp = camera.position;
    for (const g of this.groups.values()) {
      g.visible = g.userData.center.distanceToSquared(cp) < max2;
    }
  }

  setVisible(on) {
    this.visible = !!on;
  }

  /** 0..1; also applies to icons whose texture has not arrived yet. */
  setOpacity(value) {
    iconOpacity = Math.max(0, Math.min(1, value));
    for (const g of this.groups.values()) {
      g.traverse((child) => {
        if (child.material) child.material.opacity = iconOpacity;
      });
    }
  }

  /** World positions of the vertices of a feature's (front) icon mesh, for tests. */
  meshOf(id) {
    const g = this.groups.get(id);
    const mesh = g && g.children[0];
    if (!mesh) return null;
    mesh.updateMatrixWorld();
    const p = mesh.geometry.getAttribute("position");
    const out = [];
    const v = new THREE.Vector3();
    for (let i = 0; i < p.count; i++) out.push(v.fromBufferAttribute(p, i).applyMatrix4(mesh.matrixWorld).toArray());
    return out;
  }

  stats() {
    let shown = 0;
    for (const g of this.groups.values()) if (g.visible) shown++;
    return { icons: this.groups.size, shown, visible: this.visible, opacity: iconOpacity };
  }
}

export const iconLayer = new IconLayer();
