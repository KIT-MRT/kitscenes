// Selection highlight drawn over the map: selected ways as yellow lines, and a
// dot on every node a move would act on (selected nodes plus the nodes of
// selected ways). Rebuilt on the next frame after the selection or the
// geometry under it changes, including every frame of a drag. Under it, in
// cyan, the hover: what a click at the pointer would select (see hover.js).
// A lanelet (a relation picked in View mode) shows as its bounds over a tint
// of its surface. Any other relation (a traffic light's regulatory element,
// picked in the Selection tab) has no geometry in the stream, so it shows as
// its members: the light, the stop line, the lanes of a right of way. Which
// members those are comes from the tab's `describe` (`setRelationMembers`).
// Members draw in red over a dark edge, wider than the yellow selection:
// yellow is hard to find on a point cloud or a busy map.
import * as THREE from "three";
import { store } from "../store.js";
import { overlayRoot } from "../scene.js";
import { selection } from "../selection.js";
import { SELECT_COLOR, HOVER_COLOR, MEMBER_COLOR, MEMBER_EDGE_COLOR } from "./style.js";
import { screenLines } from "./screenlines.js";
import { laneletTriangles } from "./lanelets.js";

// Lines a few pixels wide: a one-pixel line vanishes next to a styled marking.
const lines = screenLines({ color: SELECT_COLOR, widthPx: 3, opacity: 0.95, renderOrder: 1003 });
const dotMat = new THREE.PointsMaterial({
  color: SELECT_COLOR, size: 9, sizeAttenuation: false, depthTest: false, transparent: true,
});

const dots = new THREE.Points(new THREE.BufferGeometry(), dotMat);
dots.renderOrder = 1004;
dots.frustumCulled = false;
overlayRoot.add(lines);
overlayRoot.add(dots);

const hoverLine = screenLines({ color: HOVER_COLOR, widthPx: 5, opacity: 0.85, renderOrder: 1001 });
const hoverDot = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({
  color: HOVER_COLOR, size: 14, sizeAttenuation: false, depthTest: false, transparent: true, opacity: 0.9,
}));
hoverDot.renderOrder = 1002; // the hover sits below the selection, so a selected dot shows framed
hoverDot.frustumCulled = false;
overlayRoot.add(hoverLine);
overlayRoot.add(hoverDot);
// Surface tints: drawn over the map (the lanelet surface itself sits at the
// same height, so a depth-tested copy would flicker against it), under the lines.
const fillMesh = (color, opacity, renderOrder) => {
  const m = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({
    color, transparent: true, opacity, depthTest: false, depthWrite: false, side: THREE.DoubleSide,
  }));
  m.renderOrder = renderOrder;
  m.frustumCulled = false;
  overlayRoot.add(m);
  return m;
};
const fill = fillMesh(SELECT_COLOR, 0.22, 1000);
const hoverFill = fillMesh(HOVER_COLOR, 0.14, 999);
// A selected relation's members: a dark edge, then red lines, dots and tint.
const memberEdge = screenLines({ color: MEMBER_EDGE_COLOR, widthPx: 9, opacity: 0.6, renderOrder: 1005 });
const memberLines = screenLines({ color: MEMBER_COLOR, widthPx: 5, opacity: 1, renderOrder: 1006 });
const memberDots = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({
  color: MEMBER_COLOR, size: 13, sizeAttenuation: false, depthTest: false, transparent: true,
}));
memberDots.renderOrder = 1007;
memberDots.frustumCulled = false;
overlayRoot.add(memberEdge);
overlayRoot.add(memberLines);
overlayRoot.add(memberDots);
const memberFill = fillMesh(MEMBER_COLOR, 0.32, 1000);
let hover = null; // { type: "node" | "way" | "relation", id }
// Members of relations that are not in the stream: relation token -> [token].
let relationMembers = new Map();
let stats = { segments: 0, dots: 0, triangles: 0, members: { segments: 0, dots: 0, triangles: 0 } };

// Separate flags: the hover changes often, the selection can be 20k nodes.
let dirty = true;
let hoverDirty = true;
const markDirty = () => { dirty = true; hoverDirty = true; };
selection.on("changed", () => { dirty = true; });
store.on("clear", markDirty);
// A lanelet is drawn from its bounds, so any line may be one of them: with a
// lanelet selected or hovered, every change rebuilds it (a few lanes, cheap).
const isHovered = (f) => hover && (hover.type === "relation" || (hover.type === "way" ? hover.id === f.id
  : f.nodes.includes(hover.id)));
store.on("upsert", (f) => {
  if (selection.ways.has(f.id) || selection.relations.size || f.nodes.some((n) => selection.nodes.has(n))) {
    dirty = true;
  }
  if (isHovered(f)) hoverDirty = true;
});
store.on("remove", (f) => {
  if (selection.ways.has(f.id) || selection.relations.size) dirty = true;
  if (isHovered(f)) hoverDirty = true;
});
store.on("nodes-moved", markDirty);

/** Per frame: rebuild the highlight if anything under it changed. */
export function updateHighlight() {
  if (dirty) {
    dirty = false;
    rebuildSelection();
  }
  if (hoverDirty) {
    hoverDirty = false;
    rebuildHover();
  }
}

function rebuildSelection() {
  const seg = [];
  for (const wid of selection.ways) {
    const f = store.features.get(wid);
    if (f) wayPairs(f.pos, seg);
  }
  const tri = [];
  const pts = [];
  const member = { seg: [], tri: [], pts: [] };
  for (const rid of selection.relations) {
    const f = store.features.get(rid);
    // A lanelet is drawn as itself (yellow); any other relation as its members (red).
    if (f && f.kind === "lanelet") relationOutline(rid, seg, tri, pts);
    else relationOutline(rid, member.seg, member.tri, member.pts);
  }
  if (selection.hasEditable()) {
    for (const id of selection.effectiveNodeIds()) {
      const rec = store.node(id);
      if (rec) pts.push(rec.x, rec.y, rec.z);
    }
  }
  lines.setPairs(seg);
  setPositions(dots, pts);
  setPositions(fill, tri);
  memberEdge.setPairs(member.seg);
  memberLines.setPairs(member.seg);
  setPositions(memberDots, member.pts);
  setPositions(memberFill, member.tri);
  stats = { segments: seg.length / 6, dots: pts.length / 3, triangles: tri.length / 9,
            members: { segments: member.seg.length / 6, dots: member.pts.length / 3,
                       triangles: member.tri.length / 9 } };
}

/**
 * A lanelet's bounds (as line pairs) and surface (as triangles). Any other
 * relation draws its members, as far as they are known and streamed; a
 * member relation is drawn only if it is a lanelet, so nothing recurses.
 */
function relationOutline(id, seg, tri, pts, nested = false) {
  const f = store.features.get(id);
  if (f && f.kind === "lanelet") {
    for (const wid of [f.lanelet.left, f.lanelet.right]) {
      const w = store.features.get(wid);
      if (w) wayPairs(w.pos, seg);
    }
    const t = laneletTriangles(f);
    if (t) for (const v of t) tri.push(v);
    return;
  }
  if (nested) return;
  for (const token of relationMembers.get(id) || []) {
    if (token.startsWith("way/")) {
      const w = store.features.get(token);
      if (w && w.kind === "line") wayPairs(w.pos, seg);
    } else if (token.startsWith("node/")) {
      const rec = pts && store.node(Number(token.slice(5)));
      if (rec) pts.push(rec.x, rec.y, rec.z);
    } else if (token.startsWith("relation/")) {
      relationOutline(token, seg, tri, pts, true);
    }
  }
}

/**
 * What the Selection tab learnt about relations: token -> member tokens.
 * Replaces what was known before (only the described relations matter).
 */
export function setRelationMembers(map) {
  relationMembers = new Map(map);
  dirty = true;
  hoverDirty = true;
}

/** What the selection highlight draws now (for the tests). */
export function highlightStats() {
  return { ...stats, members: { ...stats.members } };
}

function rebuildHover() {
  const hSeg = [];
  const hPts = [];
  const hTri = [];
  if (hover && hover.type === "way") {
    const f = store.features.get(hover.id);
    if (f) wayPairs(f.pos, hSeg);
  } else if (hover && hover.type === "relation") {
    relationOutline(hover.id, hSeg, hTri, hPts);
  } else if (hover) {
    const rec = store.node(hover.id);
    if (rec) hPts.push(rec.x, rec.y, rec.z);
  }
  hoverLine.setPairs(hSeg);
  setPositions(hoverDot, hPts);
  setPositions(hoverFill, hTri);
}

function wayPairs(p, out) {
  for (let i = 3; i < p.length; i += 3) out.push(p[i - 3], p[i - 2], p[i - 1], p[i], p[i + 1], p[i + 2]);
}

function setPositions(obj, arr) {
  obj.geometry.dispose();
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(arr), 3));
  obj.geometry = g;
}

/** Show `item` ({ type, id } or null) as the hover. */
export function setHover(item) {
  if (item === hover || (item && hover && item.type === hover.type && item.id === hover.id)) return;
  hover = item;
  hoverDirty = true;
}

/** The highlight shows only in edit mode; the selection itself persists. */
export function setHighlightVisible(on) {
  lines.visible = !!on;
  dots.visible = !!on;
  hoverLine.visible = !!on;
  hoverDot.visible = !!on;
  fill.visible = !!on;
  hoverFill.visible = !!on;
  for (const m of [memberEdge, memberLines, memberDots, memberFill]) m.visible = !!on;
}
