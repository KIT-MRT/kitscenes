// The viewer's copy of the streamed map, as plain data.
//
// Rendering and picking both read from here, never from three.js objects, so
// a map with 100k+ ways stays cheap to hold, index and update. The server
// pushes a `snapshot` on connect, then `patch` messages as JOSM data changes.
//
//   Feature  { id, kind, tags, pos: Float64Array(x,y,z per vertex),
//              nodes: number[] (node id per vertex), center, bs, tile,
//              lanelet: { left, right, lrev, rrev, two, owx, arrow } for kind "lanelet" }
//   Tile     { key, features: Set<Feature>, bs }  spatial bucket, TILE_M square
//   NodeRec  { id, x, y, z, refs: [{ f, i }] }    only while the index is on
//
// Coordinates are local ENU metres around the anchor lat/lon (Z up).
import * as THREE from "three";
import { Emitter } from "./util.js";

export const TILE_M = 256;
export const VIEWPORT_ID = "viewport";
const FRAME_ELE_MIN = -200;
const FRAME_ELE_MAX = 5000;

// Rendering batches RENDER_CHUNK x RENDER_CHUNK store tiles (~1 km) into one
// draw call: a whole city in view costs a few hundred draws, while picking
// keeps the finer tiles and a one-edit rebuild stays a few milliseconds.
export const RENDER_CHUNK = 4;

/** Render chunk key ("cx,cy") of a store tile key ("tx,ty"). */
export function chunkKeyOf(tileKey) {
  const i = tileKey.indexOf(",");
  const tx = Number(tileKey.slice(0, i));
  const ty = Number(tileKey.slice(i + 1));
  return `${Math.floor(tx / RENDER_CHUNK)},${Math.floor(ty / RENDER_CHUNK)}`;
}

/** Store tiles (existing ones) of a render chunk. */
export function* tilesOfChunk(chunkKey) {
  const i = chunkKey.indexOf(",");
  const cx = Number(chunkKey.slice(0, i));
  const cy = Number(chunkKey.slice(i + 1));
  for (let dx = 0; dx < RENDER_CHUNK; dx++) {
    for (let dy = 0; dy < RENDER_CHUNK; dy++) {
      const t = store.tiles.get(`${cx * RENDER_CHUNK + dx},${cy * RENDER_CHUNK + dy}`);
      if (t) yield t;
    }
  }
}

/** "node/123" | 123 -> 123. JOSM unique ids of new nodes are negative. */
export function nodeKey(n) {
  if (typeof n === "number") return n;
  const s = String(n);
  const i = s.indexOf("/");
  return Number(i >= 0 ? s.slice(i + 1) : s);
}

export function nodeToken(id) {
  return "node/" + id;
}

function toFeature(raw) {
  const pos = Float64Array.from(Array.isArray(raw.pts) ? raw.pts : []);
  const f = {
    id: raw.id,
    kind: raw.kind || "line",
    tags: raw.tags || {},
    pos,
    nodes: (raw.nodes || []).map(nodeKey),
    center: raw.center || null,
    color: raw.color || null,
    lanelet: raw.kind === "lanelet" ? {
      left: raw.left, right: raw.right, lrev: !!raw.lrev, rrev: !!raw.rrev, two: !!raw.two,
      owx: Array.isArray(raw.owx) ? raw.owx : [], // participants a one_way:<p> tag gives the other answer
      arrow: Array.isArray(raw.arrow) ? raw.arrow : null,
    } : null,
    bs: null,
    tile: null,
  };
  updateSphere(f);
  return f;
}

/** Bounding sphere of a feature's vertices (bbox centre, max distance). */
export function updateSphere(f) {
  const p = f.pos;
  const n = p.length / 3;
  if (n === 0) {
    f.bs = { x: 0, y: 0, z: 0, r: 0 };
    return;
  }
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
  let r2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = p[i * 3] - cx, dy = p[i * 3 + 1] - cy, dz = p[i * 3 + 2] - cz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > r2) r2 = d2;
  }
  f.bs = { x: cx, y: cy, z: cz, r: Math.sqrt(r2) };
}

/** Bounding sphere enclosing all feature spheres of a tile. */
function tileSphere(tile) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const f of tile.features) {
    const s = f.bs;
    minX = Math.min(minX, s.x - s.r); maxX = Math.max(maxX, s.x + s.r);
    minY = Math.min(minY, s.y - s.r); maxY = Math.max(maxY, s.y + s.r);
    minZ = Math.min(minZ, s.z - s.r); maxZ = Math.max(maxZ, s.z + s.r);
  }
  if (minX === Infinity) return { x: 0, y: 0, z: 0, r: 0 };
  const x = (minX + maxX) / 2, y = (minY + maxY) / 2, z = (minZ + maxZ) / 2;
  return { x, y, z, r: Math.hypot(maxX - x, maxY - y, maxZ - z) };
}

class FeatureStore extends Emitter {
  constructor() {
    super();
    this.features = new Map();
    this.tiles = new Map();
    this.nodes = new Map();
    this.nodeIndexOn = false;
    this.anchor = null;
    // Who sent the map: "josm", "scene_server" (a read-only copy) or null
    // before the first snapshot.
    this.mapSource = null;
    this.lastSeq = -1;
    this._bounds = null;
  }

  /** Map features, i.e. everything but the JOSM viewport overlay. */
  featureCount() {
    return this.features.has(VIEWPORT_ID) ? this.features.size - 1 : this.features.size;
  }

  /**
   * Apply one server message. Returns { applied, type, nOps } or
   * { applied: false } for a stale / out-of-order patch.
   */
  applyMessage(msg) {
    if (typeof msg.seq === "number") {
      if (msg.type === "snapshot") {
        this.lastSeq = msg.seq; // snapshot resets the sequence baseline
      } else if (msg.seq <= this.lastSeq) {
        return { applied: false };
      } else {
        this.lastSeq = msg.seq;
      }
    }
    let nOps = 0;
    if (msg.type === "snapshot") {
      const wasEmpty = this.featureCount() === 0;
      this.clear();
      if (msg.anchor) this.anchor = msg.anchor;
      this.mapSource = typeof msg.source === "string" ? msg.source : "josm";
      const feats = msg.features || [];
      nOps = feats.length;
      for (const raw of feats) this.upsert(raw);
      this.emit("snapshot", { wasEmpty });
    } else if (msg.type === "patch") {
      const ops = msg.ops || [];
      nOps = ops.length;
      for (const op of ops) {
        if (op.op === "upsert") this.upsert(op.feature);
        else if (op.op === "remove") this.remove(op.id);
        else if (op.op === "anchor") this.anchor = { lat: op.lat, lon: op.lon };
      }
    } else if (msg.type === "clear") {
      this.clear();
    }
    this.emit("applied", msg);
    return { applied: true, type: msg.type, nOps };
  }

  upsert(raw) {
    if (!raw || raw.id === undefined || raw.id === null) return null;
    const f = toFeature(raw);
    const old = this.features.get(f.id);
    if (old) this._detach(old);
    this.features.set(f.id, f);
    this._attach(f);
    this._bounds = null;
    this.emit("upsert", f, old || null);
    return f;
  }

  remove(id) {
    const f = this.features.get(id);
    if (!f) return;
    this._detach(f);
    this.features.delete(id);
    this._bounds = null;
    this.emit("remove", f);
  }

  clear() {
    this.features.clear();
    this.tiles.clear();
    this.nodes.clear();
    this._bounds = null;
    this.emit("clear");
  }

  _attach(f) {
    if (f.kind === VIEWPORT_ID) return;
    const key = `${Math.floor(f.bs.x / TILE_M)},${Math.floor(f.bs.y / TILE_M)}`;
    let tile = this.tiles.get(key);
    if (!tile) {
      tile = { key, features: new Set(), bs: null };
      this.tiles.set(key, tile);
    }
    tile.features.add(f);
    tile.bs = null;
    f.tile = tile;
    if (this.nodeIndexOn) this._indexAdd(f);
    this.emit("tile-dirty", tile);
  }

  _detach(f) {
    const tile = f.tile;
    if (tile) {
      tile.features.delete(f);
      tile.bs = null;
      if (tile.features.size === 0) this.tiles.delete(tile.key);
      this.emit("tile-dirty", tile);
      f.tile = null;
    }
    if (this.nodeIndexOn) this._indexRemove(f);
  }

  tileSphere(tile) {
    if (!tile.bs) tile.bs = tileSphere(tile);
    return tile.bs;
  }

  // --- node index ---------------------------------------------------------------
  // Built on first use (edit mode, selection), then kept up to date per
  // feature, so a patch costs the size of the patch, not of the map.

  ensureNodeIndex() {
    if (this.nodeIndexOn) return;
    this.nodeIndexOn = true;
    this.nodes.clear();
    for (const f of this.features.values()) {
      if (f.kind !== VIEWPORT_ID) this._indexAdd(f);
    }
  }

  _indexAdd(f) {
    const n = Math.min(f.nodes.length, f.pos.length / 3);
    for (let i = 0; i < n; i++) {
      const id = f.nodes[i];
      let rec = this.nodes.get(id);
      if (!rec) {
        rec = { id, x: 0, y: 0, z: 0, refs: [] };
        this.nodes.set(id, rec);
      }
      // The latest feature to arrive carries the authoritative position.
      rec.x = f.pos[i * 3];
      rec.y = f.pos[i * 3 + 1];
      rec.z = f.pos[i * 3 + 2];
      rec.refs.push({ f, i });
    }
  }

  _indexRemove(f) {
    const n = Math.min(f.nodes.length, f.pos.length / 3);
    for (let i = 0; i < n; i++) {
      const rec = this.nodes.get(f.nodes[i]);
      if (!rec) continue;
      rec.refs = rec.refs.filter((r) => r.f !== f);
      if (rec.refs.length === 0) this.nodes.delete(rec.id);
    }
  }

  node(id) {
    this.ensureNodeIndex();
    return this.nodes.get(id) || null;
  }

  /**
   * Move nodes locally (a drag in progress): every feature using them follows.
   * `moves` is [{ id, x, y, z }]. Emits "nodes-moved" with the NodeRecs.
   */
  moveNodes(moves) {
    this.ensureNodeIndex();
    const recs = [];
    const touched = new Set();
    for (const m of moves) {
      const rec = this.nodes.get(m.id);
      if (!rec) continue;
      rec.x = m.x;
      rec.y = m.y;
      rec.z = m.z;
      for (const { f, i } of rec.refs) {
        f.pos[i * 3] = m.x;
        f.pos[i * 3 + 1] = m.y;
        f.pos[i * 3 + 2] = m.z;
        touched.add(f);
      }
      recs.push(rec);
    }
    for (const f of touched) {
      updateSphere(f);
      if (f.tile) f.tile.bs = null;
    }
    this._bounds = null;
    if (recs.length) this.emit("nodes-moved", recs);
    return recs;
  }

  /** Bounding box of the map for framing; implausible elevations count as 0. */
  mapBounds() {
    if (this._bounds) return this._bounds;
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    for (const f of this.features.values()) {
      if (f.kind === VIEWPORT_ID) continue;
      const p = f.pos;
      for (let i = 0; i < p.length; i += 3) {
        const x = p[i], y = p[i + 1];
        let z = p[i + 2];
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        if (!Number.isFinite(z) || z < FRAME_ELE_MIN || z > FRAME_ELE_MAX) z = 0;
        box.expandByPoint(v.set(x, y, z));
      }
    }
    this._bounds = box;
    return box;
  }
}

export const store = new FeatureStore();
