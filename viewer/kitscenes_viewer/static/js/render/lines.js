// Batched line rendering, per render chunk (4x4 store tiles) instead of per
// way: a city map (Karlsruhe: 144k ways) would otherwise cost 144k draw calls.
//
// Two looks, switched in the Layers tab:
//   * Simple: one indexed LineSegments per chunk, one pixel wide, coloured
//     by type. Cheapest; also what edit-mode node dots share their buffer with.
//   * Styled (default): render/ribbons.js -- each segment a quad at the
//     marking's real width with its dash / double-line pattern, plus raised
//     curbs, rails and walls when "3D" is on. One more draw call per chunk
//     for the paint, one for the raised parts.
//
// Styled has a level of detail per chunk: where a metre is under
// LOD_PX_PER_M pixels on screen, every marking is at its pixel minimum and
// its dashes merge anyway, so the chunk shows the one-pixel lines (same
// colours) instead. With a whole city in view the paint costs more than
// half the frame rate, for about half a pixel of width.
//
// A chunk is rebuilt (on the next frame) when a tile in it changes. A drag in
// progress only rewrites the moved vertices in place, in every buffer.
import * as THREE from "three";
import { store, chunkKeyOf, tilesOfChunk } from "../store.js";
import { mapRoot } from "../scene.js";
import { EDIT_COLOR } from "./style.js";
import { LINE_STYLES, lineStyleKey, lineColor } from "./linestyles.js";
import { paintMaterial, raisedMaterial, segmentGeometry, boundSegments, styleIndex } from "./ribbons.js";
import { now, plog, PROFILE } from "../util.js";

const lineMaterial = new THREE.LineBasicMaterial({ vertexColors: true });
export const LOD_PX_PER_M = 2;
const _cam = new THREE.Vector3();
const _centre = new THREE.Vector3();
const _col = new THREE.Color();
const emptyGeometry = () => new THREE.BufferGeometry();
const dotMaterial = new THREE.PointsMaterial({ color: EDIT_COLOR, size: 6, sizeAttenuation: false });

class LineLayer {
  constructor() {
    this.chunks = new Map(); // chunk key -> { key, lines, dots, paint, raised, posDirty, ribbonDirty }
    this.dirty = new Set();  // chunk keys to rebuild
    this.dotsVisible = false;
    this.styled = true;      // ribbons with patterns, else one-pixel lines
    this.raised = true;      // curbs, rails and walls as upright faces (styled only)
    store.on("tile-dirty", (tile) => this.dirty.add(chunkKeyOf(tile.key)));
    store.on("clear", () => this._clearAll());
    store.on("nodes-moved", (recs) => this._patchNodes(recs));
  }

  setDotsVisible(on) {
    this.dotsVisible = !!on;
    for (const c of this.chunks.values()) c.dots.visible = this.dotsVisible;
  }

  /** Styled ribbons (true) or one-pixel lines. The other look's buffers are freed. */
  setStyled(on) {
    if (this.styled === !!on) return;
    this.styled = !!on;
    for (const key of this.chunks.keys()) this.dirty.add(key);
  }

  /** Raised curbs, rails and walls (styled only); no rebuild needed. */
  setRaised(on) {
    this.raised = !!on;
    for (const c of this.chunks.values()) this._show(c);
  }

  /** Which of a chunk's looks is drawn: its paint up close, its lines far away or in Simple. */
  _show(c) {
    const styled = this.styled && c.hasPaint && c.near !== false;
    c.lines.visible = !styled;
    c.paint.visible = styled;
    c.raised.visible = styled && c.hasRaised && this.raised;
  }

  /**
   * Per frame, before drawing: pick each chunk's level of detail from how
   * many pixels a metre is at its nearest point.
   */
  updateDetail(camera, viewportHeight) {
    if (!this.styled) return;
    camera.getWorldPosition(_cam);
    const ortho = camera.isOrthographicCamera;
    const pxPerM1 = ortho ? viewportHeight / Math.max(1e-6, camera.top - camera.bottom)
      : viewportHeight / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    for (const c of this.chunks.values()) {
      const sphere = c.paint.geometry.boundingSphere;
      if (!c.hasPaint || !sphere) continue;
      _centre.copy(sphere.center);
      const d = Math.max(1, _cam.distanceTo(_centre) - sphere.radius);
      const near = (ortho ? pxPerM1 : pxPerM1 / d) >= LOD_PX_PER_M;
      if (near === c.near) continue;
      c.near = near;
      this._show(c);
    }
  }

  /** What is drawn, for the tests and the perf script. */
  stats() {
    let segments = 0, raised = 0, bytes = 0, lineVerts = 0;
    for (const c of this.chunks.values()) {
      lineVerts += c.lines.geometry.attributes.position ? c.lines.geometry.attributes.position.count : 0;
      segments += c.paint.geometry.instanceCount || 0;
      raised += c.raised.geometry.instanceCount || 0;
      for (const g of [c.lines.geometry, c.paint.geometry, c.raised.geometry]) {
        for (const a of Object.values(g.attributes)) if (a.isInstancedBufferAttribute || g === c.lines.geometry) bytes += a.array.byteLength;
        if (g === c.lines.geometry && g.index) bytes += g.index.array.byteLength;
      }
    }
    const near = [...this.chunks.values()].filter((c) => c.hasPaint && c.near !== false).length;
    return { styled: this.styled, raised: this.raised, chunks: this.chunks.size, nearChunks: this.styled ? near : 0,
             segments, raisedSegments: raised,
             lineVertices: lineVerts, bufferMB: +(bytes / 1e6).toFixed(1) };
  }

  /** Per frame: rebuild changed tiles, upload patched vertices. */
  update() {
    if (this.dirty.size) {
      const t0 = PROFILE ? now() : 0;
      const n = this.dirty.size;
      for (const key of this.dirty) this._rebuild(key);
      this.dirty.clear();
      if (PROFILE && n > 20) plog(`lines: ${n} chunks ${(now() - t0).toFixed(1)}ms`);
    }
    for (const c of this.chunks.values()) {
      if (!c.posDirty) continue;
      c.posDirty = false;
      c.lines.geometry.attributes.position.needsUpdate = true;
      c.lines.geometry.computeBoundingSphere();
      c.dots.geometry.boundingSphere = c.lines.geometry.boundingSphere;
      for (const g of [c.paint.geometry, c.raised.geometry]) {
        if (!g.attributes.aStart) continue;
        g.attributes.aStart.needsUpdate = true;
        g.attributes.aEnd.needsUpdate = true;
      }
    }
  }

  _rebuild(chunkKey) {
    const feats = [];
    let nVerts = 0;
    let nIdx = 0;
    for (const tile of tilesOfChunk(chunkKey)) {
      for (const f of tile.features) {
        if (f.kind !== "line") continue;
        const n = f.pos.length / 3;
        if (n < 2) continue;
        feats.push(f);
        nVerts += n;
        nIdx += 2 * (n - 1);
      }
    }
    if (feats.length === 0) {
      this._dispose(chunkKey);
      return;
    }
    const pos = new Float32Array(nVerts * 3);
    const col = new Float32Array(nVerts * 3);
    const idx = nVerts > 65535 ? new Uint32Array(nIdx) : new Uint16Array(nIdx);
    const c = this._chunk(chunkKey);
    const nSeg = nIdx / 2;
    const paint = this.styled ? newSegments(nSeg) : null;
    let nRaised = 0;
    if (this.styled) {
      for (const f of feats) {
        f.styleKey = lineStyleKey(f.tags);
        if (LINE_STYLES[f.styleKey].raise) nRaised += f.pos.length / 3 - 1;
      }
    }
    const raised = nRaised ? newSegments(nRaised) : null;
    let v = 0;
    let e = 0;
    let si = 0;
    let ri = 0;
    for (const f of feats) {
      const n = f.pos.length / 3;
      const key = lineStyleKey(f.tags);
      const color = _col.set(lineColor(f.tags, key));
      f.vChunk = c;
      f.vOff = v;
      if (paint) {
        f.sOff = si;
        f.rOff = LINE_STYLES[key].raise ? ri : -1;
        const row = styleIndex.get(key);
        const r8 = Math.round(color.r * 255), g8 = Math.round(color.g * 255), b8 = Math.round(color.b * 255);
        const rc = LINE_STYLES[key].raise ? _col.clone().set(LINE_STYLES[key].raise.color) : null;
        let along = 0;
        for (let i = 0; i < n - 1; i++) {
          const a = i * 3, b = a + 3;
          const segLen = Math.hypot(f.pos[b] - f.pos[a], f.pos[b + 1] - f.pos[a + 1], f.pos[b + 2] - f.pos[a + 2]);
          writeSegment(paint, si++, f.pos, a, b, along, along + segLen, row, r8, g8, b8);
          if (rc) {
            writeSegment(raised, ri++, f.pos, a, b, along, along + segLen, row,
              Math.round(rc.r * 255), Math.round(rc.g * 255), Math.round(rc.b * 255));
          }
          along += segLen;
        }
      }
      for (let i = 0; i < n; i++) {
        const o = (v + i) * 3;
        pos[o] = f.pos[i * 3];
        pos[o + 1] = f.pos[i * 3 + 1];
        pos[o + 2] = f.pos[i * 3 + 2];
        col[o] = color.r;
        col[o + 1] = color.g;
        col[o + 2] = color.b;
      }
      for (let i = 0; i < n - 1; i++) {
        idx[e++] = v + i;
        idx[e++] = v + i + 1;
      }
      v += n;
    }
    const posAttr = new THREE.BufferAttribute(pos, 3);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", posAttr);
    geom.setAttribute("color", new THREE.BufferAttribute(col, 3));
    geom.setIndex(new THREE.BufferAttribute(idx, 1));
    geom.computeBoundingSphere();
    const dotGeom = new THREE.BufferGeometry();
    dotGeom.setAttribute("position", posAttr);
    dotGeom.boundingSphere = geom.boundingSphere;
    c.lines.geometry.dispose();
    c.dots.geometry.dispose();
    c.lines.geometry = geom;
    c.dots.geometry = dotGeom;
    c.paint.geometry.dispose();
    c.raised.geometry.dispose();
    if (paint) {
      c.paint.geometry = segmentGeometry(paint, nSeg);
      boundSegments(c.paint.geometry, paint.start, paint.end, nSeg, 1);
    } else {
      c.paint.geometry = emptyGeometry();
    }
    if (raised) {
      c.raised.geometry = segmentGeometry(raised, nRaised);
      boundSegments(c.raised.geometry, raised.start, raised.end, nRaised, 3);
    } else {
      c.raised.geometry = emptyGeometry();
    }
    c.hasPaint = !!paint;
    c.hasRaised = !!raised;
    c.near = undefined;  // decided on the next frame
    this._show(c);
    c.posDirty = false;
  }

  _chunk(key) {
    let c = this.chunks.get(key);
    if (c) return c;
    const lines = new THREE.LineSegments(new THREE.BufferGeometry(), lineMaterial);
    const dots = new THREE.Points(new THREE.BufferGeometry(), dotMaterial);
    const paint = new THREE.Mesh(emptyGeometry(), paintMaterial);
    const raised = new THREE.Mesh(emptyGeometry(), raisedMaterial);
    dots.visible = this.dotsVisible;
    dots.renderOrder = 1000;
    mapRoot.add(lines);
    mapRoot.add(dots);
    mapRoot.add(paint);
    mapRoot.add(raised);
    c = { key, lines, dots, paint, raised, posDirty: false };
    this.chunks.set(key, c);
    return c;
  }

  _dispose(key) {
    const c = this.chunks.get(key);
    if (!c) return;
    for (const obj of [c.lines, c.dots, c.paint, c.raised]) {
      mapRoot.remove(obj);
      obj.geometry.dispose();
    }
    this.chunks.delete(key);
  }

  _clearAll() {
    for (const key of Array.from(this.chunks.keys())) this._dispose(key);
    this.dirty.clear();
  }

  // A drag in progress: rewrite the moved vertices of every feature using them.
  // Tiles already queued for a rebuild pick the new positions up from the store.
  _patchNodes(recs) {
    for (const rec of recs) {
      for (const { f, i } of rec.refs) {
        const c = f.vChunk;
        if (!c || f.kind !== "line" || this.dirty.has(c.key) || this.chunks.get(c.key) !== c) continue;
        c.lines.geometry.attributes.position.setXYZ(f.vOff + i, rec.x, rec.y, rec.z);
        // Segment i - 1 ends here, segment i starts here (dash phases keep
        // their old lengths until the chunk is rebuilt).
        const n = f.pos.length / 3;
        for (const [g, off] of [[c.paint.geometry, f.sOff], [c.raised.geometry, f.rOff]]) {
          if (!g.attributes.aStart || off === undefined || off < 0) continue;
          if (i > 0) g.attributes.aEnd.setXYZ(off + i - 1, rec.x, rec.y, rec.z);
          if (i < n - 1) g.attributes.aStart.setXYZ(off + i, rec.x, rec.y, rec.z);
        }
        c.posDirty = true;
      }
    }
  }
}

function newSegments(count) {
  return {
    start: new Float32Array(count * 3),
    end: new Float32Array(count * 3),
    dist: new Float32Array(count * 2),
    style: new Float32Array(count),
    color: new Uint8Array(count * 3),
  };
}

function writeSegment(buf, i, pos, a, b, d0, d1, row, r, g, bl) {
  buf.start[i * 3] = pos[a];
  buf.start[i * 3 + 1] = pos[a + 1];
  buf.start[i * 3 + 2] = pos[a + 2];
  buf.end[i * 3] = pos[b];
  buf.end[i * 3 + 1] = pos[b + 1];
  buf.end[i * 3 + 2] = pos[b + 2];
  buf.dist[i * 2] = d0;
  buf.dist[i * 2 + 1] = d1;
  buf.style[i] = row;
  buf.color[i * 3] = r;
  buf.color[i * 3 + 1] = g;
  buf.color[i * 3 + 2] = bl;
}

export const lineLayer = new LineLayer();
