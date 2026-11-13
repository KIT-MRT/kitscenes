// Ego trajectories of a whole city of recorded scenes.
//
// Every scene is a polyline through its frames' ego positions, a dot on every
// frame (the polyline's nodes, readable at any zoom), and an arrow per frame
// for the 6D pose once close enough to read one (heading, and a fin for
// "up", so pitch and roll read too). A city is hundreds of scenes and tens of
// thousands of frames, so everything is three draw calls: one instanced quad
// per segment, one point per frame, one instanced arrow per frame.
//
// WebGL ignores `linewidth`, so a line a user can actually see and click is a
// screen-space quad: each segment is widened in the vertex shader by a fixed
// number of pixels, after trimming it to the near plane so a segment that
// passes behind the camera does not flip across the screen.
//
// Picking is done on the CPU in screen space, the same way as for map nodes:
// nodes first (the arrows), then segments, so a click lands on a frame both
// up close and at city scale where the arrows have faded out.
import * as THREE from "three";

const LINE_VERTEX = /* glsl */`
attribute vec3 aStart;
attribute vec3 aEnd;
attribute vec3 aColor;
attribute float aScene;
uniform vec2 uResolution;
uniform float uWidth;
uniform float uSelected;
uniform float uHover;
uniform float uDim;
varying vec3 vColor;
#include <common>
#include <logdepthbuf_pars_vertex>
void main() {
  vec4 s = modelViewMatrix * vec4(aStart, 1.0);
  vec4 e = modelViewMatrix * vec4(aEnd, 1.0);
  // Perspective only: nothing is behind an orthographic camera that matters.
  if (projectionMatrix[2][3] < -0.5) {
    // The near plane in view space, read off the projection matrix.
    float nearZ = -0.5 * projectionMatrix[3][2] / projectionMatrix[2][2];
    if (s.z > nearZ && e.z > nearZ) {
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      vColor = vec3(0.0);
      return;
    }
    if (s.z > nearZ) s.xyz = mix(e.xyz, s.xyz, (nearZ - e.z) / (s.z - e.z));
    if (e.z > nearZ) e.xyz = mix(s.xyz, e.xyz, (nearZ - s.z) / (e.z - s.z));
  }
  vec4 cs = projectionMatrix * s;
  vec4 ce = projectionMatrix * e;
  vec2 dir = (ce.xy / ce.w - cs.xy / cs.w) * uResolution;
  float len = length(dir);
  dir = len > 1e-6 ? dir / len : vec2(1.0, 0.0);
  vec2 normal = vec2(-dir.y, dir.x);
  bool selected = abs(aScene - uSelected) < 0.5;
  bool hovered = abs(aScene - uHover) < 0.5;
  float width = uWidth * (selected ? 2.0 : (hovered ? 1.6 : 1.0));
  vec4 clip = position.x < 0.5 ? cs : ce;
  // Half a width past each end, so consecutive segments overlap at the joint.
  vec2 offset = normal * position.y * width * 0.5 + dir * (position.x < 0.5 ? -0.5 : 0.5) * width;
  clip.xy += offset / uResolution * 2.0 * clip.w;
  gl_Position = clip;
  vec3 colour = aColor;
  if (selected) colour = mix(aColor, vec3(1.0), 0.35);
  else if (uSelected >= 0.0) colour *= uDim;
  if (hovered && !selected) colour = mix(colour, vec3(1.0), 0.25);
  vColor = colour;
  #include <logdepthbuf_vertex>
}
`;

const ARROW_VERTEX = /* glsl */`
attribute vec3 aColor;
attribute float aScene;
uniform float uPxPerM;     // screen pixels per metre at one metre distance
uniform float uSizePx;
uniform vec2 uSizeM;       // never smaller / larger than this many metres
uniform vec2 uFade;        // fully visible nearer than x, gone beyond y
uniform float uSelected;
uniform float uDim;
varying vec3 vColor;
varying float vAlpha;
#include <common>
#include <logdepthbuf_pars_vertex>
void main() {
  vec4 centre = modelViewMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
  float dist = max(-centre.z, 1e-3);
  bool selected = abs(aScene - uSelected) < 0.5;
  // The selected scene keeps its arrows further out: it is the one being read.
  float far = selected ? uFade.y * 2.0 : uFade.y;
  vAlpha = 1.0 - smoothstep(uFade.x, far, dist);
  if (vAlpha <= 0.0) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vColor = vec3(0.0);
    return;
  }
  float size = clamp(dist * uSizePx / uPxPerM, uSizeM.x, uSizeM.y);
  gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position * size, 1.0);
  vec3 colour = aColor;
  if (selected) colour = mix(aColor, vec3(1.0), 0.35);
  else if (uSelected >= 0.0) colour *= uDim;
  vColor = colour;
  #include <logdepthbuf_vertex>
}
`;

const DOT_VERTEX = /* glsl */`
attribute vec3 aColor;
attribute float aScene;
uniform float uSize;
uniform vec2 uFade;
uniform float uSelected;
uniform float uDim;
varying vec3 vColor;
varying float vAlpha;
#include <common>
#include <logdepthbuf_pars_vertex>
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float dist = -mv.z;
  bool selected = abs(aScene - uSelected) < 0.5;
  float far = selected ? uFade.y * 2.0 : uFade.y;
  vAlpha = 1.0 - smoothstep(uFade.x, far, dist);
  if (vAlpha <= 0.0 || dist <= 0.0) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    vColor = vec3(0.0);
    return;
  }
  gl_Position = projectionMatrix * mv;
  gl_PointSize = uSize * (selected ? 1.4 : 1.0);
  vec3 colour = aColor;
  if (selected) colour = mix(aColor, vec3(1.0), 0.35);
  else if (uSelected >= 0.0) colour *= uDim;
  vColor = colour;
  #include <logdepthbuf_vertex>
}
`;

// A round dot with a dark rim, so it reads on the line it sits on.
const DOT_FRAGMENT = /* glsl */`
varying vec3 vColor;
varying float vAlpha;
#include <common>
#include <logdepthbuf_pars_fragment>
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  float r = length(d) * 2.0;
  if (r > 1.0) discard;
  #include <logdepthbuf_fragment>
  vec3 colour = r > 0.62 ? vColor * 0.25 : vColor;
  gl_FragColor = vec4(colour, vAlpha);
}
`;

const LINE_FRAGMENT = /* glsl */`
varying vec3 vColor;
#include <common>
#include <logdepthbuf_pars_fragment>
void main() {
  #include <logdepthbuf_fragment>
  gl_FragColor = vec4(vColor, 1.0);
}
`;

const ARROW_FRAGMENT = /* glsl */`
varying vec3 vColor;
varying float vAlpha;
#include <common>
#include <logdepthbuf_pars_fragment>
void main() {
  #include <logdepthbuf_fragment>
  gl_FragColor = vec4(vColor, vAlpha);
}
`;

/** A flat arrow along +x (the vehicle's forward axis) with a fin along +z. */
function arrowGeometry() {
  const v = [
    // head
    1.0, 0.0, 0.0, 0.25, 0.42, 0.0, 0.25, -0.42, 0.0,
    // shaft
    0.25, 0.15, 0.0, -0.75, 0.15, 0.0, -0.75, -0.15, 0.0,
    0.25, 0.15, 0.0, -0.75, -0.15, 0.0, 0.25, -0.15, 0.0,
    // fin: which way is up for this pose
    -0.55, 0.0, 0.0, 0.05, 0.0, 0.0, -0.55, 0.0, 0.45,
  ];
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
  return g;
}

/** One colour per scene, far apart for neighbours in the list. */
export function sceneColour(i) {
  return new THREE.Color().setHSL((i * 0.6180339887) % 1, 0.75, 0.58);
}

const _vp = new THREE.Matrix4();
const _viewport = new THREE.Vector4();

class TrajectoryLayer {
  constructor() {
    this.group = new THREE.Group();
    this.group.name = "trajectories";
    this.scenes = [];         // [{ id, frames, start (node offset) }]
    this.nodes = null;        // Float32Array xyz of every frame of every scene
    this.nodeScene = null;    // Uint32Array scene index per node
    this.nodeFrame = null;    // Uint32Array frame index per node
    this.lines = null;
    this.dots = null;
    this.arrows = null;
    this.selected = -1;
    this.hovered = -1;
    this.widthPx = 3;
    this.dotPx = 7;
    this.arrowPx = 22;
    // Arrows only once there is room to read one; dots much further out.
    this.fade = [150, 300];
    this.dotFade = [700, 1400];
    this.visible = true;
    this.lineMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uResolution: { value: new THREE.Vector2(1, 1) },
        uWidth: { value: this.widthPx },
        uSelected: { value: -1 },
        uHover: { value: -1 },
        uDim: { value: 0.55 },
      },
      vertexShader: LINE_VERTEX,
      fragmentShader: LINE_FRAGMENT,
    });
    this.dotMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uSize: { value: this.dotPx * (window.devicePixelRatio || 1) },
        uFade: { value: new THREE.Vector2(this.dotFade[0], this.dotFade[1]) },
        uSelected: { value: -1 },
        uDim: { value: 0.55 },
      },
      vertexShader: DOT_VERTEX,
      fragmentShader: DOT_FRAGMENT,
      transparent: true,
    });
    this.arrowMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uPxPerM: { value: 1 },
        uSizePx: { value: this.arrowPx },
        uSizeM: { value: new THREE.Vector2(0.35, 1.6) },
        uFade: { value: new THREE.Vector2(this.fade[0], this.fade[1]) },
        uSelected: { value: -1 },
        uDim: { value: 0.55 },
      },
      vertexShader: ARROW_VERTEX,
      fragmentShader: ARROW_FRAGMENT,
      transparent: true,
      side: THREE.DoubleSide,
    });
  }

  attach(parent) {
    parent.add(this.group);
  }

  clear() {
    for (const obj of [this.lines, this.dots, this.arrows]) {
      if (!obj) continue;
      obj.geometry.dispose();
      // An InstancedMesh's instanceMatrix buffer is only freed by its own dispose().
      if (obj.isInstancedMesh) obj.dispose();
      this.group.remove(obj);
    }
    this.lines = null;
    this.dots = null;
    this.arrows = null;
    this.scenes = [];
    this.nodes = null;
    this.setSelected(-1);
    this.setHovered(-1);
  }

  /**
   * Build from `/city.json`'s scenes: each { id, frames, pos: [x,y,z,...],
   * quat: [x,y,z,w,...] } in viewer ENU.
   */
  setScenes(scenes) {
    this.clear();
    let nodeCount = 0;
    let segCount = 0;
    for (const s of scenes) {
      const n = Math.floor(s.pos.length / 3);
      nodeCount += n;
      segCount += Math.max(0, n - 1);
    }
    this.nodes = new Float32Array(nodeCount * 3);
    this.nodeScene = new Uint32Array(nodeCount);
    this.nodeFrame = new Uint32Array(nodeCount);
    const segStart = new Float32Array(segCount * 3);
    const segEnd = new Float32Array(segCount * 3);
    const segColour = new Float32Array(segCount * 3);
    const segScene = new Float32Array(segCount);
    const arrowColour = new Float32Array(nodeCount * 3);
    const arrowScene = new Float32Array(nodeCount);
    const matrices = new Float32Array(nodeCount * 16);
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const p = new THREE.Vector3();
    const one = new THREE.Vector3(1, 1, 1);
    let node = 0;
    let seg = 0;
    scenes.forEach((s, si) => {
      const n = Math.floor(s.pos.length / 3);
      const c = sceneColour(si);
      this.scenes.push({ id: s.id, frames: n, start: node });
      for (let f = 0; f < n; f++, node++) {
        const x = s.pos[f * 3];
        const y = s.pos[f * 3 + 1];
        const z = s.pos[f * 3 + 2];
        this.nodes.set([x, y, z], node * 3);
        this.nodeScene[node] = si;
        this.nodeFrame[node] = f;
        arrowColour.set([c.r, c.g, c.b], node * 3);
        arrowScene[node] = si;
        q.set(s.quat[f * 4], s.quat[f * 4 + 1], s.quat[f * 4 + 2], s.quat[f * 4 + 3]).normalize();
        m.compose(p.set(x, y, z), q, one);
        m.toArray(matrices, node * 16);
        if (f > 0) {
          segStart.set(s.pos.slice((f - 1) * 3, f * 3), seg * 3);
          segEnd.set([x, y, z], seg * 3);
          segColour.set([c.r, c.g, c.b], seg * 3);
          segScene[seg] = si;
          seg++;
        }
      }
    });

    if (segCount) {
      const quad = new THREE.InstancedBufferGeometry();
      quad.setAttribute("position", new THREE.Float32BufferAttribute(
        [0, -1, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0], 3));
      quad.setIndex([0, 1, 2, 0, 2, 3]);
      quad.setAttribute("aStart", new THREE.InstancedBufferAttribute(segStart, 3));
      quad.setAttribute("aEnd", new THREE.InstancedBufferAttribute(segEnd, 3));
      quad.setAttribute("aColor", new THREE.InstancedBufferAttribute(segColour, 3));
      quad.setAttribute("aScene", new THREE.InstancedBufferAttribute(segScene, 1));
      quad.instanceCount = segCount;
      this.lines = new THREE.Mesh(quad, this.lineMaterial);
      // The quad's own positions say nothing about where it is drawn.
      this.lines.frustumCulled = false;
      this.lines.renderOrder = 3;
      this.group.add(this.lines);
    }
    if (nodeCount) {
      const dots = new THREE.BufferGeometry();
      dots.setAttribute("position", new THREE.BufferAttribute(this.nodes, 3));
      dots.setAttribute("aColor", new THREE.BufferAttribute(arrowColour, 3));
      dots.setAttribute("aScene", new THREE.BufferAttribute(arrowScene, 1));
      this.dots = new THREE.Points(dots, this.dotMaterial);
      this.dots.frustumCulled = false;
      this.dots.renderOrder = 4;
      this.group.add(this.dots);

      const geometry = arrowGeometry();
      geometry.setAttribute("aColor", new THREE.InstancedBufferAttribute(arrowColour, 3));
      geometry.setAttribute("aScene", new THREE.InstancedBufferAttribute(arrowScene, 1));
      this.arrows = new THREE.InstancedMesh(geometry, this.arrowMaterial, nodeCount);
      this.arrows.instanceMatrix.array.set(matrices);
      this.arrows.instanceMatrix.needsUpdate = true;
      this.arrows.frustumCulled = false;
      this.arrows.renderOrder = 5;
      this.group.add(this.arrows);
    }
    this.group.visible = this.visible;
  }

  setVisible(on) {
    this.visible = !!on;
    this.group.visible = this.visible;
  }

  setSelected(index) {
    this.selected = index;
    this.lineMaterial.uniforms.uSelected.value = index;
    this.dotMaterial.uniforms.uSelected.value = index;
    this.arrowMaterial.uniforms.uSelected.value = index;
  }

  setHovered(index) {
    this.hovered = index;
    this.lineMaterial.uniforms.uHover.value = index;
  }

  indexOf(id) {
    return this.scenes.findIndex((s) => s.id === id);
  }

  /** Per frame: the shaders need the viewport and the camera's pixel scale. */
  update(renderer, camera) {
    if (!this.lines && !this.arrows) return;
    this.dotMaterial.uniforms.uSize.value = this.dotPx * renderer.getPixelRatio();
    renderer.getViewport(_viewport);
    this.lineMaterial.uniforms.uResolution.value.set(_viewport.z, _viewport.w);
    // Pixels per metre at 1 m: viewport height over the frustum height there.
    const pxPerM = _viewport.w / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    this.arrowMaterial.uniforms.uPxPerM.value = pxPerM;
  }

  /** Bounds of everything, for framing the city. */
  box() {
    const box = new THREE.Box3();
    if (!this.nodes) return box;
    for (let i = 0; i < this.nodes.length; i += 3) {
      box.expandByPoint(new THREE.Vector3(this.nodes[i], this.nodes[i + 1], this.nodes[i + 2]));
    }
    return box;
  }

  /**
   * The frame under page pixel (x, y): { scene, frame, px } or null. Nodes
   * whose dot has faded out are not candidates; segments always are, so a
   * click on a line at city scale still picks the frame nearest to it.
   */
  pick(x, y, camera, rect, nodePx = 10, linePx = 7) {
    if (!this.nodes || !this.visible) return null;
    camera.updateMatrixWorld();
    _vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const e = _vp.elements;
    const n = this.nodes.length / 3;
    const sx = new Float32Array(n);
    const sy = new Float32Array(n);
    const ok = new Uint8Array(n);
    const camPos = camera.position;
    let best = null;
    for (let i = 0; i < n; i++) {
      const px = this.nodes[i * 3];
      const py = this.nodes[i * 3 + 1];
      const pz = this.nodes[i * 3 + 2];
      const w = e[3] * px + e[7] * py + e[11] * pz + e[15];
      if (w <= 1e-6) continue;
      sx[i] = rect.left + ((e[0] * px + e[4] * py + e[8] * pz + e[12]) / w + 1) / 2 * rect.width;
      sy[i] = rect.top + (1 - (e[1] * px + e[5] * py + e[9] * pz + e[13]) / w) / 2 * rect.height;
      ok[i] = 1;
      const dist = Math.hypot(px - camPos.x, py - camPos.y, pz - camPos.z);
      const far = this.nodeScene[i] === this.selected ? this.dotFade[1] * 2 : this.dotFade[1];
      if (dist > far) continue;
      const d = Math.hypot(sx[i] - x, sy[i] - y);
      if (d <= nodePx && (!best || d < best.px || (d === best.px && dist < best.dist))) {
        best = { node: i, px: d, dist };
      }
    }
    if (!best) {
      for (const s of this.scenes) {
        for (let f = 1; f < s.frames; f++) {
          const a = s.start + f - 1;
          const b = a + 1;
          if (!ok[a] || !ok[b]) continue;
          const dx = sx[b] - sx[a];
          const dy = sy[b] - sy[a];
          const len2 = dx * dx + dy * dy;
          const t = len2 > 0 ? Math.max(0, Math.min(1, ((x - sx[a]) * dx + (y - sy[a]) * dy) / len2)) : 0;
          const d = Math.hypot(sx[a] + t * dx - x, sy[a] + t * dy - y);
          if (d <= linePx && (!best || d < best.px)) best = { node: t < 0.5 ? a : b, px: d };
        }
      }
    }
    if (!best) return null;
    const si = this.nodeScene[best.node];
    return { scene: si, id: this.scenes[si].id, frame: this.nodeFrame[best.node], px: best.px };
  }

  /** Where frame `frame` of scene `index` is, as [x, y, z]. */
  nodePosition(index, frame) {
    const s = this.scenes[index];
    if (!s || frame < 0 || frame >= s.frames) return null;
    const i = (s.start + frame) * 3;
    return [this.nodes[i], this.nodes[i + 1], this.nodes[i + 2]];
  }

  /** Plan distance from (x, y) to the nearest frame of scene `index`. */
  distanceTo(index, x, y) {
    const s = this.scenes[index];
    if (!s) return Infinity;
    let best = Infinity;
    for (let f = 0; f < s.frames; f++) {
      const i = (s.start + f) * 3;
      best = Math.min(best, Math.hypot(this.nodes[i] - x, this.nodes[i + 1] - y));
    }
    return best;
  }

  /** Scenes with a frame within `radius` metres (in plan) of (x, y), nearest first. */
  scenesNear(x, y, radius) {
    if (!this.nodes) return [];
    const r2 = radius * radius;
    const out = [];
    for (let si = 0; si < this.scenes.length; si++) {
      const s = this.scenes[si];
      let bestD2 = Infinity;
      for (let f = 0; f < s.frames; f++) {
        const i = (s.start + f) * 3;
        const dx = this.nodes[i] - x;
        const dy = this.nodes[i + 1] - y;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestD2) bestD2 = d2;
      }
      if (bestD2 <= r2) out.push({ index: si, id: s.id, dist: Math.sqrt(bestD2) });
    }
    return out.sort((a, b) => a.dist - b.dist);
  }

  stats() {
    return {
      scenes: this.scenes.length,
      nodes: this.nodes ? this.nodes.length / 3 : 0,
      segments: this.lines ? this.lines.geometry.instanceCount : 0,
      selected: this.selected >= 0 ? this.scenes[this.selected].id : null,
      hovered: this.hovered >= 0 ? this.scenes[this.hovered].id : null,
      visible: this.visible,
    };
  }
}

export const trajectoryLayer = new TrajectoryLayer();
