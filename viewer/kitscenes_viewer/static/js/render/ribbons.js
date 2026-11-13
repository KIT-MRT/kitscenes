// Styled map lines: one instanced quad per segment, painted in the shader.
//
// Paint: each segment is widened on screen to the marking's real width at
// its distance (render/linestyles.js, metres), never narrower than the
// style's pixel minimum, after trimming it to the near plane (as the
// trajectories do). The fragment shader paints the pattern from two
// coordinates in metres: along the way (cumulative, so dashes run on across
// vertices) and across it (signed, left positive). Dashes, double lines,
// solid|dashed and the dash background cost a few instructions per pixel and
// no geometry. When the line is drawn wider than its real width (far away),
// the stripes merge into one, and dashes shorter than a few pixels merge
// into a solid line, so nothing shimmers.
//
// Raised: curbs, rails, walls and fences as upright quads in world metres on
// the same segments (their own, smaller instance buffers), lit by a fixed
// light so the faces read as faces. Rails and fences are cut out in the
// shader (posts, lattice): still opaque, still one draw call.
//
// Style parameters live in one small float texture, a row per style; each
// segment carries its style row and its colour (the `color` tag may differ).
import * as THREE from "three";
import { LINE_STYLES, STYLE_KEYS, PATTERN, RAISE } from "./linestyles.js";

const TEXELS = 4; // per style row, see buildStyleTexture

function rgb(hex) {
  const c = new THREE.Color(hex);
  return [c.r, c.g, c.b];
}

/**
 * Row per style:
 *   0: colour rgb, width (m)
 *   1: pattern, dash on (m), dash off (m), min width (px)
 *   2: gap colour rgb, has gap colour
 *   3: raise kind, bottom (m), top (m), stripe width (m)
 */
function buildStyleTexture() {
  const n = STYLE_KEYS.length;
  const data = new Float32Array(TEXELS * n * 4);
  STYLE_KEYS.forEach((key, row) => {
    const s = LINE_STYLES[key];
    const o = row * TEXELS * 4;
    data.set([...rgb(s.color), s.width], o);
    data.set([PATTERN[s.pattern || "solid"], s.dash ? s.dash[0] : 0, s.dash ? s.dash[1] : 0, s.minPx || 1.5], o + 4);
    data.set([...(s.gap ? rgb(s.gap) : [0, 0, 0]), s.gap ? 1 : 0], o + 8);
    const r = s.raise;
    data.set([r ? RAISE[r.kind] : 0, r ? r.bottom : 0, r ? r.top : 0, s.stripe || 0], o + 12);
  });
  const tex = new THREE.DataTexture(data, TEXELS, n, THREE.RGBAFormat, THREE.FloatType);
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.needsUpdate = true;
  return tex;
}

export const styleTexture = buildStyleTexture();
export const styleIndex = new Map(STYLE_KEYS.map((k, i) => [k, i]));

const STYLE_FETCH = /* glsl */`
uniform sampler2D uStyles;
uniform float uStyleRows;
vec4 styleTexel(float row, float col) {
  return texture2D(uStyles, vec2((col + 0.5) / ${TEXELS.toFixed(1)}, (row + 0.5) / uStyleRows));
}
`;

const PAINT_VERTEX = /* glsl */`
attribute vec3 aStart;
attribute vec3 aEnd;
attribute vec2 aDist;
attribute float aStyle;
attribute vec3 aColor;
uniform vec2 uResolution;
uniform float uLift;
varying vec3 vColor;
varying float vStyle;
varying float vAlong;
varying float vAcross;
varying float vWidened;
${STYLE_FETCH}
#include <common>
#include <logdepthbuf_pars_vertex>
void main() {
  vec4 s = modelViewMatrix * vec4(aStart, 1.0);
  vec4 e = modelViewMatrix * vec4(aEnd, 1.0);
  float d0 = aDist.x;
  float d1 = aDist.y;
  if (projectionMatrix[2][3] < -0.5) {
    float nearZ = -0.5 * projectionMatrix[3][2] / projectionMatrix[2][2];
    if (s.z > nearZ && e.z > nearZ) {
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      return;
    }
    if (s.z > nearZ) {
      float t = (nearZ - e.z) / (s.z - e.z);
      s.xyz = mix(e.xyz, s.xyz, t);
      d0 = mix(d1, d0, t);
    }
    if (e.z > nearZ) {
      float t = (nearZ - s.z) / (e.z - s.z);
      e.xyz = mix(s.xyz, e.xyz, t);
      d1 = mix(d0, d1, t);
    }
  }
  // A hair towards the camera, so the paint wins over lanelet surfaces and
  // tiles at the same height. The depth buffer is logarithmic in w, so this
  // moves the points (along the view ray: the picture does not change).
  if (projectionMatrix[2][3] < -0.5) {
    s.xyz *= 1.0 - uLift;
    e.xyz *= 1.0 - uLift;
  } else {
    s.z += uLift * 1000.0;
    e.z += uLift * 1000.0;
  }
  vec4 cs = projectionMatrix * s;
  vec4 ce = projectionMatrix * e;
  vec2 dir = (ce.xy / ce.w - cs.xy / cs.w) * uResolution;
  float len = length(dir);
  dir = len > 1e-6 ? dir / len : vec2(1.0, 0.0);
  vec2 normal = vec2(-dir.y, dir.x);
  bool atEnd = position.x > 0.5;
  vec4 mid = atEnd ? e : s;
  vec4 clip = atEnd ? ce : cs;
  vec4 p0 = styleTexel(aStyle, 0.0);
  vec4 p1 = styleTexel(aStyle, 1.0);
  float halfM = 0.5 * p0.w;
  // Paint lies flat on the road: offset sideways in the ground plane (world
  // horizontal, left of the way), so at street level it foreshortens like
  // paint does instead of standing up like a board.
  vec2 w = aEnd.xy - aStart.xy;
  vec2 wn = length(w) > 1e-9 ? normalize(vec2(-w.y, w.x)) : vec2(0.0, 1.0);
  vec2 wd = vec2(wn.y, -wn.x);
  vec3 sideView = mat3(modelViewMatrix) * vec3(wn, 0.0);
  vec3 alongView = mat3(modelViewMatrix) * vec3(wd, 0.0);
  vec4 flatClip = projectionMatrix * vec4(mid.xyz + sideView * position.y * halfM
                                      + alongView * (atEnd ? halfM : -halfM), 1.0);
  // How wide the flat paint comes out on screen, across the line.
  vec2 flatPx = (flatClip.xy / flatClip.w - clip.xy / clip.w) * uResolution * 0.5;
  float realHalfPx = abs(dot(flatPx, normal));
  float minHalfPx = 0.5 * p1.w;
  float widened = 1.0;
  if (realHalfPx >= minHalfPx) {
    clip = flatClip;
  } else {
    // Too thin to see (far away, or seen edge-on): widen on screen to the
    // pixel minimum instead.
    widened = minHalfPx / max(realHalfPx, 1e-6);
    vec2 offset = normal * position.y * minHalfPx + dir * (atEnd ? 1.0 : -1.0) * minHalfPx;
    clip.xy += offset / uResolution * 2.0 * clip.w;
  }
  gl_Position = clip;
  vColor = aColor;
  vStyle = aStyle;
  vAlong = (atEnd ? d1 : d0) + (atEnd ? halfM : -halfM) * widened;
  vAcross = position.y * halfM * widened;
  vWidened = widened;
  #include <logdepthbuf_vertex>
}
`;

const PAINT_FRAGMENT = /* glsl */`
varying vec3 vColor;
varying float vStyle;
varying float vAlong;
varying float vAcross;
varying float vWidened;
${STYLE_FETCH}
#include <common>
#include <logdepthbuf_pars_fragment>
bool dashOn(float on, float off) {
  float period = on + off;
  // Dashes a few pixels long would shimmer: draw them solid.
  if (fwidth(vAlong) * 6.0 > period) return true;
  return mod(vAlong, period) < on;
}
void main() {
  vec4 p0 = styleTexel(vStyle, 0.0);
  vec4 p1 = styleTexel(vStyle, 1.0);
  vec4 p2 = styleTexel(vStyle, 2.0);
  vec4 p3 = styleTexel(vStyle, 3.0);
  int pattern = int(p1.x + 0.5);
  float halfW = 0.5 * p0.w;
  bool on = true;
  bool striped = pattern == 2 || pattern == 3 || pattern == 4;
  // Drawn wider than real (far away): one stripe, no gap between them.
  if (striped && vWidened < 1.4) {
    float a = abs(vAcross);
    if (a < halfW - p3.w) discard;          // the gap between the two stripes
    bool left = vAcross > 0.0;
    // solid_dashed: solid on the left, dashed on the right, looking along
    // the way (lanelet2's naming); dashed_solid the other way round.
    bool dashedSide = (pattern == 3 && !left) || (pattern == 4 && left);
    if (dashedSide) on = dashOn(p1.y, p1.z);
  } else if (pattern == 1 || pattern == 5) {
    on = dashOn(p1.y, p1.z);
  }
  vec3 colour = vColor;
  if (!on) {
    if (p2.w < 0.5) discard;
    colour = p2.rgb;
  }
  #include <logdepthbuf_fragment>
  gl_FragColor = vec4(colour, 1.0);
  #include <colorspace_fragment>
}
`;

const RAISED_VERTEX = /* glsl */`
attribute vec3 aStart;
attribute vec3 aEnd;
attribute vec2 aDist;
attribute float aStyle;
attribute vec3 aColor;
varying vec3 vColor;
varying float vStyle;
varying float vAlong;
varying float vHeight;
varying float vShade;
${STYLE_FETCH}
#include <common>
#include <logdepthbuf_pars_vertex>
void main() {
  vec4 p3 = styleTexel(aStyle, 3.0);
  bool atEnd = position.x > 0.5;
  float h = position.y > 0.0 ? p3.z : p3.y;
  vec3 p = atEnd ? aEnd : aStart;
  p.z += h;
  vec2 along = aEnd.xy - aStart.xy;
  vec2 n = length(along) > 1e-9 ? normalize(vec2(-along.y, along.x)) : vec2(0.0, 1.0);
  // A fixed light from the south-west, so faces in different directions differ.
  vShade = 0.62 + 0.38 * abs(dot(n, normalize(vec2(-0.6, -0.8))));
  vColor = aColor;
  vStyle = aStyle;
  vAlong = atEnd ? aDist.y : aDist.x;
  vHeight = h;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  #include <logdepthbuf_vertex>
}
`;

const RAISED_FRAGMENT = /* glsl */`
varying vec3 vColor;
varying float vStyle;
varying float vAlong;
varying float vHeight;
varying float vShade;
${STYLE_FETCH}
#include <common>
#include <logdepthbuf_pars_fragment>
void main() {
  vec4 p3 = styleTexel(vStyle, 3.0);
  int kind = int(p3.x + 0.5);
  float top = p3.z;
  float shade = vShade;
  if (kind == 2) {
    // Guard rail: the band on top, posts every 2 m under it.
    bool band = vHeight > top - 0.3;
    bool post = mod(vAlong, 2.0) < 0.08 || fwidth(vAlong) > 0.05;
    if (!band && !post) discard;
    if (!band) shade *= 0.7;
  } else if (kind == 3) {
    // Fence: a 25 cm lattice with a top rail; far away a solid, darker face.
    if (fwidth(vAlong) < 0.04) {
      bool bar = mod(vAlong, 0.25) < 0.04 || mod(vHeight, 0.25) < 0.04 || vHeight > top - 0.05;
      if (!bar) discard;
    } else {
      shade *= 0.6;
    }
  }
  #include <logdepthbuf_fragment>
  gl_FragColor = vec4(vColor * shade, 1.0);
  #include <colorspace_fragment>
}
`;

const quad = (() => {
  // x: 0 start / 1 end; y: -1 right / +1 left (raised: bottom / top).
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute([0, -1, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0], 3));
  g.setIndex([0, 1, 2, 0, 2, 3]);
  return g;
})();

const sharedUniforms = {
  uStyles: { value: styleTexture },
  uStyleRows: { value: STYLE_KEYS.length },
  uResolution: { value: new THREE.Vector2(1, 1) },
  uLift: { value: 2e-5 },
};

export const paintMaterial = new THREE.ShaderMaterial({
  vertexShader: PAINT_VERTEX,
  fragmentShader: PAINT_FRAGMENT,
  uniforms: sharedUniforms,
});

export const raisedMaterial = new THREE.ShaderMaterial({
  vertexShader: RAISED_VERTEX,
  fragmentShader: RAISED_FRAGMENT,
  uniforms: { uStyles: sharedUniforms.uStyles, uStyleRows: sharedUniforms.uStyleRows },
  side: THREE.DoubleSide,
});

/** Per frame: the paint is sized in pixels, so it needs the drawing size. */
export function setRibbonResolution(width, height) {
  sharedUniforms.uResolution.value.set(width, height);
}

/** Instanced geometry over `count` segments, attribute arrays from the caller. */
export function segmentGeometry({ start, end, dist, style, color }, count) {
  const g = new THREE.InstancedBufferGeometry();
  g.index = quad.index;
  g.setAttribute("position", quad.getAttribute("position"));
  g.setAttribute("aStart", new THREE.InstancedBufferAttribute(start, 3));
  g.setAttribute("aEnd", new THREE.InstancedBufferAttribute(end, 3));
  g.setAttribute("aDist", new THREE.InstancedBufferAttribute(dist, 2));
  g.setAttribute("aStyle", new THREE.InstancedBufferAttribute(style, 1));
  g.setAttribute("aColor", new THREE.InstancedBufferAttribute(color, 3, true));
  g.instanceCount = count;
  return g;
}

/** A sphere around both ends of every segment (the quads widen by pixels; culling uses it). */
export function boundSegments(g, start, end, count, pad) {
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let i = 0; i < count; i++) {
    box.expandByPoint(v.set(start[i * 3], start[i * 3 + 1], start[i * 3 + 2]));
    box.expandByPoint(v.set(end[i * 3], end[i * 3 + 1], end[i * 3 + 2]));
  }
  if (box.isEmpty()) box.set(v.set(0, 0, 0), v.clone());
  box.expandByScalar(pad);
  g.boundingBox = box;
  g.boundingSphere = box.getBoundingSphere(new THREE.Sphere());
}
