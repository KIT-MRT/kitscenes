// Lines a fixed number of pixels wide, drawn over everything: the selection
// and hover highlight. WebGL ignores `linewidth`, and a one-pixel highlight
// disappears next to a styled marking several pixels wide, so each segment is
// an instanced quad widened on screen (the same trimming as the trajectories).
import * as THREE from "three";

const VERTEX = /* glsl */`
attribute vec3 aStart;
attribute vec3 aEnd;
uniform vec2 uResolution;
uniform float uWidth;
#include <common>
#include <logdepthbuf_pars_vertex>
void main() {
  vec4 s = modelViewMatrix * vec4(aStart, 1.0);
  vec4 e = modelViewMatrix * vec4(aEnd, 1.0);
  if (projectionMatrix[2][3] < -0.5) {
    float nearZ = -0.5 * projectionMatrix[3][2] / projectionMatrix[2][2];
    if (s.z > nearZ && e.z > nearZ) {
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
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
  vec4 clip = position.x < 0.5 ? cs : ce;
  vec2 offset = normal * position.y * uWidth * 0.5 + dir * (position.x < 0.5 ? -0.5 : 0.5) * uWidth;
  clip.xy += offset / uResolution * 2.0 * clip.w;
  gl_Position = clip;
  #include <logdepthbuf_vertex>
}
`;

const FRAGMENT = /* glsl */`
uniform vec3 uColor;
uniform float uOpacity;
#include <common>
#include <logdepthbuf_pars_fragment>
void main() {
  #include <logdepthbuf_fragment>
  gl_FragColor = vec4(uColor, uOpacity);
  #include <colorspace_fragment>
}
`;

const resolution = { value: new THREE.Vector2(1, 1) };
const quad = new THREE.Float32BufferAttribute([0, -1, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0], 3);
const quadIndex = new THREE.BufferAttribute(new Uint16Array([0, 1, 2, 0, 2, 3]), 1);

/** Per frame (with the drawing size): widths are pixels. */
export function setScreenLineResolution(w, h) {
  resolution.value.set(w, h);
}

/** A mesh of `widthPx` wide segments; `setPairs(flat)` takes x,y,z,x,y,z per segment. */
export function screenLines({ color, widthPx, opacity = 1, renderOrder = 1000 }) {
  const material = new THREE.ShaderMaterial({
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    uniforms: {
      uResolution: resolution,
      uWidth: { value: widthPx },
      uColor: { value: new THREE.Color(color) },
      uOpacity: { value: opacity },
    },
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(new THREE.InstancedBufferGeometry(), material);
  mesh.frustumCulled = false;
  mesh.renderOrder = renderOrder;
  mesh.setPairs = (pairs) => {
    mesh.geometry.dispose();
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute("position", quad);
    g.setIndex(quadIndex);
    const n = pairs.length / 6;
    const start = new Float32Array(n * 3);
    const end = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      start.set(pairs.slice(i * 6, i * 6 + 3), i * 3);
      end.set(pairs.slice(i * 6 + 3, i * 6 + 6), i * 3);
    }
    g.setAttribute("aStart", new THREE.InstancedBufferAttribute(start, 3));
    g.setAttribute("aEnd", new THREE.InstancedBufferAttribute(end, 3));
    g.instanceCount = n;
    mesh.geometry = g;
  };
  return mesh;
}
