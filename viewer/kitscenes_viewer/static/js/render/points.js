// LiDAR point clouds.
//
// One `THREE.Points` per loaded cloud, fed straight from the scene server's
// binary blob (see kitscenes_viewer/scene/points.py): float32 xyz
// relative to a float64 origin, uint8 intensity, uint8 sensor id. The origin
// becomes the object's position, so the attribute stays small and float32
// loses nothing even though ENU metres run into the kilometres.
//
// Colour is computed in the vertex shader from attributes that are already on
// the GPU, so switching between intensity / height / sensor costs no upload.
import * as THREE from "three";

// "solid" is not offered in the colour menu: it is what the single frame is
// painted when it is shown on top of the aggregate, so the two can be told
// apart at a glance.
export const COLOR_MODES = ["intensity", "height", "sensor", "solid", "class", "rgb", "instance"];
const FRAME_TINT = 0xff6b35; // warm orange: nothing in viridis comes near it
const MAGIC = "LL2PC1\0\0";
const HEADER_SIZE = 40;
const FLAG_INTENSITY = 1;
const FLAG_SENSOR = 2;
const FLAG_CLASS = 4;
const FLAG_RGB = 8;
const FLAG_INSTANCE = 16;

// lidar_top, corner left/right, left, right, front, rear — the dataset's own
// sensor_id order.
export const SENSOR_COLORS = [
  0xffffff, 0xff6b6b, 0x4fc3f7, 0x9bd24f, 0xffd23f, 0xc792ea, 0xff9f43,
];

const VERTEX = /* glsl */`
attribute float aIntensity;
attribute float aSensor;
attribute float aClass;
attribute vec3 aRGB;
// seamseg's instance number for the point, as a normalised uint16; 0 means no
// instance. Per frame only -- nothing tracks an object between frames.
attribute float aInstance;
uniform float uSize;
uniform int uMode;
uniform float uGain;
uniform float uZMin;
uniform float uZRange;
uniform vec3 uSensorColors[7];
uniform vec3 uSolid;
uniform sampler2D uPalette;
uniform sampler2D uClassFilter;   // one texel per class: red 0 hides it
uniform float uAboveGround;       // hide points higher than this over the ground
uniform vec2 uIntensityRange;     // keep reflectivity inside this
// Ground height under the point, in the same local frame as the position
// attribute: the chunks of an aggregate share one material but each has its
// own origin, so an absolute height could not come from a uniform.
attribute float aGround;
varying vec3 vColor;
#include <common>
#include <logdepthbuf_pars_vertex>

vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

// Instance numbers carry no meaning beyond "a different object", so they are
// hashed to a colour. The golden ratio keeps consecutive numbers far apart in
// hue, and jittering saturation and value separates neighbours that happen to
// land on a similar hue.
vec3 instanceColor(float id) {
  if (id < 0.5) return vec3(0.32);
  return hsv2rgb(vec3(fract(id * 0.6180339887),
                      0.55 + 0.35 * fract(id * 0.2454),
                      0.70 + 0.30 * fract(id * 0.7548)));
}

vec3 viridis(float t) {
  t = clamp(t, 0.0, 1.0);
  const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
  const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
  const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

void main() {
  // Filters first: a dropped point is pushed outside the clip volume, which
  // costs nothing and keeps the buffers untouched so a slider can be dragged
  // without re-uploading anything.
  float intensity = aIntensity;
  bool keep = intensity >= uIntensityRange.x && intensity <= uIntensityRange.y;
  if (texture2D(uClassFilter, vec2((aClass * 255.0 + 0.5) / 256.0, 0.5)).r < 0.5) {
    keep = false;
  }
  // aGround below -1e5 means "no ground known here yet" (a cloud that has
  // just arrived): keep it rather than drop a whole new frame.
  if (uAboveGround > 0.0 && aGround > -1e5 && position.z - aGround > uAboveGround) {
    keep = false;
  }
  if (!keep) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);  // behind the far plane
    gl_PointSize = 0.0;
    vColor = vec3(0.0);
    return;
  }
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  gl_PointSize = uSize;
  if (uMode == 1) {
    // World height: each cloud has its own origin, and the ramp spans them all.
    vColor = viridis(((modelMatrix * vec4(position, 1.0)).z + uZMin) / uZRange);
  } else if (uMode == 2) {
    vColor = uSensorColors[int(clamp(aSensor, 0.0, 6.0))];
  } else if (uMode == 3) {
    vColor = uSolid;
  } else if (uMode == 4) {
    // aClass is a normalised byte, so * 255 is the Vistas class id; the
    // palette is one texel per id.
    vColor = texture2D(uPalette, vec2((aClass * 255.0 + 0.5) / 256.0, 0.5)).rgb;
  } else if (uMode == 5) {
    vColor = aRGB;
  } else if (uMode == 6) {
    vColor = instanceColor(floor(aInstance * 65535.0 + 0.5));
  } else {
    vColor = viridis(clamp(aIntensity * uGain, 0.0, 1.0));
  }
  #include <logdepthbuf_vertex>
}
`;

const FRAGMENT = /* glsl */`
varying vec3 vColor;
uniform float uAlpha;
#include <common>
#include <logdepthbuf_pars_fragment>
void main() {
  #include <logdepthbuf_fragment>
  gl_FragColor = vec4(vColor, uAlpha);
}
`;

/** Parse a cloud blob: { count, origin, positions, intensity, sensor }. */
export function parseCloud(buffer) {
  const view = new DataView(buffer);
  let magic = "";
  for (let i = 0; i < 8; i++) magic += String.fromCharCode(view.getUint8(i));
  if (magic !== MAGIC) throw new Error(`not a point cloud (${JSON.stringify(magic)})`);
  const count = view.getUint32(8, true);
  const flags = view.getUint32(12, true);
  const origin = [view.getFloat64(16, true), view.getFloat64(24, true), view.getFloat64(32, true)];
  const positions = new Float32Array(buffer, HEADER_SIZE, count * 3);
  let offset = HEADER_SIZE + count * 12;
  let intensity = null;
  let sensor = null;
  if (flags & FLAG_INTENSITY) {
    intensity = new Uint8Array(buffer, offset, count);
    offset += count;
  }
  if (flags & FLAG_SENSOR) {
    sensor = new Uint8Array(buffer, offset, count);
    offset += count;
  }
  let classes = null;
  if (flags & FLAG_CLASS) {
    classes = new Uint8Array(buffer, offset, count);
    offset += count;
  }
  let rgb = null;
  if (flags & FLAG_RGB) {
    rgb = new Uint8Array(buffer, offset, count * 3);
    offset += count * 3;
  }
  let instances = null;
  if (flags & FLAG_INSTANCE) {
    // A Uint16Array view needs an even offset; without the rgb channel an
    // odd point count leaves it odd, so copy the bytes out instead.
    instances = offset % 2 === 0
      ? new Uint16Array(buffer, offset, count)
      : new Uint16Array(buffer.slice(offset, offset + count * 2));
  }
  return { count, origin, positions, intensity, sensor, classes, rgb, instances };
}

// One texel per class id, filled from the palette the scene server sends.
const paletteTexture = new THREE.DataTexture(
  new Uint8Array(256 * 4), 256, 1, THREE.RGBAFormat);
paletteTexture.needsUpdate = true;

// Same shape, but it carries "show this class" rather than a colour.
const classFilterTexture = new THREE.DataTexture(
  new Uint8Array(256 * 4).fill(255), 256, 1, THREE.RGBAFormat);
classFilterTexture.needsUpdate = true;

/** Hide the given class ids (a Set of numbers); everything else shows. */
export function setHiddenClasses(hidden) {
  const data = classFilterTexture.image.data;
  for (let i = 0; i < 256; i++) {
    const show = !hidden || !hidden.has(i);
    data[i * 4] = show ? 255 : 0;
    data[i * 4 + 3] = 255;
  }
  classFilterTexture.needsUpdate = true;
}

/** Install the Vistas colours: `rows` is 256 [r, g, b] entries. */
export function setClassPalette(rows) {
  const data = paletteTexture.image.data;
  for (let i = 0; i < 256; i++) {
    const rgb = (rows && rows[i]) || [90, 90, 90];
    data[i * 4] = rgb[0];
    data[i * 4 + 1] = rgb[1];
    data[i * 4 + 2] = rgb[2];
    data[i * 4 + 3] = 255;
  }
  paletteTexture.needsUpdate = true;
}

function makeMaterial(size, gain) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uSize: { value: size },
      uMode: { value: 0 },
      uGain: { value: gain },
      uZMin: { value: 0 },
      uZRange: { value: 10 },
      uSensorColors: { value: SENSOR_COLORS.map((c) => new THREE.Color(c)) },
      uSolid: { value: new THREE.Color(FRAME_TINT) },
      uPalette: { value: paletteTexture },
      uClassFilter: { value: classFilterTexture },
      uAboveGround: { value: 0 },
      uIntensityRange: { value: new THREE.Vector2(0, 1) },
      uAlpha: { value: 1 },
    },
    transparent: false,
    depthWrite: true,
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
  });
}

class PointLayer {
  constructor() {
    this.group = new THREE.Group();
    this.group.name = "pointClouds";
    // The single frame and the whole-scene aggregate are separate groups so
    // either can be shown alone, and separate materials because the height
    // colouring is scaled to the cloud it belongs to.
    this.frameGroup = new THREE.Group();
    this.aggregateGroup = new THREE.Group();
    // Whole-scene clouds of *other* scenes around the camera, in city mode:
    // chunks come and go as the camera moves, keyed "<scene>/<chunk>".
    this.cityGroup = new THREE.Group();
    this.cityChunks = new Map();
    this.group.add(this.frameGroup);
    this.group.add(this.aggregateGroup);
    this.group.add(this.cityGroup);
    this.cloud = null;
    this.visible = true;
    this.aggregateVisible = true;
    this.aggregateBounds = null; // { min, max } z over every chunk loaded
    this.aggregateCount = 0;
    this.aggregateHasClasses = false;
    this.aggregateHasRgb = false;
    this.mode = "intensity";
    // Tell the live frame from the aggregate by colour. Only takes effect
    // while both are on screen: a lone frame painted one flat colour would
    // throw away the intensity that makes markings readable.
    this.tintFrame = true;
    this.alpha = 1;
    this.aboveGround = 0;
    this.intensityRange = [0, 1];
    this.groundReady = false;
    this.size = 2;
    this.gain = 3; // most returns sit well below the retro-reflector ceiling
    this.material = makeMaterial(this.size, this.gain);
    this.aggregateMaterial = makeMaterial(this.size, this.gain);
    this.materials = [this.material, this.aggregateMaterial];
  }

  /** Points for one aggregate chunk (no instances: those are per frame). */
  _staticPoints(cloud) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(cloud.positions, 3));
    geometry.setAttribute("aIntensity", new THREE.BufferAttribute(
      cloud.intensity || new Uint8Array(cloud.count), 1, true));
    geometry.setAttribute("aSensor", new THREE.BufferAttribute(new Uint8Array(cloud.count), 1, false));
    geometry.setAttribute("aClass", new THREE.BufferAttribute(
      cloud.classes || new Uint8Array(cloud.count).fill(255), 1, true));
    geometry.setAttribute("aRGB", new THREE.BufferAttribute(
      cloud.rgb || new Uint8Array(cloud.count * 3).fill(90), 3, true));
    // The aggregate never carries instances; the attribute is still declared
    // so one material can be shared with the frame.
    geometry.setAttribute("aInstance", new THREE.BufferAttribute(
      new Uint16Array(cloud.count), 1, true));
    geometry.setAttribute("aGround", new THREE.BufferAttribute(
      new Float32Array(cloud.count).fill(-1e6), 1, false));
    geometry.computeBoundingSphere();
    const points = new THREE.Points(geometry, this.aggregateMaterial);
    points.position.set(cloud.origin[0], cloud.origin[1], cloud.origin[2]);
    points.renderOrder = 2;
    let min = Infinity;
    let max = -Infinity;
    for (let i = 2; i < cloud.positions.length; i += 3) {
      const z = cloud.origin[2] + cloud.positions[i];
      if (z < min) min = z;
      if (z > max) max = z;
    }
    points.userData.cloud = { ...cloud, object: points, zMin: min, zMax: max };
    return points;
  }

  /** Add one chunk of the whole-scene aggregate. */
  addAggregateChunk(buffer) {
    const cloud = parseCloud(buffer);
    const points = this._staticPoints(cloud);
    this.aggregateGroup.add(points);
    this._groundChanged();
    this.aggregateCount += cloud.count;
    if (cloud.classes) this.aggregateHasClasses = true;
    if (cloud.rgb) this.aggregateHasRgb = true;
    this._refreshStaticRange();
    return cloud.count;
  }

  /**
   * Heights are absolute metres in the static clouds, so the colour ramp has
   * to span what is actually loaded -- this scene's aggregate and every city
   * chunk around it -- or everything comes out one colour.
   */
  _refreshStaticRange() {
    let min = Infinity;
    let max = -Infinity;
    for (const group of [this.aggregateGroup, this.cityGroup]) {
      for (const child of group.children) {
        const c = child.userData.cloud;
        if (!c || !c.count) continue;
        min = Math.min(min, c.zMin);
        max = Math.max(max, c.zMax);
      }
    }
    if (min > max) {
      this.aggregateBounds = null;
      return;
    }
    this.aggregateBounds = { min, max };
    this.aggregateMaterial.uniforms.uZMin.value = -min;
    this.aggregateMaterial.uniforms.uZRange.value = Math.max(0.5, max - min);
  }

  // ---- city mode: static chunks of many scenes, loaded around the camera ----
  /** Add chunk `key` ("<scene>/<index>") of scene `sceneId`; replaces a same-key one. */
  addCityChunk(key, sceneId, buffer) {
    this.removeCityChunk(key);
    const cloud = parseCloud(buffer);
    const points = this._staticPoints(cloud);
    points.userData.sceneId = sceneId;
    points.visible = !this.hiddenCityScenes || !this.hiddenCityScenes.has(sceneId);
    this.cityGroup.add(points);
    this.cityChunks.set(key, points);
    this._groundChanged();
    this._refreshStaticRange();
    return cloud.count;
  }

  removeCityChunk(key) {
    const points = this.cityChunks.get(key);
    if (!points) return false;
    points.geometry.dispose();
    this.cityGroup.remove(points);
    this.cityChunks.delete(key);
    this._groundChanged();
    this._refreshStaticRange();
    return true;
  }

  clearCity() {
    for (const key of [...this.cityChunks.keys()]) this.removeCityChunk(key);
  }

  /**
   * Hide one scene's city chunks while that scene shows its own aggregate, so
   * the same points are not drawn twice.
   */
  setCitySceneVisible(sceneId, on) {
    this.hiddenCityScenes = this.hiddenCityScenes || new Set();
    if (on) this.hiddenCityScenes.delete(sceneId);
    else this.hiddenCityScenes.add(sceneId);
    for (const points of this.cityChunks.values()) {
      if (points.userData.sceneId === sceneId) points.visible = !!on;
    }
  }

  cityStats() {
    const scenes = new Set();
    let points = 0;
    let shown = 0;
    for (const p of this.cityChunks.values()) {
      scenes.add(p.userData.sceneId);
      points += p.userData.cloud.count;
      if (p.visible) shown += p.userData.cloud.count;
    }
    return { chunks: this.cityChunks.size, scenes: scenes.size, points, shown };
  }

  clearAggregate() {
    for (const child of [...this.aggregateGroup.children]) {
      child.geometry.dispose();
      this.aggregateGroup.remove(child);
    }
    this.aggregateCount = 0;
    this.aggregateHasClasses = false;
    this.aggregateHasRgb = false;
    this._refreshStaticRange();
  }

  setAggregateVisible(on) {
    this.aggregateVisible = !!on;
    this.aggregateGroup.visible = this.aggregateVisible;
    this._applyModes();
  }

  setTintFrame(on) {
    this.tintFrame = !!on;
    this._applyModes();
  }

  /** The aggregate always shows the chosen colouring; the frame may be tinted. */
  _applyModes() {
    const chosen = COLOR_MODES.indexOf(this.mode);
    const instance = this.mode === "instance";
    // The aggregate has no instance numbers -- seamseg does not track objects
    // between frames -- so it stays on intensity rather than going flat grey,
    // which also makes the frame's instances stand out against it.
    this.aggregateMaterial.uniforms.uMode.value =
      instance ? COLOR_MODES.indexOf("intensity") : chosen;
    // Instance colours are the whole point of that mode, so they beat the
    // tint that otherwise marks the frame out from the aggregate.
    const tinted = this.aggregateGroup.visible && this.tintFrame && !instance;
    this.material.uniforms.uMode.value = tinted ? COLOR_MODES.indexOf("solid") : chosen;
  }

  attach(parent) {
    parent.add(this.group);
  }

  /** Replace the shown single-frame cloud with one decoded blob. */
  setCloud(buffer) {
    const cloud = parseCloud(buffer);
    this.clear();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(cloud.positions, 3));
    if (cloud.intensity) {
      geometry.setAttribute("aIntensity", new THREE.BufferAttribute(cloud.intensity, 1, true));
    } else {
      geometry.setAttribute("aIntensity", new THREE.BufferAttribute(new Uint8Array(cloud.count), 1, true));
    }
    const sensor = cloud.sensor || new Uint8Array(cloud.count);
    geometry.setAttribute("aSensor", new THREE.BufferAttribute(sensor, 1, false));
    geometry.setAttribute("aClass", new THREE.BufferAttribute(
      cloud.classes || new Uint8Array(cloud.count).fill(255), 1, true));
    geometry.setAttribute("aRGB", new THREE.BufferAttribute(
      cloud.rgb || new Uint8Array(cloud.count * 3).fill(90), 3, true));
    geometry.setAttribute("aInstance", new THREE.BufferAttribute(
      cloud.instances || new Uint16Array(cloud.count), 1, true));
    geometry.setAttribute("aGround", new THREE.BufferAttribute(
      new Float32Array(cloud.count).fill(-1e6), 1, false));
    geometry.computeBoundingSphere();
    const points = new THREE.Points(geometry, this.material);
    points.position.set(cloud.origin[0], cloud.origin[1], cloud.origin[2]);
    points.frustumCulled = true;
    points.renderOrder = 2;
    this.frameGroup.add(points);
    this.cloud = { ...cloud, object: points };
    // Height colouring spans this cloud: a fixed range would be meaningless
    // in a map whose absolute heights are in the hundreds of metres.
    let min = Infinity;
    let max = -Infinity;
    for (let i = 2; i < cloud.positions.length; i += 3) {
      const z = cloud.origin[2] + cloud.positions[i];  // absolute, as the shader reads it
      if (z < min) min = z;
      if (z > max) max = z;
    }
    if (min > max) {
      min = 0;
      max = 1;
    }
    this.material.uniforms.uZMin.value = -min;
    this.material.uniforms.uZRange.value = Math.max(0.5, max - min);
    this.frameGroup.visible = this.visible;
    this._groundChanged();
    return this.cloud;
  }

  /**
   * A cloud came or went: with the above-ground filter on, the ground grid
   * is rebuilt once things settle (a stream of chunks is one rebuild, not
   * one each). Until then the new points count as "ground unknown" and stay.
   */
  _groundChanged() {
    this.groundReady = false;
    if (!(this.aboveGround > 0)) return;
    clearTimeout(this._groundTimer);
    this._groundTimer = setTimeout(() => {
      if (this.aboveGround > 0 && !this.groundReady) this.ensureGround();
    }, 250);
  }

  clear() {
    for (const child of [...this.frameGroup.children]) {
      child.geometry.dispose();
      this.frameGroup.remove(child);
    }
    this.cloud = null;
  }

  /** The single-frame cloud only. The aggregate has its own switch. */
  setVisible(on) {
    this.visible = !!on;
    this.frameGroup.visible = this.visible;
  }

  setMode(mode) {
    this.mode = COLOR_MODES.includes(mode) && mode !== "solid" ? mode : "intensity";
    this._applyModes();
  }

  /** Does the loaded data carry Vistas classes at all? */
  hasClasses() {
    return Boolean((this.cloud && this.cloud.classes) || this.aggregateHasClasses
      || [...this.cityChunks.values()].some((p) => p.userData.cloud.classes));
  }

  /** Does it carry camera colour? */
  hasRgb() {
    return Boolean((this.cloud && this.cloud.rgb) || this.aggregateHasRgb
      || [...this.cityChunks.values()].some((p) => p.userData.cloud.rgb));
  }

  /** Does the loaded *frame* carry instance numbers? The aggregate never does. */
  hasInstances() {
    return Boolean(this.cloud && this.cloud.instances);
  }

  /** How many distinct instances the loaded frame holds, 0 aside. */
  instanceCount() {
    if (!this.cloud || !this.cloud.instances) return 0;
    const seen = new Set();
    for (const id of this.cloud.instances) if (id) seen.add(id);
    return seen.size;
  }

  setSize(px) {
    this.size = Math.max(1, Math.min(12, px));
    for (const m of this.materials) m.uniforms.uSize.value = this.size;
  }

  setGain(gain) {
    this.gain = Math.max(0.25, Math.min(16, gain));
    for (const m of this.materials) m.uniforms.uGain.value = this.gain;
  }

  /** Point opacity, so map lines and icons behind the cloud stay readable. */
  setAlpha(alpha) {
    this.alpha = Math.max(0.05, Math.min(1, alpha));
    for (const m of this.materials) {
      m.uniforms.uAlpha.value = this.alpha;
      // Writing depth from translucent points would hide whatever is behind
      // them, which is the whole reason for turning the alpha down.
      m.transparent = this.alpha < 1;
      m.depthWrite = this.alpha >= 1;
      m.needsUpdate = true;
    }
  }

  setIntensityRange(low, high) {
    this.intensityRange = [Math.max(0, low), Math.min(1, high)];
    for (const m of this.materials) m.uniforms.uIntensityRange.value.set(...this.intensityRange);
  }

  /** Hide points more than `metres` above the local ground; 0 turns it off. */
  setAboveGround(metres) {
    this.aboveGround = Math.max(0, metres || 0);
    // The grid depends on the clouds, not the threshold: dragging the slider
    // must not rebuild it on every step.
    if (this.aboveGround > 0 && !this.groundReady) this.ensureGround();
    for (const m of this.materials) m.uniforms.uAboveGround.value = this.aboveGround;
  }

  /**
   * Fill every cloud's `aGround` from a coarse pillar grid over all of them.
   *
   * The lowest return in a cell is the ground as far as this filter is
   * concerned: a percentile would be tidier but needs a sort per cell, and
   * the lowest point only ever makes the filter more generous.
   */
  ensureGround(cell = 4.0) {
    const clouds = this._allClouds();
    if (!clouds.length) return;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const { positions, origin, count } of clouds) {
      for (let i = 0; i < count; i++) {
        const x = origin[0] + positions[i * 3];
        const y = origin[1] + positions[i * 3 + 1];
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    const nx = Math.max(1, Math.ceil((maxX - minX) / cell) + 1);
    const ny = Math.max(1, Math.ceil((maxY - minY) / cell) + 1);
    const grid = new Float32Array(nx * ny).fill(Infinity);
    const cellOf = (x, y) => Math.min(nx - 1, Math.floor((x - minX) / cell))
      + Math.min(ny - 1, Math.floor((y - minY) / cell)) * nx;
    for (const { positions, origin, count } of clouds) {
      for (let i = 0; i < count; i++) {
        const at = cellOf(origin[0] + positions[i * 3], origin[1] + positions[i * 3 + 1]);
        const z = origin[2] + positions[i * 3 + 2];
        if (z < grid[at]) grid[at] = z;
      }
    }
    for (const cloud of clouds) {
      const { positions, origin, count, object } = cloud;
      const attr = object.geometry.getAttribute("aGround");
      for (let i = 0; i < count; i++) {
        const at = cellOf(origin[0] + positions[i * 3], origin[1] + positions[i * 3 + 1]);
        const ground = grid[at];
        // Stored in the cloud's own frame, like `position`.
        attr.array[i] = (Number.isFinite(ground) ? ground : -1e6) - origin[2];
      }
      attr.needsUpdate = true;
    }
    this.groundReady = true;
  }

  _allClouds() {
    const out = [];
    if (this.cloud) out.push(this.cloud);
    for (const group of [this.aggregateGroup, this.cityGroup]) {
      for (const child of group.children) {
        if (child.userData.cloud) out.push(child.userData.cloud);
      }
    }
    return out;
  }

  /** {class id: points}; `everything` includes the aggregate's chunks. */
  classHistogram(everything = false) {
    const clouds = everything ? this._allClouds() : (this.cloud ? [this.cloud] : []);
    const withClasses = clouds.filter((c) => c.classes);
    if (!withClasses.length) return null;
    // Counted once per cloud and kept on it: this runs after every frame and
    // every chunk, and a city holds millions of points.
    const counts = {};
    for (const cloud of withClasses) {
      if (!cloud.histogram) {
        const own = new Uint32Array(256);
        for (let i = 0; i < cloud.count; i++) own[cloud.classes[i]]++;
        cloud.histogram = own;
      }
      cloud.histogram.forEach((n, id) => { if (n) counts[id] = (counts[id] || 0) + n; });
    }
    return counts;
  }

  stats() {
    const cloud = this.cloud;
    return {
      count: cloud ? cloud.count : 0,
      // Effective visibility: a flag on a group whose parent is hidden says
      // nothing, which is exactly how hiding the frame cloud once took the
      // aggregate with it.
      visible: this.group.visible && this.frameGroup.visible,
      mode: this.mode,
      frameMode: COLOR_MODES[this.material.uniforms.uMode.value],
      aggregateMode: COLOR_MODES[this.aggregateMaterial.uniforms.uMode.value],
      tintFrame: this.tintFrame,
      instances: this.instanceCount(),
      size: this.size,
      alpha: this.alpha,
      aboveGround: this.aboveGround,
      // Share of the frame's points with a known ground under them.
      groundKnown: cloud ? Array.from(cloud.object.geometry.getAttribute("aGround").array)
        .filter((g) => g > -1e5).length / Math.max(1, cloud.count) : 0,
      intensityRange: this.intensityRange,
      origin: cloud ? cloud.origin : null,
      aggregate: this.aggregateCount,
      aggregateChunks: this.aggregateGroup.children.length,
      aggregateVisible: this.group.visible && this.aggregateGroup.visible,
      city: this.cityStats(),
      // First point in world ENU, so a test can check where a cloud landed.
      first: cloud && cloud.count
        ? [cloud.origin[0] + cloud.positions[0],
           cloud.origin[1] + cloud.positions[1],
           cloud.origin[2] + cloud.positions[2]]
        : null,
    };
  }
}

export const pointLayer = new PointLayer();
