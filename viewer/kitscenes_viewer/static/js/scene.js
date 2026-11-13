// Renderer, scene graph roots and the reference grid.
import * as THREE from "three";

// The logarithmic depth buffer keeps depth precision from a few centimetres
// (nudging a node) out to a whole map, so the clip planes can stay fixed while
// the camera moves right up to geometry.
export const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById("app").appendChild(renderer.domElement);

export const canvas = renderer.domElement;

export const scene = new THREE.Scene();
scene.background = new THREE.Color(0x11151c);

// Ground grid + axes for spatial reference.
export const grid = new THREE.GridHelper(400, 40, 0x2a3340, 0x20262f);
grid.rotation.x = Math.PI / 2; // GridHelper is XZ by default; rotate to XY
scene.add(grid);
scene.add(new THREE.AxesHelper(5));

/** Map geometry: batched line chunks and traffic-element icons. */
export const mapRoot = new THREE.Group();
scene.add(mapRoot);

/** Markers, highlights and the JOSM viewport rectangle, drawn over the map. */
export const overlayRoot = new THREE.Group();
scene.add(overlayRoot);
