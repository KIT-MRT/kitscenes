// End-to-end checks of recorded sensor scenes: the scene bar, point clouds,
// frame stepping, colouring and the jump to the car.
//
// The scene server is faked here the way the JOSM bridge is: a small HTTP
// server serving the same `/scene.json` and `/frame/<i>/points.bin` as
// kitscenes_viewer.scene, so the browser side is tested without the 4.7 GB
// dataset. The binary layout below must stay in step with
// kitscenes_viewer/scene/points.py (there is a Python test for that
// side).
//   node viewer/tests/browser/e2e_scene.mjs [--shots DIR]
import http from "node:http";
import { openViewer, Checks, dist3, sleep } from "./harness.mjs";

const shotsIdx = process.argv.indexOf("--shots");
const shotsDir = shotsIdx > 0 ? process.argv[shotsIdx + 1] : null;
const t = new Checks();
const Z = 100;

const FRAMES = 14;
const POINTS_PER_FRAME = 500;
// Frame i puts its cloud around x = 10 * i, so "did the frame change?" is a
// question about geometry, not about a counter.
const frameOrigin = (i) => [10 * i, 0, Z];

// Mapillary Vistas ids the scene server also uses: 58 Car, 48 Pole, 31 Person.
const CLASS_CAR = 58;
const CLASS_POLE = 48;
const CLASS_NONE = 255;
const classPalette = () => {
  const rows = Array.from({ length: 256 }, () => [90, 90, 90]);
  rows[CLASS_CAR] = [0, 0, 142];
  rows[CLASS_POLE] = [153, 153, 153];
  rows[31] = [220, 20, 60];
  return rows;
};

// `instances` is what a single frame carries and the aggregate does not:
// seamseg numbers objects within one image and nothing tracks them onward.
function encodeCloud(index, count = POINTS_PER_FRAME, spacing = 10, instanced = true) {
  const origin = [spacing * index, 0, Z];
  const header = new ArrayBuffer(40);
  const view = new DataView(header);
  for (const [k, ch] of [..."LL2PC1"].entries()) view.setUint8(k, ch.charCodeAt(0));
  view.setUint32(8, count, true);
  view.setUint32(12, 1 | 2 | 4 | 8 | (instanced ? 16 : 0), true); // intensity, sensor, class, rgb, instance
  view.setFloat64(16, origin[0], true);
  view.setFloat64(24, origin[1], true);
  view.setFloat64(32, origin[2], true);
  const xyz = new Float32Array(count * 3);
  const intensity = new Uint8Array(count);
  const sensor = new Uint8Array(count);
  const classes = new Uint8Array(count);
  const rgb = new Uint8Array(count * 3);
  const instances = new Uint16Array(count);
  for (let p = 0; p < count; p++) {
    const a = (p / count) * Math.PI * 2;
    xyz[p * 3] = Math.cos(a) * 6;
    xyz[p * 3 + 1] = Math.sin(a) * 6;
    xyz[p * 3 + 2] = (p % 10) * 0.2;
    intensity[p] = p % 256;
    sensor[p] = p % 7;
    classes[p] = p % 5 === 0 ? CLASS_CAR : p % 5 === 1 ? CLASS_POLE : CLASS_NONE;
    rgb[p * 3] = p % 256;
    rgb[p * 3 + 1] = 255 - (p % 256);
    rgb[p * 3 + 2] = 128;
    // Three cars and nothing else numbered, as the real masks come out.
    instances[p] = classes[p] === CLASS_CAR ? 1 + (p % 3) : 0;
  }
  const parts = [Buffer.from(header), Buffer.from(xyz.buffer),
                 Buffer.from(intensity), Buffer.from(sensor),
                 Buffer.from(classes), Buffer.from(rgb)];
  if (instanced) parts.push(Buffer.from(instances.buffer));
  return Buffer.concat(parts);
}

const ego = {
  pos: Array.from({ length: FRAMES }, (_, i) => [10 * i, 0, Z]),
  // Heading east, level: identity in the viewer's ENU convention.
  quat: Array.from({ length: FRAMES }, () => [0, 0, 0, 1]),
};

let served = [];
let aggregateReadyAfter = 2;   // 202 with progress this many times, then ready
let aggregateAsked = 0;
let aggregateChunksServed = [];
const AGG_CHUNKS = 3;
const AGG_PER_CHUNK = 200;
let delayMs = 0;        // pretend a frame costs real decode time
let inFlight = 0;
let maxInFlight = 0;
const data = http.createServer((req, res) => {
  const cors = { "Access-Control-Allow-Origin": "*", "Cache-Control": "no-store" };
  const match = /^\/frame\/(\d+)\/points\.bin$/.exec(req.url);
  if (req.url === "/aggregate/meta.json") {
    aggregateAsked++;
    const ready = aggregateAsked > aggregateReadyAfter;
    res.writeHead(ready ? 200 : 202, { ...cors, "Content-Type": "application/json" });
    res.end(JSON.stringify(ready
      ? { ready: true,
          stats: { points: AGG_CHUNKS * AGG_PER_CHUNK, voxel_m: 0.15, min_run: 3 },
          chunks: Array.from({ length: AGG_CHUNKS }, (_, i) => (
            { index: i, count: AGG_PER_CHUNK, centre: [i * 40, 0, 100], radius: 20 })) }
      : { ready: false, progress: { state: "scanning", done: aggregateAsked, total: 9 } }));
    return;
  }
  const aggChunk = /^\/aggregate\/(\d+)\.bin$/.exec(req.url);
  if (aggChunk) {
    const index = Number(aggChunk[1]);
    aggregateChunksServed.push(index);
    res.writeHead(200, { ...cors, "Content-Type": "application/octet-stream" });
    res.end(encodeCloud(index, AGG_PER_CHUNK, 40, false));
    return;
  }
  if (req.url === "/scene.json") {
    res.writeHead(200, { ...cors, "Content-Type": "application/json" });
    res.end(JSON.stringify(sceneInfo()));
    return;
  }
  if (match) {
    const index = Number(match[1]);
    served.push(index);
    if (index >= FRAMES) {
      res.writeHead(404, cors);
      res.end("no such frame");
      return;
    }
    const body = encodeCloud(index);
    inFlight++;
    maxInFlight = Math.max(maxInFlight, inFlight);
    let done = false;
    const release = () => { if (!done) { done = true; inFlight--; } };
    // An aborted fetch closes the connection; the real server drops that work
    // at the gate, so it must not count as busy here either.
    res.on("close", release);
    const finish = () => {
      if (done) return;
      release();
      if (!res.writableEnded) {
        res.writeHead(200, { ...cors, "Content-Type": "application/octet-stream" });
        res.end(body);
      }
    };
    if (delayMs) setTimeout(finish, delayMs);
    else finish();
    return;
  }
  res.writeHead(404, cors);
  res.end("not found");
});
await new Promise((r) => data.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${data.address().port}`;
const sceneInfo = () => ({
  id: "e2e-scene-0000", base, frames: FRAMES,
  times: Array.from({ length: FRAMES }, (_, i) => i * 0.1),
  lidars: ["lidar_top", "lidar_front"],
  ego,
  cameras: {
    camera_ring_front: {
      fx: 1841, fy: 1841, cx: 1700, cy: 1100, width: 3504, height: 2272,
      // camera x = ego -y (right), camera y = ego -z (down), camera z = ego x
      // (forward), mounted 1.5 m up and 0.2 m ahead of the ego origin.
      T_ego_cam: [[0, 0, 1, 0.2], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]],
    },
    camera_ring_rear: {
      fx: 1845, fy: 1845, cx: 1760, cy: 1140, width: 3504, height: 2272,
      T_ego_cam: [[0, 0, -1, -0.2], [1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]],
    },
  },
  anchor: { lat: 49.0, lon: 8.4 },
  classes: {
    palette: classPalette(),
    names: { 58: "Car", 48: "Pole", 31: "Person" },
    unlabelled: CLASS_NONE,
    cameras: ["camera_ring_front", "camera_ring_rear"],
  },
});

const line = (id, nodes, pts) => ({ id: `way/${id}`, kind: "line", tags: { type: "line_thin" },
                                    pts: pts.flat(), nodes });
const features = [line(10, [1001, 1002], [[0, -20, Z], [60, -20, Z]])];

const v = await openViewer({ shotsDir });
const sceneState = () => v.evaluate("window.__ll2scene.state()");
const points = () => v.evaluate("window.__ll2scene.points()");
const barHidden = () => v.evaluate("document.getElementById('sceneBar').hidden");
const label = () => v.evaluate("document.getElementById('sceneFrameLabel').textContent");
async function waitFor(pred, ms = 5000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    if (await pred()) return true;
    await sleep(50);
  }
  return false;
}
/** Click an element the way a user does: a real mouse press at its centre.
 *  `#leftCol` is pointer-events:none, so a panel that forgets to opt back in
 *  looks fine and reacts only to Tab+Enter. */
async function clickElement(id) {
  const box = await v.evaluate(`(() => { const b = document.getElementById(${JSON.stringify(id)})`
    + `.getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2]; })()`);
  await v.click(box[0], box[1]);
  await sleep(120);
}
// Where the free camera looks, from its yaw/pitch (camera.js: yaw 0 is north
// along +y, counter-clockwise positive).
const lookDir = (cam) => [
  -Math.sin(cam.yaw) * Math.cos(cam.pitch),
  Math.cos(cam.yaw) * Math.cos(cam.pitch),
  Math.sin(cam.pitch),
];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const unit = (a) => { const n = Math.hypot(...a) || 1; return a.map((v) => v / n); };
const waitLoaded = (frame) => waitFor(async () => {
  const s = await sceneState();
  return s.frame === frame && !s.loading;
});

try {
  v.sendScene({ type: "snapshot", anchor: { lat: 49.0, lon: 8.4 }, features });
  t.check("map streamed", await v.waitForFeatures(1));
  t.check("no scene bar without a scene", await barHidden());

  v.sendScene({ type: "scene", scene: sceneInfo() });
  t.check("the scene announcement shows the scene bar", await waitFor(async () => !(await barHidden())));
  t.check("and loads frame 0 by itself", await waitLoaded(0));
  const uiState = () => v.evaluate("window.__ll2ui.state()");
  t.check("the side panel turns to the sensor data", (await uiState()).tab === "scene", JSON.stringify(await uiState()));
  t.check("the timeline shows under the view", !(await v.evaluate("document.getElementById('timeline').hidden"))
    && (await v.evaluate("document.body.classList.contains('hasTimeline')")));
  t.check("and the breadcrumb names the scene",
    (await v.evaluate("document.getElementById('crumbSceneName').textContent")).startsWith("e2e-scen")
    && !(await v.evaluate("document.getElementById('crumbs').hidden")));
  t.check("the camera bar offers one button per ring camera",
    (await v.evaluate("[...document.querySelectorAll('#camButtons [data-camera]')].map((b) => b.textContent).join()")) === "F,R",
    await v.evaluate("[...document.querySelectorAll('#camButtons [data-camera]')].map((b) => b.textContent).join()"));
  let p = await points();
  t.check("the point cloud is on screen", p.count === POINTS_PER_FRAME && p.visible, JSON.stringify(p));
  t.check("placed where the scene server put it",
    dist3(p.origin, frameOrigin(0)) < 1e-9, JSON.stringify(p.origin));
  t.check("the frame label counts from 0", (await label()).startsWith(`0 / ${FRAMES - 1}`), await label());
  await v.key("b");
  await v.shot("scene_points.png");

  // ---- the bar reacts to the mouse, not just to the keyboard -----------------------
  await clickElement("scenePoints");
  t.check("a real mouse click on the scene bar hides the cloud", !(await points()).visible);
  await clickElement("scenePoints");
  t.check("and shows it again", (await points()).visible);
  await clickElement("sceneNext");
  t.check("the step button works with the mouse too", await waitLoaded(1));
  await clickElement("scenePrev");
  await waitLoaded(0);

  // ---- stepping ------------------------------------------------------------------
  await v.evaluate("document.getElementById('sceneNext').click()");
  t.check("the next-frame button loads frame 1", await waitLoaded(1));
  p = await points();
  t.check("and the cloud moved with it", dist3(p.origin, frameOrigin(1)) < 1e-9, JSON.stringify(p.origin));
  await v.key(".");
  t.check("'.' steps forward too", await waitLoaded(2));
  await v.key(",");
  t.check("',' steps back", await waitLoaded(1));
  t.check("a frame already seen is not fetched again",
    served.filter((i) => i === 1).length === 1, JSON.stringify(served));

  await v.evaluate("(() => { const el = document.getElementById('sceneFrame');"
    + " el.value = '4'; el.dispatchEvent(new Event('input')); })()");
  t.check("the slider jumps to a frame", await waitLoaded(4));
  await v.evaluate("document.getElementById('scenePrev').click()");
  t.check("the previous-frame button goes back", await waitLoaded(3));

  // ---- colouring and size --------------------------------------------------------
  await v.evaluate("(() => { const el = document.getElementById('sceneMode');"
    + " el.value = 'height'; el.dispatchEvent(new Event('change')); })()");
  t.check("the colour mode switches without reloading the cloud",
    (await points()).mode === "height" && (await points()).count === POINTS_PER_FRAME);
  await v.evaluate("(() => { const el = document.getElementById('sceneSize');"
    + " el.value = '5'; el.dispatchEvent(new Event('input')); })()");
  t.check("the point size follows the slider", (await points()).size === 5);
  t.check("the legend shows the height ramp",
    await v.evaluate("!!document.querySelector('#pointLegend .ramp')"));
  await v.evaluate("(() => { const el = document.getElementById('sceneMode');"
    + " el.value = 'sensor'; el.dispatchEvent(new Event('change')); })()");
  t.check("and one swatch per LiDAR for the sensor colours",
    (await v.evaluate("document.querySelectorAll('#pointLegend .key').length")) === 7);
  await v.evaluate("(() => { const el = document.getElementById('sceneMode');"
    + " el.value = 'height'; el.dispatchEvent(new Event('change')); })()");
  await v.shot("scene_height.png");

  // ---- jump to the car -------------------------------------------------------------
  const before = await v.camera();
  const atFrame = (await sceneState()).frame;
  await v.key("k");
  await sleep(150);
  const cam = await v.camera();
  const car = ego.pos[atFrame];
  const away = Math.hypot(cam.pos[0] - car[0], cam.pos[1] - car[1]);
  t.check("K puts the camera behind the car of the current frame",
    away > 5 && away < 40 && cam.pos[2] > car[2] && dist3(cam.pos, before.pos) > 1,
    `${away.toFixed(1)} m away, ${(cam.pos[2] - car[2]).toFixed(1)} m up`);
  // The car drives +x here, so the camera belongs west of it, aimed at it.
  t.check("standing behind the car, not beside it", cam.pos[0] < car[0] - 5
    && Math.abs(cam.pos[1] - car[1]) < 1, JSON.stringify(cam.pos.map((n) => +n.toFixed(2))));
  const atCar = unit([car[0] - cam.pos[0], car[1] - cam.pos[1], car[2] - cam.pos[2]]);
  t.check("and actually looking at it", dot(lookDir(cam), atCar) > 0.999,
    `dot ${dot(lookDir(cam), atCar).toFixed(4)}, yaw ${cam.yaw.toFixed(2)}`);
  await v.shot("scene_car.png");

  // ---- hiding ----------------------------------------------------------------------
  await v.key("p");
  t.check("P hides the cloud", !(await points()).visible);
  await v.key("p");
  t.check("and shows it again", (await points()).visible);

  // ---- a refused frame is reported, not silently dropped ---------------------------
  await v.evaluate("window.__ll2scene.show(99)");
  await sleep(400);
  t.check("a frame out of range clamps instead of erroring",
    (await sceneState()).frame === FRAMES - 1 && !(await sceneState()).error,
    JSON.stringify(await sceneState()));

  // ---- stepping fast must not pile decodes on the server ---------------------------
  // Each frame costs the real server seconds of CPU; eight at once made every
  // answer ~6x slower, including the one being waited for.
  served = [];
  maxInFlight = 0;
  delayMs = 250;
  await v.evaluate("window.__ll2scene.show(6)");
  for (let i = 0; i < 6; i++) await v.key(".");
  const settled = await waitFor(async () => {
    const st = await sceneState();
    return !st.loading && st.frame === 12;
  }, 15000);
  await sleep(300);
  t.check("rapid stepping ends on the frame asked for", settled, JSON.stringify(await sceneState()));
  t.check("with only one request in flight at a time", maxInFlight === 1, `peak ${maxInFlight}`);
  t.check("and the frames stepped over are never fetched",
    served.length <= 4, `fetched ${JSON.stringify(served)}`);
  t.check("the cloud shown is the one for that frame",
    dist3((await points()).origin, frameOrigin(12)) < 1e-9, JSON.stringify((await points()).origin));
  // A look-ahead is dropped for a frame the user is actually waiting for.
  await v.evaluate("window.__ll2scene.show(13)");
  await waitLoaded(13);
  served = [];
  maxInFlight = 0;
  delayMs = 400;
  await sleep(600);           // the idle look-ahead starts (and will not finish)
  await v.evaluate("window.__ll2scene.show(5)");
  const jumped = await waitFor(async () => {
    const st = await sceneState();
    return st.frame === 5 && !st.loading;
  }, 10000);
  t.check("a jump during a look-ahead is served, not queued behind it", jumped,
    JSON.stringify(await sceneState()));
  t.check("and still only one request runs at a time", maxInFlight === 1, `peak ${maxInFlight}`);
  t.check("showing the frame that was asked for",
    dist3((await points()).origin, frameOrigin(5)) < 1e-9, JSON.stringify((await points()).origin));
  delayMs = 0;

  // ---- the whole scene as one cloud ------------------------------------------------
  const aggOf = () => v.evaluate("window.__ll2scene.aggregate()");
  const aggLabel = () => v.evaluate("document.getElementById('sceneAggregate').textContent");
  t.check("nothing is aggregated to begin with", (await aggOf()).points === 0);
  await v.key("P", { modifiers: ["shift"] });
  await sleep(300);
  t.check("Shift+P asks the server to build it", (await aggOf()).building,
    JSON.stringify(await aggOf()));
  t.check("and the button reports the progress", (await aggLabel()).startsWith("Building"),
    await aggLabel());
  const built = await waitFor(async () => {
    const a = await aggOf();
    return !a.building && a.points > 0;
  }, 15000);
  t.check("it arrives once the server says ready", built, JSON.stringify(await aggOf()));
  let agg = await aggOf();
  t.check("every chunk is loaded", agg.loaded === AGG_CHUNKS
    && agg.points === AGG_CHUNKS * AGG_PER_CHUNK, JSON.stringify(agg));
  t.check("chunks are fetched one at a time, in order",
    aggregateChunksServed.join() === "0,1,2", aggregateChunksServed.join());
  let stats = await points();
  t.check("the aggregate is on screen", stats.aggregate === AGG_CHUNKS * AGG_PER_CHUNK
    && stats.aggregateChunks === AGG_CHUNKS && stats.aggregateVisible, JSON.stringify(stats));
  t.check("and the single frame steps out of the way", !stats.visible, JSON.stringify(stats));
  await v.key("b");
  await v.shot("scene_aggregate.png");

  // ---- putting the frame back, told apart by colour --------------------------------
  await v.key("p");
  await sleep(150);
  stats = await points();
  t.check("P puts the single frame back on top of the aggregate",
    stats.visible && stats.aggregateVisible, JSON.stringify(stats));
  t.check("painted one colour so it is not mistaken for the aggregate",
    stats.frameMode === "solid" && stats.mode !== "solid", JSON.stringify(stats));
  await v.shot("scene_aggregate_frame.png");
  await v.key("u");
  await sleep(150);
  stats = await points();
  t.check("U turns the tint off and the frame is coloured like the aggregate",
    !stats.tintFrame && stats.frameMode === stats.mode, JSON.stringify(stats));
  await v.key("u");

  await v.key("P", { modifiers: ["shift"] });
  await sleep(200);
  stats = await points();
  t.check("Shift+P again hides it without throwing it away",
    !stats.aggregateVisible && (await aggOf()).points > 0);
  t.check("the frame goes back to its own colouring once it is alone",
    stats.visible && stats.frameMode === stats.mode, JSON.stringify(stats));
  await v.key("P", { modifiers: ["shift"] });
  await sleep(200);
  t.check("and shows it again without refetching",
    (await points()).aggregateVisible && aggregateChunksServed.length === AGG_CHUNKS,
    aggregateChunksServed.join());
  // Showing the aggregate put the frame away again, so P brings it back.
  await v.key("p");
  await sleep(150);
  let vis = await points();
  t.check("P puts the frame back over the aggregate", vis.visible && vis.aggregateVisible,
    JSON.stringify(vis));
  await v.key("p");
  await sleep(150);
  // Both live under one parent group; hiding the frame cloud by hiding that
  // parent took the aggregate with it, while the child's own flag still said
  // "visible". The hook reports effective visibility for that reason.
  vis = await points();
  t.check("P hides the frame cloud but leaves the aggregate",
    !vis.visible && vis.aggregateVisible, JSON.stringify(vis));
  t.check("and the aggregate is still actually drawn",
    (await v.evaluate("window.__ll2test.stats().drawCalls")) > 0);

  // ---- Vistas classes from the seamseg masks ---------------------------------------
  const classesOf = () => v.evaluate("window.__ll2scene.classes()");
  let cls = await classesOf();
  t.check("the scene brings its class table", cls.available && cls.car === "Car",
    JSON.stringify({ available: cls.available, car: cls.car, cameras: cls.cameras }));
  t.check("and the points carry a class each", cls.inData, JSON.stringify(cls));
  const hist = cls.histogram || {};
  t.check("which decode to the classes the server sent",
    hist[CLASS_CAR] === POINTS_PER_FRAME / 5 && hist[CLASS_POLE] === POINTS_PER_FRAME / 5
    && hist[CLASS_NONE] === (POINTS_PER_FRAME * 3) / 5, JSON.stringify(hist));
  t.check("the colour menu offers class only for a scene that has it",
    await v.evaluate("!!document.querySelector('#sceneMode option[value=\"class\"]')"));
  await v.evaluate("(() => { const el = document.getElementById('sceneMode');"
    + " el.value = 'class'; el.dispatchEvent(new Event('change')); })()");
  await sleep(150);
  t.check("selecting it colours by class", (await points()).mode === "class",
    JSON.stringify(await points()));
  await v.shot("scene_classes.png");
  t.check("the camera colour shows up as a mode too",
    (await classesOf()).rgb
    && (await v.evaluate("!!document.querySelector('#sceneMode option[value=\"rgb\"]')")));
  await v.evaluate("(() => { const el = document.getElementById('sceneMode');"
    + " el.value = 'rgb'; el.dispatchEvent(new Event('change')); })()");
  await sleep(150);
  t.check("and selecting it colours from the cameras", (await points()).mode === "rgb");

  // ---- one colour per object in the frame -------------------------------------------
  await v.key("p");  // the frame cloud was hidden above; instances live on it
  t.check("the frame carries three distinct instance numbers",
    (await classesOf()).instances === 3, JSON.stringify(await classesOf()));
  t.check("the colour menu offers instance",
    await v.evaluate("!!document.querySelector('#sceneMode option[value=\"instance\"]')"));
  await v.evaluate("(() => { const el = document.getElementById('sceneMode');"
    + " el.value = 'instance'; el.dispatchEvent(new Event('change')); })()");
  await sleep(150);
  let inst = await points();
  t.check("selecting it colours the frame by instance", inst.frameMode === "instance",
    JSON.stringify(inst));
  // The frame is normally painted one flat colour while the aggregate shows,
  // which would defeat the whole mode.
  t.check("the tint steps aside for it", inst.tintFrame && inst.frameMode !== "solid");
  // Nothing numbers an object across frames, so the aggregate has no
  // instances; grey would be the honest colour but a useless picture.
  // Instances without camera colour and an odd point count start at an odd
  // byte, where a Uint16Array view throws.
  const oddInstances = await v.evaluate(`import('/js/render/points.js').then((m) => {
    const n = 5;
    const buf = new ArrayBuffer(40 + n * 12 + n * 3 + n * 2);
    const dv = new DataView(buf);
    [...'LL2PC1'].forEach((c, i) => dv.setUint8(i, c.charCodeAt(0)));
    dv.setUint32(8, n, true);
    dv.setUint32(12, 1 | 2 | 4 | 16, true);
    for (let i = 0; i < n; i++) dv.setUint16(40 + n * 15 + 2 * i, i + 1, true);
    return Array.from(m.parseCloud(buf).instances);
  })`);
  t.check("instances read at an odd offset too", JSON.stringify(oddInstances) === "[1,2,3,4,5]",
    JSON.stringify(oddInstances));
  t.check("and the aggregate stays on intensity", inst.aggregateMode === "intensity",
    JSON.stringify(inst));
  await v.shot("scene_instances.png");
  await v.evaluate("(() => { const el = document.getElementById('sceneMode');"
    + " el.value = 'intensity'; el.dispatchEvent(new Event('change')); })()");

  // ---- filters ---------------------------------------------------------------------
  const filtersOf = () => v.evaluate("window.__ll2scene.filters()");
  const setRange = (id, value) => v.evaluate(
    `(() => { const el = document.getElementById('${id}');`
    + ` el.value = '${value}'; el.dispatchEvent(new Event('input')); })()`);
  await v.evaluate("document.getElementById('sceneFilterBtn').click()");
  t.check("the Filter button opens the panel",
    await v.evaluate("!document.getElementById('sceneFilters').hidden"));
  t.check("a chip per class present in the data",
    (await v.evaluate("document.querySelectorAll('#classChips label').length")) === 2,
    await v.evaluate("document.getElementById('classChips').textContent"));

  await v.evaluate("document.getElementById('classPresetCars').click()");
  let filters = await filtersOf();
  t.check("the vehicles preset hides the car class", filters.hidden.includes(CLASS_CAR),
    JSON.stringify(filters.hidden));
  t.check("and leaves other classes alone", !filters.hidden.includes(CLASS_POLE));
  t.check("the button says how much is filtered",
    (await v.evaluate("document.getElementById('sceneFilterBtn').textContent")).startsWith("Filter ·"));
  await v.shot("scene_filtered.png");
  // Filtering is a shader uniform: nothing is refetched or re-uploaded.
  t.check("filtering does not refetch anything",
    aggregateChunksServed.length === AGG_CHUNKS, aggregateChunksServed.join());
  t.check("and the points are all still loaded", (await points()).count === POINTS_PER_FRAME);

  await v.evaluate("document.getElementById('classShowAll').click()");
  t.check("Show all clears the class filter", (await filtersOf()).hidden.length === 0);

  // Inverted: the ticked classes are the only ones drawn.
  await clickElement("classMode");
  filters = await filtersOf();
  t.check("Only with nothing ticked filters nothing yet", filters.only && filters.hidden.length === 0,
    JSON.stringify(filters));
  t.check("the switch says Only", (await v.evaluate("document.getElementById('classMode').textContent")) === "Only");
  await v.evaluate("document.getElementById('classPresetCars').click()");
  filters = await filtersOf();
  t.check("Only + Vehicles hides everything but vehicles",
    !filters.hidden.includes(CLASS_CAR) && filters.hidden.includes(CLASS_POLE)
    && filters.hidden.includes(CLASS_NONE), JSON.stringify(filters.hidden.length));
  t.check("the car chip reads as shown, not hidden",
    await v.evaluate("(() => { const l = document.querySelector('#classChips label[data-class-id=\"58\"]');"
      + " return l.classList.contains('only') && !l.classList.contains('hidden'); })()"));
  t.check("and the button says so", (await v.evaluate("document.getElementById('sceneFilterBtn').textContent"))
    .startsWith("Filter · only"), await v.evaluate("document.getElementById('sceneFilterBtn').textContent"));
  await v.shot("scene_only_cars.png");
  await clickElement("classMode");
  filters = await filtersOf();
  t.check("back to Hide, the same ticks hide the vehicles again",
    !filters.only && filters.hidden.includes(CLASS_CAR) && !filters.hidden.includes(CLASS_POLE));
  await v.evaluate("document.getElementById('classShowAll').click()");

  await setRange("pointAlpha", 40);
  t.check("the opacity slider fades the points", Math.abs((await points()).alpha - 0.4) < 1e-9,
    JSON.stringify(await points()));
  await setRange("pointAlpha", 100);

  await setRange("aboveGround", 2);
  t.check("the above-ground filter takes a height", (await filtersOf()).aboveGround === 2
    && (await points()).aboveGround === 2);
  t.check("and labels it in metres",
    (await v.evaluate("document.getElementById('aboveGroundLabel').textContent")) === "2.0 m");
  // A frame stepped to while the filter is on gets its ground too. It used
  // to keep "ground unknown", which the filter read as "far above" -- the
  // whole new frame vanished.
  const filteredFrame = (await sceneState()).frame;
  await v.key(".");
  await waitLoaded(filteredFrame + 1);
  t.check("a frame stepped to under the filter gets its ground",
    await waitFor(async () => (await points()).groundKnown > 0.99, 3000),
    String((await points()).groundKnown));
  await setRange("aboveGround", 0);
  t.check("zero turns it off again",
    (await v.evaluate("document.getElementById('aboveGroundLabel').textContent")) === "off");

  await setRange("intensityMin", 50);
  t.check("the intensity floor is applied",
    Math.abs((await points()).intensityRange[0] - 0.5) < 1e-9, JSON.stringify(await points()));
  await setRange("intensityMin", 0);
  await v.evaluate("document.getElementById('sceneFilterBtn').click()");

  // ---- camera lock ---------------------------------------------------------------
  await v.evaluate("window.__ll2scene.show(2)");
  await waitLoaded(2);
  const rigOf = () => v.evaluate("window.__ll2scene.rig()");
  // The lock is the recorded camera's perspective: a map view gives way to it.
  await v.key("v");
  t.check("(ortho on before locking)", (await v.evaluate("window.__ll2map.ortho()")).on);
  await clickElement("sceneLock");
  // A reconnect replays the scene announcement; nothing about the scene
  // changed, so nothing may be dropped -- the lock least of all.
  const lockedBefore = (await v.evaluate("window.__ll2scene.rig()")).on;
  v.sendScene({ type: "scene", scene: sceneInfo() });
  await sleep(400);
  t.check("the same scene announced again keeps the lock",
    lockedBefore && (await v.evaluate("window.__ll2scene.rig()")).on);
  t.check("locking the camera leaves the ortho view", !(await v.evaluate("window.__ll2map.ortho()")).on
    && !(await v.evaluate("document.getElementById('orthoBtn').classList.contains('on')")));
  await sleep(200);
  let rigNow = await rigOf();
  t.check("the lock turns on with the front camera", rigNow.on && rigNow.camera === "camera_ring_front",
    JSON.stringify(rigNow));
  const lockCam = await v.camera();
  const mount = [ego.pos[2][0] + 0.2, ego.pos[2][1], ego.pos[2][2] + 1.5];
  t.check("the camera sits at the mount of that sensor", dist3(lockCam.pos, mount) < 1e-6,
    `${JSON.stringify(lockCam.pos.map((n) => +n.toFixed(3)))} vs ${JSON.stringify(mount)}`);
  t.check("the view is letterboxed to the image aspect",
    Math.abs(rigNow.viewport.width / rigNow.viewport.height - 3504 / 2272) < 0.01,
    JSON.stringify(rigNow.viewport));

  // The whole point of the lock: the render must reproduce the recorded
  // camera's pinhole projection, principal point and all.
  const K = sceneInfo().cameras.camera_ring_front;
  const worst = await v.evaluate(`(() => {
    const K = ${JSON.stringify(K)};
    const ego = ${JSON.stringify(ego.pos[2])};
    let worst = 0, sample = null;
    for (const [dx, dy, dz] of [[20,0,0],[35,6,2],[12,-4,-1],[60,15,5],[25,-9,3]]) {
      // camera frame: z forward (ego +x), x right (ego -y), y down (ego -z)
      const Xc = -dy, Yc = -(dz - 1.5), Zc = dx - 0.2;
      const want = { u: K.fx * Xc / Zc + K.cx, v: K.fy * Yc / Zc + K.cy };
      const got = window.__ll2scene.project([ego[0]+dx, ego[1]+dy, ego[2]+dz]);
      const d = Math.hypot(got.u - want.u, got.v - want.v);
      if (d > worst) { worst = d; sample = { want, got }; }
    }
    return { worst, sample };
  })()`);
  t.check("the render matches the recorded camera's pinhole projection",
    worst.worst < 0.01, `worst ${worst.worst.toFixed(4)} px — ${JSON.stringify(worst.sample)}`);

  t.check("the recorded image is blended over the render",
    rigNow.overlay && rigNow.imageSrc.endsWith("/frame/2/camera/camera_ring_front.jpg"), JSON.stringify(rigNow));
  await v.key("o");
  t.check("O takes the image away but keeps the lock",
    !(await rigOf()).overlay && (await rigOf()).on);
  await v.key("o");

  await v.key("]");
  await sleep(150);
  rigNow = await rigOf();
  t.check("] walks to the next ring camera", rigNow.camera === "camera_ring_rear", rigNow.camera);
  // view.yaw is the unlocked camera's state; while locked the orientation
  // lives in the camera matrix, so compare where it actually looks.
  const dirs = await v.evaluate(`(() => {
    const r = window.__ll2scene.rig();
    const p = window.__ll2scene.project;
    const ego = ${JSON.stringify(ego.pos[2])};
    // A point 20 m ahead of the car is behind a rear-facing camera.
    return { ahead: p([ego[0] + 20, ego[1], ego[2]]), rear: r.camera };
  })()`);
  t.check("and the rear camera looks the other way",
    dirs.ahead && dirs.ahead.behind === true, JSON.stringify(dirs));
  await v.key("[");

  await v.evaluate("window.__ll2scene.show(4)");
  await waitLoaded(4);
  t.check("stepping a frame moves the locked view and its image",
    (await rigOf()).imageSrc.endsWith("/frame/4/camera/camera_ring_front.jpg")
    && dist3((await v.camera()).pos, mount) > 1,
    (await rigOf()).imageSrc);

  // ---- releasing the lock leaves the view where it is ------------------------------
  const locked = await v.camera();
  const lockedDir = locked.dir;
  await v.key("L", { modifiers: ["shift"] });
  await sleep(200);
  t.check("Shift+L releases the lock", !(await rigOf()).on);
  const freed = await v.camera();
  t.check("the camera stays exactly where the lock left it",
    dist3(freed.pos, locked.pos) < 1e-9,
    `${JSON.stringify(locked.pos)} -> ${JSON.stringify(freed.pos)}`);
  t.check("still looking the same way", dot(freed.dir, unit(lockedDir)) > 0.9999,
    `${JSON.stringify(lockedDir.map((n) => +n.toFixed(4)))} -> `
    + `${JSON.stringify(freed.dir.map((n) => +n.toFixed(4)))}`);
  t.check("and yaw/pitch now describe that direction",
    dot(lookDir(freed), freed.dir) > 0.9999, `dot ${dot(lookDir(freed), freed.dir).toFixed(5)}`);
  t.check("and the ordinary projection is back", Math.abs(freed.fov - 55) < 1e-6
    && (await rigOf()).viewport === null, String(freed.fov));

  // The map view and the lock exclude each other in both directions.
  await v.key("L", { modifiers: ["shift"] });
  await sleep(200);
  await v.key("v");
  await sleep(200);
  t.check("V during the camera lock releases the lock for the map view",
    !(await rigOf()).on && (await v.evaluate("window.__ll2map.ortho()")).on);
  await v.key("v");

  // ---- the camera bar ---------------------------------------------------------------
  const camButton = (label) => v.evaluate(`(() => { const b = [...document.querySelectorAll('#camButtons [data-camera]')]`
    + `.find((e) => e.textContent === ${JSON.stringify(label)}); const r = b.getBoundingClientRect();`
    + " return [r.left + r.width / 2, r.top + r.height / 2]; })()");
  let at = await camButton("R");
  await v.click(at[0], at[1]);
  await sleep(200);
  rigNow = await rigOf();
  t.check("a ring button looks through that camera", rigNow.on && rigNow.camera === "camera_ring_rear", JSON.stringify(rigNow));
  t.check("the bar says it is locked", await v.evaluate("document.getElementById('camBar').classList.contains('locked')"));
  t.check("and the status bar says how to get out", (await uiState()).hints.includes("release"), (await uiState()).hints);
  await v.shot("scene_locked.png");
  at = await camButton("F");
  await v.click(at[0], at[1]);
  await sleep(150);
  t.check("another ring button switches camera, still locked",
    (await rigOf()).on && (await rigOf()).camera === "camera_ring_front");
  at = await camButton("F");
  await v.click(at[0], at[1]);
  await sleep(150);
  t.check("the lit button lets go again", !(await rigOf()).on);
  await clickElement("sceneLock");
  await v.key("Escape");
  await sleep(150);
  t.check("Esc releases the lock", !(await rigOf()).on);

  // ---- following the car ---------------------------------------------------------------
  await v.evaluate("window.__ll2scene.show(2)");
  await waitLoaded(2);
  await v.key("k");
  await sleep(100);
  await clickElement("sceneFollow");
  t.check("Follow turns on", (await sceneState()).follow
    && (await v.evaluate("document.getElementById('sceneFollow').classList.contains('on')")));
  let c2 = await v.camera();
  await v.evaluate("window.__ll2scene.show(6)");
  await waitLoaded(6);
  let c6 = await v.camera();
  const rel = (c, i) => [c.pos[0] - ego.pos[i][0], c.pos[1] - ego.pos[i][1], c.pos[2] - ego.pos[i][2]];
  t.check("the camera keeps its place relative to the car while the frames change",
    dist3(rel(c2, 2), rel(c6, 6)) < 1e-6 && Math.abs(c6.yaw - c2.yaw) < 1e-9,
    `${JSON.stringify(rel(c2, 2))} -> ${JSON.stringify(rel(c6, 6))}`);
  await v.key("K", { modifiers: ["shift"] });
  t.check("Shift+K turns it off", !(await sceneState()).follow);
  c6 = await v.camera();
  await v.evaluate("window.__ll2scene.show(8)");
  await waitLoaded(8);
  t.check("and then the camera stays put", dist3(c6.pos, (await v.camera()).pos) < 1e-9);

  // A car that turns: the camera turns with it, about the car.
  const turning = sceneInfo();
  turning.id = "e2e-scene-turning";
  const headingOf = (i) => (i * 10 * Math.PI) / 180;
  turning.ego = {
    pos: ego.pos,
    quat: ego.pos.map((_, i) => [0, 0, Math.sin(headingOf(i) / 2), Math.cos(headingOf(i) / 2)]),
  };
  v.sendScene({ type: "scene", scene: turning });
  await waitFor(async () => (await v.evaluate("document.getElementById('sceneName').textContent")) === turning.id);
  await v.evaluate("window.__ll2scene.show(2)");
  await waitLoaded(2);
  await v.key("k");
  await v.key("K", { modifiers: ["shift"] });
  c2 = await v.camera();
  await v.evaluate("window.__ll2scene.show(5)");
  await waitLoaded(5);
  const c5 = await v.camera();
  const turn = headingOf(5) - headingOf(2);
  const r2 = rel(c2, 2);
  const want = [r2[0] * Math.cos(turn) - r2[1] * Math.sin(turn), r2[0] * Math.sin(turn) + r2[1] * Math.cos(turn), r2[2]];
  t.check("a turning car takes the camera round with it",
    dist3(rel(c5, 5), want) < 1e-6 && Math.abs(Math.sin(c5.yaw - c2.yaw - turn)) < 1e-9,
    `${JSON.stringify(rel(c5, 5).map((n) => +n.toFixed(3)))} vs ${JSON.stringify(want.map((n) => +n.toFixed(3)))}`);
  // While the lock drives the view, Follow waits; afterwards it carries on
  // from wherever the lock left the camera, without a jump.
  await v.key("L", { modifiers: ["shift"] });
  await v.evaluate("window.__ll2scene.show(7)");
  await waitLoaded(7);
  await v.key("L", { modifiers: ["shift"] });
  await sleep(150);
  const c7 = await v.camera();
  await v.evaluate("window.__ll2scene.show(8)");
  await waitLoaded(8);
  const c8 = await v.camera();
  t.check("after the lock, Follow moves on one frame's worth", dist3(c7.pos, c8.pos) < 15
    && dist3(c7.pos, c8.pos) > 1, `${dist3(c7.pos, c8.pos).toFixed(2)} m`);
  await v.key("K", { modifiers: ["shift"] });

  t.check("no page errors", v.errors.length === 0, v.errors.join(" | "));
} finally {
  await v.close();
  data.close();
}
console.log(t.failures ? `\n${t.failures} of ${t.count} FAILED` : `\nall ${t.count} passed`);
process.exit(t.failures ? 1 : 0);
