// End-to-end checks of city mode: every scene of a city as trajectories,
// picking one by clicking its pose arrows, and the static clouds of the
// scenes around the camera coming and going.
//
// The scene server's city mode is faked: `/city.json`, `/city/status.json`
// and `/scene/<id>/...` with the same shapes as kitscenes_viewer.scene (the
// Python side has its own tests). Four scenes: two side by side (their clouds
// overlap), one turning north, one 4 km away, and one of the near ones has no
// cloud built yet.
//   node viewer/tests/browser/e2e_city.mjs [--shots DIR]
import http from "node:http";
import { openViewer, Checks, dist3, sleep } from "./harness.mjs";

const shotsIdx = process.argv.indexOf("--shots");
const shotsDir = shotsIdx > 0 ? process.argv[shotsIdx + 1] : null;
const t = new Checks();
const Z = 100;
const FRAMES = 30;
const AGG = 300;           // points per aggregate chunk
const FRAME_POINTS = 200;

const yawQuat = (deg) => {
  const h = (deg * Math.PI) / 360;
  return [0, 0, Math.sin(h), Math.cos(h)];
};
const SCENES = [
  { id: "east-1", split: "train", built: true, at: (i) => [2 * i, 0, Z + 2], quat: yawQuat(0) },
  { id: "north-2", split: "val", built: true, at: (i) => [100, -30 + 2 * i, Z + 2], quat: yawQuat(90) },
  { id: "far-3", split: "test", built: true, at: (i) => [3000 + 2 * i, 3000, Z + 2], quat: yawQuat(0) },
  { id: "overlap-4", split: "val", built: false, at: (i) => [2 * i, 6, Z + 2], quat: yawQuat(0) },
];
const byId = Object.fromEntries(SCENES.map((s) => [s.id, s]));
const CLASS_CAR = 58;

function encodeCloud(origin, count) {
  const header = new ArrayBuffer(40);
  const view = new DataView(header);
  for (const [k, ch] of [..."LL2PC1"].entries()) view.setUint8(k, ch.charCodeAt(0));
  view.setUint32(8, count, true);
  view.setUint32(12, 1 | 4, true); // intensity + class
  view.setFloat64(16, origin[0], true);
  view.setFloat64(24, origin[1], true);
  view.setFloat64(32, origin[2], true);
  const xyz = new Float32Array(count * 3);
  const intensity = new Uint8Array(count);
  const classes = new Uint8Array(count);
  for (let p = 0; p < count; p++) {
    const a = (p / count) * Math.PI * 2;
    xyz[p * 3] = Math.cos(a) * 8;
    xyz[p * 3 + 1] = Math.sin(a) * 8;
    xyz[p * 3 + 2] = -2 + (p % 10) * 0.2;
    intensity[p] = p % 256;
    classes[p] = p % 4 === 0 ? CLASS_CAR : 255;
  }
  return Buffer.concat([Buffer.from(header), Buffer.from(xyz.buffer),
                        Buffer.from(intensity), Buffer.from(classes)]);
}

const chunksOf = (s) => [
  { index: 0, count: AGG, centre: s.at(0), radius: 30 },
  { index: 1, count: AGG, centre: s.at(FRAMES - 1), radius: 30 },
];

let base = "";
const metaAsked = {};       // scene id -> times its aggregate meta was asked for
const chunksServed = [];    // "<id>/<n>"
const framesServed = [];    // "<id>/<i>"
const sceneJsonAsked = [];

const sceneInfo = (s) => ({
  id: s.id, base: `${base}/scene/${s.id}`, frames: FRAMES,
  times: Array.from({ length: FRAMES }, (_, i) => i * 0.1),
  lidars: ["lidar_top"],
  ego: { pos: Array.from({ length: FRAMES }, (_, i) => s.at(i)),
         quat: Array.from({ length: FRAMES }, () => s.quat) },
  cameras: {
    camera_ring_front: {
      fx: 1841, fy: 1841, cx: 1700, cy: 1100, width: 3504, height: 2272,
      T_ego_cam: [[0, 0, 1, 0.2], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]],
    },
  },
  anchor: { lat: 49.0, lon: 8.4 },
});

const data = http.createServer((req, res) => {
  const cors = { "Access-Control-Allow-Origin": "*", "Cache-Control": "no-store" };
  const json = (obj, code = 200) => {
    res.writeHead(code, { ...cors, "Content-Type": "application/json" });
    res.end(JSON.stringify(obj));
  };
  if (req.url === "/city.json") {
    json({
      city: "testcity",
      anchor: { lat: 49.0, lon: 8.4 },
      base,
      classes: { palette: Array.from({ length: 256 }, (_, i) => (i === CLASS_CAR ? [0, 0, 142] : [90, 90, 90])),
                 names: { 58: "Car" }, unlabelled: 255 },
      scenes: SCENES.map((s) => {
        const pos = Array.from({ length: FRAMES }, (_, i) => s.at(i)).flat();
        const xs = pos.filter((_, k) => k % 3 === 0);
        const ys = pos.filter((_, k) => k % 3 === 1);
        return { id: s.id, split: s.split, frames: FRAMES, base: `${base}/scene/${s.id}`, pos,
                 quat: Array.from({ length: FRAMES }, () => s.quat).flat(),
                 bbox: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
                 aggregate: s.built };
      }),
    });
    return;
  }
  if (req.url === "/city/status.json") {
    json(Object.fromEntries(SCENES.map((s) => [s.id, { aggregate: s.built, building: false, open: false }])));
    return;
  }
  const m = /^\/scene\/([^/]+)(\/.*)$/.exec(req.url);
  const s = m && byId[m[1]];
  if (!s) {
    json({ error: "not found" }, 404);
    return;
  }
  const rest = m[2];
  if (rest === "/scene.json") {
    sceneJsonAsked.push(s.id);
    json(sceneInfo(s));
    return;
  }
  if (rest === "/aggregate/meta.json") {
    metaAsked[s.id] = (metaAsked[s.id] || 0) + 1;
    // An unbuilt scene "builds" for two polls once someone asks for it.
    if (!s.built) {
      if (metaAsked[s.id] <= 2) {
        json({ ready: false, progress: { state: "scanning", done: metaAsked[s.id], total: 3 } }, 202);
        return;
      }
      s.built = true;
    }
    json({ ready: true, stats: { points: 2 * AGG }, chunks: chunksOf(s) });
    return;
  }
  let mm = /^\/aggregate\/(\d+)\.bin$/.exec(rest);
  if (mm) {
    const chunk = chunksOf(s)[Number(mm[1])];
    chunksServed.push(`${s.id}/${mm[1]}`);
    res.writeHead(200, { ...cors, "Content-Type": "application/octet-stream" });
    res.end(encodeCloud(chunk.centre, AGG));
    return;
  }
  mm = /^\/frame\/(\d+)\/points\.bin$/.exec(rest);
  if (mm) {
    framesServed.push(`${s.id}/${mm[1]}`);
    res.writeHead(200, { ...cors, "Content-Type": "application/octet-stream" });
    res.end(encodeCloud(s.at(Number(mm[1])), FRAME_POINTS));
    return;
  }
  json({ error: "not found" }, 404);
});
await new Promise((r) => data.listen(0, "127.0.0.1", r));
base = `http://127.0.0.1:${data.address().port}`;

const line = (id, nodes, pts) => ({ id: `way/${id}`, kind: "line", tags: { type: "line_thin" },
                                    pts: pts.flat(), nodes });
const features = [line(10, [1001, 1002], [[-20, -40, Z], [140, 40, Z]])];

const v = await openViewer({ shotsDir });
const city = () => v.evaluate("window.__ll2city.state()");
const scene = () => v.evaluate("window.__ll2scene.state()");
async function waitFor(pred, ms = 6000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    if (await pred()) return true;
    await sleep(50);
  }
  return false;
}
async function clickElement(id) {
  const box = await v.evaluate(`(() => { const b = document.getElementById(${JSON.stringify(id)})`
    + `.getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2]; })()`);
  await v.click(box[0], box[1]);
  await sleep(120);
}
const shown = (id) => v.evaluate(`getComputedStyle(document.getElementById(${JSON.stringify(id)})).display !== "none"`
  + ` && !document.getElementById(${JSON.stringify(id)}).closest("[hidden]")`);
/** Page pixel of frame `frame` of scene `id`. */
async function nodePx(id, frame) {
  const p = await v.evaluate(`window.__ll2city.nodePosition(${JSON.stringify(id)}, ${frame})`);
  return v.project(p);
}

try {
  v.sendScene({ type: "snapshot", anchor: { lat: 49.0, lon: 8.4 }, features });
  t.check("map streamed", await v.waitForFeatures(1));
  t.check("no city bar before a city", await v.evaluate("document.getElementById('cityBar').hidden"));

  v.sendScene({ type: "city", city: { name: "testcity", base, scenes: SCENES.length,
                                      anchor: { lat: 49.0, lon: 8.4 } } });
  t.check("the city announcement shows the city bar",
    await waitFor(async () => !(await v.evaluate("document.getElementById('cityBar').hidden"))));
  t.check("and loads every trajectory", await waitFor(async () => (await city()).tracks.scenes === 4));
  let c = await city();
  t.check("one arrow per frame, one segment between frames",
    c.tracks.nodes === 4 * FRAMES && c.tracks.segments === 4 * (FRAMES - 1), JSON.stringify(c.tracks));
  t.check("named after the city", (await v.hud("cityName")).startsWith("testcity · 4 scenes"),
    await v.hud("cityName"));
  t.check("no scene is open yet", c.selected === null && sceneJsonAsked.length === 0);
  t.check("nothing is loaded on arrival", Object.keys(metaAsked).length === 0 && chunksServed.length === 0);
  // Colour, size and filters act on the city's clouds too; the frame tools
  // need a scene and step aside.
  t.check("the point controls are there without a scene",
    (await shown("sceneMode")) && (await shown("sceneFilterBtn")));
  t.check("the per-scene controls are not",
    !(await shown("scenePoints")) && !(await shown("sceneLock")) && !(await shown("sceneFrame")));

  await v.key("b");
  await sleep(400);
  await v.shot("city_tracks.png");

  // ---- picking ---------------------------------------------------------------------
  let px = await nodePx("east-1", 10);
  const pick = await v.evaluate(`window.__ll2city.pick(${px[0]}, ${px[1]})`);
  t.check("an arrow picks its scene and frame", pick && pick.id === "east-1" && pick.frame === 10,
    JSON.stringify(pick));
  await v.hover(px[0], px[1]);
  await sleep(80);
  c = await city();
  t.check("hovering names the scene and frame",
    (c.hover || "").startsWith("east-1 · train · frame 10 / 29"), c.hover);
  t.check("and widens its track", c.tracks.hovered === "east-1");
  t.check("with a hand cursor", (await v.evaluate("document.querySelector('canvas').style.cursor")) === "pointer");

  await v.click(px[0], px[1]);
  t.check("clicking an arrow opens that scene", await waitFor(async () => (await scene()).base === `${base}/scene/east-1`));
  const selAfterTrack = await v.evaluate("window.__ll2test.selection()");
  t.check("and does not also select the map under it", selAfterTrack.ways.length === 0 && selAfterTrack.nodes.length === 0,
    JSON.stringify(selAfterTrack));
  t.check("at the frame that was clicked", await waitFor(async () => {
    const s = await scene();
    return s.frame === 10 && !s.loading;
  }), JSON.stringify(await scene()));
  t.check("the hover label goes once the scene is open", (await city()).hover === null);
  let pts = await v.evaluate("window.__ll2scene.points()");
  t.check("and its point cloud is that frame's", pts.origin && dist3(pts.origin, byId["east-1"].at(10)) < 1e-6,
    JSON.stringify(pts.origin));
  c = await city();
  t.check("the picked track is highlighted", c.tracks.selected === "east-1");
  t.check("the frame tools are back", (await shown("scenePoints")) && (await shown("sceneLock")));
  t.check("and Deselect is offered", !(await v.evaluate("document.getElementById('cityDeselect').disabled")));
  await v.shot("city_picked.png");
  // Behind the car, the poses read as arrows on the dots.
  await v.key("k");
  await sleep(300);
  await v.shot("city_arrows.png");
  await v.key("b");
  await sleep(300);

  px = await nodePx("north-2", 5);
  await v.click(px[0], px[1]);
  t.check("clicking another scene switches to it",
    await waitFor(async () => (await scene()).base === `${base}/scene/north-2` && (await scene()).frame === 5));
  t.check("and its frame replaces the old one", await waitFor(async () => {
    const p = await v.evaluate("window.__ll2scene.points()");
    return p.origin && dist3(p.origin, byId["north-2"].at(5)) < 1e-6;
  }));
  t.check("the highlight follows", (await city()).tracks.selected === "north-2");

  // Between two arrows, the line still picks -- the nearer frame of the two.
  const a = await nodePx("east-1", 20);
  const b = await nodePx("east-1", 21);
  const between = await v.evaluate(`window.__ll2city.pick(${(a[0] * 3 + b[0]) / 4}, ${(a[1] * 3 + b[1]) / 4})`);
  t.check("a click between two arrows picks the nearer frame",
    between && between.id === "east-1" && between.frame === 20, JSON.stringify(between));

  await clickElement("cityDeselect");
  t.check("Deselect closes the scene", await waitFor(async () => (await scene()).base === null));
  t.check("and the city stays", (await city()).tracks.scenes === 4 && (await city()).tracks.selected === null);
  t.check("its frame cloud is gone", (await v.evaluate("window.__ll2scene.points()")).count === 0);
  t.check("the per-scene controls step aside again", !(await shown("scenePoints")));

  // ---- the scene list and the breadcrumb ----------------------------------------------
  const crumbShown = (part) => v.evaluate(`!document.querySelector('#crumbs [data-crumb="${part}"]').hidden`
    + " && !document.getElementById('crumbs').hidden");
  t.check("the breadcrumb names the city alone once the scene is closed",
    (await crumbShown("city")) && !(await crumbShown("scene"))
    && (await v.hud("crumbCity")) === "testcity", await v.hud("crumbCity"));
  const listed = () => v.evaluate("[...document.querySelectorAll('#sceneList [data-scene]')].map((e) => e.dataset.scene)");
  let ids = await listed();
  t.check("the scene list has every scene of the city", ids.length === 4, JSON.stringify(ids));
  t.check("nearest the camera first, the far one last", ids[3] === "far-3", JSON.stringify(ids));
  await v.evaluate("(() => { const e = document.getElementById('sceneSearch'); e.value = 'val';"
    + " e.dispatchEvent(new Event('input')); })()");
  ids = await listed();
  t.check("the filter narrows it by split or id", ids.length === 2 && ids.includes("north-2") && ids.includes("overlap-4")
    && (await v.hud("sceneListCount")) === "(2 of 4)", `${JSON.stringify(ids)} ${await v.hud("sceneListCount")}`);
  const item = await v.evaluate("(() => { const r = document.querySelector('#sceneList [data-scene=\"north-2\"]').getBoundingClientRect();"
    + " return [r.left + r.width / 2, r.top + r.height / 2]; })()");
  await v.click(item[0], item[1]);
  t.check("a click in the list opens that scene", await waitFor(async () => (await scene()).base === `${base}/scene/north-2`));
  await waitFor(async () => !(await scene()).loading);
  const camList = await v.camera();
  const car0 = byId["north-2"].at(0);
  t.check("and goes to its car, wherever the camera was",
    Math.hypot(camList.pos[0] - car0[0], camList.pos[1] - car0[1]) < 40, JSON.stringify(camList.pos));
  t.check("the breadcrumb now names the scene too", (await crumbShown("scene"))
    && (await v.hud("crumbSceneName")) === "north-2", await v.hud("crumbSceneName"));
  t.check("and the list marks it as open",
    await v.evaluate("document.querySelector('#sceneList [data-scene=\"north-2\"]').classList.contains('current')"));
  await v.evaluate("(() => { const e = document.getElementById('sceneSearch'); e.value = '';"
    + " e.dispatchEvent(new Event('input')); })()");
  await clickElement("cityDeselect");
  t.check("the city crumb closes the scene again", await waitFor(async () => (await scene()).base === null));
  await v.key("b");  // back to the overview the checks below expect
  await sleep(300);

  // In edit mode a click belongs to the map.
  await v.key("e");
  px = await nodePx("east-1", 3);
  const asked = sceneJsonAsked.length;
  await v.click(px[0], px[1]);
  await sleep(300);
  t.check("in edit mode a click does not open a scene",
    sceneJsonAsked.length === asked && (await scene()).base === null);
  await v.key("e");

  // ---- clouds around the camera ----------------------------------------------------
  await clickElement("cityAuto");
  t.check("Auto loads the built clouds around the camera", await waitFor(async () => {
    const s = await city();
    return s.loaded.length === 4 && !s.busy && s.queue === 0;
  }), JSON.stringify((await city()).loaded));
  c = await city();
  t.check("from both near scenes",
    JSON.stringify(c.loaded) === JSON.stringify(["east-1/0", "east-1/1", "north-2/0", "north-2/1"]));
  t.check("but never starts a build", !metaAsked["overlap-4"] && c.load["overlap-4"] === "unbuilt",
    JSON.stringify({ asked: metaAsked, load: c.load }));
  t.check("and says what it left out", (await v.hud("cityStatus")).includes("1 not built"),
    await v.hud("cityStatus"));
  t.check("nothing from the scene 4 km away", !metaAsked["far-3"] && !chunksServed.some((k) => k.startsWith("far-3")));
  await clickElement("cityAuto");

  await clickElement("cityLoad");
  t.check("Load builds the missing one and loads it too", await waitFor(async () => {
    const s = await city();
    return s.loaded.length === 6 && !s.busy;
  }, 15000), JSON.stringify(await city()));
  t.check("after asking until it was built", metaAsked["overlap-4"] >= 3, JSON.stringify(metaAsked));
  c = await city();
  t.check("overlapping clouds simply add up",
    c.points.scenes === 3 && c.points.points === 6 * AGG, JSON.stringify(c.points));
  t.check("the status line counts them", (await v.hud("cityStatus")).includes("from 3 scenes"),
    await v.hud("cityStatus"));
  t.check("classes in the city clouds are offered without a scene",
    await v.evaluate("!!document.querySelector('#sceneMode option[value=\"class\"]')"));
  await v.shot("city_loaded.png");

  // The picked scene's own whole-scene cloud replaces its city chunks.
  px = await nodePx("east-1", 4);
  await v.click(px[0], px[1]);
  await waitFor(async () => (await scene()).base === `${base}/scene/east-1`);
  await v.key("P", { modifiers: ["shift"] });
  t.check("its own whole scene loads", await waitFor(async () =>
    (await v.evaluate("window.__ll2scene.aggregate()")).loaded === 2));
  c = await city();
  t.check("and its city chunks step aside, the others stay",
    c.points.shown === 4 * AGG && c.points.points === 6 * AGG, JSON.stringify(c.points));
  await v.key("P", { modifiers: ["shift"] });
  await sleep(150);
  t.check("they come back when it is turned off", (await city()).points.shown === 6 * AGG);

  await clickElement("cityUnload");
  t.check("Unload drops every city cloud", (await city()).points.chunks === 0);
  await clickElement("cityLoad");
  t.check("and Load brings them back from the server's cache",
    await waitFor(async () => (await city()).loaded.length === 6 && !(await city()).busy));

  // Drive to the far scene: everything near the old spot is dropped again.
  await v.evaluate("window.__ll2city.open('far-3', 0)");
  await waitFor(async () => (await scene()).base === `${base}/scene/far-3`);
  await v.key("k");
  t.check("moving 4 km away drops the clouds behind", await waitFor(async () =>
    (await city()).points.chunks === 0, 4000), JSON.stringify((await city()).points));
  t.check("without fetching anything there on its own", !chunksServed.some((k) => k.startsWith("far-3")));

  await v.evaluate("document.getElementById('cityRadius').value = '600';"
    + " document.getElementById('cityRadius').dispatchEvent(new Event('input'))");
  t.check("the radius slider sets the radius", (await city()).radius === 600
    && (await v.hud("cityRadiusLabel")) === "600 m" && (await v.hud("cityLoad")) === "Load 600 m");

  // A reloaded tab finds the city again: the hub keeps the announcement.
  await v.cdp("Page.reload", {});
  await waitFor(async () => v.evaluate("typeof window.__ll2city === 'object'").catch(() => false), 20000);
  t.check("a reloaded tab gets the city back from the hub",
    await waitFor(async () => (await city()).tracks.scenes === 4, 10000));
  t.check("no page errors", v.errors.length === 0, v.errors.join(" | "));
} catch (err) {
  t.check(`no exception (${err && err.stack ? err.stack : err})`, false);
} finally {
  await v.close();
  data.close();
}
console.log(t.failures ? `\n${t.failures} of ${t.count} FAILED` : `\nall ${t.count} passed`);
process.exit(t.failures ? 1 : 0);
