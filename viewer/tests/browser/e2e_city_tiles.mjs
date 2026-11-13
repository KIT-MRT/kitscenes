// End-to-end checks of city mode with merged tiles: a server that merges the
// static clouds of all scenes reaching a tile (`/city/tiles.json`,
// `/city/tile/<key>.bin`, kitscenes_viewer/scene/tiles.py) instead of
// sending each scene's own chunks.
//
// The server is faked. Two drives side by side share tile "0_0" (one is not
// built at first), one scene has a tile of its own, one is 4 km away. A
// merged tile has *fewer* points than its scenes together: what the browser
// draws must be the tile, fetched again (not added again) when one more of
// its scenes is built.
//   node viewer/tests/browser/e2e_city_tiles.mjs [--shots DIR]
import http from "node:http";
import { openViewer, Checks, sleep } from "./harness.mjs";

const shotsIdx = process.argv.indexOf("--shots");
const shotsDir = shotsIdx > 0 ? process.argv[shotsIdx + 1] : null;
const t = new Checks();
const Z = 100;
const FRAMES = 30;
const ONE = 300;           // points of a tile with one scene
const MORE = 20;           // each further scene adds this few: the voxels are shared

const SCENES = [
  { id: "east-1", built: true, at: (i) => [2 * i, 0, Z + 2], tile: "0_0" },
  { id: "overlap-4", built: false, at: (i) => [2 * i, 6, Z + 2], tile: "0_0" },
  { id: "north-2", built: true, at: (i) => [100, -30 + 2 * i, Z + 2], tile: "0_1" },
  { id: "far-3", built: true, at: (i) => [3000 + 2 * i, 3000, Z + 2], tile: "23_23" },
];
const byId = Object.fromEntries(SCENES.map((s) => [s.id, s]));
const TILES = { "0_0": [30, 3, Z], "0_1": [100, 0, Z], "23_23": [3030, 3000, Z] };

function encodeCloud(origin, count) {
  const header = new ArrayBuffer(40);
  const view = new DataView(header);
  for (const [k, ch] of [..."LL2PC1"].entries()) view.setUint8(k, ch.charCodeAt(0));
  view.setUint32(8, count, true);
  view.setUint32(12, 1, true); // intensity
  view.setFloat64(16, origin[0], true);
  view.setFloat64(24, origin[1], true);
  view.setFloat64(32, origin[2], true);
  const xyz = new Float32Array(count * 3);
  for (let p = 0; p < count; p++) {
    const a = (p / count) * Math.PI * 2;
    xyz[p * 3] = Math.cos(a) * 8;
    xyz[p * 3 + 1] = Math.sin(a) * 8;
    xyz[p * 3 + 2] = -2 + (p % 10) * 0.2;
  }
  return Buffer.concat([Buffer.from(header), Buffer.from(xyz.buffer), Buffer.alloc(count, 100)]);
}

const builtIn = (key) => SCENES.filter((s) => s.tile === key && s.built).map((s) => s.id).sort();

let base = "";
const metaAsked = {};
const tilesServed = [];     // "<key>@<version>"
const sceneChunksServed = [];

const data = http.createServer((req, res) => {
  const cors = { "Access-Control-Allow-Origin": "*", "Cache-Control": "no-store" };
  const json = (obj, code = 200) => {
    res.writeHead(code, { ...cors, "Content-Type": "application/json" });
    res.end(JSON.stringify(obj));
  };
  const url = new URL(req.url, base);
  if (url.pathname === "/city.json") {
    json({
      city: "testcity", anchor: { lat: 49.0, lon: 8.4 }, base,
      classes: { palette: Array.from({ length: 256 }, () => [90, 90, 90]), names: {}, unlabelled: 255 },
      scenes: SCENES.map((s) => {
        const pos = Array.from({ length: FRAMES }, (_, i) => s.at(i)).flat();
        const xs = pos.filter((_, k) => k % 3 === 0);
        const ys = pos.filter((_, k) => k % 3 === 1);
        return { id: s.id, split: "train", frames: FRAMES, base: `${base}/scene/${s.id}`, pos,
                 quat: Array.from({ length: FRAMES }, () => [0, 0, 0, 1]).flat(),
                 bbox: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
                 aggregate: s.built };
      }),
    });
    return;
  }
  if (url.pathname === "/city/status.json") {
    json(Object.fromEntries(SCENES.map((s) => [s.id, { aggregate: s.built, building: false, open: false }])));
    return;
  }
  if (url.pathname === "/city/tiles.json") {
    const x = Number(url.searchParams.get("x"));
    const y = Number(url.searchParams.get("y"));
    const r = Number(url.searchParams.get("r"));
    const tiles = [];
    for (const [key, centre] of Object.entries(TILES)) {
      const ids = builtIn(key);
      if (!ids.length || Math.hypot(centre[0] - x, centre[1] - y) > r + 90) continue;
      tiles.push({ key, centre, radius: 90, version: ids.join(","), ids });
    }
    json({ tile_m: 128, tiles });
    return;
  }
  let mm = /^\/city\/tile\/(-?\d+_-?\d+)\.bin$/.exec(url.pathname);
  if (mm) {
    const ids = builtIn(mm[1]);
    tilesServed.push(`${mm[1]}@${ids.join(",")}`);
    res.writeHead(200, { ...cors, "Content-Type": "application/octet-stream" });
    res.end(encodeCloud(TILES[mm[1]], ONE + MORE * (ids.length - 1)));
    return;
  }
  const m = /^\/scene\/([^/]+)(\/.*)$/.exec(url.pathname);
  const s = m && byId[m[1]];
  if (s && m[2] === "/aggregate/meta.json") {
    metaAsked[s.id] = (metaAsked[s.id] || 0) + 1;
    if (!s.built) {  // "builds" for two polls once someone asks
      if (metaAsked[s.id] <= 2) {
        json({ ready: false, progress: { state: "scanning", done: metaAsked[s.id], total: 3 } }, 202);
        return;
      }
      s.built = true;
    }
    json({ ready: true, stats: { points: ONE }, chunks: [{ index: 0, count: ONE, centre: s.at(0), radius: 30 }] });
    return;
  }
  if (s && /^\/aggregate\/\d+\.bin$/.test(m[2])) sceneChunksServed.push(s.id);
  json({ error: "not found" }, 404);
});
await new Promise((r) => data.listen(0, "127.0.0.1", r));
base = `http://127.0.0.1:${data.address().port}`;

const v = await openViewer({ shotsDir });
const city = () => v.evaluate("window.__ll2city.state()");
async function waitFor(pred, ms = 8000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    if (await pred()) return true;
    await sleep(50);
  }
  return false;
}

try {
  v.sendScene({ type: "snapshot", anchor: { lat: 49.0, lon: 8.4 }, features: [] });
  v.sendScene({ type: "city", city: { name: "testcity", base, scenes: SCENES.length,
                                      anchor: { lat: 49.0, lon: 8.4 } } });
  t.check("the city arrives", await waitFor(async () => (await city()).tracks.scenes === 4));
  // Look at the near scenes (the overview would fit the one 4 km away as well).
  await v.evaluate("window.__ll2city.lookAt([50, 0, 100], 150)");
  await sleep(300);

  await v.evaluate("window.__ll2city.setAuto(true)");
  t.check("Auto loads the merged tiles around the camera", await waitFor(async () => {
    const s = await city();
    return s.loaded.length === 2 && !s.busy && s.queue === 0;
  }), JSON.stringify((await city()).loaded));
  let c = await city();
  t.check("tiles, not each scene's chunks", JSON.stringify(c.loaded) === JSON.stringify(["tile:0_0", "tile:0_1"])
    && sceneChunksServed.length === 0, JSON.stringify({ loaded: c.loaded, sceneChunksServed }));
  t.check("one point set per tile", c.points.points === 2 * ONE, JSON.stringify(c.points));
  t.check("the status line says merged, from the built scenes", (await v.hud("cityStatus")).includes("from 2 scenes, merged"),
    await v.hud("cityStatus"));
  t.check("the unbuilt one is named, not built", (await v.hud("cityStatus")).includes("1 not built")
    && !metaAsked["overlap-4"], await v.hud("cityStatus"));
  t.check("nothing from 4 km away", !tilesServed.some((k) => k.startsWith("23_23")));
  await v.evaluate("window.__ll2city.setAuto(false)");
  await v.shot("city_tiles.png");

  await v.evaluate("window.__ll2city.loadHere(true)");
  t.check("Load builds the missing scene", await waitFor(async () => (await city()).load["overlap-4"] === "ready", 15000),
    JSON.stringify((await city()).load));
  t.check("and its tile is fetched again with both scenes", await waitFor(async () =>
    tilesServed.includes("0_0@east-1,overlap-4") && !(await city()).busy, 15000), JSON.stringify(tilesServed));
  c = await city();
  t.check("replacing the old tile, not adding to it",
    c.points.chunks === 2 && c.points.points === 2 * ONE + MORE, JSON.stringify(c.points));
  t.check("the status line counts three scenes", (await v.hud("cityStatus")).includes("from 3 scenes, merged"),
    await v.hud("cityStatus"));
  t.check("the unchanged tile is not fetched again", tilesServed.filter((k) => k.startsWith("0_1@")).length === 1,
    JSON.stringify(tilesServed));

  await v.evaluate("window.__ll2city.unload()");
  t.check("Unload drops every tile", (await city()).points.chunks === 0);
  await v.evaluate("window.__ll2city.loadHere(true)");
  t.check("and Load brings them back", await waitFor(async () => (await city()).points.chunks === 2));

  await v.evaluate("window.__ll2city.lookAt([3030, 3000, 100], 120)");
  await sleep(200);
  await v.evaluate("window.__ll2city.evict()");
  t.check("moving 4 km away drops the tiles behind", (await city()).points.chunks === 0,
    JSON.stringify((await city()).points));
  t.check("no page errors", v.errors.length === 0, v.errors.join(" | "));
} catch (err) {
  t.check(`no exception (${err && err.stack ? err.stack : err})`, false);
} finally {
  await v.close();
  data.close();
}
console.log(t.failures ? `\n${t.failures} of ${t.count} FAILED` : `\nall ${t.count} passed`);
process.exit(t.failures ? 1 : 0);
