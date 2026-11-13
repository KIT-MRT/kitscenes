// End-to-end checks of the map view: the orthographic camera and the web-map
// tiles under it (XYZ, TMS, WMS, and the viewer server's tile proxy for
// servers without CORS).
//
// Two local tile servers stand in for OpenStreetMap / a GeoServer: one sends
// CORS headers, one does not (WebGL cannot use its images directly).
//   node viewer/tests/browser/e2e_mapview.mjs [--shots DIR]
import http from "node:http";
import { openViewer, Checks, sleep } from "./harness.mjs";

const shotsIdx = process.argv.indexOf("--shots");
const shotsDir = shotsIdx > 0 ? process.argv[shotsIdx + 1] : null;
const t = new Checks();
const Z = 100;
const ANCHOR = { lat: 49.0, lon: 8.4 };
// A valid 1x1 PNG; WebGL is happy to stretch it over a tile.
const PNG = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==", "base64");

const seen = { cors: [], wms: [], nocors: [] };
function tileServer(withCors, log) {
  const srv = http.createServer((req, res) => {
    log.push(req.url);
    const headers = { "Content-Type": "image/png", "Content-Length": PNG.length };
    if (withCors) headers["Access-Control-Allow-Origin"] = "*";
    res.writeHead(200, headers);
    res.end(PNG);
  });
  return new Promise((r) => srv.listen(0, "127.0.0.1", () => r(srv)));
}
const corsSrv = await tileServer(true, seen.cors);
const wmsLog = [];
const corsBase = `http://127.0.0.1:${corsSrv.address().port}`;
const nocorsSrv = await tileServer(false, seen.nocors);
const nocorsHost = `127.0.0.1:${nocorsSrv.address().port}`;

// Independent tile maths, so the viewer's is checked rather than reused.
const R = 6378137.0;
const MERC = Math.PI * R;
const rad = (d) => (d * Math.PI) / 180;
const deg = (r) => (r * 180) / Math.PI;
const enu = (lat, lon) => [rad(lon - ANCHOR.lon) * Math.cos(rad(ANCHOR.lat)) * R, rad(lat - ANCHOR.lat) * R];
const tileOf = (lat, lon, z) => {
  const n = 2 ** z;
  const x = Math.floor(((lon + 180) / 360) * n);
  const y = Math.floor(((1 - Math.log(Math.tan(rad(lat)) + 1 / Math.cos(rad(lat))) / Math.PI) / 2) * n);
  return [x, y];
};
const swCornerEnu = (z, x, y) => {
  const s = (2 * MERC) / 2 ** z;
  const lon = deg((-MERC + x * s) / R);
  const lat = deg(2 * Math.atan(Math.exp((MERC - (y + 1) * s) / R)) - Math.PI / 2);
  return enu(lat, lon);
};

const line = (id, nodes, pts) => ({ id: `way/${id}`, kind: "line", tags: { type: "line_thin" },
                                    pts: pts.flat(), nodes });
const features = [
  line(10, [1001, 1002], [[-100, -60, Z], [100, -60, Z]]),
  line(11, [1003, 1004], [[-100, 60, Z], [100, 60, Z]]),
];

const v = await openViewer({ shotsDir, serverArgs: ["--tile-proxy-host", nocorsHost] });
const cam = () => v.camera();
const map = (expr) => v.evaluate(`window.__ll2map.${expr}`);
async function waitFor(pred, ms = 8000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    if (await pred()) return true;
    await sleep(60);
  }
  return false;
}
async function clickElement(id) {
  const box = await v.evaluate(`(() => { const b = document.getElementById(${JSON.stringify(id)})`
    + `.getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2]; })()`);
  await v.click(box[0], box[1]);
  await sleep(150);
}
const px = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const frame = () => v.evaluate("new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))");

try {
  v.sendScene({ type: "snapshot", anchor: ANCHOR, features });
  t.check("map streamed", await v.waitForFeatures(2));
  await v.key("b");
  await frame();

  // ---- orthographic --------------------------------------------------------------
  const ground = [[0, -60, Z], [0, 60, Z]];
  let a = await v.project(ground[0]);
  let b = await v.project(ground[1]);
  const spanPersp = px(a, b);
  const lowP = await v.project([80, -60, Z]);
  const highP = await v.project([80, -60, Z + 30]);
  t.check("perspective: a point 30 m higher lands elsewhere", px(lowP, highP) > 5, px(lowP, highP).toFixed(1));

  await v.key("v");
  await frame();
  let c = await cam();
  t.check("V turns the ortho view on", (await map("ortho()")).on
    && await v.evaluate("document.getElementById('orthoBtn').classList.contains('on')"));
  t.check("looking straight down, north up", Math.abs(c.pitch + Math.PI / 2) < 1e-9 && Math.abs(c.yaw) < 1e-9,
    JSON.stringify([c.yaw, c.pitch]));
  const lowO = await v.project([80, -60, Z]);
  const highO = await v.project([80, -60, Z + 30]);
  t.check("ortho: height no longer moves a point on screen", px(lowO, highO) < 0.01, px(lowO, highO).toFixed(4));
  a = await v.project(ground[0]);
  b = await v.project(ground[1]);
  t.check("the ground keeps its scale across the switch", Math.abs(px(a, b) / spanPersp - 1) < 0.01,
    `${spanPersp.toFixed(1)} -> ${px(a, b).toFixed(1)} px`);
  t.check("the HUD gives the scale", (await v.hud("camAng")).includes("cm/px"), await v.hud("camAng"));

  // Picking is done on the CPU with its own projection maths.
  const mid = await v.project([30, 60, Z]);
  const hit = await v.evaluate(`window.__ll2test.pick(${mid[0]}, ${mid[1]})`);
  t.check("a click picks the line under it in ortho", hit && hit.id === "way/11", JSON.stringify(hit));
  const node = await v.project([100, -60, Z]);
  // View mode picks ways; nodes are Edit mode's (or the setting's).
  await v.key("e");
  const hitNode = await v.evaluate(`window.__ll2test.pick(${node[0]}, ${node[1]})`);
  await v.key("e");
  t.check("and a node in Edit mode", hitNode && hitNode.type === "node", JSON.stringify(hitNode));

  // The wheel zooms about the point under the cursor.
  const target = [60, 60, Z];
  const before = await v.project(target);
  await v.wheel(before[0], before[1], -100, 3);
  await frame();
  const after = await v.project(target);
  a = await v.project(ground[0]);
  b = await v.project(ground[1]);
  t.check("the wheel zooms in", px(a, b) > spanPersp * 1.3, `${spanPersp.toFixed(1)} -> ${px(a, b).toFixed(1)} px`);
  t.check("keeping the point under the cursor in place", px(before, after) < 1.5,
    `${before.map((n) => n.toFixed(1))} -> ${after.map((n) => n.toFixed(1))}`);
  t.check("and stays straight down", Math.abs((await cam()).pitch + Math.PI / 2) < 1e-9);

  // Left-drag pans like a map: the grabbed point follows the cursor.
  const grabbed = [0, 60, Z];
  const grab = await v.project(grabbed);
  const yawBefore = (await cam()).yaw;
  await v.drag("left", grab[0], grab[1], 120, 80);
  await frame();
  const dropped = await v.project(grabbed);
  t.check("a left-drag pans the map with the cursor",
    px(dropped, [grab[0] + 120, grab[1] + 80]) < 1.5, `${dropped.map((n) => n.toFixed(1))}`);
  t.check("without turning it", Math.abs((await cam()).yaw - yawBefore) < 1e-9);
  // Right-drag turns it; it never tilts.
  await v.drag("right", 640, 400, 120, 80);
  c = await cam();
  t.check("a right-drag turns the map", Math.abs(c.yaw) > 0.1, c.yaw.toFixed(3));
  t.check("but never tilts it", Math.abs(c.pitch + Math.PI / 2) < 1e-9, c.pitch.toFixed(6));
  await v.key("b");
  await frame();
  c = await cam();
  t.check("B is north up again, still ortho", Math.abs(c.yaw) < 1e-9 && (await map("ortho()")).on);
  await v.shot("mapview_ortho.png");

  // ---- tiles ----------------------------------------------------------------------
  t.check("tiles are off until asked for", !(await map("tiles()")).on && seen.cors.length === 0);
  const problems = await map(`setSource(${JSON.stringify({ type: "xyz", url: "not a url" })})`);
  t.check("a bad source is refused with a reason", problems.length > 0, problems.join("; "));
  await map(`setSource(${JSON.stringify({
    type: "xyz", url: `${corsBase}/xyz/{z}/{x}/{y}.png`, maxZoom: 19, attribution: "Test tiles",
  })})`);
  await clickElement("tilesBtn");
  t.check("the Tiles button turns them on", (await map("tiles()")).on);
  t.check("tiles arrive", await waitFor(async () => (await map("tiles()")).shown > 0));
  let st = await map("tiles()");
  t.check("one screen of them, at a zoom matching the view", st.zoom >= 16 && st.zoom <= 19 && st.wanted <= 40,
    JSON.stringify({ zoom: st.zoom, wanted: st.wanted }));
  const [tx, ty] = tileOf(ANCHOR.lat, ANCHOR.lon, st.zoom);
  t.check("including the tile under the map", seen.cors.some((u) => u === `/xyz/${st.zoom}/${tx}/${ty}.png`),
    `${st.zoom}/${tx}/${ty} in ${seen.cors.slice(0, 4).join(" ")}`);
  t.check("the grid steps aside", !(await map("gridVisible()")));
  t.check("and the credit is shown", (await v.hud("tileAttribution")) === "Test tiles"
    && !(await v.evaluate("document.getElementById('tileAttribution').hidden")));
  // Each tile corner goes through Mercator -> lat/lon -> the viewer's ENU.
  const key = `${st.zoom}/${tx}/${ty}`;
  await waitFor(async () => (await map("readyKeys()")).includes(key));
  const corner = await map(`tileCorner(${JSON.stringify(key)})`);
  const want = swCornerEnu(st.zoom, tx, ty);
  t.check("tiles sit where the map's own lat/lon put them", corner && px(corner, want) < 0.01,
    `${corner && corner.map((n) => n.toFixed(3))} vs ${want.map((n) => n.toFixed(3))}`);
  await v.shot("mapview_tiles.png");
  // What the screen shows: the map's white line on top of the (red) tiles.
  // Tiles drawn as transparent objects once covered the whole map, because
  // three draws transparent objects after every opaque one.
  const onLine = await v.project([0, 60, Z]);
  let lineSeen = null;
  for (let dy = -2; dy <= 2 && !lineSeen; dy++) {
    const p = await v.pixel(onLine[0], onLine[1] + dy);
    // White over red tiles; a one-pixel line on a pixel boundary is drawn
    // half-strength across two rows (far chunks draw one-pixel lines).
    if (p[1] > 90 && p[2] > 90) lineSeen = p;
  }
  t.check("the map lines are drawn over the tiles", !!lineSeen, JSON.stringify(lineSeen));
  const offLine = await v.pixel(onLine[0], onLine[1] + 40);
  t.check("and the tiles fill the ground around them", offLine[0] > 150 && offLine[1] < 60,
    JSON.stringify(offLine));

  await v.evaluate("(() => { const el = document.getElementById('tileOpacity'); el.value = '40';"
    + " el.dispatchEvent(new Event('input')); })()");
  t.check("the opacity slider fades them", Math.abs((await map("tiles()")).opacity - 0.4) < 1e-9);
  await frame();
  const faded = await v.pixel(onLine[0], onLine[1] + 40);
  t.check("toward the background colour", faded[0] < offLine[0] - 60, `${offLine} -> ${faded}`);

  // TMS counts rows from the bottom.
  await map(`setSource(${JSON.stringify({ type: "tms", url: `${corsBase}/tms/{z}/{x}/{y}.png` })})`);
  t.check("TMS flips the row", (await map("tileUrl(3, 1, 2)")) === `${corsBase}/tms/3/1/5.png`,
    await map("tileUrl(3, 1, 2)"));

  // WMS: one GetMap per tile, in EPSG:3857, with the tile's own bounds.
  seen.cors.length = 0;
  await map(`setSource(${JSON.stringify({
    type: "wms", url: `${corsBase}/geoserver/wms`, layers: "ws:roads", format: "image/png",
    version: "1.3.0", transparent: true, maxZoom: 20,
  })})`);
  t.check("a WMS answers as tiles", await waitFor(async () => (await map("tiles()")).shown > 0));
  st = await map("tiles()");
  const req = new URL(corsBase + seen.cors.find((u) => u.startsWith("/geoserver/wms")));
  const q = Object.fromEntries(req.searchParams.entries());
  t.check("as GetMap requests", q.SERVICE === "WMS" && q.REQUEST === "GetMap" && q.LAYERS === "ws:roads"
    && q.CRS === "EPSG:3857" && q.WIDTH === "256" && q.TRANSPARENT === "TRUE", JSON.stringify(q));
  const bbox = q.BBOX.split(",").map(Number);
  const size = (2 * MERC) / 2 ** st.zoom;
  t.check("one tile's bounds each", Math.abs(bbox[2] - bbox[0] - size) < 0.01
    && Math.abs(bbox[3] - bbox[1] - size) < 0.01, `${(bbox[2] - bbox[0]).toFixed(3)} vs ${size.toFixed(3)}`);

  // UTM (EPSG:25832, what German state servers serve), against pyproj.
  const UTM_REF = [
    [32, 50.1005, 8.6746, 476728.2057, 5549855.5473],   // Frankfurt
    [32, 49.0094, 8.4044, 456444.6194, 5428671.6160],   // Karlsruhe
    [32, 48.7133, 9.0028, 500205.9719, 5395585.5028],   // Sindelfingen
    [32, 47.6, 6.1, 282014.4585, 5275918.8891],         // 2.9 degrees west of the meridian
    [32, 54.5, 13.3, 778389.4512, 6047665.5509],        // 4.3 east, beyond the zone
    [33, 50.1005, 8.6746, 47786.2700, 5568990.6262],    // 6.3 degrees out
  ];
  let worstUtm = 0;
  for (const [zone, lat, lon, e, n] of UTM_REF) {
    const got = await map(`utm(${lat}, ${lon}, ${zone})`);
    worstUtm = Math.max(worstUtm, Math.hypot(got[0] - e, got[1] - n));
  }
  t.check("UTM agrees with pyproj to the millimetre, even outside the zone", worstUtm < 0.001,
    `${(worstUtm * 1000).toFixed(3)} mm`);

  seen.cors.length = 0;
  await map(`setSource(${JSON.stringify({
    type: "wms", url: `${corsBase}/lgl/ows?`, layers: "IMAGES_DOP_20_RGB", format: "image/png",
    version: "1.1.1", crs: "EPSG:25832", maxZoom: 20,
  })})`);
  t.check("a UTM WMS answers as tiles", await waitFor(async () => (await map("tiles()")).shown > 0));
  st = await map("tiles()");
  const utmReq = seen.cors.find((u) => u.startsWith("/lgl/ows?"));
  const uq = Object.fromEntries(new URL(corsBase + utmReq).searchParams.entries());
  t.check("asked in EPSG:25832 (SRS, as WMS 1.1.1 spells it)", uq.SRS === "EPSG:25832" && !uq.CRS
    && uq.VERSION === "1.1.1" && !utmReq.includes("?&"), utmReq.slice(0, 120));
  // Which tile was that? Find it among the ready ones by its box.
  const ub = uq.BBOX.split(",").map(Number);
  let matched = null;
  for (const k of await map("readyKeys()")) {
    const [kz, kx, ky] = k.split("/").map(Number);
    const box = await map(`wmsBox(${kz}, ${kx}, ${ky})`);
    if (Math.abs(box.minx - ub[0]) < 0.002 && Math.abs(box.maxy - ub[3]) < 0.002) matched = { k, kz, kx, ky, box };
  }
  t.check("each request is one tile's box", !!matched, uq.BBOX);
  if (matched) {
    // The box encloses the tile's corners in UTM, and only just.
    const s3857 = (2 * MERC) / 2 ** matched.kz;
    const cornerLL = [];
    for (const [ix, iy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      const mx = -MERC + (matched.kx + ix) * s3857;
      const my = MERC - (matched.ky + iy) * s3857;
      cornerLL.push([deg(2 * Math.atan(Math.exp(my / R)) - Math.PI / 2), deg(mx / R)]);
    }
    const cornersUtm = [];
    for (const [lat, lon] of cornerLL) cornersUtm.push(await map(`utm(${lat}, ${lon}, 32)`));
    const es = cornersUtm.map((c) => c[0]);
    const ns = cornersUtm.map((c) => c[1]);
    t.check("the box encloses the tile in UTM, and only just",
      Math.abs(Math.min(...es) - ub[0]) < 0.01 && Math.abs(Math.max(...es) - ub[2]) < 0.01
      && Math.abs(Math.min(...ns) - ub[1]) < 0.01 && Math.abs(Math.max(...ns) - ub[3]) < 0.01,
      `${es.map((v) => v.toFixed(2))} / ${uq.BBOX}`);
    t.check("which is wider than the tile: the grids are turned against each other",
      ub[2] - ub[0] > Math.abs(es[1] - es[0]) + 0.1);
    // Every vertex samples the image where its own lat/lon lies.
    const uvs = await map(`tileUvs(${JSON.stringify(matched.k)})`);
    const sw = cornersUtm[2];  // (0, 1): west, south
    const wantU = (sw[0] - ub[0]) / (ub[2] - ub[0]);
    const wantV = (sw[1] - ub[1]) / (ub[3] - ub[1]);
    t.check("the south-west vertex samples its own spot in the image",
      Math.abs(uvs[0] - wantU) < 1e-6 && Math.abs(uvs[1] - wantV) < 1e-6,
      `${uvs[0].toFixed(5)},${uvs[1].toFixed(5)} vs ${wantU.toFixed(5)},${wantV.toFixed(5)}`);
    t.check("and none samples outside it", uvs.every((u) => u > -1e-9 && u < 1 + 1e-9));
  }
  t.check("an unknown CRS is refused", (await map(`setSource(${JSON.stringify({
    type: "wms", url: corsBase, layers: "x", crs: "EPSG:31467" })})`)).length > 0);

  // The form: presets fill it, custom types open their fields.
  await clickElement("tilesMenuBtn");
  t.check("▾ opens the source panel", !(await v.evaluate("document.getElementById('tilePanel').hidden")));
  t.check("showing the WMS fields for a WMS", !(await v.evaluate("document.getElementById('tileWms').hidden")));
  await v.evaluate("(() => { const el = document.getElementById('tilePreset'); el.value = 'osm';"
    + " el.dispatchEvent(new Event('change')); })()");
  t.check("the OpenStreetMap preset fills in its URL", (await v.evaluate("document.getElementById('tileUrl').value"))
    === "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    && await v.evaluate("document.getElementById('tileUrl').disabled")
    && await v.evaluate("document.getElementById('tileWms').hidden"));
  await v.evaluate("(() => { const el = document.getElementById('tilePreset'); el.value = 'hessen_dop20';"
    + " el.dispatchEvent(new Event('change')); })()");
  t.check("the Hessen preset is a WMS with its layer filled in",
    !(await v.evaluate("document.getElementById('tileWms').hidden"))
    && (await v.evaluate("document.getElementById('tileLayers').value")) === "he_dop20_rgb");
  await v.evaluate("(() => { const el = document.getElementById('tilePreset'); el.value = 'bw_dop20';"
    + " el.dispatchEvent(new Event('change')); })()");
  t.check("the Baden-Württemberg preset asks in EPSG:25832",
    (await v.evaluate("document.getElementById('tileCrs').value")) === "EPSG:25832"
    && (await v.evaluate("document.getElementById('tileLayers').value")) === "IMAGES_DOP_20_RGB");
  const note = () => v.evaluate("(() => { const n = document.getElementById('tilePresetNote');"
    + " return { hidden: n.hidden, text: n.textContent, warn: n.classList.contains('warnText') }; })()");
  t.check("only a preset with something to say has a note", (await note()).hidden, JSON.stringify(await note()));
  await clickElement("tilesMenuBtn");

  // A server without CORS: WebGL refuses its images...
  seen.nocors.length = 0;
  await map(`setSource(${JSON.stringify({ type: "xyz", url: `http://localhost:${nocorsSrv.address().port}/{z}/{x}/{y}.png` })})`);
  t.check("a server without CORS cannot be drawn directly",
    await waitFor(async () => (await map("tiles()")).failed > 0) && (await map("tiles()")).ready === 0);
  t.check("and the user is told what to do about it", await waitFor(async () => {
    const toast = await v.evaluate("window.__ll2test.toast()");
    return toast && toast.text.includes("--tile-proxy-host");
  }, 3000));
  // ...but through the viewer server, for a host the user allowed, it works.
  await map(`setSource(${JSON.stringify({ type: "xyz", url: `http://${nocorsHost}/{z}/{x}/{y}.png` })})`);
  t.check("an allowed host is fetched through the viewer server",
    await waitFor(async () => (await map("tiles()")).shown > 0), JSON.stringify(await map("tiles()")));
  t.check("which really asked that server", seen.nocors.length > 0);
  const refused = await v.evaluate("fetch('/tileproxy?url=' + encodeURIComponent('http://example.invalid/1/1/1.png'))"
    + ".then((r) => r.status)");
  t.check("the proxy refuses every other host", refused === 403, String(refused));
  const notHttp = await v.evaluate(`fetch('/tileproxy?url=' + encodeURIComponent('file:///etc/passwd')).then((r) => r.status)`);
  t.check("and anything that is not http(s)", notHttp === 403, String(notHttp));

  // Tiles in perspective too, laid on the ground.
  await v.key("v");
  await frame();
  t.check("V goes back to perspective", !(await map("ortho()")).on);
  await v.drag("left", 640, 400, 0, -120);  // tilt toward the horizon
  await sleep(300);
  t.check("tiles also cover the ground in perspective", await waitFor(async () => (await map("tiles()")).shown > 0));
  await v.shot("mapview_perspective_tiles.png");

  await clickElement("tilesBtn");
  t.check("Tiles off hides them and brings the grid back",
    !(await map("tiles()")).on && await map("gridVisible()")
    && await v.evaluate("document.getElementById('tileAttribution').hidden"));
  t.check("no page errors", v.errors.length === 0, v.errors.join(" | "));
} catch (err) {
  t.check(`no exception (${err && err.stack ? err.stack : err})`, false);
} finally {
  await v.close();
  corsSrv.close();
  nocorsSrv.close();
}
void wmsLog;
console.log(t.failures ? `\n${t.failures} of ${t.count} FAILED` : `\nall ${t.count} passed`);
process.exit(t.failures ? 1 : 0);
