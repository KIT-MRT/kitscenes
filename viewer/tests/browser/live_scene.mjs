// Drive the real scene server against a real recorded scene and report what
// the browser actually got. Not part of the hermetic suite (e2e_scene.mjs is):
// this one needs the dataset on disk and a Python with kitscenes-viewer's
// requirements (PYTHON=/path/to/python, default python3).
//
//   node viewer/tests/browser/live_scene.mjs <scene dir> [--frame N] [--shots DIR]
//        [--aggregate] [--tiles [xyz url]] [--wms <url> <layer> [crs]]
//
// It is the end-to-end answer to "do the map and the LiDAR line up?", so it
// prints the height of the cloud under the car next to the map's own `ele`.
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { openViewer, sleep } from "./harness.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
// The package from this checkout, installed or not.
const SERVER_DIR = path.resolve(HERE, "../..");

const sceneDir = process.argv[2];
if (!sceneDir || sceneDir.startsWith("--")) {
  console.error("usage: live_scene.mjs <scene dir> [--frame N] [--shots DIR]");
  process.exit(2);
}
const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const frame = Number(arg("--frame", 110));
const shotsDir = arg("--shots", null);

const v = await openViewer({ shotsDir });
let scene = null;
try {
  scene = spawn(process.env.PYTHON || "python3",
    ["-m", "kitscenes_viewer.scene", "--scene", sceneDir,
     "--viewer-port", String(v.ports.http), "--ingest-port", String(v.ports.ingest),
     "--http-port", "0"],
    { cwd: SERVER_DIR, stdio: ["ignore", "pipe", "pipe"] });
  scene.stdout.on("data", (d) => process.stdout.write(`[scene] ${d}`));
  scene.stderr.on("data", (d) => process.stderr.write(`[scene] ${d}`));

  const ready = await (async () => {
    for (let i = 0; i < 120; i++) {
      const s = await v.evaluate("window.__ll2scene ? window.__ll2scene.state() : null");
      if (s && s.frames) return s;
      await sleep(500);
    }
    return null;
  })();
  if (!ready) throw new Error("the scene server never announced a scene");
  console.log(`scene: ${ready.frames} frames, data from ${ready.base}`);
  console.log(`map features: ${await v.evaluate("window.__ll2test.stats().features")}`);

  await v.evaluate(`window.__ll2scene.show(${frame})`);
  for (let i = 0; i < 120 && (await v.evaluate("window.__ll2scene.state()")).loading; i++) await sleep(500);
  const pts = await v.evaluate("window.__ll2scene.points()");
  console.log(`frame ${frame}: ${pts.count} points, origin ${pts.origin.map((n) => n.toFixed(1))}`);

  // Where the cloud sits under the car, against the map nodes around it.
  const car = await v.evaluate(`window.__ll2scene.ego(${frame})`);
  const compare = await v.evaluate(`(() => {
    const car = ${JSON.stringify(car)};
    const g = window.__ll2sceneDebug.cloudZNear(car, 12);
    const m = window.__ll2sceneDebug.mapZNear(car, 12);
    return { ground: g, map: m };
  })()`);
  console.log(`cloud ground under the car: ${compare.ground.p10?.toFixed(2)} m `
    + `(${compare.ground.count} pts)   map ele there: ${compare.map.median?.toFixed(2)} m `
    + `(${compare.map.count} nodes)`);
  if (compare.ground.p10 != null && compare.map.median != null) {
    console.log(`=> map vs cloud: ${((compare.ground.p10 - compare.map.median) * 100).toFixed(1)} cm`);
  }

  // The whole scene as one static cloud (built on the server, then cached).
  if (process.argv.includes("--aggregate")) {
    await v.evaluate("window.__ll2scene.showAggregate(true)");
    const ok = await (async () => {
      for (let i = 0; i < 600; i++) {
        const a = await v.evaluate("window.__ll2scene.aggregate()");
        if (a.error) throw new Error(a.error);
        if (!a.building && a.points) return a;
        await sleep(1000);
      }
      return null;
    })();
    console.log(`aggregate: ${ok ? `${ok.points} points in ${ok.chunks} chunks` : "timed out"}`);
    if (ok && ok.stats) console.log(`  ${JSON.stringify(ok.stats)}`);
    await v.key("b");
    await sleep(1500);
    await v.shot("live_aggregate_bev.png");
    // Street level with the single frame hidden: only the aggregate is left,
    // so any smeared traffic would be obvious.
    await v.evaluate(`window.__ll2scene.show(${frame})`);
    await sleep(1500);
    await v.key("p");
    await v.key("k");
    await sleep(1500);
    await v.shot("live_aggregate_car.png");
    await v.evaluate("(() => { const el = document.getElementById('sceneSize');"
      + " el.value = '1'; el.dispatchEvent(new Event('input')); })()");
    await sleep(800);
    await v.shot("live_aggregate_car_fine.png");
    const setMode = async (mode) => {
      await v.evaluate(`(() => { const el = document.getElementById('sceneMode');`
        + ` el.value = '${mode}'; el.dispatchEvent(new Event('change')); })()`);
      await sleep(900);
    };
    const classes = await v.evaluate("window.__ll2scene.classes()");
    console.log(`classes: available=${classes.available} inData=${classes.inData} rgb=${classes.rgb}`);
    // The live frame is tinted orange while the aggregate is up; hide it so
    // these shots show the aggregate's own colouring.
    await v.key("p");
    await sleep(400);
    await setMode("class");
    await v.shot("live_aggregate_class.png");
    await setMode("rgb");
    await v.shot("live_aggregate_rgb.png");
    // Instances live on the frame, so bring it back and sit in the car.
    await v.key("p");
    await sleep(400);
    await v.key("k");
    await sleep(1200);
    const inst = await v.evaluate("window.__ll2scene.classes()");
    console.log(`instances in this frame: ${inst.instances}`);
    await setMode("instance");
    await v.shot("live_frame_instances.png");
    await setMode("intensity");
  }

  await v.key("k");
  await sleep(1200);
  await v.evaluate("document.getElementById('sceneFilterBtn').click()");
  await v.evaluate("document.getElementById('classPresetCars').click()");
  await sleep(500);
  console.log(`filter: ${await v.evaluate("document.getElementById('sceneFilterBtn').textContent")}`);
  await v.shot("live_no_vehicles.png");
  await v.evaluate("document.getElementById('classShowAll').click()");
  await v.evaluate("(() => { const el = document.getElementById('pointAlpha');"
    + " el.value = '35'; el.dispatchEvent(new Event('input')); })()");
  await sleep(400);
  await v.shot("live_points_faded.png");
  await v.evaluate("(() => { const el = document.getElementById('pointAlpha');"
    + " el.value = '100'; el.dispatchEvent(new Event('input')); })()");
  await v.evaluate("document.getElementById('sceneFilterBtn').click()");
  await sleep(300);
  console.log(`icons: ${JSON.stringify(await v.evaluate("window.__ll2test.icons()"))}`);
  await v.shot("live_car.png");
  await v.evaluate("(() => { const el = document.getElementById('iconAlpha');"
    + " el.value = '45'; el.dispatchEvent(new Event('input')); })()");
  await sleep(400);
  await v.shot("live_car_icons_faded.png");
  await v.evaluate("(() => { const el = document.getElementById('iconAlpha');"
    + " el.value = '100'; el.dispatchEvent(new Event('input')); })()");

  // Camera lock + overlay: the alignment check this whole thing is for.
  await v.evaluate("document.getElementById('sceneLock').click()");
  await sleep(2500); // let the recorded JPEG arrive
  const rig = await v.evaluate("window.__ll2scene.rig()");
  console.log(`lock: ${rig.camera}, viewport ${rig.viewport.width}x${rig.viewport.height}, `
    + `image ${rig.imageSrc ? "loaded" : "none"}`);
  await v.shot("live_overlay.png");
  await v.evaluate("(() => { const el = document.getElementById('sceneOpacity');"
    + " el.value = '100'; el.dispatchEvent(new Event('input')); })()");
  await sleep(600);
  await v.shot("live_overlay_full.png");
  await v.evaluate("(() => { const el = document.getElementById('sceneOpacity');"
    + " el.value = '55'; el.dispatchEvent(new Event('input')); })()");
  // Map over the photo with the cloud out of the way: this is the view that
  // says whether the *map* is right, rather than whether the poses are.
  await v.key("p");
  await sleep(500);
  await v.shot("live_overlay_map.png");
  await v.key("p");
  await v.key("o");
  await sleep(400);
  await v.shot("live_overlay_off.png");
  await v.key("L", { modifiers: ["shift"] });
  await sleep(300);
  await v.key("b");
  await sleep(600);
  await v.shot("live_bev.png");
  // --tiles [url]: the orthographic map view over web tiles (OSM by default),
  // to see the map and the cloud against the background imagery.
  const ti = process.argv.indexOf("--tiles");
  const wi = process.argv.indexOf("--wms");
  if (ti > 0 || wi > 0) {
    const url = ti > 0 && process.argv[ti + 1] && !process.argv[ti + 1].startsWith("--") ? process.argv[ti + 1] : null;
    await v.key("b");
    await sleep(500);
    await v.evaluate("window.__ll2map.setOrtho(true)");
    if (url) await v.evaluate(`window.__ll2map.setSource(${JSON.stringify({ type: "xyz", url, maxZoom: 19 })})`);
    if (wi > 0) {
      const problems = await v.evaluate(`window.__ll2map.setSource(${JSON.stringify({
        type: "wms", url: process.argv[wi + 1], layers: process.argv[wi + 2], format: "image/jpeg",
        version: "1.3.0", maxZoom: 21,
        crs: process.argv[wi + 3] && !process.argv[wi + 3].startsWith("--") ? process.argv[wi + 3] : "EPSG:3857" })})`);
      if (problems.length) throw new Error(`WMS source: ${problems.join("; ")}`);
    }
    await v.evaluate("window.__ll2map.setTiles(true)");
    for (let i = 0; i < 60; i++) {
      const st = await v.evaluate("window.__ll2map.tiles()");
      if ((st.wanted && st.shown >= st.wanted) || st.failed) break;
      await sleep(500);
    }
    const st = await v.evaluate("window.__ll2map.tiles()");
    console.log(`tiles: ${st.source.url}, zoom ${st.zoom}, ${st.shown}/${st.wanted} shown, ${st.failed} failed`);
    await v.shot("live_tiles_ortho.png");
    await v.wheel(640, 400, -100, 6);
    await sleep(3000);
    await v.shot("live_tiles_ortho_close.png");
  }
  console.log(`page errors: ${v.errors.length ? v.errors.join(" | ") : "none"}`);
} finally {
  if (scene) scene.kill("SIGINT");
  await v.close();
}
