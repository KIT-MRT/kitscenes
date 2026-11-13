// Drive the real scene server in city mode against real KITScenes data and
// report what the browser got. Not part of the hermetic suite (e2e_city.mjs
// is): this one needs the dataset and a Python with kitscenes-viewer's
// requirements (PYTHON=/path/to/python, default python3).
//
//   node viewer/tests/browser/live_city.mjs <KITSCENES_ROOT | data | split>
//        [--map city.osm] [--city frankfurt] [--scene <id>] [--frame N]
//        [--load] [--shots DIR]
//
// It opens one scene by clicking its trajectory, checks that the city's
// trajectory and the opened scene's own ego track agree, compares the cloud
// under the car with the map, and with --load pulls in the static clouds
// around the camera the way the Load button does.
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { openViewer, sleep, dist3 } from "./harness.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
// The package from this checkout, installed or not.
const SERVER_DIR = path.resolve(HERE, "../..");

const root = process.argv[2];
if (!root || root.startsWith("--")) {
  console.error("usage: live_city.mjs <KITSCENES_ROOT | data | split> [--map osm] [--city name] "
    + "[--scene id] [--frame N] [--load] [--shots DIR]");
  process.exit(2);
}
const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const frame = Number(arg("--frame", 110));
const shotsDir = arg("--shots", null);
const extra = [];
if (arg("--map")) extra.push("--map", arg("--map"));
if (arg("--city")) extra.push("--city-name", arg("--city"));

const v = await openViewer({ shotsDir });
let server = null;
const waitFor = async (pred, ms) => {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    if (await pred()) return true;
    await sleep(250);
  }
  return false;
};
const city = () => v.evaluate("window.__ll2city.state()");
try {
  const t0 = Date.now();
  server = spawn(process.env.PYTHON || "python3",
    ["-m", "kitscenes_viewer.scene", "--city", root, ...extra,
     "--viewer-port", String(v.ports.http), "--ingest-port", String(v.ports.ingest),
     "--http-port", "0"],
    { cwd: SERVER_DIR, stdio: ["ignore", "pipe", "pipe"] });
  server.stdout.on("data", (d) => process.stdout.write(`[server] ${d}`));
  server.stderr.on("data", (d) => process.stderr.write(`[server] ${d}`));

  if (!(await waitFor(async () => (await city()).tracks.scenes > 0, 120000))) {
    throw new Error("the city never arrived");
  }
  const c0 = await city();
  console.log(`city: ${c0.name}, ${c0.tracks.scenes} scenes, ${c0.tracks.nodes} poses, `
    + `in the browser after ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  console.log(`map features: ${await v.evaluate("window.__ll2test.stats().features")}`);
  await v.key("b");
  await sleep(800);
  await v.shot("live_city_bev.png");

  // Open a scene by clicking its trajectory, as a user would.
  const chosen = arg("--scene", null) || (await v.evaluate("window.__ll2city.ids()"))[0];
  const node = await v.evaluate(`window.__ll2city.nodePosition(${JSON.stringify(chosen)}, ${frame})`);
  if (!node) throw new Error(`scene ${chosen} has no frame ${frame}`);
  // Get the node on screen: stand back from it, then click it.
  await v.evaluate(`window.__ll2city.lookAt(${JSON.stringify(node)}, 60)`);
  await sleep(500);
  const px = await v.project(node);
  await v.click(px[0], px[1]);
  if (!(await waitFor(async () => {
    const s = await v.evaluate("window.__ll2scene.state()");
    return s.base && s.base.endsWith(`/scene/${chosen}`) && s.frame === frame && !s.loading;
  }, 180000))) throw new Error("clicking the trajectory did not open the scene");
  console.log(`opened ${chosen} at frame ${frame} by clicking its trajectory`);

  const ego = await v.evaluate(`window.__ll2scene.ego(${frame})`);
  console.log(`city trajectory vs the scene's own ego track at frame ${frame}: `
    + `${(dist3(ego, node) * 1000).toFixed(1)} mm (positions are sent to the cm)`);
  const compare = await v.evaluate(`(() => {
    const car = ${JSON.stringify(ego)};
    return { ground: window.__ll2sceneDebug.cloudZNear(car, 12),
             map: window.__ll2sceneDebug.mapZNear(car, 12) };
  })()`);
  console.log(`cloud ground under the car: ${compare.ground.p10?.toFixed(2)} m, `
    + `map ele there: ${compare.map.median?.toFixed(2)} m (${compare.map.count} nodes)`);
  if (compare.ground.p10 != null && compare.map.median != null) {
    console.log(`=> map vs cloud: ${((compare.ground.p10 - compare.map.median) * 100).toFixed(1)} cm`);
  }
  await v.key("k");
  await sleep(1200);
  await v.shot("live_city_scene.png");

  if (process.argv.includes("--load")) {
    const t1 = Date.now();
    await v.evaluate("window.__ll2city.loadHere(true)");
    const done = await waitFor(async () => {
      const s = await city();
      const pendingStates = Object.values(s.load).filter((x) => ["meta", "queued", "building"].includes(x));
      return !pendingStates.length && !s.busy && s.queue === 0 && s.loaded.length > 0;
    }, 1800000);
    const s = await city();
    console.log(`load: ${done ? "done" : "timed out"} in ${((Date.now() - t1) / 1000).toFixed(1)} s, `
      + `${s.points.points} points in ${s.points.chunks} chunks from ${s.points.scenes} scenes`);
    await v.shot("live_city_loaded.png");
    await v.key("b");
    await sleep(800);
    await v.shot("live_city_loaded_bev.png");
  }
  console.log(`page errors: ${v.errors.length ? v.errors.join(" | ") : "none"}`);
} finally {
  if (server) server.kill("SIGINT");
  await v.close();
}
