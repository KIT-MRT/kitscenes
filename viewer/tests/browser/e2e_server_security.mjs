// The viewer server's defences, without a browser: who may talk to it
// (Host names, cross-site requests), what the ingest port accepts, and what
// the tile proxy fetches and passes on.
//   node viewer/tests/browser/e2e_server_security.mjs
import http from "node:http";
import net from "node:net";
import path from "node:path";
import { spawn } from "node:child_process";
import { Checks, sleep, VIEWER_DIR } from "./harness.mjs";

const t = new Checks();
const PNG = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==", "base64");
const freePort = () => new Promise((r) => { const s = net.createServer(); s.listen(0, "127.0.0.1", () => { const p = s.address().port; s.close(() => r(p)); }); });

// A tile host serving a PNG, an SVG and a PNG with a lying length.
const upstream = http.createServer((req, res) => {
  if (req.url.startsWith("/evil.svg")) {
    res.writeHead(200, { "Content-Type": "image/svg+xml" });
    res.end("<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>");
  } else if (req.url.startsWith("/short.png")) {
    res.writeHead(200, { "Content-Type": "image/png", "Content-Length": String(PNG.length + 50) });
    res.end(PNG);  // the connection closes before the promised length
  } else {
    res.writeHead(200, { "Content-Type": "image/png" });
    res.end(PNG);
  }
});
await new Promise((r) => upstream.listen(0, "127.0.0.1", r));
const up = upstream.address().port;
const httpPort = await freePort();
const ingestPort = await freePort();
const server = spawn("python3", [path.join(VIEWER_DIR, "server.py"), "--host", "127.0.0.1",
  "--http-port", String(httpPort), "--ingest-port", String(ingestPort), "--no-browser",
  "--tile-proxy-host", `127.0.0.1:${up}`], { stdio: ["ignore", "pipe", "pipe"] });
let log = "";
server.stdout.on("data", (d) => { log += d; });
server.stderr.on("data", (d) => { log += d; });

function get(p, headers = {}, method = "GET", body = null) {
  // The server reads Content-Length; node would otherwise send the body chunked.
  if (body) headers = { ...headers, "Content-Length": Buffer.byteLength(body) };
  return new Promise((resolve) => {
    const req = http.request({ host: "127.0.0.1", port: httpPort, path: p, method, headers }, (res) => {
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks) }));
    });
    req.on("error", (e) => resolve({ status: 0, error: String(e) }));
    if (body) req.write(body);
    req.end();
  });
}
const proxy = (url, headers) => get(`/tileproxy?url=${encodeURIComponent(url)}`, headers);

try {
  for (let i = 0; i < 100 && (await get("/healthz")).status !== 200; i++) await sleep(100);

  // ---- who may talk to it --------------------------------------------------------
  t.check("JOSM's way in works: Host 127.0.0.1", (await get("/healthz")).status === 200);
  t.check("and Host localhost", (await get("/healthz", { Host: `localhost:${httpPort}` })).status === 200);
  t.check("a foreign Host name is refused (DNS rebinding)",
    (await get("/state", { Host: `rebind.attacker.example:${httpPort}` })).status === 403);
  t.check("a cross-site request may not read the state",
    (await get("/state", { "Sec-Fetch-Site": "cross-site" })).status === 403);
  t.check("the page's own requests may", (await get("/state", { "Sec-Fetch-Site": "same-origin" })).status === 200);
  t.check("a foreign Origin may not send commands to JOSM",
    (await get("/command", { Origin: "https://attacker.example", "Content-Type": "text/plain" }, "POST",
      JSON.stringify({ type: "command", ops: [] }))).status === 403);
  t.check("nor shut the server down",
    (await get("/shutdown", { Origin: "https://attacker.example" }, "POST")).status === 403);
  t.check("the viewer page itself may",
    (await get("/command", { Origin: `http://127.0.0.1:${httpPort}`, "Content-Type": "application/json" }, "POST",
      JSON.stringify({ type: "command", ops: [] }))).status === 200);
  t.check("cross-site navigation to the page still works",
    (await get("/", { "Sec-Fetch-Site": "cross-site" })).status === 200);

  // ---- the ingest port ------------------------------------------------------------
  // A browser's no-cors POST to the ingest port: an HTTP request whose body
  // is a valid message. It must not get in.
  await new Promise((resolve) => {
    const sock = net.connect(ingestPort, "127.0.0.1", () => {
      sock.write("POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: text/plain\r\n\r\n"
        + JSON.stringify({ type: "city", city: { base: "http://attacker.example", name: "x" } }) + "\n");
    });
    sock.on("close", resolve);
    sock.on("error", resolve);
    setTimeout(() => { sock.destroy(); resolve(); }, 1500);
  });
  let state = JSON.parse((await get("/state")).body.toString());
  t.check("an HTTP request on the ingest port is dropped, body and all", !state.city, JSON.stringify(state.city));
  // A bridge sending NaN: Python reads it, browsers cannot.
  await new Promise((resolve) => {
    const sock = net.connect(ingestPort, "127.0.0.1", () => {
      sock.write('{"type":"city","city":{"base":"http://127.0.0.1:1","anchor":{"lat":NaN,"lon":8}}}\n');
      sock.write('[1, 2, 3]\n');
      sock.write(JSON.stringify({ type: "city", city: { base: "http://127.0.0.1:2", name: "ok" } }) + "\n");
      setTimeout(() => { sock.end(); resolve(); }, 400);
    });
  });
  state = JSON.parse((await get("/state")).body.toString());
  t.check("NaN and non-objects are refused, the next message still counts",
    state.city && state.city.name === "ok", JSON.stringify(state.city));

  // ---- the tile proxy ------------------------------------------------------------
  let r = await proxy(`http://127.0.0.1:${up}/1/2/3.png`);
  t.check("an allowed host's PNG comes through", r.status === 200 && r.body.equals(PNG));
  t.check("marked nosniff and sandboxed", r.headers["x-content-type-options"] === "nosniff"
    && /sandbox/.test(r.headers["content-security-policy"] || ""));
  t.check("SVG is refused: it can carry script", (await proxy(`http://127.0.0.1:${up}/evil.svg`)).status === 502);
  t.check("a tile cut short is not passed on", (await proxy(`http://127.0.0.1:${up}/short.png`)).status !== 200);
  t.check("user info in the URL is refused",
    (await proxy(`http://evil@127.0.0.1:${up}/1/2/3.png`)).status === 403);
  t.check("another port on the allowed host is refused",
    (await proxy(`http://127.0.0.1:${httpPort}/state`)).status === 403);
  t.check("another host is refused", (await proxy("http://example.invalid/1.png")).status === 403);
  t.check("so is anything not http(s)", (await proxy("file:///etc/passwd")).status === 403);
  const weird = await proxy(`http://127.0.0.1:${up}/x€.png`);
  t.check("odd characters in an upstream error do not break the answer", weird.status > 0, String(weird.status));
} catch (err) {
  t.check(`no exception (${err && err.stack ? err.stack : err})`, false);
} finally {
  server.kill();
  upstream.close();
}
if (t.failures) console.log(log.slice(-2000));
console.log(t.failures ? `\n${t.failures} of ${t.count} FAILED` : `\nall ${t.count} passed`);
process.exit(t.failures ? 1 : 0);
