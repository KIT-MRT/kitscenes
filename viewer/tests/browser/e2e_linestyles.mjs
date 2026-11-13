// End-to-end checks of the styled map lines (render/ribbons.js), read off
// real screen pixels: dashes and their gaps in metres, the dash background,
// real widths up close, double lines and their sides, the pixel minimum far
// away, the `color` tag, raised walls on and off, the Simple look, drags,
// the highlight over a wide marking and the legend.
//   node viewer/tests/browser/e2e_linestyles.mjs [--shots DIR]
import { openViewer, Checks, sleep } from "./harness.mjs";

const shotsIdx = process.argv.indexOf("--shots");
const shotsDir = shotsIdx > 0 ? process.argv[shotsIdx + 1] : null;
const t = new Checks();
const Z = 100;
const X0 = 1000; // clear of the reference grid around the origin

// Every line runs east (+x) from X0, 30 m, nodes every 10 m.
let nextNode = 1;
const line = (id, y, tags) => {
  const nodes = [0, 1, 2, 3].map(() => nextNode++);
  return { id: `way/${id}`, kind: "line", tags, nodes,
           pts: [0, 10, 20, 30].flatMap((dx) => [X0 + dx, y, Z]) };
};
const features = [
  line(1, 0, { type: "line_thin", subtype: "dashed" }),
  line(2, 3, { type: "line_thick", subtype: "dashed" }),
  line(3, 6, { type: "line_thin", subtype: "solid_solid" }),
  line(4, 9, { type: "line_thin", subtype: "solid_dashed" }),
  line(5, 12, { type: "stop_line" }),
  line(6, 15, { type: "line_thin", color: "yellow" }),
  line(7, 18, { type: "wall" }),
  line(8, 21, { type: "line_thin" }),
];

const v = await openViewer({ shotsDir });
const frames = (n = 3) => v.evaluate(`new Promise((r) => { let k = ${n}; const f = () => (--k <= 0 ? r() : requestAnimationFrame(f)); requestAnimationFrame(f); })`);
const lines = () => v.evaluate("window.__ll2lines.stats()");
const topDown = async (x, y, h) => {
  await v.evaluate(`window.__ll2test.setView([${x}, ${y}, ${Z + h}], 0, -Math.PI / 2)`);
  await frames();
};
const px = async (x, y, z = Z) => {
  const s = await v.project([x, y, z]);
  return v.pixel(s[0], s[1]);
};
const bright = (p) => (p[0] + p[1] + p[2]) / 3;
const isLine = (p) => bright(p) > 170;
const isGap = (p) => Math.abs(p[0] - 0x7d) < 25 && Math.abs(p[2] - 0x8b) < 25;  // DASH_BG
const isBackground = (p) => bright(p) < 60;
const show = (p) => JSON.stringify(p);

try {
  v.sendScene({ type: "snapshot", anchor: { lat: 49.0, lon: 8.4 }, features });
  t.check("map streamed", await v.waitForFeatures(features.length));
  await frames(4);
  let s = await lines();
  t.check("styled by default: one quad per segment", s.styled && s.segments === 8 * 3, JSON.stringify(s));
  t.check("the wall's segments also get raised faces", s.raisedSegments === 3, JSON.stringify(s));
  t.check("the segment carries its style and distance along the way",
    JSON.stringify((await v.evaluate("window.__ll2lines.segment('way/1', 1)")).slice(2))
      === JSON.stringify([[10, 20], "line_thin/dashed"]), JSON.stringify(await v.evaluate("window.__ll2lines.segment('way/1', 1)")));

  // ---- dashes in metres: 3 m paint, 6 m gap, running on across vertices ----------
  await topDown(X0 + 10, 4, 20);
  await v.shot("lines_dashes.png");
  let a = await px(X0 + 1.5, 0), b = await px(X0 + 4.5, 0), c = await px(X0 + 10.5, 0), d = await px(X0 + 13, 0);
  t.check("a thin dashed line is painted 3 m on", isLine(a) && isLine(c), `${show(a)} ${show(c)}`);
  t.check("and 6 m off, the phase carried across the vertex at 10 m", isBackground(b) && isBackground(d), `${show(b)} ${show(d)}`);
  a = await px(X0 + 1.5, 3);
  b = await px(X0 + 4.5, 3);
  t.check("a thick dashed line fills its gaps with the dash background", isLine(a) && isGap(b), `${show(a)} ${show(b)}`);
  await topDown(X0 + 10, 15, 20);
  const yellow = await px(X0 + 5, 15);
  t.check("color=yellow paints the marking yellow", yellow[0] > 200 && yellow[1] > 170 && yellow[2] < 90, show(yellow));

  // ---- real widths up close ----------------------------------------------------------------
  await topDown(X0 + 1, 6, 4);
  await v.shot("lines_close.png");
  a = await px(X0 + 1, 6);
  b = await px(X0 + 1, 6.12);
  c = await px(X0 + 1, 5.88);
  t.check("a double line is two 12 cm stripes with a 12 cm gap", isBackground(a) && isLine(b) && isLine(c),
    `${show(a)} ${show(b)} ${show(c)}`);
  await topDown(X0 + 1, 12, 4);
  a = await px(X0 + 1, 12.2);
  t.check("a stop line is 50 cm wide, a thin line 12 cm", isLine(a), show(a));
  await topDown(X0 + 1, 21, 4);
  b = await px(X0 + 1, 21.2);
  t.check("(20 cm off a thin line is road)", isBackground(b), show(b));
  await topDown(X0 + 4.5, 9, 4);
  a = await px(X0 + 4.5, 9.12);
  b = await px(X0 + 4.5, 8.88);
  t.check("solid_dashed: solid on the left of the way's direction", isLine(a), show(a));
  t.check("and dashed on the right, here in a gap", isGap(b), show(b));

  // ---- far away: never thinner than its pixel minimum -----------------------------------------
  await topDown(X0 + 15, 21, 300);
  await v.shot("lines_far.png");
  a = await px(X0 + 15, 21);
  b = await px(X0 + 15, 24);
  t.check("from 300 m up a 12 cm line still shows", bright(a) > 70 && isBackground(b), `${show(a)} ${show(b)}`);
  // From 600 m a metre is under 2 px: the chunk draws one-pixel lines.
  await topDown(X0 + 15, 10, 600);
  t.check("far away the chunk draws the cheap one-pixel lines instead",
    (await lines()).nearChunks === 0, JSON.stringify(await lines()));
  await topDown(X0 + 15, 10, 20);
  t.check("up close it is paint again", (await lines()).nearChunks === 1, JSON.stringify(await lines()));

  // ---- raised walls ---------------------------------------------------------------------------
  await v.evaluate(`window.__ll2test.setView([${X0 + 15}, 3, ${Z + 1}], 0, 0)`);
  await frames();
  await v.shot("lines_wall.png");
  a = await px(X0 + 15, 18, Z + 1);
  t.check("a wall stands 2 m high", !isBackground(a), show(a));
  // Paint is flat: at street level a 50 cm stop line is a thin sliver, not a
  // 50 cm board facing the camera.
  b = await px(X0 + 15, 12, Z + 0.2);
  t.check("at street level paint lies flat on the road", isBackground(b), show(b));
  await v.evaluate("window.__ll2lines.setRaised(false)");
  await frames();
  b = await px(X0 + 15, 18, Z + 1);
  t.check("3D off: the wall is a flat line again", isBackground(b), show(b));
  t.check("and the switch says so", !(await v.evaluate("document.getElementById('raisedBtn').classList.contains('on')")));
  await v.evaluate("window.__ll2lines.setRaised(true)");
  await frames();

  // ---- the Simple look ------------------------------------------------------------------------------
  await v.evaluate("window.__ll2lines.setStyled(false)");
  await frames(3);
  s = await lines();
  t.check("Simple frees the quads and draws one-pixel lines", !s.styled && s.segments === 0 && s.lineVertices === 32,
    JSON.stringify(s));
  t.check("the 3D switch is off-limits in Simple", await v.evaluate("document.getElementById('raisedBtn').disabled"));
  await v.evaluate("window.__ll2lines.setStyled(true)");
  await frames(3);
  s = await lines();
  t.check("and Styled builds them again", s.styled && s.segments === 24, JSON.stringify(s));

  // ---- a drag moves the quads with the node -------------------------------------------------------
  await v.evaluate(`void window.__ll2test.moveLocal([{ id: ${features[7].nodes[1]}, x: ${X0 + 10}, y: 23, z: ${Z} }])`);
  await frames();
  const s0 = await v.evaluate("window.__ll2lines.segment('way/8', 0)");
  const s1 = await v.evaluate("window.__ll2lines.segment('way/8', 1)");
  t.check("a node moved in a drag moves both segments that meet there",
    s0[1][1] === 23 && s1[0][1] === 23, `${JSON.stringify(s0)} ${JSON.stringify(s1)}`);

  // ---- the highlight shows over a wide marking ----------------------------------------------------------
  await topDown(X0 + 5, 12, 4);
  await v.key("e");
  const at = await v.project([X0 + 5, 12, Z]);
  await v.click(at[0], at[1]);
  await frames();
  t.check("(the stop line is selected)", (await v.evaluate("window.__ll2test.selection()")).ways.join() === "way/5");
  a = await px(X0 + 5, 12);
  t.check("its highlight shows on top of the 50 cm paint", a[0] > 200 && a[1] > 150 && a[2] < 120, show(a));
  await v.shot("lines_selected.png");
  await v.key("Escape");
  await v.key("e");

  // ---- the legend ---------------------------------------------------------------------------------------------
  await v.evaluate("document.getElementById('lineLegendBtn').click()");
  const rows = await v.evaluate("window.__ll2lines.legendRows()");
  t.check("the legend draws the styles from the same table", rows > 25, String(rows));
  t.check("no page errors", v.errors.length === 0, v.errors.join(" | "));
} catch (err) {
  t.check("no exception", false, String(err && err.stack ? err.stack : err));
} finally {
  await v.close();
}
console.log(t.failures ? `\n${t.failures} of ${t.count} FAILED` : `\nall ${t.count} passed`);
process.exit(t.failures ? 1 : 0);
