// How each Lanelet2 line type looks in 3D: the curated counterpart of the
// `way` rules in lanelet2/lines.mapcss. Plain data and plain functions, no
// three.js, so viewer/tests/browser/check_line_styles.mjs can hold it against
// the MapCSS file and fail when a rule there has no entry here.
//
// Same visual vocabulary as JOSM (white markings, dashes, double lines, the
// dash background colour, grey curbs, green road borders), with two changes:
//   * sizes are real metres, not pixels: a marking is as wide as the paint
//     (with a pixel minimum so it never disappears), a dash as long as a
//     painted dash. Typical German values (RMS); tune them here.
//   * colours that vanish on the dark canvas (JOSM's black or dark slate
//     grey) are lifted, keeping their hue.
// Physical borders also get a shape (`raise`), drawn when "3D curbs, rails
// and walls" is on; without it they are flat line styles like the rest.

/** Patterns, as the shader numbers them. */
export const PATTERN = { solid: 0, dashed: 1, double: 2, solid_dashed: 3, dashed_solid: 4, dotted: 5 };
/** Raised shapes, as the shader numbers them. */
export const RAISE = { none: 0, solid: 1, rail: 2, fence: 3 };

const WHITE = "#f2f2f2";
const DASH_BG = "#7d838b";  // JOSM's dashes-background-color: darkgray
const THIN = 0.12;          // "schmaler Strich"
const THICK = 0.25;         // "Breitstrich"
const CITY_DASH = [3, 6];   // Leitlinie in town: 3 m line, 6 m gap
const THICK_DASH = [3, 3];  // unterbrochener Breitstrich in town

/**
 * Every style. `width` is the whole marking (both stripes and the gap of a
 * double line), `stripe` one stripe of a double line, `dash` [on, off] in
 * metres, `gap` the colour between dashes (none: see-through), `minPx` the
 * narrowest it is drawn on screen. `raise`: { kind, bottom, top } in metres
 * above the line, and the colour of the raised part.
 */
export const LINE_STYLES = {
  default: { label: "Other line", color: "#8899aa", width: 0.05, minPx: 1 },

  line_thin: { label: "Thin line", color: WHITE, width: THIN },
  "line_thin/dashed": { label: "Thin dashed", color: WHITE, width: THIN, pattern: "dashed", dash: CITY_DASH },
  "line_thin/solid_solid": { label: "Thin double", color: WHITE, width: 3 * THIN, stripe: THIN, pattern: "double" },
  "line_thin/solid_dashed": { label: "Thin solid | dashed", color: WHITE, width: 3 * THIN, stripe: THIN,
    pattern: "solid_dashed", dash: CITY_DASH, gap: DASH_BG },
  "line_thin/dashed_solid": { label: "Thin dashed | solid", color: WHITE, width: 3 * THIN, stripe: THIN,
    pattern: "dashed_solid", dash: CITY_DASH, gap: DASH_BG },
  line_thick: { label: "Thick line", color: WHITE, width: THICK },
  "line_thick/dashed": { label: "Thick dashed", color: WHITE, width: THICK, pattern: "dashed", dash: THICK_DASH,
    gap: DASH_BG },
  "line_thick/solid_solid": { label: "Thick double", color: WHITE, width: 2 * THICK + THIN, stripe: THICK,
    pattern: "double" },
  "line_thick/solid_dashed": { label: "Thick solid | dashed", color: WHITE, width: 2 * THICK + THIN, stripe: THICK,
    pattern: "solid_dashed", dash: THICK_DASH, gap: DASH_BG },
  "line_thick/dashed_solid": { label: "Thick dashed | solid", color: WHITE, width: 2 * THICK + THIN, stripe: THICK,
    pattern: "dashed_solid", dash: THICK_DASH, gap: DASH_BG },
  stop_line: { label: "Stop line", color: WHITE, width: 0.5 },
  pedestrian_marking: { label: "Pedestrian marking", color: WHITE, width: THICK, pattern: "dashed", dash: [0.5, 0.5],
    gap: DASH_BG },
  bike_marking: { label: "Bike marking", color: "#9acd32", width: THICK },
  "bike_marking/dashed": { label: "Bike marking dashed", color: "#9acd32", width: THICK, pattern: "dashed", dash: [1, 1] },
  "marking/halt": { label: "Halt marking", color: "#ff4d42", width: THIN, pattern: "dashed", dash: [1, 0.2] },
  "zig-zag": { label: "Zig-zag", color: "#ff9f1a", width: THIN },
  zebra_marking: { label: "Zebra marking", color: "#4caf50", width: 0.5 },
  virtual: { label: "Virtual", color: "#9fb3c8", width: 0.05, minPx: 1, pattern: "dotted", dash: [0.3, 0.7] },

  curbstone: { label: "Curb, high", color: "#8d939a", width: 0.15,
    raise: { kind: "solid", bottom: 0, top: 0.12, color: "#a3a8ad" } },
  "curbstone/low": { label: "Curb, low", color: "#b4b9bf", width: 0.15,
    raise: { kind: "solid", bottom: 0, top: 0.03, color: "#b4b9bf" } },
  "curbstone/low/entry": { label: "Curb, sidewalk entry", color: "#b4b9bf", width: 0.15, pattern: "dashed",
    dash: [0.5, 0.5], raise: { kind: "solid", bottom: 0, top: 0.03, color: "#b4b9bf" } },
  "curbstone_regular_road/low": { label: "Curb, low (regular road)", color: "#b4b9bf", width: 0.15,
    pattern: "dashed", dash: [1, 1], raise: { kind: "solid", bottom: 0, top: 0.03, color: "#b4b9bf" } },
  road_border: { label: "Road border", color: "#4caf50", width: 0.15 },
  guard_rail: { label: "Guard rail", color: "#8fb3c0", width: 0.1,
    raise: { kind: "rail", bottom: 0, top: 0.75, color: "#a9c4cf" } },
  jersey_barrier: { label: "Jersey barrier", color: "#9aa3ad", width: 0.3,
    raise: { kind: "solid", bottom: 0, top: 0.8, color: "#b5bcc4" } },
  wall: { label: "Wall", color: "#9a8a7a", width: 0.2, raise: { kind: "solid", bottom: 0, top: 2.0, color: "#8c7d6e" } },
  fence: { label: "Fence", color: "#9acd32", width: 0.05, minPx: 1.5,
    raise: { kind: "fence", bottom: 0, top: 1.6, color: "#8fbf3f" } },
  rail: { label: "Rail track", color: "#40e0d0", width: 1.505, stripe: 0.07, pattern: "double" },
  building: { label: "Building", color: "#8a8a8a", width: 0.2, pattern: "dashed", dash: [4, 0.5] },
  visualization: { label: "Visualization", color: "#9a9a9a", width: 0.05, minPx: 1 },
  "trajectory/car": { label: "Trajectory, car", color: "#46c46a", width: 0.2, pattern: "dashed", dash: [1, 1] },
  "trajectory/bike": { label: "Trajectory, bike", color: "#ff9f1a", width: 0.2, pattern: "dashed", dash: [1, 1] },
  "trajectory/special_vehicle": { label: "Trajectory, special vehicle", color: "#5b8cff", width: 0.2, pattern: "dashed",
    dash: [1, 1] },
  "trajectory/pedestrian": { label: "Trajectory, pedestrian", color: "#9a9a9a", width: 0.2, pattern: "dashed",
    dash: [1, 1] },

  // Lines that carry an icon: kept thin, the icon says what they are.
  traffic_sign: { label: "Traffic sign", color: "#4fa3ff", width: 0.05, minPx: 1.2 },
  traffic_light: { label: "Traffic light", color: "#ffd23f", width: 0.05, minPx: 1.2 },
  traffic_light_bikes: { label: "Traffic light, bikes", color: "#ffd23f", width: 0.05, minPx: 1.2 },
  traffic_light_pedestrians: { label: "Traffic light, pedestrians", color: "#ffd23f", width: 0.05, minPx: 1.2 },
  traffic_light_misc: { label: "Traffic light, other", color: "#ffd23f", width: 0.05, minPx: 1.2 },
  arrow: { label: "Arrow marking", color: "#ffd23f", width: 0.05, minPx: 1.2 },
  symbol: { label: "Symbol marking", color: "#ffd23f", width: 0.05, minPx: 1.2 },
};

// The `color` tag (lines.mapcss: `color: tag(color)` on line_thin/thick).
const TAG_COLORS = {
  white: WHITE, yellow: "#ffd400", red: "#ff4d42", blue: "#4f8fff", green: "#4caf50", orange: "#ff9f1a",
};

/** The key of the style a way with `tags` gets: "type/subtype", "type", or "default". */
export function lineStyleKey(tags) {
  const t = (tags && tags.type) || "";
  const st = (tags && tags.subtype) || "";
  if (t === "curbstone" && st === "low" && tags.sidewalk_entry === "true") return "curbstone/low/entry";
  if (t === "curbstone" && st === "high") return "curbstone";
  if ((t === "line_thin" || t === "line_thick") && st === "solid") return t;
  if (st && LINE_STYLES[`${t}/${st}`]) return `${t}/${st}`;
  if (LINE_STYLES[t]) return t;
  return "default";
}

/** The colour a way is drawn in: its `color` tag where the style allows, else the style's. */
export function lineColor(tags, key = lineStyleKey(tags)) {
  const c = tags && tags.color;
  if (c && (key.startsWith("line_thin") || key.startsWith("line_thick"))) {
    if (TAG_COLORS[c]) return TAG_COLORS[c];
    if (/^#[0-9a-f]{6}$/i.test(c)) return c;
  }
  return LINE_STYLES[key].color;
}

/** Style keys in a fixed order: the row of each in the shader's style table. */
export const STYLE_KEYS = Object.keys(LINE_STYLES);
