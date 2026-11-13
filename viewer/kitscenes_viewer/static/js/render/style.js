// Colours and traffic-element icons by Lanelet2 tag. Loose, for orientation;
// not MapCSS parity.

export const VIEWPORT_COLOR = 0x9aa3af; // gray outline of the JOSM viewport on the floor
export const EDIT_COLOR = 0x46c46a;
export const SELECT_COLOR = 0xffd23f;
export const HOVER_COLOR = 0x3ff0ff; // what a click at the pointer would select
// Members of a selected relation (a traffic light rule's light and stop line):
// red over a dark edge, so they stand out on point clouds where yellow drowns.
export const MEMBER_COLOR = 0xff2440;
export const MEMBER_EDGE_COLOR = 0x000000;
export const PIVOT_COLOR = 0x4fc3f7;
export const ZOOM_COLOR = 0x9dffc0;

// --- traffic-element icons (style_images, served by the Python server) -----
const ARROW_ICONS = {
  straight: "pf-g.png",
  left: "pf-l.png",
  right: "pf-r.png",
  left_right: "pf-lr.png",
  straight_left: "pf-gl.png",
  straight_right: "pf-gr.png",
};
const SYMBOL_ICONS = {
  bicycle: "bike.png",
  bus: "bus.png",
  "30": "30.png",
  "50": "50.png",
  "70": "70.png",
};

export function isFacedIconType(type) {
  return type === "traffic_sign" || (type && type.indexOf("traffic_light") === 0);
}

export function iconSpec(feature) {
  const tags = feature.tags || {};
  const t = tags.type || "";
  const st = tags.subtype || "";
  const bike = tags["participant:bicycle"];

  if (t === "arrow") {
    const file = ARROW_ICONS[st];
    return file ? { file } : null;
  }
  if (t === "symbol") {
    const file = SYMBOL_ICONS[st];
    return file ? { file } : null;
  }
  if (t === "traffic_light_bikes") return { file: "traffic_light_bikes.svg" };
  if (t === "traffic_light_pedestrians") return { file: "traffic_light_pedestrians.svg" };
  if (t === "traffic_light_misc") return { file: "traffic_light_misc.svg" };
  if (t === "traffic_light") {
    return { file: bike === "yes" ? "traffic_light_bikes.svg" : "traffic_light.png" };
  }
  if (t === "traffic_sign") {
    if (st && (st.indexOf("us") === 0 || st.indexOf("se") === 0)) return { file: "traffic_sign.png" };
    if (st && st.indexOf("de") === 0) {
      const id = st.slice(2).replace(/_/g, ".");
      if (id) return { file: id + ".png" };
    }
    return { file: "traffic_sign.png" };
  }
  return null;
}
