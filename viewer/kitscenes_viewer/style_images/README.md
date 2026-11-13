# Traffic element icons

PNG/SVG assets for HD map visualization, served by the viewer at `/style_images/<file>`.

`server.py --icons-dir DIR` serves another directory instead.

`kitscenes.visualization` (camera reprojection, top-down view) does not read this
directory: it ships its own copy of the icons it needs in `kitscenes/visualization/icons/`
(mapping in `kitscenes.visualization.style.ICONS`), with the three `traffic_light_*.svg`
rendered to PNG so it needs no SVG renderer. Keep both copies in sync when an icon changes.
`$KITSCENES_TE_ICON_PATH` overrides the kitscenes copy.

Generic fallbacks used when a specific sign file is missing:

- `traffic_light.png` — traffic lights
- `traffic_sign.png` — signs and unknown types
