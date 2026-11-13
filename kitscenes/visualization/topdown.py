"""The ML converter's top-down view of map labels (matplotlib).

Road borders, lane dividers (a grey underlay tells solid and dashed apart),
centerlines, crosswalks, stop lines and traffic element icons, with the
edges between traffic elements and to the lanes they apply to, around the
ego car::

    from kitscenes.map import MLConverter, Region
    from kitscenes.visualization import plot_top_down

    labels = MLConverter.for_scene(scene, region=Region(60, 60, 30, 30))(frame.pose)
    fig, ax = plot_top_down(labels)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np

from kitscenes.visualization import _deps, style

if TYPE_CHECKING:
    from kitscenes.map import MapLabels

#: Size of the traffic element icons (metres).
ICON_SIZE = 3.0
#: Edges longer than this fraction of the view are left out (they cross the whole plot).
MAX_EDGE_FRACTION = 0.3


def plot_top_down(
    labels: "MapLabels",
    ax=None,
    *,
    orientation: str = "vertical",
    extent: Optional[tuple[float, float]] = None,
    show_edges: bool = True,
    show_legend: bool = True,
    exclude_te_ids: Iterable[int] = (),
    show_untyped_signs: bool = False,
):
    """Draw *labels* from above.

    Args:
        orientation: ``"vertical"`` (driving up) or ``"horizontal"`` (driving right).
        extent: ``(longitudinal, lateral)`` half extents shown, in metres
            (default: exactly the labels' region, also when it is asymmetric).
        show_edges: Draw the edges between traffic elements and to their lanes
            (those longer than 0.3 of the view are left out).
        show_legend: Draw the legend.
        exclude_te_ids: Traffic elements not to draw, by map id.
        show_untyped_signs: Also draw the traffic signs without a type
            (:attr:`~kitscenes.map.MapLabels.untyped_sign_ids`).

    Returns:
        ``(fig, ax)``.
    """
    plt = _deps.matplotlib("pyplot")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 20) if orientation == "vertical" else (20, 10))
    else:
        fig = ax.figure
    xy = _orient(orientation)
    if extent is None:
        r = labels.region
        ahead, behind, left, right = r.ahead, r.behind, r.left, r.right
    else:
        ahead = behind = float(extent[0])
        left = right = float(extent[1])
    # vertical: driving up, left of the car on the left of the plot; horizontal: driving right
    xlim, ylim = ((-left, right), (-behind, ahead)) if orientation == "vertical" else ((-behind, ahead), (-right, left))
    present: list[str] = []
    lines = labels.lines

    _car(ax, orientation)
    for name in style.BORDER_TYPES:
        for line in lines.get(name, []):
            ax.plot(*xy(line), color=style.LINE_COLOURS[name], linewidth=7.0, zorder=5)
        present += [name] * bool(lines.get(name))
    for name in style.DIVIDER_TYPES:
        for line in lines.get(name, []):
            if name == "Dashed":
                ax.plot(*xy(line), color="gray", linewidth=12.0, linestyle=(0, (5, 5)), zorder=5)
            elif name in ("Solid", "SolidSolid"):
                ax.plot(*xy(line), color="gray", linewidth=12.0, zorder=5)
            ax.plot(*xy(line), color=style.LINE_COLOURS.get(name, "purple"), linewidth=3.0, zorder=5)
        present += [name] * bool(lines.get(name))
    for line in lines.get("Centerline", []):
        ax.plot(*xy(line), color=style.LINE_COLOURS["Centerline"], linestyle="dashed", linewidth=2.0, zorder=4)
    present += ["Centerline"] * bool(lines.get("Centerline"))
    for line in lines.get("BikeCenterline", []):
        ax.plot(*xy(line), color=style.LINE_COLOURS["BikeCenterline"], linestyle="dotted", linewidth=2.5, zorder=4)
    present += ["BikeCenterline"] * bool(lines.get("BikeCenterline"))
    for name in style.CROSSING_TYPES:
        for line in lines.get(name, []):
            ax.plot(*xy(style.crossing_outline(line)), color=style.LINE_COLOURS[name], linewidth=5.0, zorder=6)
        present += [name] * bool(lines.get(name))

    excluded = frozenset(exclude_te_ids)
    if not show_untyped_signs:
        excluded |= labels.untyped_sign_ids
    for name, elements in labels.traffic_elements.items():
        for element, map_id in zip(elements, labels.traffic_element_ids[name]):
            if map_id in excluded:
                continue
            if name == "StopLine":
                ax.plot(*xy(element), color="red", linewidth=4.0, zorder=7)
                present.append("StopLine")
            else:
                _icon(ax, name, element[:, :2].mean(axis=0), ICON_SIZE, xy)
    if show_edges:
        _edges(ax, labels, xy, xlim, ylim, excluded)
    if show_legend:
        _legend(ax, dict.fromkeys(present), "lower right" if orientation == "vertical" else "upper right")

    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    return fig, ax


def render_top_down(labels: "MapLabels", *, dpi: int = 100, **kwargs) -> np.ndarray:
    """:func:`plot_top_down` as an RGB image (uint8)."""
    import io

    matplotlib = _deps.matplotlib()

    matplotlib.use("Agg")
    plt = _deps.matplotlib("pyplot")
    from PIL import Image

    fig, _ = plot_top_down(labels, **kwargs)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _orient(orientation: str):
    if orientation == "vertical":
        return lambda pts: (-np.asarray(pts)[..., 1], np.asarray(pts)[..., 0])
    if orientation == "horizontal":
        return lambda pts: (np.asarray(pts)[..., 0], np.asarray(pts)[..., 1])
    raise ValueError("orientation must be 'vertical' or 'horizontal'")


def _car(ax, orientation: str) -> None:
    path = style.car_image()
    if path is None:
        return
    from PIL import Image

    image = Image.open(path)
    image = image.rotate(-90 if orientation == "vertical" else 180, expand=True)
    scale = 600
    ax.imshow(image, extent=[-image.width / scale, image.width / scale, -image.height / scale, image.height / scale],
              zorder=10)


def _icon(ax, te_type: str, centre: np.ndarray, size: float, xy) -> None:
    icon = style.icon_rgba(te_type)
    cx, cy = xy(centre)
    if icon is None:
        ax.plot(cx, cy, marker="o", color="gray", markersize=6, markeredgecolor="black", markeredgewidth=0.5, zorder=8)
        return
    h, w = icon.shape[:2]
    width, height = (size, size * h / w) if w > h else (size * w / h, size)
    ax.imshow(icon, extent=[cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2], zorder=8, alpha=0.9)


def _edges(ax, labels: "MapLabels", xy, xlim, ylim, excluded: frozenset) -> None:
    longest = MAX_EDGE_FRACTION * max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    def draw(source, target, width: float) -> None:
        (sx, sy), (tx, ty) = xy(source), xy(target)
        inside = all(xlim[0] <= x <= xlim[1] for x in (sx, tx)) and all(ylim[0] <= y <= ylim[1] for y in (sy, ty))
        if inside and np.hypot(tx - sx, ty - sy) <= longest:
            ax.annotate("", xy=(tx, ty), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color="darkgray", lw=width, alpha=0.7), zorder=1)

    def element(te_type: str, index: int) -> Optional[np.ndarray]:
        items = labels.traffic_elements.get(te_type, [])
        if index >= len(items) or labels.traffic_element_ids[te_type][index] in excluded:
            return None
        return items[index][:, :2]

    for a, i, b, j in labels.te_to_te:
        if a in style.ARROW_TYPES or b in style.ARROW_TYPES:
            continue
        source, target = element(a, i), element(b, j)
        if source is not None and target is not None:
            draw(source.mean(axis=0), target.mean(axis=0), 3)
    centerlines = labels.lines.get("Centerline", [])
    for te_type, i, k in labels.te_to_centerline:
        source = element(te_type, i)
        if te_type in style.ARROW_TYPES or source is None or k >= len(centerlines):
            continue
        centre = source.mean(axis=0)
        draw(centre, _point_in_band(centerlines[k][:, :2], centre, 3.0, 5.0), 2)


def _point_in_band(line: np.ndarray, source: np.ndarray, near: float, far: float) -> np.ndarray:
    """The first point along *line* between *near* and *far* metres from *source*, else its closest point."""
    dense = _densify(line, 0.1)
    d = np.linalg.norm(dense - source, axis=1)
    inside = np.flatnonzero((d >= near) & (d <= far))
    return dense[inside[0]] if len(inside) else dense[int(np.argmin(d))]


def _densify(line: np.ndarray, step: float) -> np.ndarray:
    from kitscenes.map.sampling import Spacing, resample

    return resample(line, Spacing(step)) if len(line) > 1 else line


def _legend(ax, present: dict, loc: str) -> None:
    Line2D = _deps.matplotlib("lines").Line2D

    names = style.LEGEND_NAMES
    handles, texts = [], []
    for name in names:
        if name not in present:
            continue
        colour = "red" if name == "StopLine" else style.LINE_COLOURS.get(name, "purple")
        dash = "--" if name == "Centerline" else ":" if name == "BikeCenterline" else "-"
        line = Line2D([0], [0], color=colour, linestyle=dash, linewidth=3 if name in style.BORDER_TYPES else 2)
        if name in ("Dashed", "Solid", "SolidSolid", "SolidDashed", "DashedSolid"):
            under = Line2D([0], [0], color="gray", linewidth=8.0, linestyle=(0, (1, 1)) if name == "Dashed" else "-")
            handles.append((under, line))
        else:
            handles.append(line)
        texts.append(names[name])
    if handles:
        ax.legend(handles, texts, loc=loc, fontsize=15, framealpha=0.9, edgecolor="black").set_zorder(100)
