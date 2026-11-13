"""Vectorised line drawing on torch image tensors (for the video renderer).

All lines of a frame are one batch of 2D segments: clipped to their rectangle, sampled at
sub-pixel steps, dashed, and stamped with a disk. Where stamps overlap, the highest
``(priority, index)`` wins, so the layering does not depend on the device.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

#: Reference depth (m) at which dashes have their nominal length (``reprojection._dashed``).
DASH_REFERENCE_DEPTH = 25.0
#: Bounds of the depth factor of the dash length (near dashes longer, far ones shorter).
DASH_FACTOR_RANGE = (0.45, 1.75)


@lru_cache(maxsize=64)
def _disk(radius: float, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel offsets ``(dx, dy)`` of a disk of *radius* (at least the centre pixel)."""
    r = int(math.ceil(radius))
    rng = torch.arange(-r, r + 1, device=device)
    dy, dx = torch.meshgrid(rng, rng, indexing="ij")
    keep = dx * dx + dy * dy <= radius * radius + 0.5
    return dx[keep], dy[keep]


def clip_segments(p0: torch.Tensor, p1: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
    """Liang-Barsky for many segments at once, each with its own rectangle ``lo``..``hi`` (S, 2).

    Returns:
        ``(t0, t1, keep)``: the kept part of segment i is ``p0 + t * (p1 - p0)`` for
        ``t0[i] <= t <= t1[i]``; ``keep`` is False where nothing is inside.
    """
    d = p1 - p0
    t0 = torch.zeros(len(p0), device=p0.device, dtype=p0.dtype)
    t1 = torch.ones(len(p0), device=p0.device, dtype=p0.dtype)
    keep = torch.ones(len(p0), dtype=torch.bool, device=p0.device)
    for p, q in ((-d[:, 0], p0[:, 0] - lo[:, 0]), (d[:, 0], hi[:, 0] - p0[:, 0]),
                 (-d[:, 1], p0[:, 1] - lo[:, 1]), (d[:, 1], hi[:, 1] - p0[:, 1])):
        par = p == 0
        keep &= ~(par & (q < 0))
        t = q / torch.where(par, torch.ones_like(p), p)
        t0 = torch.where(~par & (p < 0), torch.maximum(t0, t), t0)
        t1 = torch.where(~par & (p > 0), torch.minimum(t1, t), t1)
    keep &= t0 <= t1
    return t0, t1, keep


def project_segments(P0: torch.Tensor, P1: torch.Tensor, M: torch.Tensor, near: float = 0.1):
    """3D segments (S, 3) -> pixel segments in every camera, clipped at the near plane in 3D.

    Args:
        M: (C, 4, 4) projections whose row 2 gives the depth (``[K @ T_sensor_base[:3]; 0 0 0 1]``).
        near: Smallest depth kept (m), :data:`kitscenes.projection.MIN_DEPTH` by default.

    Returns:
        ``(p0, p1, d0, d1, keep)``: (C, S, 2) end points, (C, S) depths of the
        (clipped) ends and (C, S) whether any part of the segment is in front of the camera.
    """
    ones = torch.ones_like(P0[:, :1])
    h0 = torch.cat([P0, ones], 1) @ M.transpose(-1, -2)
    h1 = torch.cat([P1, ones], 1) @ M.transpose(-1, -2)
    w0, w1 = h0[..., 2], h1[..., 2]
    keep = (w0 >= near) | (w1 >= near)
    dw = torch.where((w1 - w0).abs() < 1e-9, torch.full_like(w0, 1e-9), w1 - w0)
    t = ((near - w0) / dw).clamp(0, 1)[..., None]
    cut = h0 + t * (h1 - h0)
    h0 = torch.where((w0 < near)[..., None], cut, h0)
    h1 = torch.where((w1 < near)[..., None], cut, h1)
    d0, d1 = h0[..., 2].clamp(min=near), h1[..., 2].clamp(min=near)
    return h0[..., :2] / d0[..., None], h1[..., :2] / d1[..., None], d0, d1, keep


def _group_start(group: torch.Tensor) -> torch.Tensor:
    """Index of the first element of each element's run of equal *group* values (runs are contiguous)."""
    n = len(group)
    idx = torch.arange(n, device=group.device)
    start = torch.ones(n, dtype=torch.bool, device=group.device)
    if n > 1:
        start[1:] = group[1:] != group[:-1]
    return torch.cummax(torch.where(start, idx, torch.zeros_like(idx)), 0)[0]


def _dash_period(depth, dash, gap, min_dash, min_gap, scaled):
    """``(period, on)`` in pixels of the dash pattern at *depth* (as ``reprojection._dashed``)."""
    lo, hi = DASH_FACTOR_RANGE
    factor = torch.where(scaled, (DASH_REFERENCE_DEPTH / depth.clamp(min=1.0)).clamp(lo, hi), torch.ones_like(depth))
    on = torch.maximum(min_dash, dash * factor)
    off = torch.maximum(min_gap, gap * factor)
    return on + off, on


class Segments:
    """A batch of 2D segments with their drawing attributes (all tensors on one device).

    Attributes:
        p0, p1: (S, 2) end points in frame pixels.
        colour: (S, 3) uint8 RGB.
        radius: (S,) disk radius in pixels (the line is about ``2 * radius`` wide).
        alpha: (S,) opacity over the image (1: opaque).
        priority: (S,) int, higher on top (0 .. 65535).
        bounds: (S, 4) ``x0, y0, x1, y1``: the rectangle the segment is drawn in (inclusive).
        group: (S,) int: the polyline each segment belongs to; segments of one
            polyline are contiguous, in order (dashes run on along them).
        dash, gap: (S,) nominal dash and gap length in pixels (dash 0: solid).
        min_dash, min_gap: (S,) lower bounds of the dash and gap length.
        depth_scaled: (S,) bool: dash lengths shrink with depth (cameras), or not (BEV).
        d0, d1: (S,) depth of the ends (only used where ``depth_scaled``).
    """

    FIELDS = ("p0", "p1", "colour", "radius", "alpha", "priority", "bounds", "group", "dash", "gap",
              "min_dash", "min_gap", "depth_scaled", "d0", "d1")

    def __init__(self, **fields) -> None:
        missing = [f for f in self.FIELDS if f not in fields]
        if missing:
            raise TypeError(f"Segments needs {missing}")
        for name in self.FIELDS:
            setattr(self, name, fields[name])

    def __len__(self) -> int:
        return len(self.p0)

    @classmethod
    def cat(cls, parts: list["Segments"], group_counts: Optional[list[int]] = None) -> Optional["Segments"]:
        """The parts in one batch, their groups kept apart (*group_counts*: an upper bound of each part's
        group ids, known on the host, saves waiting for the GPU)."""
        if group_counts is not None:
            group_counts = [c for p, c in zip(parts, group_counts) if p is not None and len(p)]
        parts = [p for p in parts if p is not None and len(p)]
        if not parts:
            return None
        offset, groups = 0, []
        for k, p in enumerate(parts):  # keep the groups of different parts apart
            if group_counts is not None:
                groups.append(p.group + offset)
                offset += int(group_counts[k])
            else:
                groups.append(p.group - p.group.min() + offset)
                offset = int(groups[-1].max()) + 1
        fields = {f: torch.cat([getattr(p, f) for p in parts]) for f in cls.FIELDS if f != "group"}
        return cls(group=torch.cat(groups), **fields)


def draw_segments(img: torch.Tensor, seg: Optional[Segments], step: float = 0.7, max_points: int = 8192) -> torch.Tensor:
    """Draw *seg* into *img* (3, H, W) uint8, in place, in one pass.

    Dashes start at the beginning of each polyline (its first segment) and
    run on along it; in cameras (``depth_scaled``) a dash at depth d is
    ``max(min_dash, dash * clip(25 / d, 0.45, 1.75))`` long (likewise the
    gaps), measured along the line, with the depth interpolated perspective-
    correctly (1/d is linear along the projected segment).
    """
    if seg is None or len(seg) == 0:
        return img
    H, W = img.shape[1:]
    dev = img.device
    full = seg.p1 - seg.p0
    length = full.norm(dim=1)
    # dash phase in cycles at the start of each segment: integral of 1 / period along the polyline
    dashed = seg.dash > 0
    inv0, inv1 = 1.0 / seg.d0, 1.0 / seg.d1
    dmid = 2.0 / (inv0 + inv1)
    g = [1.0 / _dash_period(d, seg.dash, seg.gap, seg.min_dash, seg.min_gap, seg.depth_scaled)[0]
         for d in (seg.d0, dmid, seg.d1)]
    cycles = torch.where(dashed, length * (g[0] + 4 * g[1] + g[2]) / 6, torch.zeros_like(length))
    excl = torch.cumsum(cycles, 0) - cycles
    u0 = excl - excl[_group_start(seg.group)]
    # clip to each segment's rectangle (grown by its radius); segments left out get no points (no compaction
    # here: every boolean selection on the GPU makes the host wait for it)
    r = seg.radius[:, None]
    t0, t1, keep = clip_segments(seg.p0, seg.p1, seg.bounds[:, :2] - r, seg.bounds[:, 2:] + r)
    n = torch.where(keep, torch.clamp(torch.ceil(length * (t1 - t0) / step), 1, max_points).long() + 1,
                    torch.zeros_like(length, dtype=torch.long))
    s = torch.repeat_interleave(torch.arange(len(seg), device=dev), n)  # the segment of each point (waits once)
    if len(s) == 0:
        return img
    first = torch.cumsum(n, 0) - n
    k = torch.arange(len(s), device=dev) - first[s]
    tau = k.to(t0.dtype) / (n[s] - 1).clamp(min=1).to(t0.dtype)
    t = t0[s] + tau * (t1 - t0)[s]  # parameter along the whole (unclipped) segment
    pts = seg.p0[s] + t[:, None] * full[s]
    # dashes (computed for every point: asking whether any segment is dashed would wait for the GPU)
    a, b, c = g[0][s], g[1][s], g[2][s]
    u = u0[s] + length[s] * (a * t + (-3 * a + 4 * b - c) * t * t / 2 + (2 * a - 4 * b + 2 * c) * t * t * t / 3)
    depth = 1.0 / ((1 - t) * inv0[s] + t * inv1[s])
    period, on = _dash_period(depth, seg.dash[s], seg.gap[s], seg.min_dash[s], seg.min_gap[s], seg.depth_scaled[s])
    on_mask = ~dashed[s] | ((u - torch.floor(u)) * period < on)
    # stamp disks; radii in a few buckets (one disk per bucket, cut to each point's own radius): the points
    # sorted by bucket, split by the bucket sizes (one wait), all selections in one compaction at the end
    radius = seg.radius[s]
    bucket = torch.ceil(radius).clamp(min=1).long()
    order = torch.argsort(bucket)
    sizes = torch.bincount(bucket, minlength=1)
    counts = sizes.tolist()
    flats, keys, oks = [], [], []
    start = 0
    for R, count in enumerate(counts):
        if not count:
            continue
        idx = order[start:start + count]
        start += count
        dx, dy = _disk(float(R), str(dev))
        q, sg, rq = pts[idx], s[idx], radius[idx]
        inside = (dx * dx + dy * dy)[None].float() <= rq[:, None] * rq[:, None] + 0.5
        X = (torch.floor(q[:, 0] + 0.5).long()[:, None] + dx[None])
        Y = (torch.floor(q[:, 1] + 0.5).long()[:, None] + dy[None])
        sg = sg[:, None].expand_as(X)
        bd = seg.bounds[sg]
        ok = inside & on_mask[idx][:, None] & (X >= bd[..., 0]) & (X <= bd[..., 2]) & (Y >= bd[..., 1]) & \
            (Y <= bd[..., 3]) & (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        flats.append((Y * W + X).reshape(-1))
        keys.append((seg.priority[sg].long() * (1 << 24) + sg).reshape(-1))
        oks.append(ok.reshape(-1))
    keep_px = torch.cat(oks).nonzero().flatten()
    flat, key = torch.cat(flats)[keep_px], torch.cat(keys)[keep_px]
    if len(flat) == 0:
        return img
    if H * W < (1 << 23):  # one sort on (pixel, key)
        order = torch.argsort(flat * (1 << 40) + key)
    else:  # two stable sorts: by key, then by pixel
        order = torch.argsort(key)
        order = order[torch.sort(flat[order], stable=True).indices]
    flat, key = flat[order], key[order]
    last = torch.ones_like(flat, dtype=torch.bool)
    last[:-1] = flat[1:] != flat[:-1]
    flat, winner = flat[last], key[last] % (1 << 24)
    colour = seg.colour[winner].float()
    alpha = seg.alpha[winner][:, None]
    view = img.view(3, -1)
    under = view[:, flat].t().float()
    out = (alpha * colour + (1 - alpha) * under).round_().clamp_(0, 255).to(torch.uint8)
    view[:, flat] = out.t()
    return img


def polylines_to_segments(polylines, closed: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N_i, D) arrays -> ``(P0, P1, line_index)`` numpy arrays of their segments (closed: back to the start)."""
    P0, P1, idx = [], [], []
    for i, pl in enumerate(polylines):
        pl = np.asarray(pl, np.float32)
        if closed and len(pl) > 2 and not np.array_equal(pl[0], pl[-1]):
            pl = np.concatenate([pl, pl[:1]])
        if len(pl) == 1:
            pl = np.concatenate([pl, pl])
        if len(pl) < 2:
            continue
        P0.append(pl[:-1])
        P1.append(pl[1:])
        idx.append(np.full(len(pl) - 1, i, np.int64))
    if not P0:
        dims = np.asarray(polylines[0]).shape[-1] if len(polylines) else 3
        return np.zeros((0, dims), np.float32), np.zeros((0, dims), np.float32), np.zeros(0, np.int64)
    return np.concatenate(P0), np.concatenate(P1), np.concatenate(idx)


def arrow_heads(polylines, length: float, angle_deg: float = 28.0) -> tuple[np.ndarray, np.ndarray]:
    """The two wing segments at the end of each 2D polyline (direction of its last segment)."""
    a0, a1 = [], []
    ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    for pl in polylines:
        pl = np.asarray(pl, np.float32)
        if len(pl) < 2:
            continue
        d = pl[-1] - pl[-2]
        norm = float(np.linalg.norm(d))
        if norm < 1e-6:
            continue
        d = -d / norm * length
        for s in (sa, -sa):
            a0.append(pl[-1])
            a1.append(pl[-1] + np.array([ca * d[0] - s * d[1], s * d[0] + ca * d[1]], np.float32))
    if not a0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    return np.stack(a0), np.stack(a1)


def rgb_to_nv12(img: torch.Tensor) -> torch.Tensor:
    """(3, H, W) uint8 RGB (H, W even) -> NV12 (H * 3 / 2, W) uint8, BT.709 limited range."""
    x = img.float() / 255.0
    r, g, b = x[0], x[1], x[2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556
    cr = (r - y) / 1.5748
    Y = (16 + 219 * y).round_().clamp_(0, 255).to(torch.uint8)
    uv = F.avg_pool2d(torch.stack([cb, cr])[None], 2)[0]
    U = (128 + 224 * uv[0]).round_().clamp_(0, 255).to(torch.uint8)
    V = (128 + 224 * uv[1]).round_().clamp_(0, 255).to(torch.uint8)
    h, w = Y.shape
    return torch.cat([Y, torch.stack([U, V], dim=-1).reshape(h // 2, w)], 0)


def paste_rgba(img: torch.Tensor, icon: torch.Tensor, x0: int, y0: int, alpha: float,
               rect: Optional[tuple[int, int, int, int]] = None) -> None:
    """Blend an RGBA icon (4, h, w) uint8 with its top-left corner at (x0, y0) into *img*, in place.

    *rect* ``(x0, y0, x1, y1)`` (exclusive end) limits where it is drawn (its tile).
    """
    H, W = img.shape[1:]
    X0, Y0, X1, Y1 = rect if rect is not None else (0, 0, W, H)
    h, w = icon.shape[1:]
    a0, b0 = max(x0, X0, 0), max(y0, Y0, 0)
    a1, b1 = min(x0 + w, X1, W), min(y0 + h, Y1, H)
    if a1 <= a0 or b1 <= b0:
        return
    part = icon[:, b0 - y0:b1 - y0, a0 - x0:a1 - x0].float()
    a = part[3:4] / 255.0 * alpha
    region = img[:, b0:b1, a0:a1].float()
    img[:, b0:b1, a0:a1] = (a * part[:3] + (1 - a) * region).round_().clamp_(0, 255).to(torch.uint8)
