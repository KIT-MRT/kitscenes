"""Trajectory matching: an HMM (Newson & Krumm 2009) over road candidates, solved with Viterbi.

Internal; see :meth:`kitscenes.sdmap.SDMap.match_trajectory`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from kitscenes.sdmap._graph import RoadGraph


def key_poses(xy: np.ndarray, step_m: float) -> np.ndarray:
    """Indices of the poses used for matching: the first, then each one at least *step_m* from the last kept, and the last."""
    keep = [0]
    last = xy[0]
    for i in range(1, len(xy)):
        if np.hypot(*(xy[i] - last)) >= step_m:
            keep.append(i)
            last = xy[i]
    if keep[-1] != len(xy) - 1:
        keep.append(len(xy) - 1)
    return np.array(keep, dtype=np.int64)


def viterbi(
    g: RoadGraph, xy: np.ndarray, yaw: Optional[np.ndarray], *, sigma_m: float, max_distance: float,
    max_heading_error: float, beta_m: float, public_only: bool = False, uturn_penalty_m: float = 30.0,
) -> list[tuple[list[int], list[tuple[int, float]]]]:
    """The most likely road positions ``(road, s)`` of the poses *xy*, as pieces of consecutive matched poses:
    ``[(pose indices, positions), ...]``.

    Emission: Gaussian in the distance to the road (*sigma_m*) and in the
    heading error (0.35 rad). Transition: exponential in the difference
    between the driving distance and the straight distance of two consecutive
    poses (*beta_m*); a U-turn at a graph node costs *uturn_penalty_m* more. A pose without candidates, or without any reachable
    candidate, ends a piece; the next piece starts after it.
    """
    cands: list[list[tuple[int, float, float, float]]] = []
    for i in range(len(xy)):
        h = None if yaw is None else float(yaw[i])
        cands.append(g.snap(xy[i], h, max_distance, max_heading_error, public_only=public_only)[:12])

    def emission(c) -> float:
        _, _, d, err = c
        return -0.5 * (d / sigma_m) ** 2 - 0.5 * (err / 0.35) ** 2

    pieces: list[tuple[list[int], list[tuple[int, float]]]] = []
    piece_idx: list[int] = []
    score: list[float] = []
    back: list[list[int]] = []
    layers: list[list] = []

    def close() -> None:
        if not layers:
            return
        j = int(np.argmax(score))
        out = []
        for k in range(len(layers) - 1, -1, -1):
            r, s, _, _ = layers[k][j]
            out.append((piece_idx[k], r, s))
            j = back[k][j]
        out.reverse()
        pieces.append(([i for i, _, _ in out], [(r, s) for _, r, s in out]))

    prev_xy = None
    for i, layer in enumerate(cands):
        if not layer:
            close()
            layers, back, score, piece_idx = [], [], [], []
            prev_xy = None
            continue
        if not layers:
            layers = [layer]
            back = [[-1] * len(layer)]
            score = [emission(c) for c in layer]
            piece_idx = [i]
            prev_xy = xy[i]
            continue
        straight = float(np.hypot(*(xy[i] - prev_xy)))
        bound = 2.0 * straight + 4 * max_distance + 20.0
        prev = layers[-1]
        trans = np.full((len(prev), len(layer)), -np.inf)
        for a, (ra, sa, _, _) in enumerate(prev):
            dist = g.distances_from(ra, sa, bound, uturn_penalty=uturn_penalty_m)
            for b, (rb, sb, _, _) in enumerate(layer):
                if rb == ra and sb >= sa - 1.0:
                    route = max(sb - sa, 0.0)
                elif rb in dist:
                    route = dist[rb] + sb
                else:
                    continue
                trans[a, b] = -abs(route - straight) / beta_m
        total = np.asarray(score)[:, None] + trans
        best_prev = np.argmax(total, axis=0)
        best = total[best_prev, np.arange(len(layer))]
        if not np.isfinite(best).any():
            close()
            layers = [layer]
            back = [[-1] * len(layer)]
            score = [emission(c) for c in layer]
            piece_idx = [i]
            prev_xy = xy[i]
            continue
        layers.append(layer)
        back.append(best_prev.tolist())
        score = [float(best[b]) + emission(layer[b]) for b in range(len(layer))]
        piece_idx.append(i)
        prev_xy = xy[i]
    close()
    return pieces


def connect(g: RoadGraph, positions: list[tuple[int, float]], uturn_penalty_m: float = 30.0
            ) -> Optional[tuple[list[int], float, float]]:
    """The road sequence through consecutive matched positions: ``(roads, start_s, end_s)``, None if a gap stays."""
    roads = [positions[0][0]]
    for (ra, sa), (rb, sb) in zip(positions[:-1], positions[1:]):
        if rb == ra and sb >= sa - 1.0:
            continue
        found = g.shortest_path(ra, sa, rb, sb, restrictions=False, max_cost=2000.0, uturn_penalty=uturn_penalty_m)
        if found is None:
            return None
        path = found[1]
        roads.extend(path[1:])
    # drop immediate back-and-forth repeats (a -> b -> a) that matching noise can create
    clean: list[int] = []
    for r in roads:
        if clean and clean[-1] == r:
            continue
        clean.append(r)
    return clean, float(positions[0][1]), float(positions[-1][1])
