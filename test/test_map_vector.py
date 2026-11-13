"""The geometry behind the vector-map label options (kitscenes.map._vector, numpy only)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kitscenes.map import _lanelet2, _vector


@dataclasses.dataclass
class _Lane:  # the fields of kitscenes.map.Lane that crossings() reads
    id: int
    left: int
    right: int
    left_points: np.ndarray
    right_points: np.ndarray

    @property
    def polygon(self) -> np.ndarray:
        return np.concatenate([self.left_points, self.right_points[::-1]])


def _crosswalk(lane_id: int, x0: float, x1: float, y0: float, y1: float, z: float = 100.0, left: int = 0) -> _Lane:
    """A rectangular crosswalk lanelet walked in +x: left bound at y1, right bound at y0."""
    return _Lane(lane_id, left or lane_id * 10, lane_id * 10 + 1,
                 np.array([[x0, y1, z], [x1, y1, z]]), np.array([[x0, y0, z], [x1, y0, z]]))


def test_crosswalks_that_share_an_edge_are_one_crossing() -> None:
    # 1 and 2 continue each other (a crossing split at x = 4); 3 touches 2 at one corner only; 4 is apart
    lanes = [_crosswalk(1, 0, 4, 0, 3), _crosswalk(2, 4, 9, 0, 3), _crosswalk(3, 9, 12, 3, 8), _crosswalk(4, 20, 24, 0, 3)]
    found = {c.lane_ids: c for c in _vector.crossings(lanes, zebra_lines={30})}
    assert set(found) == {(1, 2), (3,), (4,)}
    merged = found[(1, 2)]
    assert _vector.signed_area(merged.ring) == pytest.approx(27.0)  # counter-clockwise, 9 x 3
    assert len(merged.ring) == 6  # the shared edge is gone, its two nodes stay on the outline
    assert not merged.zebra and found[(3,)].zebra
    np.testing.assert_allclose(merged.lo, [0, 0])
    np.testing.assert_allclose(merged.hi, [9, 3])


def test_crosswalks_that_overlap_stay_apart() -> None:
    """Outlines that are not one simple ring after merging (here: one lane on top of another) are not merged."""
    lanes = [_crosswalk(1, 0, 4, 0, 3), _crosswalk(2, 0, 4, 0, 3)]
    assert sorted(c.lane_ids for c in _vector.crossings(lanes, set())) == [(1,), (2,)]


def test_clip_ring_interpolates_z_and_close_ring() -> None:
    ring = np.array([[-5.0, -1, 0], [5, -1, 10], [5, 1, 10], [-5, 1, 0]])  # rises 1 m per metre in x
    clipped = _vector.clip_ring(ring, 2.0, 15.0)
    assert _vector.signed_area(clipped) == pytest.approx(8.0)
    np.testing.assert_allclose(clipped[:, 2], clipped[:, 0] + 5.0)  # z along the cut edges
    closed = _vector.close_ring(clipped)
    assert np.array_equal(closed[0], closed[-1]) and len(closed) == len(clipped) + 1
    np.testing.assert_allclose(closed[0, :2], [-2, -1])  # starts at the smallest (x, y)
    assert len(_vector.clip_ring(ring + [100, 0, 0], 2.0, 15.0)) == 0


def test_cut_to_band() -> None:
    line = np.column_stack([np.arange(11.0), np.zeros(11)])
    z = np.array([0, 0, 1, 3, 5, 7, 5, 3, 1, 0, 0], dtype=float)
    pieces = _vector.cut_to_band(np.column_stack([line, z]), z, -1.0, 4.0)
    assert len(pieces) == 2
    np.testing.assert_allclose(pieces[0][-1], [3.5, 0, 4.0])  # cut where the line leaves the band
    np.testing.assert_allclose(pieces[1][0], [6.5, 0, 4.0])
    assert all((p[:, 2] <= 4.0).all() for p in pieces)
    assert _vector.cut_to_band(line, np.zeros(11), -1.0, 1.0)[0] is line  # all inside: untouched
    assert _vector.cut_to_band(line[:2], np.array([0.0, 2.0]), -1.0, 1.0001)[0].shape == (2, 2)
    tiny = _vector.cut_to_band(line[:2], np.array([0.0, 100.0]), -1.0, 1.0)  # 0.01 m inside: dropped
    assert tiny == []


def test_arc_position_and_keys() -> None:
    u = np.array([[-40.0, 5], [40, 5], [40, -5], [-40, -5]])
    assert _vector.arc_position(u, [-30, 5]) == pytest.approx(10.0)
    assert _vector.arc_position(u, [30, -5]) == pytest.approx(80 + 10 + 10)
    key = _vector.instance_key("divider", (1, 2), 0)
    assert key == _vector.instance_key("divider", [1, 2], 0)  # the same in every process (no hash seed)
    assert len({key, _vector.instance_key("divider", (1, 2), 1), _vector.instance_key("boundary", (1, 2), 0),
                _vector.instance_key("divider", (12,), 0)}) == 4
    assert key == -2515768836991503790  # blake2b-64 of "divider|1,2|0": a format change shows here


def test_reusable_map_data_by_version() -> None:
    """Scene chaining needs the converter of lanelet2 1.2.3+kitscenes2 (repeated MapData.processAll)."""
    assert not _lanelet2.has_reusable_map_data("1.2.3+kitscenes1")
    assert _lanelet2.has_reusable_map_data("1.2.3+kitscenes2")
    assert _lanelet2.has_reusable_map_data("1.2.3+kitscenes12")
    assert _lanelet2.has_reusable_map_data("1.2.4+kitscenes1")  # a later base version has the fix
    assert not _lanelet2.has_reusable_map_data("1.2.3")
    assert not _lanelet2.has_reusable_map_data("1.2.2+kitscenes5")
