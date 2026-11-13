"""Lidar rows: the stored row index, the scan layouts, range images and the no-return opt-in."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kitscenes import Scene
from kitscenes._fusion import fuse_point_clouds
from kitscenes.errors import LabelsUnavailable
from kitscenes.points import ScanLayout, scan_layout
from release_fixture import make_scene, write_ground_labels, write_lidar, write_organized_lidar

OT128, XT32_LEFT, VELODYNE = 7, 1, 0


def _scene(tmp_path: Path, sensor_id: int, **kwargs) -> tuple[Scene, np.ndarray]:
    scene_dir = make_scene(tmp_path / "data" / "train" / "s", cameras=(), radars=(), lidars=("lidar_top",))
    slots = write_organized_lidar(scene_dir, "lidar_top", 1, sensor_id=sensor_id, **kwargs)
    return Scene(scene_dir, split="train"), slots


# ---------------------------------------------------------------------------
# ScanLayout
# ---------------------------------------------------------------------------


def test_layout_from_metadata() -> None:
    ot = scan_layout(7, 7200, 128, 921_600)
    assert ot == ScanLayout("Hesai OT128", "column-major", 128, 3600, 2, -0.1)
    assert ot.num_rows == 921_600 and ot.shape == (2, 128, 3600)
    xt = scan_layout(2, 3008, 32, 96_256)
    assert xt.model == "Hesai PandarXT-32" and (xt.columns, xt.returns, xt.azimuth_step_deg) == (1504, 2, -0.18)
    vel = scan_layout(0, 1708, 128, 218_624)
    assert vel.order == "ring-major" and vel.shape == (1, 128, 1708)
    assert vel.azimuth_step_deg == pytest.approx(-0.2108, abs=1e-4)
    assert scan_layout(5, 88_829, 1, 88_829) is None  # Seyond: unorganized
    assert scan_layout(7, 7200, 128, 921_599) is None  # metadata does not match the rows
    assert scan_layout(None, 7200, 128, 921_600) is None


@pytest.mark.parametrize("layout", [ScanLayout("h", "column-major", 4, 5, 2, -1.0),
                                    ScanLayout("v", "ring-major", 4, 5, 1, -1.0)])
def test_position_and_row_are_inverse(layout: ScanLayout) -> None:
    rows = np.arange(layout.num_rows)
    ring, column, ret = layout.position(rows)
    np.testing.assert_array_equal(layout.row(ring, column, ret), rows)
    assert ring.max() == 3 and column.max() == 4 and ret.max() == layout.returns - 1
    if layout.order == "column-major":
        np.testing.assert_array_equal(ring, rows % 4)  # the Hesai store the rings innermost
    else:
        np.testing.assert_array_equal(ring, rows // 5)


# ---------------------------------------------------------------------------
# Row index and range images on stored sweeps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sensor_id", [OT128, XT32_LEFT, VELODYNE])
def test_range_image_follows_the_layout(tmp_path: Path, sensor_id: int) -> None:
    scene, slots = _scene(tmp_path, sensor_id)
    cloud = scene.frame(1).lidar("lidar_top", frame="sensor", deskew=False, rows=True)
    assert cloud.layout is not None and cloud.layout.shape == ((2 if sensor_id else 1), 4, 5)
    np.testing.assert_array_equal(cloud.attrs["row"], np.arange(len(slots)))
    ring, column, ret = cloud.scan_position()
    np.testing.assert_array_equal(np.column_stack([ring, column, ret]), slots)
    np.testing.assert_array_equal(ring, cloud.attrs["ring"])  # the stored ring column agrees

    image = cloud.range_image()
    r, c, k = np.meshgrid(np.arange(4), np.arange(5), np.arange(cloud.layout.returns), indexing="ij")
    np.testing.assert_allclose(image[k, r, c], 0.005 * (2000 + 1000 * r + 2 * c + k), atol=1e-5)
    rings = cloud.range_image("ring")
    assert (rings == np.arange(4)[None, :, None]).all()  # image row = ring, ring 0 on top

    # The trap: pcl_width x pcl_height reshaped as an organized PCL cloud.
    naive = np.asarray(cloud.attrs["ring"]).reshape(4, -1)
    if cloud.layout.order == "column-major":
        assert not (naive == np.arange(4)[:, None]).all()
    else:
        assert (naive == np.arange(4)[:, None]).all()


def test_range_image_needs_the_sensor_frame_and_a_layout(tmp_path: Path) -> None:
    scene, _ = _scene(tmp_path, OT128)
    f = scene.frame(1)
    with pytest.raises(ValueError, match="sensor frame"):
        f.lidar("lidar_top").range_image()
    assert f.lidar("lidar_top").range_image("reflectivity").shape == (2, 4, 5)  # attributes in any frame
    with pytest.raises(ValueError, match="no scan layout"):
        f.fused_lidar().range_image("reflectivity")  # fused
    write_lidar(scene.files.path, "lidar_top", [2])  # unorganized, like the Seyond lidars
    unorganized = scene.frame(2).lidar("lidar_top")
    assert unorganized.layout is None
    with pytest.raises(ValueError, match="no scan layout"):
        unorganized.scan_position()


# ---------------------------------------------------------------------------
# No-returns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sensor_id", [OT128, VELODYNE])
def test_no_returns_are_dropped_by_default_and_kept_on_request(tmp_path: Path, sensor_id: int) -> None:
    scene, slots = _scene(tmp_path, sensor_id, no_return_slots=[(0, 1), (2, 3)], empty_slots=[(1, 4)])
    f = scene.frame(1)
    returns = 2 if sensor_id else 1
    no_return_rows = np.flatnonzero(np.isin(slots[:, 0] * 10 + slots[:, 1], [1, 23]))
    assert len(no_return_rows) == 2 * returns

    stored = f.lidar("lidar_top", rows=True)
    assert len(stored) == len(slots) - len(no_return_rows) and not stored.no_return.any()
    np.testing.assert_array_equal(stored.attrs["row"], np.setdiff1d(np.arange(len(slots)), no_return_rows))
    assert (~stored.measured).sum() == returns  # the empty slot is kept, unmeasured
    image = stored.range_image("reflectivity")
    assert np.isnan(image[:, 0, 1]).all() and (image[:, 1, 4] == 0).all()  # dropped slot: fill; empty slot: 0

    default = f.lidar("lidar_top")  # the measured points only
    assert default.measured.all() and len(default) == int(stored.measured.sum())
    np.testing.assert_array_equal(default.attrs["row"], stored.attrs["row"][stored.measured])
    np.testing.assert_array_equal(default.xyz, stored.xyz[stored.measured])
    assert np.isnan(default.range_image("reflectivity")[:, 1, 4]).all()  # range images work from attrs["row"]
    with pytest.raises(ValueError, match="rows=True"):
        f.lidar("lidar_top", no_returns=True)
    with pytest.raises(ValueError, match="rows=True"):
        f.ground_mask("lidar_top", no_returns=True)

    full = f.lidar("lidar_top", rows=True, no_returns=True)
    assert len(full) == len(slots)
    np.testing.assert_array_equal(full.attrs["row"], np.arange(len(slots)))
    np.testing.assert_array_equal(np.flatnonzero(full.no_return), no_return_rows)
    assert "no_return=" in repr(full)
    expected_t = f.t_s + slots[no_return_rows, 1] * 1e-4  # their firing times are kept
    np.testing.assert_allclose(full.attrs["timestamp"][full.no_return], expected_t)
    assert (full.range_image("reflectivity")[:, 0, 1] == -1).all()

    # Never moved: at the origin in every frame, and dropped by fusion.
    for frame in ("base", "sensor", "road", "world"):
        cloud = f.lidar("lidar_top", frame=frame, rows=True, no_returns=True)
        assert not cloud.xyz[cloud.no_return].any() and not cloud.measured[cloud.no_return].any()
        np.testing.assert_allclose(cloud.xyz[cloud.measured], f.lidar("lidar_top", frame=frame).xyz)
    fused = fuse_point_clouds([full])
    assert len(fused) == int(full.measured.sum()) and fused.measured.all()
    assert len(f.fused_lidar()) == len(fused)  # the default path is unchanged


def test_ground_mask_with_no_returns(tmp_path: Path) -> None:
    scene, slots = _scene(tmp_path, OT128, no_return_slots=[(3, 0)], empty_slots=[(1, 2)])
    no_return = (slots[:, 0] == 3) & (slots[:, 1] == 0)
    labels = [i % 2 == 0 for i in range(int((~no_return).sum()))]  # one label per row without the no-returns
    for i in range(3):
        if i != 1:
            write_lidar(scene.files.path, "lidar_top", [i], n=len(labels))
        write_ground_labels(scene.files.path, "lidar_top", i, labels)
    scene = Scene(scene.files.path, split="train")
    f = scene.frame(1)

    mask = f.ground_mask("lidar_top", rows=True)
    stored = f.lidar("lidar_top", rows=True)
    assert mask.tolist() == labels and len(mask) == len(stored)

    full = f.ground_mask("lidar_top", rows=True, no_returns=True)
    cloud = f.lidar("lidar_top", rows=True, no_returns=True)
    assert len(full) == len(cloud) and not full[cloud.no_return].any()  # no label: False
    np.testing.assert_array_equal(full[~cloud.no_return], mask)

    # The mask and the "ground" column line up with the cloud, whatever its rows.
    points = f.ground_mask("lidar_top")
    np.testing.assert_array_equal(points, mask[stored.measured])
    assert len(points) == len(f.lidar("lidar_top")) < len(stored)
    for kwargs in ({}, {"rows": True}, {"rows": True, "no_returns": True}):
        labelled = f.lidar("lidar_top", labels=True, frame="road", **kwargs)
        np.testing.assert_array_equal(labelled["ground"], f.ground_mask("lidar_top", **kwargs))
        np.testing.assert_array_equal(labelled.ground, labelled.attrs["ground"])
    assert "ground" not in f.lidar("lidar_top").attrs


def test_ground_labels_travel_with_the_cloud(tmp_path: Path) -> None:
    scene_dir = make_scene(tmp_path / "data" / "train" / "s", cameras=(), radars=(), lidars=("lidar_top", "lidar_front"))
    for i in range(3):
        write_ground_labels(scene_dir, "lidar_top", i, [True] * 20)
    f = Scene(scene_dir, split="train").frame(1)
    assert f.lidar("lidar_top", labels=True)["ground"].all()
    fused = f.fused_lidar(["lidar_top"], labels=True)
    assert fused["ground"].all() and len(fused["ground"]) == len(fused)

    # A lidar with a sweep but without labels: an error, or the cloud without the column.
    with pytest.raises(LabelsUnavailable):
        f.lidar("lidar_front", labels=True)
    assert "ground" not in f.lidar("lidar_front", labels=True, missing="none").attrs
    with pytest.raises(LabelsUnavailable):
        f.fused_lidar(labels=True)
    both = f.fused_lidar(labels=True, missing="none")
    assert "ground" not in both.attrs and len(both) == len(f.fused_lidar())
    assert f.lidar("lidar_rear", labels=True, missing="none") is None  # None only without the sweep itself
