"""Colour, class and instance per point from the cameras (kitscenes.painting)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kitscenes import Scene
from kitscenes.instances import UNLABELLED
from kitscenes.painting import UNSEEN, label, paint
from release_fixture import instances_file, make_scene

# The fixture cameras sit at (1.5, 0, 1.6) in the base frame, look along +x, 64 x 48 px, focal 50.
WHITE_CORNER = [11.5, 2.0, 2.6]  # lands at u 22, v 19: the white top-left block of the image
PLAIN = [11.5, -2.0, 0.6]  # u 42, v 29: the plain colour (0, 128, 64) of frame 0
BEHIND = [-5.0, 0.0, 1.6]
ON_THE_MASK = [11.5, 3.4, 4.4]  # u 15, v 10: inside the front camera's mask rows 5-14, columns 5-24


def _scene(tmp_path: Path) -> Scene:
    scene_dir = make_scene(tmp_path / "s", n=1, cameras=("camera_ring_front", "camera_ring_rear"))
    mask = np.zeros((48, 64), bool)
    mask[5:15, 5:25] = True
    folder = scene_dir / "processed" / "seamseg_instances" / "camera_ring_front"
    folder.mkdir(parents=True)
    (folder / "0000000000.json").write_text(instances_file([(mask, 0.8, 48)]))
    return Scene(scene_dir, split="train")


def test_paint(tmp_path: Path) -> None:
    frame = _scene(tmp_path).frame(0)
    rgb, camera = paint(frame, np.array([WHITE_CORNER, PLAIN, BEHIND]))
    np.testing.assert_allclose(rgb[0], [255, 255, 255], atol=12)  # JPEG
    np.testing.assert_allclose(rgb[1], [0, 128, 64], atol=12)
    assert rgb[2].tolist() == list(UNSEEN) and camera.tolist() == [0, 0, -1]
    full, _ = paint(frame, np.array([WHITE_CORNER, PLAIN]), reduce=1)
    np.testing.assert_allclose(full.astype(int), rgb[:2].astype(int), atol=12)


def test_label(tmp_path: Path) -> None:
    frame = _scene(tmp_path).frame(0)
    classes, instances, camera = label(frame, np.array([ON_THE_MASK, PLAIN, BEHIND]))
    assert classes.tolist() == [48, UNLABELLED, UNLABELLED]  # Pole
    assert instances.tolist() == [1, 0, 0] and camera.tolist() == [0, -1, -1]
    assert frame.instances("camera_ring_front").names()[instances[0] - 1] == "Pole"


def _behind(point, factor: float = 2.0) -> list[float]:
    """A point on the front camera's ray through *point*, *factor* times as far."""
    centre = np.array([1.5, 0.0, 1.6])
    return (centre + factor * (np.asarray(point) - centre)).tolist()


def test_a_hidden_point_takes_what_is_in_front_of_it(tmp_path: Path) -> None:
    """Painting does not check visibility: a point behind a nearer one gets the same pixel's colour and class."""
    frame = _scene(tmp_path).frame(0)
    rgb, camera = paint(frame, np.array([WHITE_CORNER, _behind(WHITE_CORNER)]))
    assert rgb[0].tolist() == rgb[1].tolist() and camera.tolist() == [0, 0]
    classes, _, camera = label(frame, np.array([ON_THE_MASK, _behind(ON_THE_MASK)]))
    assert classes.tolist() == [48, 48] and camera.tolist() == [0, 0]


def test_paint_and_label_a_downscaled_camera(tmp_path: Path) -> None:
    """A downscaled camera still paints and labels (masks stay at the release size)."""
    from kitscenes.transcode import Transcoder, transcode_camera

    scene = _scene(tmp_path)
    with Transcoder("cpu", jobs=1) as t:
        transcode_camera(scene.files.path, "camera_ring_front", "half", t)
    frame = Scene(scene.files.path, split="train").frame(0)
    rgb, camera = paint(frame, np.array([WHITE_CORNER, PLAIN, BEHIND]), reduce=1)
    np.testing.assert_allclose(rgb[0], [255, 255, 255], atol=16)
    assert camera.tolist() == [0, 0, -1]
    classes, _, _ = label(frame, np.array([ON_THE_MASK, PLAIN]))
    assert classes.tolist() == [48, UNLABELLED]
