"""Instance masks: COCO RLE decoding, label images, Frame.instances."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kitscenes import Availability, LabelsUnavailable, Scene
from kitscenes.instances import UNLABELLED, InstanceMasks, decode_rle, palette, rle_runs
from release_fixture import encode_rle, instances_file, make_scene, pack_scene, write_ground_labels


def test_rle_round_trip() -> None:
    rng = np.random.default_rng(0)
    for shape in ((7, 5), (40, 64), (1, 1)):
        mask = rng.random(shape) > 0.6
        mask[:, :1] = False
        np.testing.assert_array_equal(decode_rle(encode_rle(mask), *shape), mask)
    assert rle_runs(encode_rle(np.zeros((3, 3), bool))).tolist() == [9]
    with pytest.raises(ValueError):
        rle_runs("@")  # 0x10 without continuation: a sign-extended, negative run


def test_label_images_confident_wins() -> None:
    a = np.zeros((48, 64), bool)
    a[10:30, 10:30] = True
    b = np.zeros((48, 64), bool)
    b[20:40, 20:40] = True
    inst = InstanceMasks.from_json(instances_file([(a, 0.9, 58), (b, 0.5, 31)]), "camera_ring_front")
    assert len(inst) == 2 and inst.names() == ["Car", "Person"] and inst.image_size == (64, 48)
    labels, ids = inst.label_images()
    assert labels[25, 25] == 58 and ids[25, 25] == 1  # overlap: the car (0.9) wins
    assert labels[35, 35] == 31 and ids[35, 35] == 2
    assert labels[0, 0] == UNLABELLED and ids[0, 0] == 0
    np.testing.assert_array_equal(inst.mask(1), b)
    assert (inst.label_image(min_score=0.6) == 31).sum() == 0
    assert palette()[labels].shape == (48, 64, 3)
    assert len(inst.select(inst.classes == 31)) == 1


def test_frame_instances(tmp_path: Path) -> None:
    scene_dir = make_scene(tmp_path / "s", n=2, cameras=("camera_ring_front", "camera_ring_rear"))
    mask = np.zeros((48, 64), bool)
    mask[5:15, 5:25] = True
    folder = scene_dir / "processed" / "seamseg_instances" / "camera_ring_front"
    folder.mkdir(parents=True)
    (folder / "0000000000.json").write_text(instances_file([(mask, 0.8, 48)]))
    f = Scene(scene_dir, split="train").frame(0)
    inst = f.instances("camera_ring_front")
    assert inst.names() == ["Pole"] and inst.label_image().shape == f.image("camera_ring_front").shape[:2]
    for camera, index, split, state in (
        ("camera_ring_rear", 0, "train", Availability.NOT_IN_RELEASE),  # the masks are here, none for it
        ("camera_ring_front", 1, "train", Availability.DROPPED),
        ("camera_ring_front", 0, "test", Availability.WITHHELD_BY_SPLIT),
    ):
        with pytest.raises(LabelsUnavailable) as err:
            Scene(scene_dir, split=split).frame(index).instances(camera)
        assert err.value.state is state


def test_labels_not_in_the_release(tmp_path: Path) -> None:
    """Labels the release does not have for a sensor are NOT_IN_RELEASE, not NOT_DOWNLOADED."""
    scene_dir = make_scene(tmp_path / "data" / "train" / "s", n=2, cameras=("camera_ring_front", "camera_ring_rear"),
                           lidars=("lidar_top", "lidar_front"))
    folder = scene_dir / "processed" / "seamseg_instances" / "camera_ring_front"
    folder.mkdir(parents=True)
    (folder / "0000000000.json").write_text(instances_file([]))
    write_ground_labels(scene_dir, "lidar_top", 0, [False] * 20)
    pack_scene(scene_dir)
    scene = Scene(scene_dir)
    assert scene.availability("instances/camera_ring_front", 0) is Availability.PRESENT
    assert scene.availability("instances/camera_ring_rear", 0) is Availability.NOT_IN_RELEASE
    assert scene.availability("ground_labels/lidar_front", 0) is Availability.NOT_IN_RELEASE
    with pytest.raises(LabelsUnavailable) as err:
        scene.frame(0).instances("camera_ring_rear")
    assert err.value.state is Availability.NOT_IN_RELEASE and "kitscenes download" not in str(err.value)

    for archive in ("processed_seamseg_instances.zip", "processed_ground_segmentation.zip"):
        (scene_dir / archive).unlink()  # index.parquet still lists what the release has
    scene = Scene(scene_dir)
    assert scene.availability("instances/camera_ring_front", 0) is Availability.NOT_DOWNLOADED
    assert scene.availability("ground_labels/lidar_top", 0) is Availability.NOT_DOWNLOADED
    assert scene.availability("instances/camera_ring_rear", 0) is Availability.NOT_IN_RELEASE
    assert scene.availability("ground_labels/lidar_front", 0) is Availability.NOT_IN_RELEASE


def _reference_runs(counts: str) -> list[int]:
    """The plain loop of pycocotools' rleFrString, to check the vectorized parser against."""
    runs, p = [], 0
    while p < len(counts):
        x = k = 0
        more = True
        while more:
            c = ord(counts[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and c & 0x10:
                x |= -1 << (5 * k)
        if len(runs) > 2:
            x += runs[-2]
        runs.append(x)
    return runs


def test_parse_rles_batch_and_lookup() -> None:
    from kitscenes.instances import parse_rles

    rng = np.random.default_rng(1)
    masks = [rng.random((48, 64)) > p for p in (0.3, 0.6, 0.95, 1.0)]
    masks[1][10:30, 5:60] = True  # long runs: multi-character numbers and negative deltas
    strings = [encode_rle(m) for m in masks] + [""]
    runs, offsets = parse_rles(strings)
    for k, s in enumerate(strings):
        assert runs[offsets[k]:offsets[k + 1]].tolist() == _reference_runs(s)
    with pytest.raises(ValueError, match="truncated"):
        parse_rles(["0", "P"])  # 'P' has the continuation bit and nothing follows

    inst = InstanceMasks.from_json(instances_file([(masks[0], 0.9, 31), (masks[1], 0.5, 58), (masks[2], 0.7, 48)]), "x")
    classes, ids = inst.label_images()
    rows, cols = rng.integers(0, 48, 5000), rng.integers(0, 64, 5000)
    found, which = inst.lookup(rows, cols)
    np.testing.assert_array_equal(found, classes[rows, cols])
    np.testing.assert_array_equal(which, ids[rows, cols])
    high, _ = inst.lookup(rows, cols, min_score=0.8)
    np.testing.assert_array_equal(high, inst.label_images(0.8)[0][rows, cols])
