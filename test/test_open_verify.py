"""kitscenes.open(verify=True): CRC-32 of release-zip members."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

import kitscenes
from kitscenes.errors import ChecksumError
from release_fixture import make_scene, pack_scene


def _zip_root(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    pack_scene(make_scene(root / "data" / "train" / "s1", n=2, cameras=("camera_ring_front",), lidars=(), radars=()))
    make_scene(root / "data" / "val" / "s2", n=2, cameras=("camera_ring_front",), lidars=(), radars=())  # extracted
    return root


def _corrupt_first_image(root: Path) -> None:
    scene = kitscenes.open(root)["s1"]
    _, offset, size, _ = scene.files.storage.packed("camera_ring_front/0000000000.jpg")
    path = root / "data" / "train" / "s1" / "camera_ring_front.zip"
    data = bytearray(path.read_bytes())
    data[offset + size // 2] ^= 0xFF
    path.write_bytes(bytes(data))


def test_verify_is_off_by_default_and_reaches_every_zip_scene(tmp_path: Path) -> None:
    root = _zip_root(tmp_path)
    assert kitscenes.open(root).verify is False and kitscenes.open(root)["s1"].files.storage.verify is False
    ds = kitscenes.open(root, verify=True)
    assert ds.verify and ds["s1"].files.storage.verify
    assert ds.filter(splits="train")["s1"].files.storage.verify  # sub-datasets keep it
    assert pickle.loads(pickle.dumps(ds))["s1"].files.storage.verify  # and so do DataLoader workers
    assert not hasattr(ds["s2"].files.storage, "verify")  # extracted scenes have no checksum to compare


def test_verify_catches_a_corrupt_member(tmp_path: Path) -> None:
    root = _zip_root(tmp_path)
    _corrupt_first_image(root)
    kitscenes.open(root)["s1"].frame(0).image_bytes("camera_ring_front")  # read as stored without verify
    with pytest.raises(ChecksumError, match="CRC mismatch") as err:
        kitscenes.open(root, verify=True)["s1"].frame(0).image_bytes("camera_ring_front")
    assert isinstance(err.value, kitscenes.DataFormatError) and not isinstance(err.value, OSError)
    assert len(kitscenes.open(root, verify=True)["s1"].frame(1).image_bytes("camera_ring_front")) > 0


def test_a_truncated_camera_zip_is_a_checksum_error(tmp_path: Path) -> None:
    root = _zip_root(tmp_path)
    scene = kitscenes.open(root)["s1"]
    _, offset, size, _ = scene.files.storage.packed("camera_ring_front/0000000001.jpg")
    with open(root / "data" / "train" / "s1" / "camera_ring_front.zip", "r+b") as fh:
        fh.truncate(offset + size - 1)
    frame = kitscenes.open(root)["s1"].frame(1)
    with pytest.raises(ChecksumError, match="Short read"):
        frame.image_bytes("camera_ring_front")
    with pytest.raises(ChecksumError, match="Short read"):  # the path of kitscenes.decode
        frame.sensor_file("camera_ring_front").read()
    assert len(kitscenes.open(root)["s1"].frame(0).image_bytes("camera_ring_front")) > 0
