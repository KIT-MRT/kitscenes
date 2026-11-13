"""kitscenes.decode: JPEG decoding at the model's size with matching intrinsics (CPU always, GPU with CUDA)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import kitscenes  # noqa: E402
from kitscenes import decode  # noqa: E402
from kitscenes.decode import decode_cameras, decode_images, scale_intrinsics  # noqa: E402
from kitscenes.rig import Camera  # noqa: E402
from kitscenes.samples import SampleDataset, SampleSpec, collate  # noqa: E402
from release_fixture import make_scene, pack_scene  # noqa: E402


def _has_nvjpeg() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import torchvision.io  # noqa: F401
    except ImportError:
        return False
    return True


GPU = pytest.mark.skipif(not _has_nvjpeg(), reason="needs CUDA and torchvision")


def _jpeg(pixels: np.ndarray, quality: int = 95) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _block_image(width: int = 640, height: int = 480, x0: int = 200, y0: int = 144, side: int = 16) -> bytes:
    """Black with one white square: its centre is at ((x0 + x0 + side - 1) / 2, ...) in pixel-centre coordinates."""
    pixels = np.zeros((height, width, 3), np.uint8)
    pixels[y0:y0 + side, x0:x0 + side] = 255
    return _jpeg(pixels)


def _centroid(image: torch.Tensor) -> np.ndarray:
    """(u, v) brightness centroid of a (3, H, W) image, pixel centres on integers."""
    weight = image.float().mean(0).cpu().numpy()
    weight = np.where(weight > 40, weight, 0.0)
    v, u = np.indices(weight.shape)
    return np.array([(u * weight).sum(), (v * weight).sum()]) / weight.sum()


def _camera(width: int = 640, height: int = 480) -> Camera:
    return Camera("cam", np.eye(4), K=np.array([[500.0, 0, 319.5], [0, 480.0, 239.5], [0, 0, 1]]), image_size=(width, height))


@pytest.mark.parametrize("factor", [0.5, 0.25, 0.125, 0.3, 1.7])
def test_scale_intrinsics_is_camera_scaled(factor: float) -> None:
    camera = _camera()
    np.testing.assert_allclose(scale_intrinsics(camera.K, factor), camera.scaled(factor).K)
    got = scale_intrinsics(torch.from_numpy(camera.K), (factor, factor))
    assert isinstance(got, torch.Tensor) and got.dtype == torch.float64
    np.testing.assert_allclose(got.numpy(), camera.scaled(factor).K)


def test_scale_intrinsics_per_axis_and_batched() -> None:
    K = _camera().K
    out = scale_intrinsics(K, (0.5, 0.25))
    np.testing.assert_allclose(out[0], [250.0, 0, 159.5])
    np.testing.assert_allclose(out[1], [0, 120.0, 59.5])
    np.testing.assert_allclose(out[2], [0, 0, 1])
    factors = np.array([[0.5, 0.5], [0.25, 0.25]])
    batched = scale_intrinsics(np.stack([K, K]), factors)  # (2, 3, 3) with (2, 2)
    np.testing.assert_allclose(batched[1], _camera().scaled(0.25).K)
    broadcast = scale_intrinsics(torch.from_numpy(K).float(), torch.from_numpy(factors))  # one K, two factors
    assert broadcast.shape == (2, 3, 3) and broadcast.dtype == torch.float32
    np.testing.assert_allclose(broadcast[0].numpy(), _camera().scaled(0.5).K, rtol=1e-6)


def test_cpu_full_size_matches_pillow() -> None:
    from PIL import Image

    data = _block_image()
    out = decode_images([data, data], device="cpu")
    assert out.images.shape == (2, 3, 480, 640) and out.images.dtype == torch.uint8
    assert out.scale.tolist() == [[1.0, 1.0], [1.0, 1.0]] and out.K is None
    reference = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    assert np.abs(out.images[0].permute(1, 2, 0).numpy().astype(int) - reference).max() <= 2


@pytest.mark.parametrize("kwargs, shape, factors", [
    (dict(scale=0.5), (240, 320), (0.5, 0.5)),            # DCT 1/2
    (dict(scale=0.25), (120, 160), (0.25, 0.25)),         # DCT 1/4
    (dict(scale=0.3), (144, 192), (0.3, 0.3)),            # DCT 1/2, then resized
    (dict(size=(320, 200)), (200, 320), (0.5, 200 / 480)),  # non-uniform
    (dict(scale=(0.5, 0.25)), (120, 320), (0.5, 0.25)),
])
@pytest.mark.parametrize("interpolation", ["bilinear", "area"])
def test_cpu_resize_keeps_pixel_centres(kwargs, shape, factors, interpolation) -> None:
    """The square's centre lands where the scaled K projects it (Camera.scaled convention)."""
    camera = _camera()
    ray = np.linalg.inv(camera.K) @ [207.5, 151.5, 1.0]  # the centre of the 16 px square at (200, 144)
    out = decode_images([_block_image()], device="cpu", K=camera.K, interpolation=interpolation, **kwargs)
    assert tuple(out.images.shape[2:]) == shape
    np.testing.assert_allclose(out.scale[0].numpy(), factors)
    np.testing.assert_allclose(out.K[0], scale_intrinsics(camera.K, factors))
    expected = (out.K[0] @ ray)[:2]
    np.testing.assert_allclose(_centroid(out.images[0]), expected, atol=0.1)


def test_cpu_without_opencv_uses_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    camera = _camera()
    out = decode_images([_block_image()], scale=0.25, device="cpu", K=camera.K)
    assert out.images.shape == (1, 3, 120, 160)
    expected = (out.K[0] @ np.linalg.inv(camera.K) @ [207.5, 151.5, 1.0])[:2]
    np.testing.assert_allclose(_centroid(out.images[0]), expected, atol=0.1)


def test_float_output_and_inputs() -> None:
    data = _block_image()
    as_uint8 = decode_images([data], device="cpu", scale=0.5).images
    variants = [bytearray(data), memoryview(data), np.frombuffer(data, np.uint8), torch.frombuffer(bytearray(data), dtype=torch.uint8)]
    for variant in variants:
        assert torch.equal(decode_images([variant], device="cpu", scale=0.5).images, as_uint8)
    as_float = decode_images([data], device="cpu", scale=0.5, dtype=torch.float32).images
    assert as_float.dtype == torch.float32 and 0.0 <= float(as_float.min()) and float(as_float.max()) <= 1.0
    assert (as_float * 255 - as_uint8.float()).abs().max() <= 0.5 + 1e-4


def test_chunks_and_reduction() -> None:
    data = [_block_image(x0=200 + 8 * k) for k in range(5)]
    whole = decode_images(data, device="cpu", scale=0.25)
    chunked = decode_images(data, device="cpu", scale=0.25, chunk=2)
    assert torch.equal(whole.images, chunked.images)
    assert decode._reduction(640, 480, (160, 120)) == 4
    assert decode._reduction(640, 480, (192, 144)) == 2
    assert decode._reduction(640, 480, (161, 120)) == 2
    assert decode._reduction(640, 480, (700, 500)) == 1


def test_errors() -> None:
    small, large = _jpeg(np.zeros((48, 64, 3), np.uint8)), _block_image()
    with pytest.raises(ValueError, match="different sizes"):
        decode_images([small, large], device="cpu")
    assert decode_images([small, large], device="cpu", size=(32, 24)).images.shape == (2, 3, 24, 32)
    with pytest.raises(ValueError, match="size or scale"):
        decode_images([small], device="cpu", size=(32, 24), scale=0.5)
    with pytest.raises(ValueError, match="no images"):
        decode_images([], device="cpu")
    with pytest.raises(ValueError, match="interpolation"):
        decode_images([small], device="cpu", interpolation="lanczos")
    with pytest.raises(ValueError, match="dtype"):
        decode_images([small], device="cpu", dtype=torch.int32)
    with pytest.raises(ValueError, match="CUDA"):
        decode_images([small], device="cpu", backend="nvjpeg")


def test_missing_torchvision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torchvision", None)
    monkeypatch.setitem(sys.modules, "torchvision.io", None)
    with pytest.raises(kitscenes.MissingDependencyError, match="torchvision"):
        decode_images([_block_image()], device="cuda", backend="nvjpeg")


def test_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(kitscenes.MissingDependencyError, match=r"kitscenes\[torch\]"):
        decode_images([_block_image()], device="cpu")
    np.testing.assert_allclose(scale_intrinsics(_camera().K, 0.5), _camera().scaled(0.5).K)  # numpy needs no torch


# --- SampleSpec(frame="base", image="bytes") + collate + decode_cameras ---------------------------------------------


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for name, n in (("scene_a", 4), ("scene_b", 5)):
        scene = make_scene(tmp_path / "data" / "train" / name, n=n, cameras=("camera_ring_front", "camera_ring_rear"))
        pack_scene(scene)
    return tmp_path


def test_bytes_with_reduce_decode_like_decoded(root: Path) -> None:
    """image="bytes" + reduce=2 + decode_cameras gives what image="decoded" + reduce=2 gives (CHW)."""
    ds = kitscenes.open(root)
    cams = ("camera_ring_front", "camera_ring_rear")
    as_bytes = SampleSpec(frame="base", cameras=cams, image="bytes", reduce=2)
    samples = SampleDataset(ds, as_bytes)
    s = samples[1]
    assert s["camera"]["camera_ring_front"]["image"][:2] == b"\xff\xd8"
    np.testing.assert_allclose(s["camera"]["camera_ring_front"]["K"], ds["scene_a"].rig.cameras["camera_ring_front"].K)
    batch = collate([samples[0], samples[1], samples[6]])
    out = decode_cameras(batch, as_bytes, device="cpu")
    assert isinstance(batch["camera"]["camera_ring_front"]["image"], list)  # the input is not modified
    reference = collate([SampleDataset(ds, SampleSpec(frame="base", cameras=cams, reduce=2))[k] for k in (0, 1, 6)])
    for name in cams:
        image = out["camera"][name]["image"]
        assert image.shape == (3, 3, 24, 32) and image.dtype == torch.uint8
        expected = reference["camera"][name]["image"]
        assert (image.int() - expected.int()).abs().max() <= 3  # OpenCV vs Pillow DCT scaling
        np.testing.assert_allclose(out["camera"][name]["K"].numpy(), reference["camera"][name]["K"].numpy())
        assert torch.equal(out["camera"][name]["T_cam_base"], batch["camera"][name]["T_cam_base"])
    assert out["meta"] == batch["meta"] and out["pose"] is batch["pose"]
    sized = decode_cameras(batch, as_bytes, device="cpu", size=(40, 20))["camera"]["camera_ring_front"]
    assert sized["image"].shape == (3, 3, 20, 40)
    full_K = ds["scene_a"].rig.cameras["camera_ring_front"].K
    np.testing.assert_allclose(sized["K"][0].numpy(), scale_intrinsics(full_K, (40 / 64, 20 / 48)))


def test_decode_cameras_clips_missing_and_decoded(root: Path) -> None:
    (root / "data/train/scene_a/camera_ring_rear.zip").unlink()
    ds = kitscenes.open(root)
    cams = ("camera_ring_front", "camera_ring_rear")
    clips = SampleDataset(ds, SampleSpec(frame="base", cameras=("camera_ring_front",), image="bytes", clip_length=2))
    batches = decode_cameras(collate([clips[0], clips[1]]), scale=0.5, device="cpu")
    assert isinstance(batches, list) and len(batches) == 2
    assert batches[1]["camera"]["camera_ring_front"]["image"].shape == (2, 3, 24, 32)

    loose = SampleDataset(ds, SampleSpec(frame="base", cameras=cams, image="bytes", require=False))
    batch = collate([loose[0], loose[5]])  # scene_a has no rear camera
    out = decode_cameras(batch, scale=0.5, device="cpu")["camera"]["camera_ring_rear"]
    assert out["image"].shape == (2, 3, 24, 32) and np.asarray(out["valid"]).tolist() == [False, True]
    assert not out["image"][0].any()  # the black image of the missing camera
    np.testing.assert_allclose(out["K"][1], ds["scene_b"].rig.cameras["camera_ring_rear"].scaled(0.5).K)

    decoded = collate([SampleDataset(ds, SampleSpec(frame="base", cameras=("camera_ring_front",)))[k] for k in (0, 1)])
    with pytest.raises(TypeError, match="already decoded"):
        decode_cameras(decoded, device="cpu")


# --- GPU -------------------------------------------------------------------------------------------------


@GPU
@pytest.mark.parametrize("kwargs", [dict(), dict(scale=0.25), dict(scale=0.3), dict(size=(320, 200))])
def test_gpu_matches_cpu(kwargs) -> None:
    camera = _camera()
    data = [_block_image(x0=200 + 8 * k) for k in range(3)]
    gpu = decode_images(data, device="cuda", K=camera.K, **kwargs)
    cpu = decode_images(data, device="cpu", K=camera.K, **kwargs)
    assert gpu.images.is_cuda and gpu.images.shape == cpu.images.shape
    np.testing.assert_allclose(gpu.K, cpu.K)
    for k in range(3):
        expected = (gpu.K[k] @ np.linalg.inv(camera.K) @ [207.5 + 8 * k, 151.5, 1.0])[:2]
        np.testing.assert_allclose(_centroid(gpu.images[k]), expected, atol=0.15)
    assert (gpu.images.cpu().float() - cpu.images.float()).abs().mean() < 2.0


@GPU
def test_gpu_one_call_per_image_for_old_torchvision(monkeypatch: pytest.MonkeyPatch) -> None:
    data = [_block_image(x0=200 + 8 * k) for k in range(3)]
    batched = decode_images(data, device="cuda", scale=0.5).images
    monkeypatch.setattr(decode, "_TV_BATCHED", (99, 0))
    assert torch.equal(decode_images(data, device="cuda", scale=0.5).images, batched)


@GPU
def test_gpu_without_torchvision_decodes_on_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    data = [_block_image()]
    expected = decode_images(data, device="cpu", scale=0.3).images
    monkeypatch.setitem(sys.modules, "torchvision", None)
    monkeypatch.setitem(sys.modules, "torchvision.io", None)
    with pytest.warns(UserWarning, match="torchvision is not installed"):
        got = decode_images(data, device="cuda", scale=0.3).images
    assert got.is_cuda and (got.cpu().int() - expected.int()).abs().max() <= 1


@GPU
def test_gpu_decode_cameras(root: Path) -> None:
    spec = SampleSpec(frame="base", cameras=("camera_ring_front",), image="bytes", reduce=2)
    samples = SampleDataset(kitscenes.open(root), spec)
    out = decode_cameras(collate([samples[0], samples[1]]), spec, device="cuda", dtype=torch.float16)
    image = out["camera"]["camera_ring_front"]["image"]
    assert image.is_cuda and image.dtype == torch.float16 and image.shape == (2, 3, 24, 32)
    assert not out["camera"]["camera_ring_front"]["K"].is_cuda  # K stays where it was
