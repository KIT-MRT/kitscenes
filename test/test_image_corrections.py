"""kitscenes.image_corrections: the warp of a camera's stored images, and every path that applies it."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import kitscenes
from kitscenes._layout import IMAGE_CORRECTIONS
from kitscenes.errors import CalibrationError, ImageSizeError
from kitscenes.image_corrections import ImageCorrection, load_image_corrections, stored_correction, warp_image
from kitscenes.pack import pack_scene
from kitscenes.samples import SampleDataset, SampleSpec, collate
from kitscenes.transcode import Transcoder, transcode
from release_fixture import camera_entry, make_scene, pack_scene as fixture_pack_scene

LEFT = "camera_base_front_left_rect"
RIGHT = "camera_base_front_right_rect"
F, CU, CV = 1843.0, 1139.0, 1754.0

#: The test scenes' images: small, but smooth enough for shifts to be measured.
W, H = 256, 192

#: Exponents of the stereo files' coefficient lists, to write tests in their order.
DX_POWERS = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1))
DY_POWERS = ((0, 0), (1, 0), (0, 1), (1, 1))


def _terms(coefficients, powers) -> tuple:
    return tuple((i, j, float(c)) for (i, j), c in zip(powers, coefficients) if c)


def _corr(dx=(), dy=(), size=(2272, 3488)) -> ImageCorrection:
    return ImageCorrection(RIGHT, _terms(dx, DX_POWERS), _terms(dy, DY_POWERS), F, CU, CV, size, era="test")


def _json_terms(coefficients, powers) -> list:
    return [{"x_power": i, "y_power": j, "coefficient": c} for i, j, c in _terms(coefficients, powers)]


def _entry(dx, dy, focal=F, cu=CU, cv=CV, size=(2272, 3488), **over) -> dict:
    entry = {
        "type": "polynomial_remap",
        "resolution": {"width": size[0], "height": size[1]},
        "normalization": {"focal_length": focal, "principal_point_u": cu, "principal_point_v": cv},
        "dx": _json_terms(dx, DX_POWERS),
        "dy": _json_terms(dy, DY_POWERS),
        "other_variants": {
            "full": {"dx": _json_terms((0.8356, 0.1121, 0.8654, -1.0905, -1.0755), DX_POWERS),
                     "dy": _json_terms((-4.2101, 0.6656, 0.6819, 1.2859), DY_POWERS)},
            "note": "text beside the variants",
        },
        "scale_excess_pct": 0.81,
        "caveats": ["validated on one held-out scene"],
    }
    entry.update(over)
    return entry


def _doc(corrections=None, **over) -> dict:
    if corrections is None:
        corrections = {RIGHT: _entry((0.7916, 0.0, 0.9602), (-4.2745, 0.4712))}
    return {"schema": "kitscenes.image_corrections/1", "era": "velodyne_A", "corrections": corrections, **over}


def _write(scene: Path, doc: dict) -> Path:
    path = scene / IMAGE_CORRECTIONS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def _shift(dx: float = -4.5, dy: float = -2.25, camera: str = RIGHT) -> dict:
    """A file that shifts *camera* by a constant, in pixels of the (W, H) image (sub-pixel: integer shifts
    need no interpolation)."""
    return _doc({camera: _entry((dx,), (dy,), 200.0, W / 2, H / 2, (W, H))})


def _smooth(brightness: float = 0.0) -> np.ndarray:
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    wave = 90 + 60 * np.sin(u / 11.0) + 50 * np.cos(v / 9.0) + brightness
    return np.clip(np.stack([wave, wave[:, ::-1], wave[::-1]], -1), 0, 255).astype(np.uint8)


def _stereo_scene(path: Path, doc=None, n: int = 2, brightness: float = 0.0) -> Path:
    """A scene with the base pair; with *doc*, also a corrections file."""
    scene = make_scene(path, n=n, cameras=())
    calib_path = scene / "calibration" / "calib.json"
    calib = json.loads(calib_path.read_text())
    for camera in (LEFT, RIGHT):
        calib[camera] = camera_entry(W, H, 200.0)
        (scene / camera).mkdir()
        for i in range(n):
            Image.fromarray(_smooth(brightness)).save(scene / camera / f"{i:010d}.jpg", quality=98)
    calib_path.write_text(json.dumps(calib))
    if doc is not None:
        _write(scene, doc)
    return scene


def _stored(frame, camera: str = RIGHT) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(frame.image_bytes(camera))).convert("RGB"))


def _mean_diff(a, b) -> float:
    return float(np.abs(np.asarray(a).astype(float) - np.asarray(b).astype(float))[8:-8, 8:-8].mean())


# --------------------------------------------------------------- polynomials

def _offsets(u, v, c: ImageCorrection) -> tuple[float, float]:
    """``(dx, dy)`` of the backward map at one pixel."""
    dx, dy = c._offsets(np.float64(u), np.float64(v))
    return float(dx), float(dy)


def test_offsets_are_the_polynomials_in_normalised_coordinates() -> None:
    assert _offsets(CU, CV, _corr(dx=(0.8, -0.5, 0.9, -0.3, -0.2)))[0] == pytest.approx(0.8)
    x = (0.0 - CU) / F
    assert _offsets(0.0, CV, _corr(dy=(1.0, 2.0, 0.0, 0.0)))[1] == pytest.approx(1.0 + 2.0 * x)


def test_terms_of_any_power() -> None:
    """The format is not limited to the terms of today's files."""
    c = ImageCorrection(RIGHT, ((3, 2, 2.0),), ((0, 4, -1.0), (0, 4, 3.0)), 10.0, 0.0, 0.0, (100, 100))
    assert _offsets(5.0, 20.0, c) == pytest.approx((2.0 * 0.5 ** 3 * 2.0 ** 2, 2.0 * 2.0 ** 4))


def test_is_identity() -> None:
    assert _corr().is_identity and _corr(dx=(0.0,), dy=(0.0,)).is_identity
    assert ImageCorrection(RIGHT, ((0, 0, 0.0),), (), F, CU, CV, (10, 10)).is_identity
    assert not _corr(dy=(0.5,)).is_identity


# ------------------------------------------------------------------- scaling

def test_scaled_evaluates_the_same_point_in_smaller_pixels() -> None:
    c = _corr(dx=(0.8, -0.5, 0.9, -0.3, -0.2), dy=(1.0, 2.0, 0.5, 0.3))
    s = c.scaled(0.25)
    u, v = 500.0, 2000.0
    us, vs = (u + 0.5) * 0.25 - 0.5, (v + 0.5) * 0.25 - 0.5  # the pixel centres of Camera.scaled
    assert _offsets(us, vs, s) == pytest.approx(tuple(0.25 * d for d in _offsets(u, v, c)))


def test_scaled_is_per_axis_and_composes() -> None:
    c = _corr(dx=(-2.0,), dy=(4.0,))
    s = c.scaled((0.5, 0.25))
    assert _offsets(0, 0, s) == (-1.0, 1.0)
    assert c.scaled(0.5).scaled(0.5).scale == c.scaled(0.25).scale == (0.25, 0.25)
    with pytest.raises(ValueError, match="> 0"):
        c.scaled(0.0)


# --------------------------------------------------------------------- I/O

def test_load_gives_nothing_when_absent(tmp_path: Path) -> None:
    assert load_image_corrections(tmp_path) == {}


def test_load_default_and_variant(tmp_path: Path) -> None:
    _write(tmp_path, _doc())
    corr = load_image_corrections(tmp_path)[RIGHT]
    assert (corr.camera, corr.variant, corr.era, corr.resolution) == (RIGHT, "default", "velodyne_A", (2272, 3488))
    assert corr.dx == ((0, 0, 0.7916), (2, 0, 0.9602)) and corr.dy == ((0, 0, -4.2745), (1, 0, 0.4712))
    assert abs(corr.scale_excess_pct - 0.81) < 1e-12 and corr.caveats

    full = load_image_corrections(tmp_path, "full")[RIGHT]
    assert full.variant == "full" and len(full.dx) == 5


def test_load_rejects_unknown_variant(tmp_path: Path) -> None:
    _write(tmp_path, _doc())
    with pytest.raises(CalibrationError, match="banana"):
        load_image_corrections(tmp_path, "banana")
    with pytest.raises(CalibrationError, match="note"):  # text beside the variants is not one
        load_image_corrections(tmp_path, "note")


@pytest.mark.parametrize("doc, match", [
    (_doc(schema="something/else"), "schema"),
    (_doc({"camera_x": _entry((), ())}), "unknown camera"),
    (_doc({RIGHT: _entry((), (), type="lookup_table")}), "lookup_table"),
    (_doc({RIGHT: {**_entry((), ()), "dx": [{"x_power": 0}]}}), "malformed"),
    (_doc({RIGHT: {**_entry((), ()), "dx": [{"x_power": -1, "y_power": 0, "coefficient": 1.0}]}}), "malformed"),
    (_doc({RIGHT: {**_entry((), ()), "dx": [{"x_power": 0.5, "y_power": 0, "coefficient": 1.0}]}}), "malformed"),
    (_doc({RIGHT: _entry((), (), focal=0.0)}), "focal_length"),
    ({"schema": "kitscenes.image_corrections/1"}, "no corrections"),
])
def test_load_rejects_malformed_files(tmp_path: Path, doc: dict, match: str) -> None:
    _write(tmp_path, doc)
    with pytest.raises(CalibrationError, match=match):
        load_image_corrections(tmp_path)


def test_load_rejects_broken_json(tmp_path: Path) -> None:
    _write(tmp_path, {}).write_text("{")
    with pytest.raises(CalibrationError, match="JSON"):
        load_image_corrections(tmp_path)


# -------------------------------------------------------------------- warp

def test_identity_correction_leaves_image_unchanged() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(40, 60), dtype=np.uint8)
    out = warp_image(img, _corr(size=(60, 40)))
    assert out.shape == img.shape and out.dtype == img.dtype
    assert np.array_equal(out, img) and out is not img


def test_backward_map_moves_content_against_dx() -> None:
    """corrected(u, v) = stored(u + dx, v): dx < 0 moves the content toward larger u."""
    img = np.zeros((9, 21), dtype=np.float64)
    img[4, 10] = 1.0
    out = warp_image(img, _corr(dx=(-3.0,), size=(21, 9)))
    assert int(np.argmax(out[4])) == 13, f"peak at {int(np.argmax(out[4]))}, expected 13"


def test_backward_map_moves_content_against_dy() -> None:
    img = np.zeros((21, 9), dtype=np.float64)
    img[10, 4] = 1.0
    out = warp_image(img, _corr(dy=(2.0,), size=(9, 21)))
    assert int(np.argmax(out[:, 4])) == 8, f"peak at {int(np.argmax(out[:, 4]))}, expected 8"


def test_warp_refuses_an_image_of_another_size() -> None:
    c = _corr(dx=(1.0,), size=(64, 48))
    with pytest.raises(ImageSizeError, match="scaled"):
        warp_image(np.zeros((24, 32), np.uint8), c)
    assert warp_image(np.zeros((24, 32), np.uint8), c.scaled(0.5)).shape == (24, 32)


@pytest.fixture
def no_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)  # import cv2 raises ImportError


def _curved(h: int, w: int) -> ImageCorrection:
    """Real coefficients on an image of (h, w): the strongest shifts of the release, with all terms."""
    return ImageCorrection(RIGHT, _terms((0.8, -0.3, 0.96, -1.1, -1.0), DX_POWERS),
                           _terms((-4.27, 0.67, 0.68, 1.29), DY_POWERS), 0.81 * w, (w - 1) / 2, (h - 1) / 2, (w, h))


def test_pillow_fallback_moves_content_the_same_way(no_opencv: None) -> None:
    img = np.zeros((21, 21), dtype=np.uint8)
    img[10, 10] = 200
    out = warp_image(img, _corr(dx=(-3.0,), dy=(2.0,), size=(21, 21)))
    assert out.dtype == np.uint8 and np.unravel_index(int(np.argmax(out)), out.shape) == (8, 13)
    assert out[8, 13] == 200 and out.sum() == 200  # an integer shift is exact


def test_pillow_fallback_matches_opencv_bilinear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Over tile boundaries, at the edges, for an image that is not a multiple of the tile size."""
    cv2 = pytest.importorskip("cv2")
    img = np.asarray(Image.fromarray(_smooth()).resize((300, 230), Image.BICUBIC))
    corr = _curved(230, 300)
    want = warp_image(img, corr, cv2.INTER_LINEAR)
    monkeypatch.setitem(sys.modules, "cv2", None)
    got = warp_image(img, corr)
    diff = got.astype(int) - want.astype(int)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert np.abs(diff).max() <= 1 and np.abs(diff).mean() < 0.05
    assert abs(diff.mean()) < 0.01  # no darkening: Pillow truncates on uint8, so this needs float


def test_pillow_fallback_repeats_the_edges(no_opencv: None) -> None:
    img = np.full((64, 80, 3), 200, dtype=np.uint8)
    out = warp_image(img, _corr(dx=(-5.5,), dy=(-4.5,), size=(80, 64)))
    assert (out == 200).all()  # Pillow itself would fill the strips with black


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("colour", [False, True])
def test_pillow_fallback_keeps_shape_and_dtype(no_opencv: None, dtype, colour: bool) -> None:
    img = (_smooth() if colour else _smooth()[..., 0]).astype(dtype)
    out = warp_image(img, _curved(H, W))
    assert out.shape == img.shape and out.dtype == img.dtype
    assert _mean_diff(out, img) > 1


def test_pillow_fallback_scaled_shift(no_opencv: None) -> None:
    img = np.zeros((40, 60), dtype=np.uint8)
    img[20, 30] = 200
    corr = _corr(dx=(-8.0,), dy=(4.0,), size=(120, 80)).scaled(0.5)
    assert np.unravel_index(int(np.argmax(warp_image(img, corr))), img.shape) == (18, 34)


def test_frame_image_without_opencv(tmp_path: Path, no_opencv: None) -> None:
    frame = kitscenes.Scene(_stereo_scene(tmp_path / "s", _shift())).frame(0)
    assert _mean_diff(frame.image(RIGHT), _stored(frame)) > 3


def test_warp_preserves_shape_for_colour_images() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(16, 24, 3), dtype=np.uint8)
    out = warp_image(img, _corr(dx=(0.5,), size=(24, 16)))
    assert out.shape == img.shape and out.dtype == img.dtype


def test_warp_rejects_bad_dimensionality() -> None:
    with pytest.raises(ValueError, match="2D or 3D"):
        warp_image(np.zeros(5), _corr())


# ------------------------------------------------------------ Scene and Frame

def test_a_scene_without_the_file_is_not_touched(tmp_path: Path) -> None:
    scene = kitscenes.Scene(_stereo_scene(tmp_path / "s"))
    frame = scene.frame(0)
    assert scene.files.image_corrections == {}
    np.testing.assert_array_equal(frame.image(RIGHT), _stored(frame))


@pytest.mark.parametrize("packed", [False, True])
def test_frame_image_corrects_the_listed_camera_only(tmp_path: Path, packed: bool) -> None:
    path = _stereo_scene(tmp_path / "s", _shift())
    if packed:
        fixture_pack_scene(path)
    scene = kitscenes.Scene(path)
    frame = scene.frame(1)
    corr = scene.files.image_corrections[RIGHT]
    stored = _stored(frame)
    corrected = frame.image(RIGHT)
    np.testing.assert_array_equal(corrected, warp_image(stored, corr))
    assert _mean_diff(corrected, stored) > 3
    np.testing.assert_array_equal(frame.image(LEFT), _stored(frame, LEFT))


def test_any_camera_can_have_a_correction(tmp_path: Path) -> None:
    doc = _doc({LEFT: _entry((2.5,), (), 200.0, W / 2, H / 2, (W, H)),
                RIGHT: _entry((), (1.5,), 200.0, W / 2, H / 2, (W, H))})
    scene = kitscenes.Scene(_stereo_scene(tmp_path / "s", doc))
    frame = scene.frame(0)
    assert set(scene.files.image_corrections) == {LEFT, RIGHT}
    np.testing.assert_array_equal(frame.image(LEFT), warp_image(_stored(frame, LEFT), scene.files.image_corrections[LEFT]))
    np.testing.assert_array_equal(frame.image(RIGHT), warp_image(_stored(frame), scene.files.image_corrections[RIGHT]))
    assert not np.array_equal(frame.image(LEFT), _stored(frame, LEFT))


def test_opt_out_gives_the_stored_pixels_and_does_not_share_the_cache(tmp_path: Path) -> None:
    frame = kitscenes.Scene(_stereo_scene(tmp_path / "s", _shift())).frame(0)
    corrected = frame.image(RIGHT)
    stored = frame.image(RIGHT, image_correction=False)
    np.testing.assert_array_equal(stored, _stored(frame))
    assert not np.array_equal(corrected, stored)
    np.testing.assert_array_equal(frame.image(RIGHT), corrected)


def test_an_entry_without_coefficients_changes_nothing(tmp_path: Path) -> None:
    doc = _doc({RIGHT: _entry((), (), 200.0, W / 2, H / 2, (W, H))})  # as velodyne_B
    frame = kitscenes.Scene(_stereo_scene(tmp_path / "s", doc)).frame(0)
    assert frame.scene.files.image_corrections[RIGHT].is_identity
    np.testing.assert_array_equal(frame.image(RIGHT), _stored(frame))


def test_malformed_file_raises(tmp_path: Path) -> None:
    frame = kitscenes.Scene(_stereo_scene(tmp_path / "s", _doc(schema="nope"))).frame(0)
    with pytest.raises(CalibrationError, match="schema"):
        frame.image(RIGHT)
    assert frame.image(LEFT, image_correction=False).shape == (H, W, 3)


def test_a_file_for_another_image_size_is_refused(tmp_path: Path) -> None:
    frame = kitscenes.Scene(_stereo_scene(tmp_path / "s", _doc())).frame(0)  # for 2272x3488
    with pytest.raises(ImageSizeError, match="2272x3488"):
        frame.image(RIGHT)


@pytest.mark.parametrize("reduce", [2, 4])
def test_reduced_images_get_the_scaled_shift(tmp_path: Path, reduce: int) -> None:
    """A reduced corrected image is the corrected image, reduced; the unscaled shift would be off by far more."""
    frame = kitscenes.Scene(_stereo_scene(tmp_path / "s", _shift(-8.0, -8.0))).frame(0)
    want = np.asarray(Image.fromarray(frame.image(RIGHT)).resize((W // reduce, H // reduce), Image.BOX))
    got = frame.image(RIGHT, reduce=reduce)
    corr = frame.scene.files.image_corrections[RIGHT]
    wrong = dict(dx=corr.dx, dy=corr.dy, resolution=(W // reduce, H // reduce))  # the same pixels at the smaller size
    unscaled = warp_image(frame.image(RIGHT, reduce=reduce, image_correction=False),
                          ImageCorrection(RIGHT, focal_length=200.0, principal_point_u=0.0, principal_point_v=0.0, **wrong))
    assert got.shape == want.shape
    assert _mean_diff(got, want) < 0.5 * _mean_diff(unscaled, want)


def test_transcoded_scene_keeps_the_file_and_gets_the_scaled_shift(tmp_path: Path) -> None:
    root = tmp_path / "root"
    path = _stereo_scene(root / "data" / "train" / "s", _shift(-8.0, -8.0))
    full = kitscenes.Scene(path).frame(0).image(RIGHT)
    with Transcoder("cpu", jobs=1) as cpu:
        result = transcode(root, "half", out=tmp_path / "half", transcoder=cpu)
    assert not result.failed
    scene = kitscenes.open(tmp_path / "half").scene("s")
    assert scene.files.image_corrections[RIGHT].scale == (1.0, 1.0)
    assert stored_correction(scene, RIGHT).scale == (0.5, 0.5)
    got = scene.frame(0).image(RIGHT)
    want = np.asarray(Image.fromarray(full).resize((W // 2, H // 2), Image.BOX))
    assert got.shape == want.shape and _mean_diff(got, want) < 0.5 * _mean_diff(_stored(scene.frame(0)), want)


def test_pack_keeps_the_file(tmp_path: Path) -> None:
    src = _stereo_scene(tmp_path / "src" / "s", _shift())
    pack_scene(src, tmp_path / "out" / "s")
    assert RIGHT in kitscenes.Scene(tmp_path / "out" / "s").files.image_corrections


# ------------------------------------------------------------------- samples

def _dataset(root: Path, **scenes) -> "kitscenes.Dataset":
    for name, doc in scenes.items():
        _stereo_scene(root / "data" / "train" / name, doc)
    return kitscenes.open(root)


def test_decoded_samples_are_corrected(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, s=_shift())
    frame = ds.scene("s").frame(0)
    on = SampleDataset(ds, SampleSpec(cameras=(RIGHT,)))[0]["camera"][RIGHT]
    off = SampleDataset(ds, SampleSpec(cameras=(RIGHT,), image_correction=False))[0]["camera"][RIGHT]
    np.testing.assert_array_equal(on["image"], frame.image(RIGHT))
    np.testing.assert_array_equal(off["image"], _stored(frame))
    assert set(on) == set(off) and "image_correction" not in on


def test_bytes_samples_carry_the_correction_only_where_there_is_one(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, a=_shift(), b=None)
    spec = SampleSpec(cameras=(LEFT, RIGHT), image="bytes")
    samples = {s["meta"]["scene"]: s["camera"] for s in (SampleDataset(ds, spec)[i] for i in (0, 2))}
    assert isinstance(samples["a"][RIGHT]["image"], bytes)
    assert samples["a"][RIGHT]["image_correction"] == ds.scene("a").files.image_corrections[RIGHT]
    assert "image_correction" not in samples["a"][LEFT] and "image_correction" not in samples["b"][RIGHT]
    off = SampleDataset(ds, SampleSpec(cameras=(RIGHT,), image="bytes", image_correction=False))[0]
    assert "image_correction" not in off["camera"][RIGHT]


def test_collate_without_corrections_adds_nothing(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, a=None)
    spec = SampleSpec(cameras=(RIGHT,), image="bytes")
    assert "image_correction" not in collate([SampleDataset(ds, spec)[0]])["camera"][RIGHT]


# -------------------------------------------------------- decode_cameras (torch)

def _devices() -> list:
    torch = pytest.importorskip("torch")
    return ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]


def _chw(image) -> np.ndarray:
    return image.permute(1, 2, 0).cpu().numpy()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decode_cameras_matches_frame_image(tmp_path: Path, device: str) -> None:
    """Scenes with and without a correction in one batch, each with its own."""
    if device not in _devices():
        pytest.skip("needs a GPU")
    from kitscenes.decode import decode_cameras

    ds = _dataset(tmp_path, a=_shift(), b=None, c=_shift(3.25, 5.5))
    spec = SampleSpec(cameras=(RIGHT,), image="bytes")
    samples = SampleDataset(ds, spec)
    batch = collate([samples[i] for i in (0, 2, 4)])
    assert batch["camera"][RIGHT]["image_correction"][1] is None
    out = decode_cameras(batch, spec, device=device, backend="cpu")
    entry = out["camera"][RIGHT]
    assert "image_correction" not in entry
    for k, scene in enumerate("abc"):
        frame = ds.scene(scene).frame(0)
        assert _mean_diff(_chw(entry["image"][k]), frame.image(RIGHT)) < 1.0  # bicubic against the CPU warp
    assert _mean_diff(_chw(entry["image"][0]), _stored(ds.scene("a").frame(0))) > 3
    np.testing.assert_array_equal(_chw(entry["image"][1]), _chw(decode_cameras(
        collate([samples[2]]), spec, device=device, backend="cpu")["camera"][RIGHT]["image"][0]))  # no file: plain decode


def test_decode_cameras_opt_out_keeps_the_stored_image(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from kitscenes.decode import decode_cameras

    ds = _dataset(tmp_path, a=_shift())
    spec = SampleSpec(cameras=(RIGHT,), image="bytes", image_correction=False)
    out = decode_cameras(collate([SampleDataset(ds, spec)[0]]), spec, device="cpu", backend="cpu")
    frame = ds.scene("a").frame(0)
    assert _mean_diff(_chw(out["camera"][RIGHT]["image"][0]), _stored(frame)) < 1.0


def test_decode_cameras_scales_the_shift_per_axis(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from kitscenes.decode import decode_cameras

    ds = _dataset(tmp_path, a=_shift(-8.0, -8.0))
    spec = SampleSpec(cameras=(RIGHT,), image="bytes")
    batch = collate([SampleDataset(ds, spec)[0]])
    size = (W // 2, H // 4)
    got = _chw(decode_cameras(batch, spec, size=size, device="cpu", backend="cpu")["camera"][RIGHT]["image"][0])
    want = np.asarray(Image.fromarray(ds.scene("a").frame(0).image(RIGHT)).resize(size, Image.BOX))
    plain = np.asarray(Image.fromarray(_stored(ds.scene("a").frame(0))).resize(size, Image.BOX))
    assert got.shape == want.shape and _mean_diff(got, want) < 0.5 * _mean_diff(plain, want)


def test_decode_cameras_with_a_camera_some_samples_lack(tmp_path: Path) -> None:
    """A sample without the camera's image has it black and not valid: only the valid one is warped."""
    pytest.importorskip("torch")
    from kitscenes.decode import decode_cameras
    from kitscenes.samples import _black_jpeg

    ds = _dataset(tmp_path, a=_shift())
    spec = SampleSpec(cameras=(RIGHT,), image="bytes", require=False)
    sample = SampleDataset(ds, spec)[0]
    assert sample["camera"][RIGHT]["valid"] and "image_correction" in sample["camera"][RIGHT]
    w, h = ds.scene("a").rig.cameras[RIGHT].image_size
    lacking = {k: v for k, v in sample["camera"][RIGHT].items() if k != "image_correction"}
    lacking = {**sample, "camera": {RIGHT: {**lacking, "image": _black_jpeg(w, h), "valid": False}}}
    out = decode_cameras(collate([sample, lacking]), spec, device="cpu", backend="cpu")["camera"][RIGHT]
    assert "image_correction" not in out and out["valid"].tolist() == [True, False]
    assert _mean_diff(_chw(out["image"][0]), ds.scene("a").frame(0).image(RIGHT)) < 1.0
    assert int(out["image"][1].max()) <= 1  # the black image stays black


def test_apply_image_corrections_checks_and_leaves_its_input() -> None:
    torch = pytest.importorskip("torch")
    from kitscenes.decode import apply_image_corrections

    images = torch.from_numpy(_smooth()).permute(2, 0, 1)[None].contiguous()
    before = images.clone()
    corr = _corr(dx=(-4.5,), size=(W, H))
    with pytest.raises(ValueError, match="corrections"):
        apply_image_corrections(images, [corr, None])
    with pytest.raises(ImageSizeError):
        apply_image_corrections(images, [corr], scale=0.5)
    out = apply_image_corrections(images, [corr])
    assert torch.equal(images, before) and not torch.equal(out, images) and out.dtype == images.dtype
    assert torch.equal(apply_image_corrections(images, [None]), images)
    as_float = apply_image_corrections(images.float() / 255, [corr])
    assert as_float.dtype == torch.float32 and float(as_float.min()) >= 0 and float(as_float.max()) <= 1


def test_mapping_gpu_decode_applies_the_correction(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from kitscenes.mapping import _decode_on_device

    scene = kitscenes.Scene(_stereo_scene(tmp_path / "s", _shift()))
    frame = scene.frame(0)
    out = _decode_on_device([{RIGHT: frame.image_bytes(RIGHT), LEFT: frame.image_bytes(LEFT)}], "cuda", scene)[0]
    assert _mean_diff(out[RIGHT].cpu().numpy(), frame.image(RIGHT)) < 3  # nvJPEG, bicubic
    assert _mean_diff(out[RIGHT].cpu().numpy(), _stored(frame)) > 5
    assert _mean_diff(out[LEFT].cpu().numpy(), frame.image(LEFT)) < 3
