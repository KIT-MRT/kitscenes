"""Image corrections: warps that bring a camera's stored images onto ``calib.json``.

The stored images of a camera can differ from what ``calib.json`` describes by a
small field that depends on when the scene was recorded, and that a pinhole
camera cannot express. Today this is ``camera_base_front_right_rect``: the
calibration has one ideal rectified pair (identical intrinsics, co-oriented
cameras, baseline along camera x), while the stored right image carries a
vertical disparity (up to about 4 px) and a disparity offset (around 1 px), which
degrade row-based matchers and bias stereo depth.

A scene that needs this ships ``calibration/image_corrections.json`` next to
``calib.json``, with one entry per camera. An entry is a backward map::

    corrected(u, v) = stored(u + dx(x, y), v + dy(x, y))
    x = (u - principal_point_u) / focal_length,  y = (v - principal_point_v) / focal_length

with ``dx`` and ``dy`` in pixels, each a sum of ``coefficient * x^i * y^j`` terms.

**A scene without that file, a camera without an entry and an entry without
coefficients are not touched.** Otherwise the correction is applied wherever the
API decodes the image:

- :meth:`kitscenes.Frame.image` and ``SampleSpec(image="decoded")``, at any
  ``reduce``, also for a camera transcoded to a smaller size;
- :func:`kitscenes.decode.decode_cameras` for ``SampleSpec(image="bytes")``;
- the colour of voxel maps on the GPU (:mod:`kitscenes.mapping`).

NVIDIA DALI (:mod:`kitscenes.dali`) yields the stored JPEGs; apply
:func:`kitscenes.decode.apply_image_corrections` to those images yourself.

Pass ``image_correction=False`` (:meth:`kitscenes.Frame.image`,
:class:`kitscenes.SampleSpec`) for the stored pixels. The stored JPEG itself
(:meth:`kitscenes.Frame.image_bytes`, :meth:`kitscenes.Frame.sensor_file`) never
changes: decoding it yourself gives the uncorrected image, which
:func:`warp_image` corrects::

    from kitscenes.image_corrections import warp_image

    corr = scene.files.image_corrections.get(camera)        # None if it needs no correction
    if corr is not None:
        image = warp_image(image, corr.scaled(1 / reduce))
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

from kitscenes._layout import IMAGE_CORRECTIONS
from kitscenes._storage import DirStorage, Storage
from kitscenes.constants import CAMERA_NAMES
from kitscenes.errors import CalibrationError, ImageSizeError

if TYPE_CHECKING:  # pragma: no cover
    from kitscenes.scene import Scene

__all__ = ["ImageCorrection", "load_image_corrections", "warp_image"]

_SCHEMA = "kitscenes.image_corrections/1"
_TYPE = "polynomial_remap"

Factor = Union[float, tuple[float, float]]
#: One polynomial term: ``(x_power, y_power, coefficient)``.
Term = tuple[int, int, float]


def _polynomial(terms: Sequence[Term], x, y):
    """The sum of the terms; only arithmetic, so numpy arrays and torch tensors both work.

    With ``x`` a row and ``y`` a column, a term in one of them stays one-dimensional.
    """
    total = 0.0
    for x_power, y_power, coefficient in terms:
        term = coefficient
        if x_power:
            term = term * x ** x_power
        if y_power:
            term = term * y ** y_power
        total = total + term
    return total


@dataclass(frozen=True)
class ImageCorrection:
    """The correction of one camera of one scene.

    Attributes:
        camera: The camera it applies to.
        dx: Terms ``(x_power, y_power, coefficient)`` of the horizontal offset of
            the backward map, in pixels of the release's image.
        dy: Likewise, the vertical offset.
        focal_length: Normalises coordinates (release's image).
        principal_point_u: Normalises coordinates (release's image).
        principal_point_v: Normalises coordinates (release's image).
        resolution: ``(width, height)`` of the release's image.
        era: The recording period the coefficients belong to.
        variant: ``"default"``, the recommended coefficients, or the name of one of
            the entry's ``other_variants`` (:func:`load_image_corrections`).
        scale_excess_pct: Depth-scale error the warp does not correct, in percent.
        caveats: Free-text warnings of the file.
        scale: ``(sx, sy)``, the size of the images this applies to relative to the
            release's (:meth:`scaled`). The other attributes stay in release pixels.
    """

    camera: str
    dx: tuple[Term, ...]
    dy: tuple[Term, ...]
    focal_length: float
    principal_point_u: float
    principal_point_v: float
    resolution: tuple[int, int]
    era: str = "unknown"
    variant: str = "default"
    scale_excess_pct: float = 0.0
    caveats: tuple[str, ...] = ()
    scale: tuple[float, float] = (1.0, 1.0)

    @property
    def is_identity(self) -> bool:
        """Whether all coefficients are zero, so that there is nothing to apply."""
        return not any(term[2] for term in (*self.dx, *self.dy))

    def scaled(self, factor: Factor) -> "ImageCorrection":
        """The correction for an image resized by *factor* (one factor or ``(sx, sy)``).

        Pixel centres are on integers, as in :meth:`kitscenes.Camera.scaled`:
        pixel ``u`` of the resized image is ``(u + 0.5) / sx - 0.5`` of the original.
        """
        sx, sy = (float(factor), float(factor)) if np.ndim(factor) == 0 else (float(factor[0]), float(factor[1]))
        if not (sx > 0 and sy > 0):
            raise ValueError(f"factor must be > 0, got {factor}")
        return replace(self, scale=(self.scale[0] * sx, self.scale[1] * sy))

    def _normalised(self, u, v):
        """Pixel coordinates -> the normalised ``(x, y)`` of the polynomials."""
        sx, sy = self.scale
        x = ((u + 0.5) / sx - 0.5 - self.principal_point_u) / self.focal_length
        y = ((v + 0.5) / sy - 0.5 - self.principal_point_v) / self.focal_length
        return x, y

    def _offsets(self, u, v):
        """``(dx, dy)`` in pixels of this image at pixels ``(u, v)`` of the corrected image."""
        x, y = self._normalised(u, v)
        return _polynomial(self.dx, x, y) * self.scale[0], _polynomial(self.dy, x, y) * self.scale[1]

    def _sampling_maps(self, u, v):
        """Where the corrected image at pixels ``(u, v)`` reads the stored image."""
        dx, dy = self._offsets(u, v)
        return u + dx, v + dy

    def _check_size(self, height: int, width: int) -> None:
        """An image of another size than this correction's would be warped wrongly without a sign of it."""
        want_w, want_h = self.resolution[0] * self.scale[0], self.resolution[1] * self.scale[1]
        if abs(width - want_w) > 1 or abs(height - want_h) > 1:
            raise ImageSizeError(
                f"{self.camera}: the image is {width}x{height}, but this correction is for "
                f"{want_w:g}x{want_h:g}. For a resized image use ImageCorrection.scaled().")


def _terms(block, key: str) -> tuple[Term, ...]:
    terms = []
    for term in block[key]:
        x_power, y_power = term["x_power"], term["y_power"]
        if not all(isinstance(p, int) and not isinstance(p, bool) and p >= 0 for p in (x_power, y_power)):
            raise ValueError(f"powers must be integers >= 0, got {x_power!r}, {y_power!r}")
        terms.append((x_power, y_power, float(term["coefficient"])))
    return tuple(terms)


def _entry(camera: str, entry: dict, variant: Optional[str], era: str, source: str) -> ImageCorrection:
    where = f"{source}, {camera}"
    if camera not in CAMERA_NAMES:
        raise CalibrationError(f"{source}: unknown camera {camera!r}")
    if not isinstance(entry, dict) or entry.get("type") != _TYPE:
        kind = entry.get("type") if isinstance(entry, dict) else None
        raise CalibrationError(f"{where}: unsupported type {kind!r}, expected {_TYPE!r}")
    block = entry
    if variant is not None:
        block = entry.get("other_variants", {}).get(variant)
        if not isinstance(block, dict):
            raise CalibrationError(f"{where}: no variant {variant!r}")
    try:
        norm, size = entry["normalization"], entry["resolution"]
        corr = ImageCorrection(
            camera=camera,
            dx=_terms(block, "dx"),
            dy=_terms(block, "dy"),
            focal_length=float(norm["focal_length"]),
            principal_point_u=float(norm["principal_point_u"]),
            principal_point_v=float(norm["principal_point_v"]),
            resolution=(int(size["width"]), int(size["height"])),
            era=era,
            variant=variant or "default",
            scale_excess_pct=float(entry.get("scale_excess_pct", 0.0)),
            caveats=tuple(entry.get("caveats", ())),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"{where}: malformed entry ({exc!r})") from exc
    if not corr.focal_length > 0:
        raise CalibrationError(f"{where}: focal_length must be > 0, got {corr.focal_length}")
    return corr


def load_image_corrections(
    scene: Union[str, Path, Storage], variant: Optional[str] = None,
) -> dict[str, ImageCorrection]:
    """Load the image corrections a scene ships (``scene.files.image_corrections`` loads the default).

    Args:
        scene: The scene folder, i.e. the one holding ``calibration/calib.json``,
            or the storage of a scene (``scene.files.storage``).
        variant: None for the recommended coefficients, else the name of one of
            the entries' ``other_variants`` (e.g. ``"full"``).

    Returns:
        ``{camera: correction}``, empty if the scene ships no file. A camera that is
        not listed needs no correction.

    Raises:
        CalibrationError: If the file is malformed, or an entry lacks *variant*.
    """
    storage = DirStorage(scene) if isinstance(scene, (str, Path)) else scene
    if not storage.exists(IMAGE_CORRECTIONS):
        return {}
    source = storage.describe(IMAGE_CORRECTIONS)
    try:
        doc = json.loads(storage.read_bytes(IMAGE_CORRECTIONS))
    except ValueError as exc:
        raise CalibrationError(f"{source}: not valid JSON ({exc})") from exc
    if not isinstance(doc, dict) or doc.get("schema") != _SCHEMA:
        schema = doc.get("schema") if isinstance(doc, dict) else None
        raise CalibrationError(f"{source}: unsupported schema {schema!r}, expected {_SCHEMA!r}")
    corrections = doc.get("corrections")
    if not isinstance(corrections, dict):
        raise CalibrationError(f"{source}: no corrections")
    era = str(doc.get("era", "unknown"))
    return {camera: _entry(camera, entry, variant, era, source) for camera, entry in corrections.items()}


def stored_correction(scene: "Scene", camera: str) -> Optional[ImageCorrection]:
    """*camera*'s correction at the size of its stored images, None if it has none."""
    corr = scene.files.image_corrections.get(camera)
    if corr is None:
        return None
    variant = scene.files.image_variants.get(camera)  # transcoded: as scene.rig scales K
    return corr if variant is None else corr.scaled(variant.scale)


def _maps(shape: Sequence[int], corr: ImageCorrection) -> tuple[np.ndarray, np.ndarray]:
    """Remap coordinates (:meth:`ImageCorrection._sampling_maps`), (H, W) float64."""
    h, w = int(shape[0]), int(shape[1])
    u = np.arange(w, dtype=np.float64)[None, :]
    v = np.arange(h, dtype=np.float64)[:, None]
    mapx, mapy = corr._sampling_maps(u, v)
    return np.ascontiguousarray(np.broadcast_to(mapx, (h, w))), np.ascontiguousarray(np.broadcast_to(mapy, (h, w)))


# The maps of a full-size image are 63 MB. A process sees few of them: one per recording
# period and image size. Four covers a batch mixing all periods of one camera.
@functools.lru_cache(maxsize=4)
def _maps_float32(h: int, w: int, corr: ImageCorrection) -> tuple[np.ndarray, np.ndarray]:
    mapx, mapy = _maps((h, w), corr)
    return mapx.astype(np.float32), mapy.astype(np.float32)


#: Tile size of the Pillow fallback, in pixels: the polynomials are bilinear within a tile, which at
#: this size is within 0.01 grey levels of the exact warp (256 px: 0.03).
_MESH_STEP = 64


@functools.lru_cache(maxsize=4)
def _pillow_mesh(h: int, w: int, corr: ImageCorrection) -> tuple[int, list]:
    """``(pad, mesh)`` for ``Image.transform(MESH)``: the tiles of an image padded by *pad* on every side.

    Pillow's coordinates are continuous, with pixel ``u`` covering ``[u, u + 1)``, and it fills what
    falls outside the source with black: *pad* keeps every read inside the padded image.
    """
    xs = np.append(np.arange(0, w, _MESH_STEP), w)
    ys = np.append(np.arange(0, h, _MESH_STEP), h)
    gx, gy = np.meshgrid(xs.astype(np.float64) - 0.5, ys.astype(np.float64) - 0.5)
    mapx, mapy = corr._sampling_maps(gx, gy)
    outside = max(-mapx.min(), mapx.max() - (w - 1), -mapy.min(), mapy.max() - (h - 1), 0.0)
    pad = int(np.ceil(outside)) + 2
    sx, sy = (mapx + 0.5 + pad).tolist(), (mapy + 0.5 + pad).tolist()
    mesh = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            box = (int(xs[i]), int(ys[j]), int(xs[i + 1]), int(ys[j + 1]))
            mesh.append((box, (sx[j][i], sy[j][i], sx[j + 1][i], sy[j + 1][i],  # NW, SW, SE, NE
                               sx[j + 1][i + 1], sy[j + 1][i + 1], sx[j][i + 1], sy[j][i + 1])))
    return pad, mesh


def _warp_pillow(img: np.ndarray, corr: ImageCorrection) -> np.ndarray:
    """Bilinear warp when OpenCV is unavailable, the edges repeated as ``cv2.BORDER_REPLICATE`` does.

    In float, one channel at a time: on uint8 Pillow truncates, which would darken
    the image by half a grey level (against the left image of a stereo pair).
    """
    from PIL import Image

    h, w = img.shape[:2]
    pad, mesh = _pillow_mesh(h, w, corr)
    src = img if img.ndim == 3 else img[:, :, None]
    padded = np.pad(src, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    channels = [
        np.asarray(Image.fromarray(np.ascontiguousarray(padded[:, :, c], dtype=np.float32))
                   .transform((w, h), Image.MESH, mesh, resample=Image.BILINEAR))
        for c in range(src.shape[2])
    ]
    out = np.stack(channels, -1) if img.ndim == 3 else channels[0]
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(img.dtype, copy=False)


def warp_image(
    image: np.ndarray,
    corr: ImageCorrection,
    interpolation: Optional[int] = None,
) -> np.ndarray:
    """Correct a stored image.

    Do not apply this to :meth:`kitscenes.Frame.image`, which is corrected already.

    Args:
        image: The stored image, ``(H, W)`` or ``(H, W, C)``.
        corr: The camera's correction, at the size of this image
            (:meth:`ImageCorrection.scaled`).
        interpolation: An OpenCV interpolation flag. Defaults to
            ``cv2.INTER_LANCZOS4`` when OpenCV is present. Without OpenCV, which
            kitscenes does not require, Pillow does a bilinear warp (about 0.5 s
            for a full-size image, against some 30 ms) and this is ignored.

    Returns:
        A new corrected image, same shape and dtype as the input.

    Raises:
        ImageSizeError: If the image does not have the size of *corr*.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D image, got shape {image.shape}")
    h, w = image.shape[:2]
    corr._check_size(h, w)
    if corr.is_identity:
        return image.copy()
    try:
        import cv2
    except ImportError:
        return _warp_pillow(image, corr)
    mapx, mapy = _maps_float32(h, w, corr)
    flag = cv2.INTER_LANCZOS4 if interpolation is None else interpolation
    return cv2.remap(image, mapx, mapy, flag, borderMode=cv2.BORDER_REPLICATE)
