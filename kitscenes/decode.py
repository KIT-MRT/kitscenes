"""JPEG decoding at the model's input size, batched on the GPU, with the intrinsics to match.

The fast training path: the ``DataLoader`` workers only read the JPEG bytes
(``SampleSpec(image="bytes")``), and the training loop decodes and resizes
each batch on the GPU (nvJPEG through ``torchvision.io.decode_jpeg``)::

    import kitscenes
    from kitscenes.decode import decode_cameras

    spec = kitscenes.SampleSpec(cameras=("camera_ring_front",), image="bytes", reduce=4)
    loader = DataLoader(kitscenes.SampleDataset(ds, spec), batch_size=8, num_workers=4,
                        collate_fn=kitscenes.collate)
    for batch in loader:
        batch = decode_cameras(batch, spec, device="cuda")       # or size=(800, 520), scale=0.3
        batch["camera"]["camera_ring_front"]["image"]           # (8, 3, H/4, W/4) uint8 on the GPU
        batch["camera"]["camera_ring_front"]["K"]               # (8, 3, 3), scaled to match

:func:`decode_images` is the same for a list of JPEGs (``frame.image_bytes()``,
``frame.sensor_file(camera).read()``). Resizing keeps pixel centres on
integers, as :meth:`kitscenes.Camera.scaled` does: :func:`scale_intrinsics`
maps the full-size ``K`` to the resized image, per axis.

:func:`decode_cameras` also applies the image corrections of the samples
(:mod:`kitscenes.image_corrections`); :func:`apply_image_corrections` is that
step alone, on the device of the images, e.g. after a DALI pipeline.

Backends: ``torchvision.io.decode_jpeg`` on a CUDA device (one batched call
per chunk with torchvision >= 0.19, one call per image before), otherwise the
CPU (OpenCV, else Pillow) with JPEG DCT scaling to the largest power-of-two
reduction (1/2, 1/4, 1/8) that is not smaller than the target, then resizing
the rest with torch. The CPU path is for debugging and machines without a
GPU; it is several times slower. Needs torch (``pip install
"kitscenes[torch]"``); torchvision for the GPU path (install the build that
matches your torch).
"""

from __future__ import annotations

import io
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Sequence, Union

import numpy as np

from kitscenes.errors import MissingDependencyError

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from kitscenes.samples import SampleSpec

__all__ = ["DecodedImages", "decode_images", "decode_cameras", "scale_intrinsics", "apply_image_corrections"]

#: Oldest torch decoding works with (``torchvision.io.decode_jpeg`` with ``device=``, checked on torch 1.11 /
#: torchvision 0.12); the API's global :data:`kitscenes._optional.TORCH_MIN_VERSION` is for the GPU voxel maps.
#: Note: torchvision's nvJPEG leaks GPU memory with CUDA < 11.6; on such builds decode on the CPU or with DALI.
DECODE_TORCH_MIN_VERSION = "1.10"

#: Interpolation modes of :func:`decode_images` (``torch.nn.functional.interpolate``;
#: bilinear and bicubic with antialiasing when shrinking).
INTERPOLATIONS = ("bilinear", "bicubic", "area", "nearest")

#: torchvision decodes a list of JPEGs in one call from this version on.
_TV_BATCHED = (0, 19)

Size = tuple[int, int]
Scale = Union[float, tuple[float, float]]


class DecodedImages(NamedTuple):
    """What :func:`decode_images` returns.

    Attributes:
        images: (N, 3, H, W) RGB on the device; uint8, or float in [0, 1].
        scale: (N, 2) float64 on the CPU: ``(sx, sy)`` = output size / JPEG size,
            per axis (the factors of :func:`scale_intrinsics`).
        K: The given intrinsics scaled to the output (:func:`scale_intrinsics`),
            same type, dtype and device; None if none were given.
    """

    images: "torch.Tensor"
    scale: "torch.Tensor"
    K: Optional[Any] = None


def scale_intrinsics(K, scale):
    """Intrinsics for an image resized by *scale*, pixel centres on integers (as :meth:`kitscenes.Camera.scaled`).

    Row 0 of ``K`` is multiplied by ``sx`` and row 1 by ``sy``, then
    ``cx += 0.5 * sx - 0.5`` and ``cy += 0.5 * sy - 0.5``: the centre of the
    top-left pixel stays the centre of the resized one's. With
    ``sx = sy = f`` this is ``Camera.scaled(f).K``.

    Args:
        K: (..., 3, 3) numpy array or torch tensor.
        scale: A factor for both axes, ``(sx, sy)``, or (..., 2) factors
            (e.g. :attr:`DecodedImages.scale`), broadcast against ``K``'s
            leading dimensions.

    Returns:
        The scaled ``K``, same type (and dtype, device) as *K*, floating point.
    """
    if _is_tensor(K):
        import torch

        K = K if K.is_floating_point() else K.double()
        s = torch.as_tensor(scale, dtype=K.dtype).to(K.device)
        if s.ndim == 0:
            s = s.repeat(2)
        sx, sy = s[..., 0], s[..., 1]
        one = torch.ones_like(sx)
        rows = torch.stack([sx, sy, one], -1)[..., :, None]  # (..., 3, 1)
        shift = torch.stack([0.5 * sx - 0.5, 0.5 * sy - 0.5, one - 1], -1)  # (..., 3)
        out = K * rows
        return torch.cat([out[..., :2], out[..., 2:] + shift[..., :, None]], -1)
    K = np.asarray(K)
    K = K if np.issubdtype(K.dtype, np.floating) else K.astype(np.float64)
    s = np.asarray(scale, dtype=K.dtype)
    if s.ndim == 0:
        s = np.repeat(s, 2)
    sx, sy = s[..., 0], s[..., 1]
    one = np.ones_like(sx)
    out = K * np.stack([sx, sy, one], -1)[..., :, None]
    out[..., :, 2] += np.stack([0.5 * sx - 0.5, 0.5 * sy - 0.5, one - 1], -1)
    return out


def decode_images(
    data: Sequence[Any],
    *,
    size: Optional[Size] = None,
    scale: Optional[Scale] = None,
    device="cuda",
    dtype=None,
    interpolation: str = "bilinear",
    K=None,
    backend: str = "auto",
    chunk: int = 32,
    image_corrections=None,
) -> DecodedImages:
    """Decode JPEGs into one (N, 3, H, W) RGB tensor on *device*, resized to *size* or by *scale*.

    Args:
        data: The JPEGs: ``bytes``, ``bytearray``, ``memoryview``, 1-D uint8
            numpy arrays or CPU tensors.
        size: Output ``(width, height)`` (the order of
            :attr:`kitscenes.Camera.image_size`), the same for every image.
        scale: Output size / JPEG size, one factor or ``(sx, sy)``; the output
            is ``round(width * sx) x round(height * sy)``. Neither: full size.
        device: Torch device of the output (and of the decoding, on CUDA).
        dtype: ``torch.uint8`` (default) or a float dtype (values in [0, 1]).
        interpolation: One of :data:`INTERPOLATIONS`.
        K: Optional full-size intrinsics, (3, 3) or (N, 3, 3), numpy or torch;
            returned scaled to the output in :attr:`DecodedImages.K`.
        backend: ``"auto"`` (nvJPEG on a CUDA device if torchvision is
            installed, else the CPU), ``"nvjpeg"`` or ``"cpu"`` (OpenCV or
            Pillow, then moved to *device*).
        chunk: Images per decoding call (bounds the full-size images held
            on the device at once).
        image_corrections: Optional corrections of the JPEGs, one per image or None,
            as the samples hold them, applied after resizing (:func:`apply_image_corrections`).

    Returns:
        :class:`DecodedImages` ``(images, scale, K)``.

    Raises:
        MissingDependencyError: Without torch (or torch < 1.10), or without
            torchvision for ``backend="nvjpeg"``.
        ValueError: For both *size* and *scale*, images that would come out in
            different sizes (pass *size*), or a bad argument.
    """
    from kitscenes._optional import require_torch

    torch = require_torch("kitscenes.decode", DECODE_TORCH_MIN_VERSION)
    if size is not None and scale is not None:
        raise ValueError("pass size or scale, not both")
    if size is not None:
        size = (int(size[0]), int(size[1]))
        if min(size) < 1:
            raise ValueError(f"size must be positive (width, height), got {size}")
    if scale is not None:
        sx, sy = (float(scale), float(scale)) if np.ndim(scale) == 0 else (float(scale[0]), float(scale[1]))
        if not (sx > 0 and sy > 0):
            raise ValueError(f"scale must be > 0, got {scale}")
        scale = (sx, sy)
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}")
    dtype = torch.uint8 if dtype is None else dtype
    if dtype != torch.uint8 and not dtype.is_floating_point:
        raise ValueError(f"dtype must be torch.uint8 or a float dtype, got {dtype}")
    if backend not in ("auto", "nvjpeg", "cpu"):
        raise ValueError(f"backend must be 'auto', 'nvjpeg' or 'cpu', got {backend!r}")
    data = list(data)
    if not data:
        raise ValueError("no images to decode")
    device = torch.device(device)
    use_nvjpeg = _use_nvjpeg(backend, device)

    images, factors = [], []
    for start in range(0, len(data), max(1, chunk)):
        part = data[start:start + max(1, chunk)]
        if use_nvjpeg:
            decoded = _decode_nvjpeg(part, device)
            full = [(int(im.shape[2]), int(im.shape[1])) for im in decoded]
        else:
            full = [_jpeg_size(d) for d in part]
            decoded = [_decode_cpu(d, wh, _target(*wh, size, scale)).to(device) for d, wh in zip(part, full)]
        for image, (w, h) in zip(decoded, full):
            out_w, out_h = _target(w, h, size, scale)
            images.append(_resize(torch, image, out_w, out_h, interpolation, dtype))
            factors.append((out_w / w, out_h / h))
        del decoded
    shapes = {tuple(im.shape) for im in images}
    if len(shapes) > 1:
        raise ValueError(f"the images come out in different sizes {sorted(s[1:] for s in shapes)}; "
                         "pass size=(width, height), or decode each camera separately")
    scale_t = torch.tensor(factors, dtype=torch.float64)
    scaled_K = None
    if K is not None:
        scaled_K = scale_intrinsics(K, scale_t.to(K.dtype) if _is_tensor(K) else scale_t.numpy())
    stacked = torch.stack(images)
    if image_corrections is not None:
        stacked = apply_image_corrections(stacked, image_corrections, scale_t)
    return DecodedImages(stacked, scale_t, scaled_K)


def apply_image_corrections(images, corrections, scale=None):
    """Warp decoded images on their device (:mod:`kitscenes.image_corrections`).

    Bicubic, so the result differs from :func:`kitscenes.image_corrections.warp_image`
    (Lanczos with OpenCV, else bilinear) by interpolation only.

    Args:
        images: (N, 3, H, W) tensor, uint8 or float in [0, 1], not yet corrected.
        corrections: One :class:`~kitscenes.image_corrections.ImageCorrection` per image,
            for its stored JPEG, or None to leave that image as it is: the
            ``"image_correction"`` of a batch.
        scale: Size of *images* / size of the stored JPEG: one factor, ``(sx, sy)``,
            or (N, 2). Default: the size of the JPEG.

    Returns:
        The corrected images, a new tensor with the type of *images*.

    Raises:
        MissingDependencyError: Without torch (or torch < 1.10).
        ValueError: If *images* and *corrections* do not have the same length.
        ImageSizeError: If *scale* does not fit the size of the images.
    """
    from kitscenes._optional import require_torch

    torch = require_torch("kitscenes.decode", DECODE_TORCH_MIN_VERSION)
    corrections = list(corrections)
    if len(corrections) != len(images):
        raise ValueError(f"{len(images)} images, but {len(corrections)} corrections")
    factors = np.ones((len(images), 2)) if scale is None else (
        scale.detach().cpu().numpy() if _is_tensor(scale) else np.asarray(scale, dtype=np.float64))
    factors = np.broadcast_to(factors, (len(images), 2))
    out = images.clone()
    for k, (corr, factor) in enumerate(zip(corrections, factors)):
        if corr is not None and not corr.is_identity:
            out[k] = _warp_on_device(torch, images[k], corr.scaled(tuple(factor)))
    return out


def _warp_on_device(torch, image, corr):
    """*image* (3, H, W) warped with *corr*, at the size of the image."""
    _, h, w = image.shape
    corr._check_size(h, w)
    u = torch.arange(w, dtype=torch.float32, device=image.device)[None, :]
    v = torch.arange(h, dtype=torch.float32, device=image.device)[:, None]
    mapx, mapy = corr._sampling_maps(u, v)
    grid = torch.stack([(mapx * (2 / max(w - 1, 1)) - 1).expand(h, w), (mapy * (2 / max(h - 1, 1)) - 1).expand(h, w)], -1)
    warped = torch.nn.functional.grid_sample(image[None].float(), grid[None], mode="bicubic", padding_mode="border",
                                             align_corners=True)[0]
    if image.dtype == torch.uint8:
        return warped.round_().clamp_(0, 255).to(torch.uint8)
    return warped.clamp_(0, 1).to(image.dtype)


def decode_cameras(
    batch: Union[dict, list],
    spec: Optional["SampleSpec"] = None,
    *,
    size: Optional[Size] = None,
    scale: Optional[Scale] = None,
    device="cuda",
    dtype=None,
    interpolation: str = "bilinear",
    backend: str = "auto",
) -> Union[dict, list]:
    """Decode the JPEG bytes of a collated batch (``SampleSpec(image="bytes")``) on *device*, with ``K`` to match.

    Every camera's ``image`` (a list of B JPEGs after :func:`kitscenes.collate`)
    becomes a (B, 3, H, W) tensor on *device* (channels first, like the
    batches of ``image="decoded"``) and its ``K`` (the full-size ``K``
    of the bytes) the ``K`` of the decoded images, on the device it was on.
    The rest of the batch is passed through. Clips (a list of batches) give
    a list. An entry's ``"image_correction"`` (:class:`kitscenes.SampleSpec`) is applied and removed.

    Args:
        batch: A collated batch, or the list of batches of a clip.
        spec: The batch's spec: its ``reduce`` is the default scale (``1 / reduce``).
        size, scale, device, dtype, interpolation, backend: As in
            :func:`decode_images` (*size* or *scale* overrides ``spec.reduce``).

    Returns:
        A new batch (the input is not modified).

    Raises:
        TypeError: If a camera's images are already decoded (``image="decoded"``).
    """
    if isinstance(batch, list):
        return [decode_cameras(b, spec, size=size, scale=scale, device=device, dtype=dtype,
                               interpolation=interpolation, backend=backend) for b in batch]
    if size is None and scale is None and spec is not None and spec.reduce > 1:
        scale = 1 / spec.reduce
    options = dict(size=size, scale=scale, device=device, dtype=dtype, interpolation=interpolation, backend=backend)
    cameras = {}
    for name, entry in batch.get("camera", {}).items():
        _check_bytes(name, entry["image"])
        rest = {k: v for k, v in entry.items() if k != "image_correction"}
        decoded = decode_images(entry["image"], K=entry["K"], image_corrections=entry.get("image_correction"), **options)
        cameras[name] = {**rest, "image": decoded.images, "K": decoded.K}
    return {**batch, "camera": cameras}


def _check_bytes(name: str, images) -> None:
    if not isinstance(images, (list, tuple)) or any(getattr(image, "ndim", 1) > 1 for image in images):
        raise TypeError(f"camera {name}: the images are already decoded; decode_cameras needs "
                        "SampleSpec(image='bytes')")


def _is_tensor(x) -> bool:
    return type(x).__module__.split(".", 1)[0] == "torch"


def _target(w: int, h: int, size: Optional[Size], scale: Optional[tuple[float, float]]) -> Size:
    """The output (width, height) of a w x h image."""
    if size is not None:
        return size
    if scale is not None:
        return max(1, round(w * scale[0])), max(1, round(h * scale[1]))
    return w, h


def _use_nvjpeg(backend: str, device) -> bool:
    if backend == "cpu":
        return False
    if backend == "nvjpeg":
        if device.type != "cuda":
            raise ValueError(f"backend='nvjpeg' decodes on a CUDA device, not {device}")
        _torchvision()
        return True
    if device.type != "cuda":
        return False
    try:
        _torchvision()
    except MissingDependencyError:
        warnings.warn("kitscenes.decode: torchvision is not installed; decoding on the CPU and resizing on "
                      f"{device} (install torchvision for nvJPEG decoding on the GPU)", stacklevel=3)
        return False
    return True


def _torchvision():
    """torchvision.io, or MissingDependencyError with what to install."""
    try:
        import torchvision.io

        return torchvision
    except ImportError as exc:
        raise MissingDependencyError(
            f"Decoding JPEGs on the GPU needs torchvision ({exc}). Install the torchvision build that "
            "matches your torch: https://pytorch.org/get-started/locally/"
        ) from exc


def _tv_batched(torchvision) -> bool:
    found = tuple(int(p) for p in torchvision.__version__.split("+", 1)[0].split(".")[:2] if p.isdigit())
    return found >= _TV_BATCHED


def _as_uint8_tensor(data):
    import torch

    if _is_tensor(data):
        return data.reshape(-1).to("cpu", torch.uint8)
    if isinstance(data, np.ndarray):
        flat = np.ascontiguousarray(data.reshape(-1), dtype=np.uint8)
        return torch.from_numpy(flat if flat.flags.writeable else flat.copy())  # torch warns on read-only arrays
    return torch.frombuffer(bytearray(data), dtype=torch.uint8)


def _as_bytes(data) -> Union[bytes, memoryview, np.ndarray]:
    if _is_tensor(data):
        return data.reshape(-1).cpu().numpy()
    return data


def _decode_nvjpeg(part: Sequence[Any], device) -> list:
    """JPEGs -> (3, H, W) uint8 tensors on *device*, full size, with torchvision (nvJPEG)."""
    torchvision = _torchvision()
    from torchvision.io import ImageReadMode, decode_jpeg

    tensors = [_as_uint8_tensor(d) for d in part]
    if _tv_batched(torchvision):
        return list(decode_jpeg(tensors, mode=ImageReadMode.RGB, device=device))
    return [decode_jpeg(t, mode=ImageReadMode.RGB, device=device) for t in tensors]


def _decode_on_device(images: Sequence[Any], device, chunk: int = 48) -> list:
    """JPEGs -> full-size (3, H, W) uint8 tensors on a CUDA *device*, in chunks of batched nvJPEG calls."""
    out: list = []
    for start in range(0, len(images), chunk):
        out.extend(_decode_nvjpeg(images[start:start + chunk], device))
    return out


def _jpeg_size(data) -> Size:
    """(width, height) from the JPEG header (Pillow reads only the header)."""
    from PIL import Image

    with Image.open(io.BytesIO(_as_bytes(data))) as img:
        return img.size


def _reduction(w: int, h: int, target: Size) -> int:
    """The largest JPEG DCT reduction (1, 2, 4, 8) whose output is at least *target*."""
    for r in (8, 4, 2):
        if -(-w // r) >= target[0] and -(-h // r) >= target[1]:
            return r
    return 1


def _decode_reduced(data: bytes, reduce: int = 1) -> np.ndarray:
    """A JPEG -> (h, w, 3) uint8 RGB array, DCT-reduced by *reduce* (1, 2, 4 or 8): OpenCV if installed, else Pillow."""
    try:
        import cv2
    except ImportError:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            if reduce > 1:
                img.draft("RGB", (img.size[0] // reduce, img.size[1] // reduce))
            return np.array(img.convert("RGB"))
    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
            8: cv2.IMREAD_REDUCED_COLOR_8}[reduce]
    array = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    if array is None:
        raise ValueError("not a decodable JPEG")
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def _decode_cpu(data, full: Size, target: Size):
    """A w x h JPEG -> (3, h', w') uint8 CPU tensor, DCT-reduced towards *target*."""
    import torch

    array = _decode_reduced(_as_bytes(data), _reduction(*full, target))
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)


def _resize(torch, image, out_w: int, out_h: int, interpolation: str, dtype):
    """(3, h, w) uint8 -> (3, out_h, out_w) of *dtype* (float in [0, 1]), on the image's device."""
    if image.shape[1:] == (out_h, out_w):
        return image if dtype == torch.uint8 else image.to(dtype) / 255
    x = image[None].float()
    if interpolation in ("bilinear", "bicubic"):
        x = torch.nn.functional.interpolate(x, size=(out_h, out_w), mode=interpolation, align_corners=False,
                                            antialias=out_h < image.shape[1] or out_w < image.shape[2])
    else:
        x = torch.nn.functional.interpolate(x, size=(out_h, out_w), mode=interpolation)
    x = x[0]
    if dtype == torch.uint8:
        return x.round_().clamp_(0, 255).to(torch.uint8)
    return (x / 255).clamp_(0, 1).to(dtype)
