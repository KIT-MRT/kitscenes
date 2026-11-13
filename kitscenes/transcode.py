"""Downscaled camera images: re-encode the release JPEGs at 1/2, 1/4 or 1/8 size to save disk.

A recipe (:class:`ImageRecipe`) is the reduction, the JPEG quality and the
chroma subsampling. Each transcoded camera of a scene has a record
``<scene>/.transcoded/<camera>.json`` (:class:`ImageVariant`); ``calib.json``
stays as released and ``scene.rig`` follows the record. Backends: ``cpu``
(Pillow) and ``gpu`` (NVIDIA DALI and nvImageCodec, in separate processes
that load nvImageCodec before anything else). A script that calls these
functions needs an ``if __name__ == "__main__":`` guard. See docs/transcode.md.
"""

from __future__ import annotations

import collections
import concurrent.futures as cf
import dataclasses
import io
import json
import logging
import math
import multiprocessing
import os
import shutil
import threading
from pathlib import Path
from typing import Callable, Iterable, Iterator, NamedTuple, Optional, Sequence, Union

from kitscenes._archive import ALIGN, ARCHIVE_SUFFIX, INDEX_FILE, write_index
from kitscenes.constants import CAMERA_NAMES
from kitscenes.errors import KITScenesError
from kitscenes.splits import _split_name

logger = logging.getLogger(__name__)

__all__ = [
    "ImageRecipe", "ImageVariant", "Transcoder", "TranscodeResult", "PRESETS",
    "transcode_jpeg", "transcode_camera", "transcode", "read_variant", "read_variants", "RECORD_DIR",
]

#: Per scene: ``<camera>.json`` (:class:`ImageVariant`) and, for a zip, ``<camera>.parquet`` (its index rows).
RECORD_DIR = ".transcoded"
RECORD_FORMAT = 1
#: Chroma subsampling -> Pillow's ``subsampling`` code.
SUBSAMPLINGS = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}
REDUCTIONS = (1, 2, 4, 8)

#: Named recipes for ``--images``: ``(reduce, quality, subsampling)``.
PRESETS: dict[str, tuple[int, int, str]] = {
    "half": (2, 90, "4:2:0"),
    "quarter": (4, 90, "4:2:0"),
    "eighth": (8, 90, "4:2:0"),
    "full": (1, 90, "4:2:0"),
}
#: ``--images`` value that means "the release's images" (undo a recipe).
ORIGINAL = "original"

#: Images per GPU batch.
GPU_BATCH = 16
#: Worker processes per GPU: one worker's CPU steps (entropy decoding and coding) leave the GPU idle half
#: the time.
GPU_WORKERS_PER_DEVICE = 2
#: CPU threads per GPU worker (entropy decoding in DALI, entropy coding in nvImageCodec).
GPU_CPU_THREADS = 8
#: Images per CPU task.
CPU_CHUNK = 4


# ---------------------------------------------------------------------------
# Recipe and record
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ImageRecipe:
    """How to re-encode camera images: reduce by *reduce* per axis, JPEG *quality*, chroma *subsampling*.

    ``reduce`` is relative to the release's images (1 keeps the size and only re-encodes).
    """

    reduce: int = 2
    quality: int = 90
    subsampling: str = "4:2:0"

    def __post_init__(self) -> None:
        if self.reduce not in REDUCTIONS:
            raise ValueError(f"reduce must be one of {REDUCTIONS}, got {self.reduce!r}")
        if not (isinstance(self.quality, int) and 1 <= self.quality <= 100):
            raise ValueError(f"quality must be an integer 1..100, got {self.quality!r}")
        if self.subsampling not in SUBSAMPLINGS:
            raise ValueError(f"subsampling must be one of {sorted(SUBSAMPLINGS)}, got {self.subsampling!r}")

    def __str__(self) -> str:
        size = "full size" if self.reduce == 1 else f"1/{self.reduce}"
        return f"{size}, q{self.quality}, {self.subsampling}"

    @classmethod
    def parse(cls, text: Union[str, "ImageRecipe"]) -> "ImageRecipe":
        """A preset (``half``, ``quarter``, ``eighth``, ``full``) or ``key=value`` pairs, comma-separated:
        ``scale=1/4,quality=85,subsampling=4:4:4`` (unset keys keep the ``half`` values, or the preset's
        when the text starts with one: ``quarter,quality=95``)."""
        if isinstance(text, ImageRecipe):
            return text
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
        if not parts:
            raise ValueError("empty image recipe")
        values = dict(zip(("reduce", "quality", "subsampling"), PRESETS["half"]))
        if "=" not in parts[0]:
            name = parts.pop(0).lower()
            if name not in PRESETS:
                raise ValueError(f"unknown image recipe {name!r}; presets: {', '.join(PRESETS)} "
                                 "(or scale=1/4,quality=85,subsampling=4:2:0)")
            values = dict(zip(("reduce", "quality", "subsampling"), PRESETS[name]))
        for part in parts:
            key, sep, value = part.partition("=")
            key, value = key.strip().lower(), value.strip()
            if not sep or not value:
                raise ValueError(f"expected key=value in the image recipe, got {part!r}")
            if key == "scale":
                num, _, den = value.partition("/")
                try:
                    factor = float(num) / float(den) if den else float(num)
                except ValueError:
                    raise ValueError(f"scale is 1, 1/2, 1/4 or 1/8, got {value!r}") from None
                if factor <= 0 or round(1 / factor) not in REDUCTIONS or abs(1 / factor - round(1 / factor)) > 1e-9:
                    raise ValueError(f"scale is 1, 1/2, 1/4 or 1/8, got {value!r}")
                values["reduce"] = round(1 / factor)
            elif key == "quality":
                try:
                    values[key] = int(value)
                except ValueError:
                    raise ValueError(f"quality is an integer, got {value!r}") from None
            elif key == "subsampling":
                values[key] = value
            else:
                raise ValueError(f"unknown key {key!r} in the image recipe (scale, quality, subsampling)")
        return cls(**values)

    def to_dict(self) -> dict:
        return {"reduce": self.reduce, "quality": self.quality, "subsampling": self.subsampling}

    @classmethod
    def from_dict(cls, d: dict) -> "ImageRecipe":
        return cls(int(d["reduce"]), int(d["quality"]), str(d["subsampling"]))


@dataclasses.dataclass(frozen=True)
class ImageVariant:
    """The record of a transcoded camera of one scene (``<scene>/.transcoded/<camera>.json``).

    Attributes:
        camera: Camera name.
        recipe: What the stored images are, relative to the release's
            (``recipe.reduce`` is the total reduction).
        image_size: ``(width, height)`` of the stored images.
        original_size: ``(width, height)`` of the release's images.
        layout: ``"zip"`` (``<camera>.zip``) or ``"folder"`` (extracted).
        source_sha256: SHA-256 of the release zip the images come from
            (from the manifest), None if unknown (e.g. an extracted copy).
        sha256, size: Of the written zip (None for a folder).
        images: Number of images.
        backend: What encoded them (``cpu (Pillow ...)``, ``gpu (DALI ..., nvImageCodec ...)``).
        kitscenes: The kitscenes version that wrote them.
        history: Earlier recipes the images went through (a camera transcoded twice), oldest first.
    """

    camera: str
    recipe: ImageRecipe
    image_size: tuple[int, int]
    original_size: tuple[int, int]
    layout: str
    source_sha256: Optional[str] = None
    sha256: Optional[str] = None
    size: Optional[int] = None
    images: int = 0
    backend: str = ""
    kitscenes: str = ""
    history: tuple[ImageRecipe, ...] = ()

    @property
    def scale(self) -> float:
        """Stored size / release size (``1 / recipe.reduce``)."""
        return 1.0 / self.recipe.reduce

    def to_json(self) -> str:
        d = {
            "format": RECORD_FORMAT, "camera": self.camera, "recipe": self.recipe.to_dict(),
            "image_size": list(self.image_size), "original_size": list(self.original_size), "layout": self.layout,
            "source_sha256": self.source_sha256, "sha256": self.sha256, "size": self.size, "images": self.images,
            "backend": self.backend, "kitscenes": self.kitscenes, "history": [r.to_dict() for r in self.history],
        }
        return json.dumps(d, indent=1) + "\n"

    @classmethod
    def from_json(cls, text: Union[str, bytes]) -> "ImageVariant":
        d = json.loads(text)
        if int(d.get("format", 1)) > RECORD_FORMAT:
            raise ValueError(f"image record format {d['format']}; this kitscenes reads up to {RECORD_FORMAT}")
        return cls(
            camera=d["camera"], recipe=ImageRecipe.from_dict(d["recipe"]), image_size=tuple(d["image_size"]),
            original_size=tuple(d["original_size"]), layout=d["layout"], source_sha256=d.get("source_sha256"),
            sha256=d.get("sha256"), size=d.get("size"), images=int(d.get("images", 0)),
            backend=d.get("backend", ""), kitscenes=d.get("kitscenes", ""),
            history=tuple(ImageRecipe.from_dict(r) for r in d.get("history", [])),
        )


def record_path(scene_dir: Path, camera: str, prev: bool = False) -> Path:
    """``.transcoded/<camera>.json`` (*prev*: ``<camera>.prev.json``, the record kept while images are swapped)."""
    return Path(scene_dir) / RECORD_DIR / f"{camera}{'.prev' if prev else ''}.json"


def overlay_path(scene_dir: Path, camera: str, prev: bool = False) -> Path:
    return Path(scene_dir) / RECORD_DIR / f"{camera}{'.prev' if prev else ''}.parquet"


def _first_image_size(folder: Path) -> Optional[tuple[int, int]]:
    from PIL import Image

    try:
        with os.scandir(folder) as entries:
            first = min((e.name for e in entries if e.name.endswith(".jpg")), default=None)
        if first is None:
            return None
        with Image.open(folder / first) as im:  # reads the header only
            return im.size
    except OSError:
        return None


def _matches(scene_dir: Path, camera: str, variant: ImageVariant) -> bool:
    """Whether the camera's images on disk are the ones *variant* describes: its zip has the recorded size, or
    (extracted) its first image the recorded image size."""
    zip_path = scene_dir / f"{camera}{ARCHIVE_SUFFIX}"
    try:
        size = zip_path.stat().st_size
    except FileNotFoundError:
        return _first_image_size(scene_dir / camera) == tuple(variant.image_size)
    return variant.size is not None and size == variant.size


def _resolve(scene_dir: Path, camera: str) -> tuple[Optional[ImageVariant], bool]:
    """``(record, is_prev)`` of the record that matches the camera's images on disk, ``(None, False)`` if none does.

    While :func:`transcode_camera` swaps images, the old record is kept as ``<camera>.prev.json`` and the new one
    is already in place, so a crash at any point leaves a record that matches the images (the release's images
    match none).
    """
    for prev in (False, True):
        path = record_path(scene_dir, camera, prev)
        try:
            variant = ImageVariant.from_json(path.read_text())
        except FileNotFoundError:
            continue
        except (ValueError, KeyError, TypeError) as err:
            logger.warning("Unreadable image record %s (%s); ignored", path, err)
            continue
        if _matches(scene_dir, camera, variant):
            return variant, prev
    return None, False


def _recorded_cameras(scene_dir: Path) -> list[str]:
    try:
        names = os.listdir(Path(scene_dir) / RECORD_DIR)
    except (FileNotFoundError, NotADirectoryError):
        return []
    found = {n.split(".", 1)[0] for n in names if n.endswith(".json")}
    return [c for c in CAMERA_NAMES if c in found]


def read_variant(scene_dir: Path | str, camera: str) -> Optional[ImageVariant]:
    """The record of *camera* in a scene folder, None if its images are the release's (no record matches them)."""
    return _resolve(Path(scene_dir), camera)[0]


def read_variants(storage) -> dict[str, ImageVariant]:
    """``{camera: record}`` of a scene's :class:`~kitscenes._storage.Storage` (a scene folder)."""
    root = getattr(storage, "root", None)
    if root is None:
        return {}
    out = {}
    for camera in _recorded_cameras(root):
        variant = read_variant(root, camera)
        if variant is not None:
            out[camera] = variant
    return out


def read_overlays(scene_dir: Path | str) -> list[bytes]:
    """The index rows of the transcoded cameras whose zip is in the scene (the overlay of the matching record)."""
    scene_dir = Path(scene_dir)
    out = []
    for camera in _recorded_cameras(scene_dir):
        variant, prev = _resolve(scene_dir, camera)
        if variant is None or not (scene_dir / f"{camera}{ARCHIVE_SUFFIX}").is_file():
            continue
        try:
            out.append(overlay_path(scene_dir, camera, prev).read_bytes())
        except FileNotFoundError:  # swapped meanwhile: the next open reads the new state
            continue
    return out


def _write_atomic(path: Path, data: Union[str, bytes]) -> Path:
    """Write *data* to ``<path>.partial`` and return that path (rename it with :func:`os.replace`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(data.encode() if isinstance(data, str) else data)
    return tmp


def drop_variant(scene_dir: Path | str, camera: str) -> None:
    """Remove a camera's records and index overlays (not its images)."""
    for prev in (False, True):
        for path in (record_path(Path(scene_dir), camera, prev), overlay_path(Path(scene_dir), camera, prev)):
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# One image (CPU)
# ---------------------------------------------------------------------------


class Source(NamedTuple):
    """Where a JPEG's bytes are: ``size`` bytes at ``offset`` in ``path`` (a zip or a file)."""

    path: str
    offset: int
    size: int


def _read(src: Union[Source, bytes]) -> bytes:
    if not isinstance(src, Source):
        return bytes(src)
    fd = os.open(src.path, os.O_RDONLY)
    try:
        data = os.pread(fd, src.size, src.offset)
    finally:
        os.close(fd)
    if len(data) != src.size:
        raise OSError(f"short read of {src.size} bytes at {src.offset} in {src.path}")
    return data


def transcode_jpeg(data: bytes, recipe: Union[ImageRecipe, str]) -> bytes:
    """One JPEG re-encoded by *recipe* on the CPU (Pillow: DCT-scaled decode, then encode). Metadata is dropped.

    Raises:
        ValueError: If it is not a JPEG, or its size does not divide by ``recipe.reduce``.
    """
    return _cpu_one(data, ImageRecipe.parse(recipe))[0]


def _cpu_one(data: bytes, recipe: ImageRecipe) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    from PIL import Image

    r = recipe.reduce
    with Image.open(io.BytesIO(data)) as im:
        if im.format != "JPEG":
            raise ValueError(f"not a JPEG ({im.format})")
        w, h = im.size
        if w % r or h % r:
            raise ValueError(f"a {w}x{h} image does not divide by {r}")
        if r > 1:
            im.draft("RGB", (w // r, h // r))
        out = im.convert("RGB")
    if out.size != (w // r, h // r):  # not a power-of-two DCT size
        out = out.resize((w // r, h // r), Image.Resampling.BOX)
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=recipe.quality, subsampling=SUBSAMPLINGS[recipe.subsampling], optimize=False)
    return buf.getvalue(), (w, h), out.size


def _cpu_task(sources: Sequence[Union[Source, bytes]], recipe: ImageRecipe):
    return [_cpu_one(_read(s), recipe) for s in sources]


def _cpu_backend_name() -> str:
    import PIL
    from PIL import features

    turbo = features.version("libjpeg_turbo") if features.check_feature("libjpeg_turbo") else None
    return f"cpu (Pillow {PIL.__version__}" + (f", libjpeg-turbo {turbo})" if turbo else ")")


def _noop() -> None:
    """Makes the pool start a worker."""


# ---------------------------------------------------------------------------
# GPU worker (own process: DALI decode + box halving, nvImageCodec encode)
# ---------------------------------------------------------------------------

_GPU: dict = {}


def _gpu_init(devices) -> None:
    """Worker start: take a device, load nvImageCodec before anything else, build the encoder, self-test."""
    device, slot = devices.get()
    # DALI pins its thread i to core i of the GPU's affinity list: workers sharing those cores
    # would not run in parallel. Give each worker its own cores (unless the user set a mask).
    if "DALI_AFFINITY_MASK" not in os.environ and hasattr(os, "sched_getaffinity"):
        cores = sorted(os.sched_getaffinity(0))
        mine = cores[slot * GPU_CPU_THREADS:(slot + 1) * GPU_CPU_THREADS]
        if len(mine) == GPU_CPU_THREADS:
            os.environ["DALI_AFFINITY_MASK"] = ",".join(map(str, mine))
    from nvidia import nvimgcodec  # first: it must load its own nvJPEG (see the module docstring)
    import nvidia.dali as dali

    _GPU.update(device=device, nvimgcodec=nvimgcodec, pipes={}, encoder=None)
    _GPU["name"] = f"gpu (DALI {dali.__version__}, nvImageCodec {nvimgcodec.__version__}, device {device})"
    from PIL import Image

    probe = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 100, 50)).save(probe, "JPEG", quality=95)
    out = _gpu_task([probe.getvalue()], ImageRecipe(2, 90, "4:2:0"))
    if out[0][2] != (32, 24):
        raise RuntimeError(f"GPU self-test gave a {out[0][2]} image, expected (32, 24)")


def _gpu_pipe(reduce: int):
    pipe = _GPU["pipes"].get(reduce)
    if pipe is not None:
        return pipe
    from nvidia.dali import fn, pipeline_def, types

    @pipeline_def(batch_size=GPU_BATCH, num_threads=GPU_CPU_THREADS, device_id=_GPU["device"], prefetch_queue_depth=1)
    def build():
        jpg = fn.external_source(name="jpg", dtype=types.UINT8, ndim=1)
        shape = fn.peek_image_shape(jpg)
        img = fn.decoders.image(jpg, device="mixed", output_type=types.RGB)
        h, w = fn.cast(shape[0], dtype=types.FLOAT), fn.cast(shape[1], dtype=types.FLOAT)
        r = 1
        while r < reduce:  # halvings with linear sampling at the pixel-centre midpoints: exact 2x2 box means
            r *= 2
            img = fn.resize(img, resize_x=w / r, resize_y=h / r, interp_type=types.INTERP_LINEAR, antialias=False)
        return img, shape

    pipe = build()
    pipe.build()
    _GPU["pipes"][reduce] = pipe
    return pipe


def _gpu_task(sources: Sequence[Union[Source, bytes]], recipe: ImageRecipe):
    import numpy as np

    nvimgcodec = _GPU["nvimgcodec"]
    datas = [_read(s) for s in sources]
    pipe = _gpu_pipe(recipe.reduce)
    pipe.feed_input("jpg", [np.frombuffer(d, np.uint8) for d in datas])
    images, shapes = pipe.run()
    if _GPU["encoder"] is None:  # after DALI made the device's CUDA context: before it, nvJPEG cannot allocate
        _GPU["encoder"] = nvimgcodec.Encoder(
            device_id=_GPU["device"], backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)],
            max_num_cpu_threads=GPU_CPU_THREADS)
    params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.QUALITY, quality_value=recipe.quality,
        chroma_subsampling=getattr(nvimgcodec.ChromaSubsampling, "CSS_" + recipe.subsampling.replace(":", "")))
    encoded = _GPU["encoder"].encode([nvimgcodec.as_image(images[i]) for i in range(len(datas))], "jpeg",
                                     params=params)
    out = []
    for i, code in enumerate(encoded):
        if code is None:
            raise RuntimeError("nvImageCodec could not encode on the GPU")
        h, w = (int(v) for v in np.array(shapes[i])[:2])
        if w % recipe.reduce or h % recipe.reduce:
            raise ValueError(f"a {w}x{h} image does not divide by {recipe.reduce}")
        oh, ow = images[i].shape()[:2]
        out.append((bytes(code), (w, h), (int(ow), int(oh))))
    return out


def _gpu_name() -> str:
    return _GPU["name"]


# ---------------------------------------------------------------------------
# Transcoder: a pool of CPU or GPU workers
# ---------------------------------------------------------------------------


class TranscoderUnavailable(KITScenesError, RuntimeError):
    """The requested transcoding backend cannot run here (e.g. ``backend="gpu"`` without DALI)."""


class Transcoder:
    """Worker processes that re-encode JPEGs by a recipe; several threads can stream archives through it.

    Args:
        backend: ``"auto"`` (the GPU if DALI and nvImageCodec work there, else
            the CPU), ``"gpu"`` or ``"cpu"``.
        jobs: CPU worker processes (default: up to 8 cores).
        gpus: CUDA devices for the GPU workers, :data:`GPU_WORKERS_PER_DEVICE` each (default ``[0]``).

    Use as a context manager, or call :meth:`close`.

    Raises:
        TranscoderUnavailable: For ``backend="gpu"`` when the GPU workers cannot start (DALI or
            nvImageCodec missing, no GPU, the self-test failed).
    """

    def __init__(self, backend: str = "auto", *, jobs: Optional[int] = None, gpus: Optional[Sequence[int]] = None) -> None:
        if backend not in ("auto", "gpu", "cpu"):
            raise ValueError(f"backend must be 'auto', 'gpu' or 'cpu', got {backend!r}")
        self.pool: Optional[cf.ProcessPoolExecutor] = None
        self.kind = "cpu"
        if backend in ("auto", "gpu"):
            try:
                self._start_gpu([d for d in (gpus or [0]) for _ in range(GPU_WORKERS_PER_DEVICE)])
            except Exception as err:
                if backend == "gpu":
                    raise TranscoderUnavailable(f"the GPU transcoder does not work here: {err}. Install DALI "
                                       "(pip install nvidia-dali-cuda120, it brings nvImageCodec) or use the CPU.") from err
                logger.info("GPU transcoding unavailable (%s); using the CPU", err)
        if self.pool is None:
            self._start_cpu(jobs or min(os.cpu_count() or 1, 8))

    def _start_gpu(self, gpus: list[int]) -> None:
        import importlib.util

        for module in ("nvidia.dali", "nvidia.nvimgcodec"):
            try:
                found = importlib.util.find_spec(module) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                raise RuntimeError(f"{module} is not installed")
        ctx = multiprocessing.get_context("spawn")  # CUDA, and a fresh process for nvImageCodec's nvJPEG
        devices = ctx.Queue()
        for slot, d in enumerate(gpus):
            devices.put((int(d), slot))
        pool = cf.ProcessPoolExecutor(len(gpus), mp_context=ctx, initializer=_gpu_init, initargs=(devices,))
        try:  # all at once: the pool starts a worker only when none is idle
            names = {f.result(timeout=300) for f in [pool.submit(_gpu_name) for _ in gpus]}
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        self.pool, self.kind, self.workers = pool, "gpu", len(gpus)
        self.name = "; ".join(sorted(names)) if len(names) > 1 else names.pop()
        self._chunk, self._task = GPU_BATCH, _gpu_task

    def _start_cpu(self, jobs: int) -> None:
        # forkserver: the workers never fork this (often multi-threaded: downloads) process; spawn where there is none
        methods = multiprocessing.get_all_start_methods()
        ctx = multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")
        self.pool = cf.ProcessPoolExecutor(jobs, mp_context=ctx)
        for f in [self.pool.submit(_noop) for _ in range(jobs)]:
            f.result()
        self.kind, self.workers, self.name = "cpu", jobs, _cpu_backend_name()
        self._chunk, self._task = CPU_CHUNK, _cpu_task

    def __enter__(self) -> "Transcoder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def __repr__(self) -> str:
        return f"Transcoder({self.name}, {self.workers} worker(s))"

    def map(self, sources: Iterable[Union[Source, bytes]], recipe: ImageRecipe,
            ) -> Iterator[tuple[bytes, tuple[int, int], tuple[int, int]]]:
        """Re-encode JPEGs (bytes, or :class:`Source` locations the workers read themselves) in order.

        Yields:
            ``(jpeg, (width, height) in, (width, height) out)`` per source.
        """
        if self.pool is None:
            raise RuntimeError("the transcoder is closed")
        window = 2 * self.workers  # tasks in flight
        pending: collections.deque = collections.deque()
        chunk: list = []

        def submit(part):
            pending.append(self.pool.submit(self._task, part, recipe))

        for src in sources:
            chunk.append(src)
            if len(chunk) == self._chunk:
                submit(chunk)
                chunk = []
                while len(pending) > window:
                    yield from pending.popleft().result()
        if chunk:
            submit(chunk)
        try:
            while pending:
                yield from pending.popleft().result()
        finally:
            for f in pending:
                f.cancel()


# ---------------------------------------------------------------------------
# One camera of one scene
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TranscodeResult:
    """What :func:`transcode` did."""

    done: list[tuple[str, str]] = dataclasses.field(default_factory=list)  # (scene folder, camera)
    skipped: list[tuple[str, str]] = dataclasses.field(default_factory=list)  # already in the recipe
    failed: list[tuple[str, str, str]] = dataclasses.field(default_factory=list)  # (scene folder, camera, error)
    bytes_before: int = 0
    bytes_after: int = 0


def _camera_sources(storage, camera: str) -> tuple[list[str], list[Source]]:
    names = [n for n in storage.list(camera) if n.endswith(".jpg")]
    sources = []
    for name in names:
        path, offset, size, _ = storage.locate(name)
        sources.append(Source(str(path), int(offset), int(size)))
    return names, sources


def transcode_camera(
    scene_dir: Path | str,
    camera: str,
    recipe: Union[ImageRecipe, str],
    transcoder: Transcoder,
    *,
    out_dir: Path | str | None = None,
    source_sha256: Optional[str] = None,
    source_zip: Path | str | None = None,
) -> Optional[ImageVariant]:
    """Re-encode one camera of a scene folder (its zip or its extracted folder) by *recipe*.

    In place (*out_dir* None) the camera's zip or folder is replaced; with
    *out_dir* (another scene folder of the same name) the result goes there
    and the source is only read. The result has the layout of the source: a
    zip stays a zip, an extracted folder a folder. A camera that is already
    in the recipe is left alone (returns None). A camera already reduced is
    reduced further from its stored images (the record keeps the history);
    a larger size than stored needs the release's images.

    Args:
        source_sha256: SHA-256 of the release zip the images come from, for the
            record (the downloader passes the manifest's).
        source_zip: Read the release's images from this zip instead of the scene
            folder (a download staged elsewhere); the result goes to *out_dir*
            (default the scene folder), replacing what is there.

    Returns:
        The new record, or None if nothing was done.

    Raises:
        ValueError: If the scene has no images of *camera*, or the recipe would enlarge them.
    """
    from kitscenes import __version__
    from kitscenes._storage import ZipStorage, open_storage
    from kitscenes.pack import check_zip, index_rows, write_aligned_zip

    recipe = ImageRecipe.parse(recipe)
    scene_dir = Path(scene_dir)
    out_dir = Path(out_dir) if out_dir is not None else scene_dir
    _finish_folder_swap(out_dir, camera)
    zip_path = scene_dir / f"{camera}{ARCHIVE_SUFFIX}"
    if source_zip is not None:
        layout, before = "zip", None
    else:
        storage = open_storage(scene_dir)
        layout = "zip" if isinstance(storage, ZipStorage) and zip_path.is_file() else "folder"
        if layout == "folder" and not (scene_dir / camera).is_dir():
            raise ValueError(f"{scene_dir} has no images of {camera} (neither {camera}.zip nor {camera}/)")
        before = read_variant(scene_dir, camera)
    have = before.recipe.reduce if before else 1
    if before is not None and before.recipe == recipe:
        return None
    if recipe.reduce < have or recipe.reduce % have:
        raise ValueError(f"{scene_dir.name}/{camera} is stored at 1/{have}; {recipe} needs the release's images "
                         f"(download them again: --images {ORIGINAL})")
    step = ImageRecipe(recipe.reduce // have, recipe.quality, recipe.subsampling)
    if source_zip is not None:
        from kitscenes.pack import read_members

        prefix = f"{scene_dir.name}/{camera}/"
        members = [m for m in read_members(source_zip) if m.name.startswith(prefix) and m.name.endswith(".jpg")]
        names = [m.name.split("/", 1)[1] for m in members]
        sources = [Source(str(source_zip), m.data_offset, m.size) for m in members]
    else:
        names, sources = _camera_sources(storage, camera)
    if not names:
        raise ValueError(f"{scene_dir.name}: {camera} has no images")
    sizes: dict[str, tuple] = {}

    def produce() -> Iterator[tuple[str, bytes]]:
        for name, (data, size_in, size_out) in zip(names, transcoder.map(sources, step)):
            if sizes.setdefault("in", size_in) != size_in or sizes.setdefault("out", size_out) != size_out:
                raise ValueError(f"{scene_dir.name}/{camera}: images of different sizes ({name}: {size_in})")
            yield name, data

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = None
    if layout == "zip":
        tmp = out_dir / f".{camera}{ARCHIVE_SUFFIX}.partial"
        try:
            members = write_aligned_zip(((f"{scene_dir.name}/{n}", d) for n, d in produce()), tmp, ALIGN)
            sha = check_zip(tmp, members)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        rows = index_rows(camera, members)
        size = tmp.stat().st_size
    else:
        tmp = out_dir / f".{camera}.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            for name, data in produce():
                target = tmp / name.split("/", 1)[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        sha = size = None
    variant = ImageVariant(
        camera=camera, recipe=recipe, image_size=tuple(sizes["out"]),
        original_size=(sizes["in"][0] * have, sizes["in"][1] * have),
        layout=layout, source_sha256=source_sha256 or (before.source_sha256 if before else None),
        sha256=sha, size=size, images=len(names), backend=transcoder.name, kitscenes=__version__,
        history=(*before.history, before.recipe) if before is not None else (),
    )
    # Swap in. The old record and overlay become <camera>.prev.*, the new ones go in place, then the images;
    # readers take the record that matches the images on disk (_resolve), so a crash at any point leaves a
    # consistent state. The prev files go last.
    record_tmp = _write_atomic(record_path(out_dir, camera), variant.to_json())
    overlay_tmp = _write_atomic(overlay_path(out_dir, camera), write_index(rows)) if rows is not None else None
    for prev_of, path in ((record_path, record_path(out_dir, camera)), (overlay_path, overlay_path(out_dir, camera))):
        if path.exists():
            os.replace(path, prev_of(out_dir, camera, True))
    if overlay_tmp is not None:
        os.replace(overlay_tmp, overlay_path(out_dir, camera))
    os.replace(record_tmp, record_path(out_dir, camera))
    if layout == "zip":
        os.replace(tmp, out_dir / f"{camera}{ARCHIVE_SUFFIX}")
    else:
        target = out_dir / camera
        old = out_dir / f".{camera}.old"
        shutil.rmtree(old, ignore_errors=True)
        if target.exists():
            os.replace(target, old)
        os.replace(tmp, target)
        shutil.rmtree(old, ignore_errors=True)
    for path in (record_path(out_dir, camera, True), overlay_path(out_dir, camera, True)):
        path.unlink(missing_ok=True)
    return variant


def _finish_folder_swap(scene_dir: Path, camera: str) -> None:
    """After a crash between the two renames of a folder swap: put the old folder back (or drop it)."""
    old = scene_dir / f".{camera}.old"
    if old.is_dir():
        if (scene_dir / camera).is_dir():
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.replace(old, scene_dir / camera)


# ---------------------------------------------------------------------------
# A root
# ---------------------------------------------------------------------------


def find_scenes(root: Path, splits: Optional[Iterable[str]] = None,
                scenes: Optional[Iterable[str]] = None) -> list[tuple[str, Path]]:
    """``(split, scene folder)`` of the scenes ``root/data/<split>/<scene_id>/``, sorted."""
    parent = root / "data"
    if not parent.is_dir():
        raise ValueError(f"{root} has no data/ folder (scenes are expected in data/<split>/<scene_id>/)")
    wanted = {_split_name(s) for s in splits} if splits else None
    ids = set(scenes) if scenes else None
    out = []
    for split_dir in sorted(p for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if wanted is not None and split_dir.name not in wanted:
            continue
        for scene in sorted(p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            if ids is not None and scene.name not in ids:
                continue
            if (scene / "timestamp.reference.txt").is_file():
                out.append((split_dir.name, scene))
    if ids is not None:
        missing = sorted(ids - {p.name for _, p in out})
        if missing:
            raise ValueError(f"scene(s) not in {root}: {missing}")
    return out


def scene_cameras(scene_dir: Path) -> list[str]:
    """Cameras with images in a scene folder (a zip or an extracted folder)."""
    return [c for c in CAMERA_NAMES if (scene_dir / f"{c}{ARCHIVE_SUFFIX}").is_file() or (scene_dir / c).is_dir()]


def _copy_scene_rest(src: Path, dst: Path, cameras: set) -> None:
    """Copy everything of a scene but the cameras being transcoded, as it is: loose files, the index, the other
    components (zips or extracted folders) and the records of cameras that stay transcoded."""
    from kitscenes._archive import COMPONENTS, LOOSE_FILES

    dst.mkdir(parents=True, exist_ok=True)
    for rel in (*LOOSE_FILES, INDEX_FILE):
        if (src / rel).is_file():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, dst / rel)
    for component in sorted(COMPONENTS):
        if component in cameras:
            continue
        zip_src = src / f"{component}{ARCHIVE_SUFFIX}"
        if component in CAMERA_NAMES and read_variant(src, component) is not None:  # keep a transcoded camera
            shutil.copytree(src / RECORD_DIR, dst / RECORD_DIR, dirs_exist_ok=True,
                            ignore=lambda d, names: [n for n in names if not n.startswith(component + ".")])
        if zip_src.is_file():
            shutil.copyfile(zip_src, dst / zip_src.name)
        else:
            for folder in COMPONENTS[component]:
                if (src / folder).is_dir():
                    shutil.copytree(src / folder, dst / folder, dirs_exist_ok=True)


def transcode(
    root: Path | str,
    recipe: Union[ImageRecipe, str] = "half",
    *,
    out: Path | str | None = None,
    splits: Optional[Iterable[str]] = None,
    scenes: Optional[Iterable[str]] = None,
    cameras: Optional[Iterable[str]] = None,
    backend: str = "auto",
    jobs: Optional[int] = None,
    gpus: Optional[Sequence[int]] = None,
    transcoder: Optional[Transcoder] = None,
    progress: Optional[Callable[[str], None]] = None,
    dry_run: bool = False,
) -> TranscodeResult:
    """Re-encode the cameras of a dataset root by *recipe*, in place or into a new root *out*.

    In place, each camera's zip (or extracted folder) is replaced one at a
    time, so the extra disk is about one archive per running camera. A root
    with ``dataset_info.json`` (a download root) remembers the recipe per
    camera, and later downloads into it apply it.

    With *out*, the source is only read. The new root gets the selected
    scenes in the layout of the source (archives or extracted folders): the
    transcoded cameras, and everything else copied as it is (also cameras
    not in *cameras* or already in the recipe). The top-level maps folders
    are copied too. ``kitscenes pack`` and ``kitscenes extract`` change the
    layout.

    Args:
        root: Dataset root (``data/<split>/<scene>/``, or ``<split>/<scene>/``).
        cameras: Cameras to transcode (default: all; names or presets such as ``ring-cameras``).
        backend, jobs, gpus: See :class:`Transcoder` (or pass a running *transcoder*).
        progress: Called with one line per camera done.
        dry_run: Only plan: the result lists the cameras that would be transcoded as ``done``
            (``bytes_before`` their size) and those already in the recipe as ``skipped``.

    Returns:
        :class:`TranscodeResult`; failed cameras are listed there, not raised.
    """
    from kitscenes.download.archives import resolve_components

    recipe = ImageRecipe.parse(recipe)
    root = Path(root)
    say = progress or (lambda line: None)
    chosen = resolve_components(list(cameras)) if cameras else None
    cams = [c for c in CAMERA_NAMES if chosen is None or c in chosen]
    if not cams:
        raise ValueError(f"no cameras selected ({list(cameras)})")
    targets = find_scenes(root, splits, scenes)
    out_root = Path(out) if out is not None else None
    if out_root is None and (splits or scenes):
        clash = sorted(c for c, r in read_policy(root).items() if c in cams and r != recipe)
        if clash:  # the root keeps one recipe per camera for later downloads
            raise ValueError(f"{root} keeps {', '.join(clash)} at {read_policy(root)[clash[0]]}; transcode them in "
                             f"every scene (no --splits/--scenes/--cities/--where), or write a new root with --out")
    release = _release_files(root)
    if out_root is not None:
        if out_root.resolve() == root.resolve():
            raise ValueError("out is the source root; leave it out to transcode in place")
    result = TranscodeResult()
    if dry_run:
        for split, scene in targets:
            for camera in (c for c in scene_cameras(scene) if c in cams):
                have = read_variant(scene, camera)
                if have is not None and have.recipe == recipe:
                    result.skipped.append((str(scene), camera))
                    continue
                result.done.append((str(scene), camera))
                result.bytes_before += _size_of(scene, camera)
                say(f"  {split}/{scene.name}/{camera}: {have.recipe if have else 'release images'} -> {recipe}")
        return result
    if out_root is not None:
        _copy_top_level(root, out_root)
    own = transcoder is None
    transcoder = transcoder or Transcoder(backend, jobs=jobs, gpus=gpus)
    say(f"transcoding with {transcoder.name} ({transcoder.workers} worker(s)), recipe {recipe}")
    try:
        jobs_ = []
        for split, scene in targets:
            present = [c for c in scene_cameras(scene) if c in cams]
            dst = (out_root / "data" / split / scene.name) if out_root is not None else scene
            if out_root is not None:  # a camera already in the recipe is copied with the rest
                same = [c for c in present if getattr(read_variant(scene, c), "recipe", None) == recipe]
                result.skipped += [(str(scene), c) for c in same]
                present = [c for c in present if c not in same]
                _copy_scene_rest(scene, dst, set(present))
            jobs_ += [(scene, dst, camera) for camera in present]
        lock = threading.Lock()

        def one(job):
            scene, dst, camera = job
            before = _size_of(scene, camera)
            try:
                variant = transcode_camera(scene, camera, recipe, transcoder, out_dir=dst,
                                           source_sha256=_release_sha(release, root, scene, camera))
            except Exception as err:  # reported, the rest goes on
                with lock:
                    result.failed.append((str(scene), camera, f"{type(err).__name__}: {err}"))
                say(f"  FAILED {scene.parent.name}/{scene.name}/{camera}: {err}")
                return
            with lock:
                if variant is None:
                    result.skipped.append((str(scene), camera))
                    return
                result.done.append((str(scene), camera))
                result.bytes_before += before
                result.bytes_after += _size_of(dst, camera)
            say(f"  {scene.parent.name}/{scene.name}/{camera}: {variant.original_size[0]}x{variant.original_size[1]}"
                f" -> {variant.image_size[0]}x{variant.image_size[1]}")

        # a few cameras at a time keep the workers busy (each streams its images through them)
        with cf.ThreadPoolExecutor(max(2, min(8, transcoder.workers // 2 or 1))) as threads:
            list(threads.map(one, jobs_))
    finally:
        if own:
            transcoder.close()
    done = {c for _, c in result.done + result.skipped}
    if out_root is None:
        update_policy(root, {c: recipe for c in done})
    else:  # a download into the new root keeps them (dataset_info.json with the recipes only)
        update_policy(out_root, {c: recipe for c in done}, create=True)
    if done and out_root is None and (splits or scenes) and (root / "dataset_info.json").is_file():
        say(f"note: {root} now keeps {', '.join(sorted(done))} at {recipe}; later downloads into it transcode them too")
    return result


def _release_files(root: Path) -> dict[str, tuple[str, int]]:
    """``{archive path: (sha256, size)}`` of the root's manifest (a download root), empty without one."""
    import csv

    manifest = root / "data" / "sequence_archives.csv"
    if not manifest.is_file():
        return {}
    with open(manifest, newline="") as fh:
        return {r["archive_path"]: (r["archive_sha256"], int(r["archive_size_bytes"])) for r in csv.DictReader(fh)
                if r.get("archive_path", "").endswith(ARCHIVE_SUFFIX)}


def _release_sha(release: dict, root: Path, scene: Path, camera: str) -> Optional[str]:
    """SHA-256 of the release zip a camera's images come from: from its record, else (release images) the
    manifest's for a zip of the release's size or the extraction marker; None if unknown."""
    from kitscenes.download.archives import EXTRACTED_DIR, read_marker

    variant = read_variant(scene, camera)
    if variant is not None:
        return variant.source_sha256
    zip_path = scene / f"{camera}{ARCHIVE_SUFFIX}"
    try:
        rel = zip_path.relative_to(root).as_posix()
    except ValueError:
        rel = None
    if zip_path.is_file():
        known = release.get(rel)
        return known[0] if known is not None and known[1] == zip_path.stat().st_size else None
    marker = read_marker(scene / EXTRACTED_DIR / f"{camera}{ARCHIVE_SUFFIX}.sha256")
    return marker[0] if marker is not None else None


def _size_of(scene: Path, camera: str) -> int:
    z = scene / f"{camera}{ARCHIVE_SUFFIX}"
    if z.is_file():
        return z.stat().st_size
    folder = scene / camera
    return sum(p.stat().st_size for p in folder.iterdir()) if folder.is_dir() else 0


def _copy_top_level(src: Path, dst: Path) -> None:
    """The root's maps folders (small, loose files) into a new root; not the aerial imagery or the state."""
    from kitscenes.constants import HD_MAPS_DIR, SD_MAPS_DIR

    dst.mkdir(parents=True, exist_ok=True)
    for name in (HD_MAPS_DIR, SD_MAPS_DIR):
        if (src / name).is_dir() and not (dst / name).exists():
            shutil.copytree(src / name, dst / name)


# ---------------------------------------------------------------------------
# The root's recipe per camera (for downloads into it)
# ---------------------------------------------------------------------------


def read_policy(root: Path | str) -> dict[str, ImageRecipe]:
    """``{camera: recipe}`` a download root applies to new downloads (from ``dataset_info.json``)."""
    from kitscenes.download.archives import read_dataset_info

    info = read_dataset_info(Path(root)) or {}
    return {c: ImageRecipe.from_dict(r) for c, r in (info.get("images") or {}).items()}


def update_policy(root: Path | str, changes: dict[str, Optional[ImageRecipe]], *, create: bool = False) -> None:
    """Set (or with None remove) cameras' recipes in ``dataset_info.json``; a root without one is left alone
    unless *create*."""
    from kitscenes.download.archives import DATASET_INFO, read_dataset_info

    root = Path(root)
    info = read_dataset_info(root)
    if info is None:
        if not create or not changes:
            return
        info = {}
    images = dict(info.get("images") or {})
    for camera, recipe in changes.items():
        if recipe is None:
            images.pop(camera, None)
        else:
            images[camera] = recipe.to_dict()
    if images == (info.get("images") or {}):
        return
    if images:
        info["images"] = dict(sorted(images.items()))
    else:
        info.pop("images", None)
    tmp = root / (DATASET_INFO + ".partial")
    tmp.write_text(json.dumps(info, indent=1) + "\n")
    os.replace(tmp, root / DATASET_INFO)


# ---------------------------------------------------------------------------
# While downloading: staged camera zips, a byte budget
# ---------------------------------------------------------------------------

#: Default staging buffer of ``kitscenes download --images`` (``--staging-gb``).
DEFAULT_STAGING_GB = 32.0
#: Release size / transcoded size at q90, 4:2:0 (see docs/transcode.md).
_RATIO_Q90_420 = {1: 1.6, 2: 6.3, 4: 28.0, 8: 80.0}


def estimate_size(size: int, recipe: ImageRecipe) -> int:
    """Rough size of a release camera zip of *size* bytes after *recipe* (for dry runs; real data varies ±30 %)."""
    ratio = _RATIO_Q90_420[recipe.reduce]
    ratio /= {"4:2:0": 1.0, "4:2:2": 1.2, "4:4:4": 1.44}[recipe.subsampling]
    ratio /= math.exp(0.11 * (recipe.quality - 90))  # ~+63 % for q95, -28 % for q85 (measured)
    return int(size / max(ratio, 0.5))


class _Budget:
    """Bytes of staged archives; a download waits while the buffer is full (one archive always fits)."""

    def __init__(self, limit: int) -> None:
        self.limit, self.used = int(limit), 0
        self._cond = threading.Condition()

    def acquire(self, n: int) -> None:
        with self._cond:
            while self.used and self.used + n > self.limit:
                self._cond.wait()
            self.used += n

    def release(self, n: int) -> None:
        with self._cond:
            self.used -= n
            self._cond.notify_all()


class DownloadTranscoding:
    """The camera side of ``kitscenes download --images``: which files to transcode, the staging buffer, the workers.

    A camera zip with a recipe is downloaded into ``ROOT/.kitscenes/staging/``
    (at most *staging_bytes* at a time: downloads wait for room), transcoded
    into its scene folder by a few archive threads that share one
    :class:`Transcoder`, then deleted. Files whose release zip is already in
    the root (downloaded before the recipe was set) are transcoded in place.

    Args:
        root: The dataset root.
        recipes: ``{camera: recipe}`` to apply.
        staging_bytes: Size of the staging buffer.
        backend, jobs, gpus: See :class:`Transcoder` (started on first use).
    """

    STAGING = "staging"

    def __init__(self, root: Path, recipes: dict[str, ImageRecipe], *, staging_bytes: float = DEFAULT_STAGING_GB * 1e9,
                 backend: str = "auto", jobs: Optional[int] = None, gpus: Optional[Sequence[int]] = None,
                 progress: Optional[Callable[[str], None]] = None) -> None:
        from kitscenes.download.archives import STATE_DIR

        self.root, self.recipes = Path(root), dict(recipes)
        self.staging = self.root / STATE_DIR / self.STAGING
        self.budget = _Budget(staging_bytes)
        self._options = dict(backend=backend, jobs=jobs, gpus=gpus)
        self._transcoder: Optional[Transcoder] = None
        self._threads: Optional[cf.ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self.say = progress or (lambda line: None)

    def recipe(self, f) -> Optional[ImageRecipe]:
        """The recipe for a manifest file (:class:`~kitscenes.download.archives.ArchiveFile`), None if it is kept."""
        return self.recipes.get(f.component) if f.component in CAMERA_NAMES else None

    def clean_staging(self) -> None:
        shutil.rmtree(self.staging, ignore_errors=True)

    def start(self) -> None:
        """Start the transcoder (raises if it cannot start): before the downloads, which wait for it."""
        with self._lock:
            if self._transcoder is None:
                self._transcoder = Transcoder(**self._options)
                self._threads = cf.ThreadPoolExecutor(max(2, min(8, self._transcoder.workers // 2 or 1)),
                                                      thread_name_prefix="kitscenes-transcode")
                self.say(f"transcoding cameras with {self._transcoder.name} ({self._transcoder.workers} worker(s))")

    def submit(self, f, *, staged: Optional[Path] = None) -> cf.Future:
        """Transcode *f*: from the staged zip (deleted afterwards, its bytes released from the budget) or in place."""
        self.start()
        scene_dir = self.root / f.path.rsplit("/", 1)[0]

        def run():
            try:
                return transcode_camera(scene_dir, f.component, self.recipes[f.component], self._transcoder,
                                        source_sha256=f.sha256, source_zip=staged)
            finally:
                if staged is not None:
                    staged.unlink(missing_ok=True)
                    self.budget.release(f.size)

        return self._threads.submit(run)

    def close(self, cancel: bool = False) -> None:
        if self._threads is not None:
            self._threads.shutdown(wait=True, cancel_futures=cancel)
        if self._transcoder is not None:
            self._transcoder.close()
        self._threads = self._transcoder = None


def variant_complete(root: Path, f, recipe: Optional[ImageRecipe], *, verify: bool = False) -> Optional[bool]:
    """For a camera file of the manifest: True if its scene holds it transcoded by *recipe* from this very release
    zip, False if the scene holds another variant of it, None if the camera is not transcoded there."""
    from kitscenes.download.archives import EXTRACTED_DIR, read_marker, sha256_file

    scene_dir = Path(root) / f.path.rsplit("/", 1)[0]
    variant = read_variant(scene_dir, f.component)
    if variant is None:
        return None
    if recipe is None or variant.recipe != recipe or variant.source_sha256 != f.sha256:
        return False
    zip_path = scene_dir / f"{f.component}{ARCHIVE_SUFFIX}"
    if zip_path.is_file():
        if variant.size is not None and zip_path.stat().st_size != variant.size:
            return False
        return not verify or variant.sha256 is None or sha256_file(zip_path) == variant.sha256
    folder = (scene_dir / f.component).is_dir()
    if variant.sha256 is not None:  # a transcoded zip, extracted: its marker holds that zip's hash
        marker = read_marker(scene_dir / EXTRACTED_DIR / f"{f.component}{ARCHIVE_SUFFIX}.sha256")
        return marker is not None and marker[0] == variant.sha256 and folder
    return variant.layout == "folder" and folder  # transcoded as a folder (from extracted release images)
