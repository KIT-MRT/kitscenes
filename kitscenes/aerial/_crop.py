"""Ego-aligned crop extraction.

roi_m = (longitudinal_m, lateral_m). Ego is the center pixel; ego forward
-> +column, ego left -> -row.

Two interpolation backends sample the same bilinear model:

- ``"scipy"`` (``scipy.ndimage.map_coordinates``, order 1): the reference.
- ``"cv2"`` (``cv2.warpAffine``, ``INTER_LINEAR``): ~20x faster (1 vs 22 ms
  for a 100 x 50 m crop); OpenCV quantizes the sample positions to 1/32 px,
  so a value differs from the reference by at most 1 level, and rarely
  (about 1 in 10^4 values on the DOP20 tiles).
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.ndimage import map_coordinates

if TYPE_CHECKING:
    from kitscenes.aerial._tiling import TileGrid


_SAMPLER_MARGIN_CELLS = 2  # extra resolution_m cells of buffer for interpolation safety

BACKENDS = ("scipy", "cv2")
_CV2_LOCK = threading.Lock()


def resolve_backend(backend: str) -> str:
    """``"auto"`` -> ``"cv2"`` if OpenCV is installed, else ``"scipy"``.

    Raises:
        ValueError: For an unknown backend.
        ImportError: For ``"cv2"`` without OpenCV.
    """
    if backend == "auto":
        try:
            import cv2  # noqa: F401
        except ImportError:
            return "scipy"
        return "cv2"
    if backend not in BACKENDS:
        raise ValueError(f"Unknown crop backend {backend!r}; use 'auto', 'scipy' or 'cv2'.")
    if backend == "cv2":
        try:
            import cv2  # noqa: F401
        except ImportError as err:
            raise ImportError('backend="cv2" needs OpenCV: pip install opencv-python-headless') from err
    return backend


def crop_sample_bbox(
    ego_xy: tuple[float, float], roi_m: tuple[float, float], resolution_m: float
) -> tuple[float, float, float, float]:
    """The EPSG:25832 window (min_e, min_n, max_e, max_n) a crop around *ego_xy* is sampled from.

    It holds the crop at any heading plus two pixels, and lies on the pixel
    grid of the raw tiles (multiples of resolution_m: tile origins are
    multiples of the tile extent), so the sampler cuts whole pixels.
    """
    longitudinal_m, lateral_m = roi_m
    ego_e, ego_n = ego_xy
    roi_radius_m = 0.5 * math.sqrt(longitudinal_m ** 2 + lateral_m ** 2)
    required_radius_m = roi_radius_m + _SAMPLER_MARGIN_CELLS * resolution_m
    # Off the grid, crops were shifted by up to 0.5 px.
    return (
        math.floor((ego_e - required_radius_m) / resolution_m) * resolution_m,
        math.floor((ego_n - required_radius_m) / resolution_m) * resolution_m,
        math.ceil((ego_e + required_radius_m) / resolution_m) * resolution_m,
        math.ceil((ego_n + required_radius_m) / resolution_m) * resolution_m,
    )


def warp_ego_aligned(
    source: np.ndarray,
    sample_bbox: tuple[float, float, float, float],
    ego_xy: tuple[float, float],
    ego_yaw: float,
    roi_m: tuple[float, float],
    resolution_m: float,
    backend: str = "scipy",
) -> np.ndarray:
    """The crop from *source*, the (H, W, 3) pixels of *sample_bbox* (row 0 is its north edge)."""
    longitudinal_m, lateral_m = roi_m
    width_px = round(longitudinal_m / resolution_m)
    height_px = round(lateral_m / resolution_m)
    ego_e, ego_n = ego_xy
    src_min_e, _src_min_n, _src_max_e, src_max_n = sample_bbox
    center_col = (width_px - 1) / 2.0
    center_row = (height_px - 1) / 2.0
    cos_yaw, sin_yaw = math.cos(ego_yaw), math.sin(ego_yaw)

    if backend == "cv2":
        import cv2

        # The affine map of the scipy branch below, crop (col, row) -> source (col, row):
        # src_col = (ego_e - min_e) / res - 0.5 + cos (col - cc) + sin (row - cr)
        # src_row = (max_n - ego_n) / res - 0.5 - sin (col - cc) + cos (row - cr)
        matrix = np.array([
            [cos_yaw, sin_yaw,
             (ego_e - src_min_e) / resolution_m - 0.5 - cos_yaw * center_col - sin_yaw * center_row],
            [-sin_yaw, cos_yaw,
             (src_max_n - ego_n) / resolution_m - 0.5 + sin_yaw * center_col - cos_yaw * center_row],
        ])
        # One thread: OpenCV's pool (a thread per core) costs ~15x the CPU time of a crop this small, and
        # crops are parallel over DataLoader workers anyway. The setting is process-wide: restored after.
        with _CV2_LOCK:
            threads = cv2.getNumThreads()
            if threads != 1:
                cv2.setNumThreads(1)
            try:
                return cv2.warpAffine(
                    np.ascontiguousarray(source), matrix, (width_px, height_px),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE,
                )
            finally:
                if threads != 1:
                    cv2.setNumThreads(threads)
    if backend != "scipy":
        raise ValueError(f"Unknown crop backend {backend!r}; use 'scipy' or 'cv2'.")

    rows, cols = np.meshgrid(
        np.arange(height_px, dtype=np.float64),
        np.arange(width_px, dtype=np.float64),
        indexing="ij",
    )
    x_body = (cols - center_col) * resolution_m  # forward (+column)
    y_body = -(rows - center_row) * resolution_m  # left (-row)

    world_e = ego_e + cos_yaw * x_body - sin_yaw * y_body
    world_n = ego_n + sin_yaw * x_body + cos_yaw * y_body

    src_col = (world_e - src_min_e) / resolution_m - 0.5
    src_row = (src_max_n - world_n) / resolution_m - 0.5

    out = np.empty((height_px, width_px, source.shape[-1]), dtype=np.uint8)
    for channel in range(source.shape[-1]):
        sampled = map_coordinates(
            source[..., channel].astype(np.float64),
            [src_row.ravel(), src_col.ravel()],
            order=1,
            mode="nearest",
        )
        out[..., channel] = sampled.reshape(height_px, width_px).round().clip(0, 255).astype(np.uint8)

    return out


class TileCoverageError(Exception):
    """A crop needs raw tiles that are not there."""


class TileSampler:
    """Pixels of an EPSG:25832 window, read from the tiles of one area.

    Args:
        area_id: For messages.
        grid: The area's tile grid.
        available: The ``(i, j)`` tiles there are.
        read: ``read(i, j)`` -> the (S, S, 3) uint8 pixels of a tile (an
            array or a memory map; only the part inside a window is copied).
        raw_cache_id: For messages.
    """

    def __init__(self, area_id: str, grid: "TileGrid", available, read: Callable[[int, int], np.ndarray],
                 raw_cache_id: str = "") -> None:
        self._area_id = area_id
        self._grid = grid
        self._available = available if isinstance(available, frozenset) else frozenset(
            (int(i), int(j)) for i, j in available)
        self._read = read
        self._raw_cache_id = raw_cache_id

    @property
    def area_id(self) -> str:
        return self._area_id

    @property
    def grid(self) -> "TileGrid":
        return self._grid

    def tile_range(self, bbox_epsg25832: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        """(min_i, min_j, max_i, max_j) of the tiles the half-open bbox touches."""
        min_e, min_n, max_e, max_n = bbox_epsg25832
        min_i, min_j = self._grid.tile_index(min_e, min_n)
        # The bbox is half-open [min, max): ceil(...) - 1 is the last tile it touches, also when max lies
        # on a tile border.
        max_i = math.ceil((max_e - self._grid.origin_e) / self._grid.tile_extent_m) - 1
        max_j = math.ceil((max_n - self._grid.origin_n) / self._grid.tile_extent_m) - 1
        return min_i, min_j, max_i, max_j

    def missing_tiles(self, bbox_epsg25832: tuple[float, float, float, float]) -> list[tuple[int, int]]:
        """Tiles *bbox_epsg25832* needs that are not available (empty if it is covered)."""
        min_i, min_j, max_i, max_j = self.tile_range(bbox_epsg25832)
        return [
            (i, j)
            for i in range(min_i, max_i + 1)
            for j in range(min_j, max_j + 1)
            if (i, j) not in self._available
        ]

    def sample_bbox(
        self, bbox_epsg25832: tuple[float, float, float, float], resolution_m: float
    ) -> np.ndarray:
        """Return (H, W, 3) uint8 pixels covering bbox_epsg25832 at resolution_m.

        Row 0 of the returned array corresponds to bbox_epsg25832's max_n
        (north) edge -- top-left-origin raster convention. Raises
        TileCoverageError if the bbox isn't fully covered by known raw tiles.
        Only the part of each tile inside the window is copied.
        """
        min_e, min_n, max_e, max_n = bbox_epsg25832
        min_i, min_j, max_i, max_j = self.tile_range(bbox_epsg25832)
        missing = self.missing_tiles(bbox_epsg25832)
        if missing:
            raise TileCoverageError(
                f"Missing raw tile coverage for area={self._area_id!r}, "
                f"raw_cache_id={self._raw_cache_id!r}: tiles {missing}"
            )

        # The window on the mosaic of tiles [min_i, max_i] x [min_j, max_j] (larger j further north,
        # nearer the mosaic's top).
        tile_extent_px = round(self._grid.tile_extent_m / resolution_m)
        mosaic_width = (max_i - min_i + 1) * tile_extent_px
        mosaic_height = (max_j - min_j + 1) * tile_extent_px
        mosaic_min_e = self._grid.origin_e + min_i * self._grid.tile_extent_m
        mosaic_max_n = self._grid.origin_n + (max_j + 1) * self._grid.tile_extent_m

        out_width = round((max_e - min_e) / resolution_m)
        out_height = round((max_n - min_n) / resolution_m)
        col_start = round((min_e - mosaic_min_e) / resolution_m)
        row_start = round((mosaic_max_n - max_n) / resolution_m)
        # out_width/out_height/col_start/row_start are each independently
        # rounded from continuous distances; near a rounding boundary their
        # sum can exceed the mosaic's pixel extent even though every needed
        # tile is present. Fail loudly instead of returning a truncated array.
        if (col_start < 0 or row_start < 0 or col_start + out_width > mosaic_width
                or row_start + out_height > mosaic_height):
            covered = (max(0, min(out_height, mosaic_height - row_start)),
                       max(0, min(out_width, mosaic_width - col_start)))
            raise ValueError(
                f"Requested bbox {bbox_epsg25832} at resolution {resolution_m} rounds to "
                f"a {out_height}x{out_width} region starting at row {row_start}, col "
                f"{col_start}, but the assembled tile mosaic (shape {(mosaic_height, mosaic_width)}) "
                f"only covers {covered} of it -- rounding-boundary mismatch, "
                f"refusing to silently return a truncated crop."
            )

        result = np.empty((out_height, out_width, 3), dtype=np.uint8)
        row_end, col_end = row_start + out_height, col_start + out_width
        for i in range(min_i, max_i + 1):
            col_off = (i - min_i) * tile_extent_px
            c0, c1 = max(col_start, col_off), min(col_end, col_off + tile_extent_px)
            if c0 >= c1:
                continue
            for j in range(min_j, max_j + 1):
                row_off = (max_j - j) * tile_extent_px
                r0, r1 = max(row_start, row_off), min(row_end, row_off + tile_extent_px)
                if r0 >= r1:
                    continue
                tile = self._read(i, j)
                result[r0 - row_start : r1 - row_start, c0 - col_start : c1 - col_start] = (
                    tile[r0 - row_off : r1 - row_off, c0 - col_off : c1 - col_off]
                )
        return result
