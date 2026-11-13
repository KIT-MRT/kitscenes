"""Aerial imagery: one ego-aligned orthophoto crop per frame (optional).

The release ships the raw DOP20 tiles of the German state WMS services
(Hesse, Baden-Württemberg; 0.2 m/px JPEG, under their own licences: see
``aerial/ATTRIBUTION.md``) in ``<dataset root>/aerial/``, as one zip per
10 km area or extracted (they come with ``kitscenes download ROOT``,
the areas of the selected scenes, unless ``--no-aerial``;
:mod:`kitscenes.aerial.tiles` has the layout). Crops are
cut from them in one of two ways:

- **On the fly** (:class:`AerialCrops`): each crop is cut when asked for.
  From the JPEG tiles directly (no preparation; each tile a crop touches
  is decoded, ~50 ms, and cached), or from a **tile store**, the tiles
  converted once into memory-mappable ``.npy`` (:func:`build_tile_store`,
  ~10 GB): a few milliseconds per crop in any order, the way to train::

    import kitscenes
    from kitscenes.aerial import build_tile_store

    ds = kitscenes.open("/data/kitscenes", splits="train")
    crops = ds.use_aerial(roi=(60, 30))            # crops from <root>/aerial, (longitudinal_m, lateral_m)
    crop = crops[ds[0].frame(10)]                  # (150, 300, 3) uint8 RGB
    crop = ds[0].frame(10).aerial()                # the same: the crops attached to the dataset

    store = build_tile_store("/scratch/kitscenes_tiles", ds.root, jobs=8)  # once
    crops = ds.use_aerial(roi=(60, 30), tiles=store)   # or tiles="/scratch/kitscenes_tiles" in a new process

- **A prepared crop cache** (:meth:`kitscenes.Dataset.prepare_aerials`):
  one PNG per frame and ROI on disk (~40 GB per ROI for the train split),
  cut from the same tiles with the same function; for tools that read
  image files::

    ds.prepare_aerials(roi=(60, 30), cache_dir="/scratch/kitscenes_aerial")
    cache = ds.use_aerial("dop20_res020_roi60x30_ego_v1", cache_dir="/scratch/kitscenes_aerial")  # later

A crop with ``roi=(longitudinal_m, lateral_m)`` is
``(round(lateral_m / resolution), round(longitudinal_m / resolution), 3)``:
the ego position is the centre pixel, forward points to +column (right) and
left to -row (up). The pose is the frame's pose (:attr:`kitscenes.Frame.pose`).

Scenes are placed on the imagery with their map origin. The test and
map_change_test splits ship no ``maps/origin.json`` but poses in their city's
map frame: they are placed with the city's origin (:func:`placement_origin`).
The test-e2e splits (poses relative to the first pose) get no crops.

CLI: ``kitscenes aerial store|prepare|check|list`` (``kitscenes aerial --help``).
"""

from __future__ import annotations

from kitscenes.aerial._crop import TileCoverageError
from kitscenes.aerial._poses import placement_origin, scene_poses
from kitscenes.aerial.cache import AerialCache, cache_ids
from kitscenes.aerial.crops import AerialCoverage, AerialCrops
from kitscenes.aerial.features import AerialFeatureCrops, AerialFeatures, NorthUpFeatures, feature_set_ids
from kitscenes.aerial.tile_store import build_tile_store
from kitscenes.aerial.tiles import AerialTiles

__all__ = [
    "AerialCache", "AerialCoverage", "AerialCrops", "AerialFeatureCrops", "AerialFeatures", "AerialTiles",
    "TileCoverageError", "build_tile_store", "cache_ids", "feature_set_ids", "NorthUpFeatures", "prepare_aerials",
    "placement_origin", "scene_poses",
]


def prepare_aerials(dataset, roi, **kwargs) -> AerialCache:
    """See :meth:`kitscenes.Dataset.prepare_aerials` (this one does not attach the cache)."""
    from kitscenes.aerial.prepare import prepare_aerials as prepare

    return prepare(dataset, roi, **kwargs)
