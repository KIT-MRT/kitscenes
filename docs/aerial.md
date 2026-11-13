# Aerial imagery (KITScenes Aerial)

[KITScenes Aerial](https://kitscenes.com/aerial/) extends KITScenes Multimodal with spatially aligned aerial imagery and learned features, which show the road structure beyond the range of the onboard sensors. The release ships two parts:

- **DOP20 orthophotos:** freely available 20 cm/px RGB tiles on a north-up grid;
- **DINOv3 feature tiles:** spatially continuous features computed from 8 cm/px municipal orthophotos ([DINOv3 features](#dinov3-features-of-8-cm-imagery), [Aerial DINOv3 features](aerial_features.md)).

`kitscenes.aerial` cuts per-frame crops from them, aligned with the poses, the lidar and the HD map labels. The benchmark tasks are aerial-only map construction, aerial-assisted online HD mapping and cross-view localization ([paper](https://arxiv.org/abs/2610.40455)). Notebook: `06_aerial_imagery.ipynb`.

**In short**

- The DOP20 tiles (~2 GB) come with every download, the DINOv3 features (~657 GB) with the full download or with `--aerial-features`.
- `crops = ds.use_aerial(roi=(60, 30))`, then `crops[frame]`: a 150 × 300 RGB crop at 0.2 m/px, ego at the centre, forward to the right.
- For training, build a tile store once (`kitscenes aerial store`): ~1 ms per crop instead of ~60 ms.
- The DINOv3 features are north-up tokens; use them north-up and let the model handle the rotation ([DINOv3 features](aerial_features.md)).
- The test-e2e splits get no crops.

```python
ds = kitscenes.open(splits="val")
crops = ds.use_aerial(roi=(60, 30))
crop = crops[ds[0][50]]          # (150, 300, 3) uint8
```

**Contents:** [Getting the tiles](#getting-the-tiles) · [Crops](#crops) · [Tiles on disk](#tiles-on-disk) · [Three ways to get crops](#three-ways-to-get-crops) · [Command line](#command-line) · [DINOv3 features](#dinov3-features-of-8-cm-imagery) · [Licence](#licence) · [Other imagery](#other-imagery-newer-older-another-source)

## Getting the tiles

The DOP20 orthophotos come from the WMS services of the German state surveying agencies: Hesse for Frankfurt, Baden-Württemberg for Karlsruhe and Sindelfingen. 2048 px tiles at 0.2 m/px, JPEG quality 95 without chroma subsampling, ~2 GB for the three 10 km areas.

They come with every download ([Download](download.md#kitscenes-aerial)):

- `kitscenes download ROOT`: every area, with the DINOv3 feature tiles (~657 GB; `--no-aerial-features` leaves them out);
- with `--splits`, `--scenes`, `--cities`, `--where` or `--max-gb`: the areas of the selected scenes (the test-e2e scenes pull none), the features only with `--aerial-features`;
- `--components none` adds them to a root with little else (the small files of the selected scenes come along), e.g. `kitscenes download ROOT --splits val --components none`;
- `--no-aerial` leaves out the tiles and the features.

They are third-party data under their own licences (`aerial/ATTRIBUTION.md`, [Licence](#licence)).

Reading the release tiles needs nothing extra; `pip install "kitscenes[aerial]"` adds OpenCV (crops ~20x faster than the scipy fallback).

## Crops

```python
ds = kitscenes.open(root, splits=["train", "val"])
crops = ds.use_aerial(roi=(60, 30))      # (longitudinal_m, lateral_m): crops cut on the fly from root/aerial
frame = ds[0].frame(50)
crop = crops[frame]                      # (150, 300, 3) uint8 RGB: ego at the centre, forward right, left up
crops.at(frame, missing="none")          # None instead of AerialUnavailable (state WITHHELD_BY_SPLIT, DROPPED)
frame.aerial()                           # a shortcut for crops[frame]: the crops attached by the last ds.use_aerial(...)
```

- `crops` is an `AerialCrops` (or, by cache ID, an `AerialCache`). Several can be used side by side, e.g. two regions: `wide = AerialCrops(crops.tiles, roi=(100, 50))`, then `wide[frame]`. `frame.aerial()` and `frame.aerial(missing="none")` read the one attached to the dataset.

- A crop of `roi=(longitudinal_m, lateral_m)` is `round(lateral_m / 0.2)` x `round(longitudinal_m / 0.2)` pixels around `frame.pose`. The ML converter's labels and the lidar line up with it in the base frame.
- The test and map_change_test splits ship no `maps/origin.json`, but their poses are in their city's map frame: they are placed with the city's origin (`kitscenes.aerial.placement_origin`, `kitscenes.map.origin.CITY_ORIGINS`), and the release tiles cover every frame. The test-e2e splits get no crops (poses relative to the first pose, by design).
- The crop IDs (`dop20_res020_roi60x30_ego_v1`, `..._roi100x50_...`) name the source, resolution, region and orientation of a crop; prepared crop caches use the same IDs.

## Tiles on disk

`root/aerial/` holds:

- `ATTRIBUTION.md`: licences and the source notes to keep when you pass on tiles or crops;
- `metadata/`: area index, tile manifest with paths relative to `aerial/`, provider metadata;
- the tiles, in one of two layouts:
  - as downloaded, one zip per area: `raw_tiles/dop20_res020_tile2048/<area>.zip` (stored, members read in place by offset);
  - extracted (`kitscenes extract ROOT`; `--no-aerial` leaves the aerial zips as they are): `raw_tiles/dop20_res020_tile2048/<area>/tile_<i>_<j>.jpg` (+ `.json`).

A zip is preferred when both are there. `kitscenes.aerial.AerialTiles` reads either, PNG tiles and tile stores, several folders at once: `ds.use_aerial(roi=..., tiles=[folder, ...])`.

## Three ways to get crops

All three use the same function: bilinear; OpenCV, or scipy as the reference. They differ by at most one level in about 1 in 10^4 values.

| | Setup | Time per crop, 60x30 / 100x50 m, random access over all tiles | Memory per process | Use |
| --- | --- | --- | --- | --- |
| On the fly from the JPEG tiles | none | 60 / 70 ms (each tile a crop touches is decoded, ~45 ms); frames in order 1-2 ms (decoded tiles are cached) | ~120 MiB + the tile cache (`cache_mb`, default 256 MiB) | trying things out, evaluation, frames in order |
| On the fly from a **tile store** | `kitscenes aerial store DIR --tiles ROOT` once: ~1 min with 4 processes, 10 GB | 0.8 / 1.7 ms (pages in the OS cache; 9 / 14 ms when they come from a local SSD) | ~120 MiB; the tiles are memory-mapped, so their pages count in RSS but are shared, reclaimable page cache | training |
| A prepared **crop cache** (PNG per frame) | `ds.prepare_aerials(roi=..., cache_dir=...)`: ~3 ms per crop with 6 processes | reading a PNG | small | tools that read image files |

- PNG tiles take ~200 ms per crop in random order.
- With 6 DataLoader workers a tile store delivers ~5000 (60x30) / ~2600 (100x50) crops per second.
- Crops from the release's JPEG tiles differ from crops of the lossless tiles they were encoded from by 1.2 levels on average (PSNR 43 dB). Crops from a tile store are identical to those from the JPEG tiles.

```python
from kitscenes.aerial import AerialCrops, AerialTiles, build_tile_store

store = build_tile_store("/scratch/kitscenes_tiles", root)            # once: .npy per tile; reruns convert only what is missing
crops = ds.use_aerial(roi=(60, 30), tiles="/scratch/kitscenes_tiles") # or tiles=store
wide = AerialCrops("/scratch/kitscenes_tiles", roi=(100, 50))         # wide[frame]; any EPSG:25832 pose: wide.crop((easting, northing), yaw)
report = crops.coverage(ds)                                           # which frames get a crop, and why the others do not
```

- `AerialCrops` pickles to its folders and parameters and opens files per process, so it goes into DataLoader workers with the dataset.
- A prepared crop cache is optional: `ds.prepare_aerials(roi=(60, 30), cache_dir="/scratch/aerial")` cuts one PNG per frame from the tiles into a writable folder (default `root/aerial`; `tiles=` other folders, only read). It cuts only what is missing and reports the frames it could not cut (`cache.coverage`, `metadata/coverage_<cache_id>.json`). Later: `cache = ds.use_aerial("dop20_res020_roi60x30_ego_v1", cache_dir="/scratch/aerial")`, then `cache[frame]`. Parallel: `workers=`.

## Command line

```bash
kitscenes aerial list ROOT                                         # the tiles and the crop caches
kitscenes aerial check ROOT --splits train val --roi 100 50       # which frames the tiles cover (exit code 2 if not all)
kitscenes aerial store /scratch/kitscenes_tiles --tiles ROOT      # the tile store (--jobs, default 4, ~130 MiB each)
kitscenes aerial prepare ROOT --splits train val --roi 60 30 --cache-dir /scratch/aerial   # a crop cache; --jobs, -v
```

## DINOv3 features of 8 cm imagery

Higher-resolution municipal orthophotos (8 cm/px) may not be redistributed, so the release ships DINOv3 feature tiles computed from them instead: the frozen DINOv3 ViT-L/16 satellite backbone (`dinov3_vitl16_pretrain_sat493m`), patch tokens of blocks 4, 11, 17 and 23 plus their class tokens, one safetensors file per 2048 px tile (163.84 m, 128 x 128 tokens of 1.28 m, bfloat16, ~128 MiB), over the HD-map areas of the three cities, test regions included (their maps and labels stay withheld). Each tile was encoded inside a 2560 px window of its neighbours, so tokens at tile edges saw the real surroundings. They come by area like the tiles: with the full download (~657 GB for the three areas; `--no-aerial-features` leaves them out) and with a filtered download only with `--aerial-features` ([Download](download.md#kitscenes-aerial)). Their terms of use (`aerial/features/<set>/TERMS.md`) forbid reconstructing the imagery.

```python
feats = ds.use_aerial_features(roi=(60, 30))                   # north-up (the default): the stored tokens + the pose
win = feats[frame]                                             # NorthUpFeatures: win.tokens (1, 1024, 57, 57), win.yaw,
                                                               # win.token_centers_ego(), win.ego_to_tokens(), win.sampling_grid()
frame.aerial_features()                                        # a shortcut for feats[frame] (what is attached to the dataset)
ego = ds.use_aerial_features(roi=(60, 30), orientation="ego", layers=(4, 11, 17, 23))
ego[frame]                                                     # (4, 1024, 23, 47), the geometry of the image crops

from kitscenes.aerial import AerialFeatures
f = AerialFeatures(ds.root)                                    # tile access: f.tile(area, i, j, layers=...), f.cls(...)
```

**North-up, not rotated.** DINOv3 features are not rotation-equivariant: resampling the token grid into the ego frame moves the tokens but not what they encode, and a head trained north-up loses about half its mIoU on such crops (measured on 24 tiles). For ego-frame tasks use the north-up window and let the model rotate (its outputs, or by attending with the pose). Why, the measurements, the recommended patterns and when to use the DOP20 images instead: [docs/aerial_features.md](aerial_features.md).

The channels are in a fixed basis that differs from the public backbone's by an unpublished orthogonal rotation (the same for every block and the class token): distances, inner products, PCA and everything a linear layer learns are unchanged, Meta's DINOv3 heads do not apply. Files are read by byte range (only the blocks asked for), from the zips in place or extracted (`kitscenes extract ROOT`); a plain install reads them (no torch, no safetensors package). A 60 x 30 m crop takes ~25 ms per block from the page cache. As a baseline, a DPT head trained on the four blocks reaches 8.6 % mIoU over the 53 HD-map classes on the val tiles (34.7 % with the 8 OpenSatMap line classes).

## Licence

The imagery comes under the terms of the state services (`aerial/ATTRIBUTION.md`):

- Baden-Württemberg: Datenlizenz Deutschland – Namensnennung 2.0 (source note such as "LGL-BW (2026) Datenlizenz Deutschland – Namensnennung – Version 2.0, www.lgl-bw.de");
- Hesse: Datenlizenz Deutschland – Zero 2.0 (no attribution required; a source note, if given, has to mention changes to the data).

The tiles were re-encoded as JPEG for the release (mean PSNR 38.6 dB against the WMS rendering). The DOP20 tiles are not covered by the dataset licence.

The DINOv3 feature tiles come under their terms of use (`aerial/features/<set>/TERMS.md`: research use, no reconstruction of the imagery) and Meta's DINOv3 licence. They come with the full download, and the dataset terms you accept on HuggingFace cover them too: the source imagery must not be reconstructed, recovered or approximated from them.

## Other imagery: newer, older, another source

The release ships the DOP20 tiles of every scene's area, so crops need no download. For imagery it does not ship (a newer or older flight, another provider), `examples/download_aerial_tiles.py` fetches tiles around chosen scenes from a WMS service into a folder of your own, in the layout and on the pixel grid of the release tiles. One use is the map_change scenes, where the road changed after it was mapped: imagery of another date next to the map and the recorded drive.

```bash
python examples/download_aerial_tiles.py ROOT --out /data/aerial_today --splits map_change_val   # --dry-run first: areas, tiles, size
```

```python
crops = ds.use_aerial(roi=(60, 30), tiles="/data/aerial_today")
crop = crops[frame]
```

- Without `--wms-url URL --layer NAME` the tiles come from the DOP20 service of the scene's state; `--param KEY=VALUE` passes a parameter on to the service. The dataset root is only read; a rerun fetches only what is missing. Needs `pip install requests`.
- A service serves its current mosaic unless a layer or a parameter says otherwise, so the result depends on the day of the download: keep the folder with your results.
- The terms of the service apply to these tiles (the script notes it in `SOURCE.md`); `aerial/ATTRIBUTION.md` covers only the released tiles.
