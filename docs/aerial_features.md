# Aerial DINOv3 features

KITScenes Aerial ships DINOv3 feature tiles of 8 cm/px municipal orthophotos for the HD-map areas of Karlsruhe, Frankfurt and
Sindelfingen. The imagery itself may not be redistributed; the features may (terms in
`aerial/features/<set>/TERMS.md` and Meta's DINOv3 licence; the dataset terms accepted on HuggingFace cover them too:
the source imagery must not be reconstructed from them). This page explains what they are, how to read them, and why north-up crops are the
recommended way to use them for tasks in the ego frame.

## What is stored

- Backbone: frozen DINOv3 ViT-L/16, satellite pretraining (`dinov3_vitl16_pretrain_sat493m`), ImageNet input
  normalisation.
- One safetensors file per 2048 px tile (163.84 m, EPSG:25832 grid per 10 km area): `maps` (4, 1024, 128, 128)
  bfloat16, the patch tokens of blocks 4, 11, 17 and 23 (one token = 16 px = 1.28 m, row 0 = north), and `cls`
  (4, 1024), the class tokens of the same forward pass.
- Each tile was encoded inside a 2560 px window of its 8 neighbours and the centre 128 x 128 tokens kept, so tokens at
  a tile edge saw the real surroundings (the manifest lists the tiles that had no neighbour on some side).
- The channels are in a fixed basis that differs from the public backbone's by an unpublished orthogonal rotation, the
  same for every block and the class token. Distances, inner products, PCA and anything a linear layer learns are
  unchanged; Meta's pretrained DINOv3 heads do not apply.

They come with the full download (~657 GB for the three areas, 104–393 GB for one; `--no-aerial-features` leaves
them out) and with a filtered download only with `--aerial-features`, for the areas of the selected scenes
([Download](download.md#kitscenes-aerial)):

```bash
kitscenes download ROOT                                                    # everything, the features included
kitscenes download ROOT --components none                                  # all areas (+ the small files of every scene)
kitscenes download ROOT --splits val --components none --aerial-features   # the areas of the val scenes
```

```python
from kitscenes.aerial import AerialFeatures

f = AerialFeatures(ROOT)                       # reads the zips in place, or the extracted files
tokens = f.tile("area_450000_5420000", 30, 35, layers=(23,))   # (1, 1024, 128, 128) float32, only block 23 read
cls = f.cls("area_450000_5420000", 30, 35)                      # (4, 1024)
```

## Crops around a frame: north-up (recommended) or ego-aligned

```python
feats = ds.use_aerial_features(roi=(60, 30))    # north-up, the default
win = feats[frame]                     # NorthUpFeatures: the stored tokens around the ego + the pose
win.tokens                             # (L, C, S, S) float32, north-up, unchanged (S = 2 * half_tokens + 1, fixed)
win.yaw, win.ego_xy, win.bbox_epsg25832

ego = ds.use_aerial_features(roi=(60, 30), orientation="ego", layers=(4, 11, 17, 23))
crop = ego[frame]                      # (4, 1024, 23, 47): ego at the centre, forward +column, left -row
```

`orientation="ego"` resamples the token grid into the ego frame (bilinear between token centres), with the geometry of
the aerial image crops (`frame.aerial()`). `orientation="north"` returns a fixed-size window of the stored tokens that
holds the ROI at any heading, with the pose and helpers to relate tokens and ego frame.

### Why north-up: the features are not rotation-equivariant

A rotated image crop is encoded after rotating, so its features describe the rotated scene. A rotated *feature* crop
only moves the tokens: each token still describes its north-up 16 x 16 px patch. A lane line running north-south is a
different feature vector than the same line running east-west, so the resampled crop is not what the backbone would
produce for the rotated imagery.

On 24 tiles (4 near the centre and 4 at 2.5-6 km of each city), the features of rotated imagery, f(rot(img)),
compared with rotated features, rot(f(img)), per token (mean cosine similarity; 1 = identical):

| | block 4 | block 11 | block 17 | block 23 |
|---|---|---|---|---|
| same input again | 1.000 | 1.000 | 1.000 | 1.000 |
| half-token shift (bilinear interpolation alone) | 0.98 | 0.96 | 0.93 | 0.93 |
| 90° (exact token rotation, no interpolation) | 0.96 | 0.91 | 0.84 | 0.85 |
| 180° | 0.95 | 0.95 | 0.89 | 0.87 |
| 30° (resampled, as `orientation="ego"`) | 0.97 | 0.94 | 0.88 | 0.88 |
| 45° (resampled) | 0.97 | 0.92 | 0.85 | 0.85 |

Early blocks are close to rotation-invariant, the deep blocks are not. Centre and outskirts behave the same. The effect
on a task: a DPT segmentation head (53 HD-map classes) trained on north-up features, scored on the rotated labels:

| mIoU (%) | north-up | 30° | 45° | 90° |
|---|---|---|---|---|
| centre, rotated imagery re-encoded | 12.7 | 12.3 | 11.5 | 11.6 |
| centre, features resampled (`"ego"`) | | 6.8 | 5.9 | 5.2 |
| outskirts, re-encoded | 15.5 | 13.1 | 12.5 | 12.2 |
| outskirts, resampled (`"ego"`) | | 9.1 | 7.7 | 6.6 |

The head handles rotated scenes when the features are right; resampled features halve it. A better sampler (e.g.
`torch.nn.functional.grid_sample`) does not help: it is the same interpolation, and the loss is in what the tokens
encode. Storing extra orientations (0/90/180/270°) would not help either: a heading still needs up to 45° of
resampling, the bad case above.

### How to use north-up features for ego-frame tasks

Keep the aerial input in the world frame and let the model handle the rotation:

1. **Predict in the world frame, transform the result.** If the output is geometry (segmentation, map elements,
   directions), compute it on the north-up tokens and rotate the *output* with the pose. Geometry rotates exactly.

   ```python
   logits = head(torch.from_numpy(win.tokens[-1])[None])                        # north-up block 23 -> (1, K, H', W')
   grid = torch.from_numpy(win.sampling_grid(roi=(60, 30), resolution_m=0.5))[None]
   ego_logits = F.grid_sample(logits, grid, align_corners=False)                # (1, K, 60, 120) ego frame
   ```

2. **Attend or sample with the pose.** Ego-frame BEV queries (or camera tokens) attend to the north-up tokens, with
   the tokens' positions given in the ego frame and the heading as input, so the network can learn how
   world-oriented features look at each heading:

   ```python
   pos = torch.from_numpy(win.token_centers_ego())     # (S, S, 2) ego-frame metres of every token: positional encoding
   ref = win.ego_to_tokens(bev_xy)                     # (N, 2) token coordinates of BEV query points (deformable attention)
   heading = torch.tensor([math.cos(win.yaw), math.sin(win.yaw)])
   ```

   With the aerial input fixed in the world frame, rotating the ego frame is a consistent augmentation (rotate the
   camera BEV, the labels and the heading; the aerial tokens stay as they are), which a resampled feature crop does
   not allow.

The equivariance is learned in pattern 2, not built in: driving data has a skewed heading distribution (street grids),
which the ego-frame rotation augmentation evens out. Sampling between token centres costs about as much as the
half-token shift above, which is small for a learned model.

Use `orientation="ego"` where only the layout matters (visualisation, nearest-neighbour lookups at fixed headings) or
next to `frame.aerial()` for a quick start, knowing the numbers above.

## DOP20 images or DOP8 features?

| | DOP20 tiles (`aerial/raw_tiles/`) | DINOv3 features (`aerial/features/`) |
|---|---|---|
| Source | public state orthophotos, 0.2 m/px | municipal orthophotos, 0.08 m/px (not redistributable) |
| What you get | RGB images | frozen features, 1.28 m tokens, 4 blocks |
| Rotation | exact: rotate the image, then encode | north-up only (see above) |
| Backbone | yours, trainable end to end | frozen DINOv3 ViT-L, no compute at training time |
| Size | ~2 GB | ~657 GB |

- **Methods that consume ego-aligned aerial images** (as in aerial-assisted online mapping) or that fine-tune the
  image backbone: DOP20. Rotation is exact and the backbone adapts; lane markings (12-15 cm) are about one pixel wide.
- **Methods that can work in the world frame or fuse with the pose** (pattern 1 or 2 above), or that want strong
  frozen features of the finer imagery without running a backbone: the DINOv3 features, north-up.
- They are complementary: the same areas, the same tile grid convention, both placed with `scene_poses`.
