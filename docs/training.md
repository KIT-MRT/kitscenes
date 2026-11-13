# Training

`SampleSpec`, `SampleDataset` and `collate` for a torch `DataLoader`; ego past and future, clips, map labels; GPU decoding. Notebook: `05_training.ipynb`; training-loop skeleton: `examples/train_loop.py`.

**Contents:** [Install](#install) · [Samples and batches](#samples-and-batches) · [Sample layout](#sample-layout) · [Missing data: `require`](#missing-data-require) · [Clips, ego history, map labels](#clips-ego-history-map-labels) · [Decoding on the GPU](#decoding-on-the-gpu) · [Frames, caches, speed](#frames-caches-speed)

## Install

Install torch first, the build for your CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) (CPU only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`), then `pip install -e ".[torch]"`.

- torch 1.13 or newer for the torch features (`SampleDataset` tensors, GPU lidar maps, `motion_torch`). `kitscenes.decode` alone works from torch 1.10, for older training stacks.
- `torchvision` is optional: GPU JPEG decoding (nvJPEG). Install the build that matches your torch. With CUDA < 11.6, torchvision's nvJPEG leaks GPU memory: decode on the CPU or with NVIDIA DALI there.
- NVIDIA DALI (`kitscenes.dali`) comes from NVIDIA's package index (`nvidia-dali`), not as an extra.
- Without torch, `collate` gives numpy arrays.

## Samples and batches

`kitscenes.SampleDataset` turns frames into dicts of arrays. Workers may fork or spawn: it pickles to scene IDs and indices. `kitscenes.collate` batches them: images and fixed-size arrays are stacked, point clouds concatenated with a `batch` index per point. `SampleSpec` says what a sample holds.

```python
from torch.utils.data import DataLoader

spec = kitscenes.SampleSpec(cameras=("camera_ring_front",), reduce=4, lidars=("lidar_top",), future_s=3.0)
samples = kitscenes.SampleDataset(ds, spec)        # frames where every sensor of the spec has data
loader = DataLoader(samples, batch_size=8, shuffle=True, num_workers=8, collate_fn=kitscenes.collate)
batch = next(iter(loader))
batch["camera"]["camera_ring_front"]["image"]     # (8, 3, H/4, W/4) uint8 (channels first);
                                                  # ["K"] scaled to match, ["T_cam_base"]
batch["lidar"]["xyz"], batch["lidar"]["batch"]    # (N, 3) base frame, (N,) which sample each point belongs to
batch["future"]["xyz_ego"], batch["future"]["valid"]   # (8, 30, 3) poses every 0.1 s ahead, in the frame's base frame
batch["pose"]["T_world_base"], batch["meta"]["token"]
```

- `SampleSpec(future_s=, past_s=, trajectory_step_s=)`: ego future and history at fixed time slots, a pose every 0.1 s by default, with `valid` for the slots the poses cover (as `frame.future()`, [End-to-end driving](e2e.md#ego-trajectory-api)).
- Samples are in the base frame by default. `SampleSpec(frame="road")` puts points, trajectories and 3D map labels in the [road frame](conventions.md#the-road-frame), the frame of the map benchmark; the `_base` keys are then named `_road` (`pose["T_world_road"]`, `camera[name]["T_cam_road"]`, in clips `pose["T_last_road"]`). `meta["frame"]` says which. Labels with `MapLabelSpec(ignore_elevation=True)` stay at z = 0 in both.
- The MapTR preset `MapLabelSpec.maptr()` is for the road frame (`frame="road"`): use it with `SampleSpec(frame="road")`; a sample in another frame raises `ValueError` (`MapLabelSpec.maptr(frame=None)` follows the sample's frame).

## Sample layout

One frame, with `SampleSpec(frame="base")` (the default; with `frame="road"` the `_base` keys are `_road`).

| Key | Contents |
|---|---|
| `meta` | `scene`, `split`, `index`, `token` (`frame.token`, `"<scene id>_<index:04d>"`: the key of info files and submissions), `t_ns`, `pose_frame`, `frame` |
| `pose` | `T_world_base` (4, 4) float64, NaN without poses |
| `ego` | causal ego motion at the frame time from the last 0.5 s of poses (`frame.ego(causal=True)`, `kitscenes.e2e.past_motion`): `velocity` (3,), `acceleration` (3,), `yaw_rate`, `speed`, float64; `valid`, False with less than 0.2 s of past poses (values 0). See [End-to-end driving](e2e.md) |
| `camera` | `{name: {image, K, T_cam_base, valid}}` for **every** camera of the spec. `image`: (H, W, 3) uint8 (a batch: (B, 3, H, W)), or the JPEG bytes with `image="bytes"`. `K` (3, 3) is that of the image: of the 1/`reduce` image, or of the full-size JPEG with `image="bytes"` (`decode_cameras` applies `reduce`). `T_cam_base` (4, 4) maps base-frame points into the camera frame (OpenCV axes): `uv ~ K @ (T_cam_base @ p_base)[:3]` |
| `lidar` | `xyz` (N, 3) float32, `reflectivity` (N,), `source` (N,) uint8 into `spec.lidars`; with `lidar="deskew_inputs"`: `{name: DeskewInputs}` |
| `radar` | `xyz` (M, 3) float32, `range_rate`, `rcs`, `source` (M,) uint8 into `spec.radars` |
| `future` | `xyz_ego` (K, 3), `quat_ego` (K, 4), `valid` (K,); K = `round(future_s / trajectory_step_s)`, slot k at `(k + 1) * step` after the frame time. `command` () int64 and `command_valid` (): the driving command of this future (a `kitscenes.navigation.Command`: 0 left, 1 straight, 2 right, 3 unknown; it leaks the future, see [End-to-end driving](e2e.md)) |
| `past` | the same for `past_s`, the oldest slot first (slot k at `(K - k) * step` before the frame time) |
| `map` | with `map_labels`: `available`, `state`, `lines` and `traffic_elements` (`{class: {xyz, length, ids, num_ids, piece, key}}`), `lane_segments` (the same fields), `topology` (`lane_to_lane` (E, 2), `te_to_centerline` (E, 3), `te_to_te` (E, 4) int64), `map_changed`. See [Map labels in samples](#map-labels-in-samples) and [Maps](maps.md#lane-segments-and-topology) |
| `sd_map` | with `sd_map` (an `SDMapSpec`): `available`, `state`, and the OpenStreetMap road pieces in the region, each resampled to `points` points: `xyz` (L · points, 3) float32 on the road, `length` (L,) int32, `way_id` (L,) int64, `road_class` (L,) int8 (`RoadClass.rank`, 0 motorway … 10 busway), `is_link`, `one_way` (L,) bool, `lanes` (L,) int8 (both directions), `speed_limit_mps` (L,) float32 (NaN unknown, inf none). © OpenStreetMap contributors, ODbL ([SD maps](sdmap.md)) |
| `navigation` | with `navigation` (a `NavigationSpec`): `available`, `state`, `command` () int64 (a `kitscenes.navigation.Command`: 0 left, 1 straight, 2 right, 3 unknown), `target_points` (K, 2) float32 (ego x, y; NaN past the found ones), `target_valid` (K,) bool; K = `num_targets` |

- Invalid trajectory slots hold xyz 0 and the identity quaternion (`Frame.future` has NaN there): mask with `valid`, don't test for NaN.
- `state` (in `map`, `sd_map`, `navigation`) is `"present"` or why there is nothing, an `Availability` value: `"withheld_by_split"` in the test splits, `"dropped"` for a frame without a pose.
- `map`: the L polylines of a class are concatenated in `xyz` (P, D) float32 (D = `MapLabelSpec.dims`), with `length` (L,) points each (with `FixedPoints(n)` every length is n: `xyz.reshape(L, n, D)`). `ids` (I,) int64 are the map element ids behind them, `num_ids` (L,) of them per polyline; `piece` (L,) is the polyline's piece of the chain and `key` (L,) int64 a stable hash of (class, ids, piece), the same for the same element in every frame where the chain is the same (always with `chaining="scene"`). Every class of the spec is a key in every sample (`MapLabelSpec.line_class_names`, `.te_class_names`). Topology edges index this sample's lists (`kitscenes.samples.TOPOLOGY_COLUMNS`): `lane_to_lane` (i, j), lane segment j follows i; `te_to_centerline` (te class, te, centerline); `te_to_te` (te class, te, other te class, other te), a te class being an index into `MapLabelSpec.te_class_names`.
- `collate` stacks fixed-size arrays (`K`, poses, `ego`, trajectories, `available`, `map_changed`). Images are stacked channels first, (B, 3, H, W), the layout `decode_cameras` gives; a single sample holds (H, W, 3) like `frame.image()`. It concatenates point clouds, polylines and SD-map pieces with a `batch` index, and topology edges with the sample as a first column ((E, 1 + C)). The converter of a scene is kept for the next frame of that scene (the last `MapLabelSpec.cache` scenes), so a sampler that stays in a scene for a while is cheaper than a fully shuffled one.

## Missing data: `require`

`SampleSpec(require=...)` does two things at once:

- `True` (default): only frames where every requested sensor has data (for clips: every frame of the clip), and anything else missing raises (a scene without a map with `map_labels`: `MapUnavailable`).
- `False`: every frame. What is missing is marked instead: a camera with `valid` False and a black image of its size (with `image="bytes"` a black JPEG; `K` and `T_cam_base` from the rig), a lidar or radar left out of the cloud, `map["available"]` False with `map["state"]` saying why.

## Clips, ego history, map labels

```python
from kitscenes.map import Region

spec = kitscenes.SampleSpec(cameras=("camera_ring_front",), reduce=4, past_s=1.0, future_s=3.0,
                            clip_length=4, clip_stride=2)   # clips of frames i-6, i-4, i-2, i
clips = DataLoader(kitscenes.SampleDataset(ds, spec), batch_size=8, collate_fn=kitscenes.collate)
batch = next(iter(clips))                          # a list of 4 batches, the oldest frame first
batch[0]["pose"]["T_last_base"]                    # (8, 4, 4): that frame's base frame -> the current one
batch[-1]["past"]["xyz_ego"]                       # (8, 10, 3) poses every 0.1 s before the current frame

label_spec = kitscenes.MapLabelSpec(grouping="m3tr", region=Region(ahead=60, behind=20, left=20, right=20))
samples = kitscenes.SampleDataset(ds, kitscenes.SampleSpec(lidars=("lidar_top",), map_labels=label_spec))
batch = kitscenes.collate([samples[0], samples[1]])
dashed = batch["map"]["lines"]["Dashed"]           # xyz (P, 3) of all polylines, length (L,), batch (L,),
                                                   # ids / num_ids (map elements), piece, key (L,) int64;
                                                   # every type of the grouping, empty where there is none

maptr = kitscenes.SampleDataset(ds, kitscenes.SampleSpec(cameras=("camera_ring_front",), reduce=4, frame="road",
                                                          map_labels=kitscenes.MapLabelSpec.maptr()))
batch = kitscenes.collate([maptr[0], maptr[1]])
divider = batch["map"]["lines"]["divider"]         # xyz.reshape(-1, 20, 3): 20 points per element, road frame
```

### Clips

`SampleSpec(clip_length=, clip_stride=)` makes a sample a list of frames of one scene, the current one last, each with `pose["T_last_base"]` (`T_last_road` with `frame="road"`). `collate` gives one batch per clip position.

### Map labels in samples

`SampleSpec(map_labels=kitscenes.MapLabelSpec(...))`: the `MLConverter` polylines around the frame's pose, collated with a batch index per polyline. Every class of the grouping (or of `classes`) is a key, empty where there is none. Options and the MapTR preset: [Maps](maps.md#maptr-preset).

Map labels need the ML converter ([Maps](maps.md#lanelet2-the-kitscenes-build)) and a map in every scene. The test splits have none: `MapUnavailable`, or `SampleSpec(require=False)` for samples whose `map["state"]` says why. `MapLabelSpec.labels(frame)` gives one frame's `sample["map"]` without a dataset.

## Decoding on the GPU

The fastest input path reads only the JPEG bytes in the workers and decodes each batch on the GPU at the model's input size, with `K` scaled to match (needs torchvision for nvJPEG):

```python
from kitscenes.decode import decode_cameras

spec = kitscenes.SampleSpec(cameras=("camera_ring_front",), image="bytes", reduce=4)
loader = DataLoader(kitscenes.SampleDataset(ds, spec), batch_size=8, shuffle=True, num_workers=4,
                    collate_fn=kitscenes.collate)
for batch in loader:
    batch = decode_cameras(batch, spec, device="cuda")   # 1/reduce; or size=(800, 520), scale=0.3
    batch["camera"]["camera_ring_front"]["image"]       # (8, 3, H/4, W/4) uint8 on the GPU
    batch["camera"]["camera_ring_front"]["K"]           # (8, 3, 3) for that size
```

- The samples carry the full-size JPEG and its full-size `K`.
- `decode_cameras` decodes with `torchvision.io.decode_jpeg(device="cuda")` (in batches with torchvision 0.19 or newer) and resizes on the GPU (bilinear, antialiased).
- It scales `K` with `kitscenes.decode.scale_intrinsics`: the pixel-centre convention of `Camera.scaled`, per axis.
- `kitscenes.decode.decode_images(jpegs, size=(w, h) or scale=..., device="cuda", K=...)` does the same for any list of JPEGs (`frame.image_bytes()`, `frame.sensor_file(camera).read()`).
- Without CUDA or torchvision it decodes on the CPU (OpenCV or Pillow, with JPEG DCT scaling to 1/2, 1/4, 1/8): for debugging.
- NVIDIA DALI reads the cameras straight from the zips (`kitscenes.dali`: `fn.readers.webdataset` → `fn.decoders.image(device="mixed")` → `fn.resize`, example in its module docs). Its resize gives the same images as `decode_images`, so `K` scales the same way.

Images per second for `camera_ring_front` (3504 × 2272, 2.3 MB JPEGs), decoded and resized; the CPU rows are one process, the GPU rows one GPU:

| | full size | 1/2 | 1/4 |
|---|---|---|---|
| CPU, `frame.image(reduce=)` with OpenCV (DCT scaling) | 15–25 | 36 | 41 |
| CPU, `frame.image(reduce=)` with Pillow (without OpenCV) | 11 | 34 | 37 |
| CPU, Pillow decode + resize | 10 | 9 | 12 |
| GPU, `decode_images` (batched nvJPEG) | 520 | 510 | 520 |
| GPU, DALI from the zips (batch 32, reading included) | 730 | 715 | 740 |

`SampleSpec(lidar="deskew_inputs")` gives the inputs of `kitscenes.motion_torch.deskew_on_device()`, for deskewing on the GPU.

## Frames, caches, speed

- `ds.frames(require=...)` is the flat, picklable list of frames behind a `SampleDataset` (`SampleDataset(ds, spec, frames)` takes one). A `DataLoader` cannot batch the frames themselves: they are views that read on access.
- Building it opens every scene once, in threads (`workers=`). On the zips that is ~6 ms per scene (about 15 s for the 2,079 scenes).
- A dataset keeps the last 32 opened scenes per process (timeline, poses, calibration, file index, map). A shuffled loader over many scenes, map labels above all, gains from `kitscenes.open(..., scene_cache=256)`.
- Nothing is cached by default. In a notebook that reads the same frames often, `kitscenes.open(..., cache_mb=1024)` keeps decoded payloads; they are then shared and read-only.
- If the spec needs a sensor you did not download, `Dataset.frames(require=...)` and `SampleDataset` raise instead of returning no frames.
