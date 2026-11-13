# Visualization and the 3D viewer

Three ways to look at the data: plots and reprojection into the cameras, MP4 videos of whole scenes rendered on the GPU, and a 3D viewer in the browser.

**In short**

- `render_frame` draws map labels, lidar and radar into the camera images; `plot_top_down` shows map labels from above. Needs `kitscenes[viz]`.
- `kitscenes video` renders the ring cameras with map labels and a BEV into one MP4, on the GPU with NVENC. Needs `kitscenes[video,map]` and ffmpeg.
- `kitscenes view` opens the 3D viewer (`kitscenes-viewer`, a separate distribution in `viewer/`).

```bash
kitscenes video "$KITSCENES_ROOT" --splits val --max-scenes 2 --out /tmp/check.mp4
```

## Top-down view and camera reprojection

Two views on the same labels and the same projection (`kitscenes.projection`, which painting and the lidar maps use too). Needs `kitscenes[viz]` ([install options](../README.md#install-options)) and, for map labels, the ML converter ([Maps](maps.md#lanelet2-the-kitscenes-build)).

```python
from kitscenes.visualization import MapLayer, render_frame, camera_grid, plot_top_down

fig, ax = plot_top_down(labels)                                   # the converter's top-down view
images = render_frame(frame, map_layer=MapLayer(scene), lidar=True, radar=True, reduce=4)   # {camera: RGB}; ~4 s
                                                                  # (reduce=1, full size: ~40 s on a CPU)
grid = camera_grid(images)                                        # the six ring cameras, 3 x 2
```

The reprojection draws:

- the map labels: styled by type, dashes by distance, icons for signs and road markings, lines clipped at the camera in 3D;
- the lidar points, coloured by distance, nearest on top (points the camera cannot see are drawn too);
- optionally the radar detections, coloured by Doppler.

Both views leave out traffic signs that have no type in the map; `plot_top_down(labels, show_untyped_signs=True)` draws them.

For a whole scene, written outside the dataset:

```bash
python examples/render_reprojection.py --split val --out /tmp/render --frames 0:200:5 --radar
```

The example calls `render_scene` (JPEGs per camera, the 3 x 2 grid and the top-down view per frame) and `write_video` (an MP4 of grid and top-down view). This path draws lidar and radar and runs without torch.

## Videos on the GPU: `kitscenes video`

A video of a selection of scenes. Every frame shows:

- a header: scene, split, city and week, time of day, weather, frame;
- the six ring cameras with the map labels drawn in, for a region of 120 x 60 m around the vehicle;
- optionally the labels from above (BEV), with arrowheads on the centerlines in driving direction;
- a legend.

The labels look as in the reprojection and the top-down view above. It is a visualization of the data: no model runs. Scenes follow each other in recording order.

```bash
kitscenes video "$KITSCENES_ROOT" --splits val --out /tmp/val.mp4                       # 1080p, cams + BEV, real time
kitscenes video ROOT --scenes ID1 ID2 --out /tmp/two.mp4 --resolution 4k --speed 2 --layout cams
kitscenes video ROOT --splits train --where weather=rain --max-scenes 5 --out /tmp/rain.mp4 --per-scene
kitscenes video ROOT --splits val --out /tmp/val.mp4 --gpus 8                             # whole scenes per process
kitscenes video ROOT --splits val --out /tmp/x.mp4 --dry-run                              # the selection and its frame rate
```

```python
from kitscenes.visualization import render_video

result = render_video(ds, "/tmp/val.mp4", splits="val", resolution="4k", layout="cams+bev", speed=2)
result.frames, result.fps, result.duration_s, result.render_fps
```

| Option | Default | |
| ------ | ------- | - |
| `--splits`, `--scenes`, `--cities`, `--lidar`, `--where COLUMN=VALUE` | all | the selection, as `kitscenes download` ([Scene table](scene_table.md)) |
| `--order` | `recording` | `recording` (scene table; scenes without a recording order, e.g. test-e2e, after) or `id` |
| `--max-scenes N` | | only the first N scenes |
| `--frames START:STOP[:STEP]`, `--frame-step K` | all | frames of each scene; every K-th of them |
| `--dry-run` | | print the selection and the frame rate, render nothing |
| `--layout` | `cams+bev` | `cams+bev`: the 3 x 2 camera grid, on the right a column with the BEV (driving up) and the legend below it. `cams`: the grid over the full width, the legend in a strip at the bottom |
| `--legend` | `compact` | `compact`: what the scene shows, the same for all its frames. `full`: every class. `none` |
| `--rear-row` | `swap` | `swap`: rear right, rear, rear left, a panorama as seen looking back. `mirror`: rear left, rear, rear right, each image mirrored like a rear-view mirror. `plain`: that order, not mirrored. The captions always name the real camera |
| `--crop-hood [FRACTION]` | off | cut the ego vehicle off the bottom of the images (per camera, `HOOD_FRACTIONS`, or one fraction of the image height), so the rest is shown larger |
| `--fit` | `cover` | `cover`: the images fill their tiles and are cut at the sides. `contain`: whole images |
| `--region` | `120x60` | labelled region, length x width in metres (cameras and BEV) |
| `--no-map`, `--grouping`, `--line-width` | | no labels; the ML converter's line type grouping; a factor on the line widths |
| `--out` | | the MP4 to write, outside the dataset root |
| `--resolution` | `1080p` | `1080p` (1920 x 1080), `2k` (2560 x 1440), `4k` (3840 x 2160) or `WxH` (even) |
| `--speed`, `--max-fps` | `1`, `60` | playback speed and the highest frame rate, see below |
| `--codec`, `--encoder` | `hevc`, `auto` | `hevc` or `h264`; `auto`: NVENC on a GPU, else `libx265` / `libx264`; `nvenc`; `software` |
| `--quality`, `--max-bitrate`, `--bitrate` | | see below |
| `--per-scene`, `--no-concat` | | also keep one MP4 per scene, in `<out stem>_scenes/`; only those |
| `--device` | `auto` | the first GPU, else the CPU; `cuda`, `cuda:K` or `cpu` |
| `--gpus`, `--procs-per-gpu` | `1`, `4` | GPUs to use and renderer processes per GPU (whole scenes each, balanced by frames) |
| `--workers` | `8` | reader processes per renderer |
| `-q` | | no progress, print only the output path |

**Speed and frame rate.** The dataset has 10 frames per second, so `--speed 1` plays at 10 fps (real time), `--speed 2` at 20 fps and `--speed 0.5` at 5 fps. Above `--max-fps` frames are dropped instead of raising the rate: `--speed 8` shows every second frame at 40 fps. With `--frame-step K` the speed stays the same: every K-th frame at 10 / K fps for `--speed 1`.

**Encoding.** Constant quality, HEVC by default, capped at `--max-bitrate` (80 Mbit/s at 4K and 10 fps, less for fewer pixels); `--quality Q` changes the level (lower is better), `--bitrate R` encodes at an average bitrate instead.

**Scenes without labels** are rendered without them, and the header says why: the test splits ship no maps.

**Throughput.** With the defaults, one GPU renders about 160 frames/s at 1080p and 80 at 4K when the images come from a fast local disk. From a network share, reading the images is usually the limit. Without a GPU everything runs on the CPU: the same video, at a few frames per second.

**Memory.** A render with the defaults needs about 10 GB at 1080p and 17 GB at 4K; `--gpus N` multiplies it.

**Requirements.**

- `kitscenes[video,map]`: torch 1.10 or newer (for your CUDA from pytorch.org), OpenCV, matplotlib and the ML converter.
- The ffmpeg binary, on `PATH` or in `$KITSCENES_FFMPEG`. For the GPU encoder it must be built with NVENC (`ffmpeg -encoders | grep nvenc`).
- Without NVENC, `--encoder auto` warns and encodes on the CPU; `--encoder nvenc` stops with the reason.
- In a container, NVENC needs the `video` driver capability: `docker run --gpus ... -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`.

## 3D viewer

`viewer/` is a separate distribution, `kitscenes-viewer`: the map, lidar, cameras and trajectories of a scene, or of every scene in a city, in the browser, built on this API.

- Install in a clone of the repository: `pip install -e . -e viewer`.
- `kitscenes view <scene | root>` opens it.
- Details: [viewer/README.md](../viewer/README.md).
