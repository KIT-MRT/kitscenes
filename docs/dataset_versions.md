# The dataset: overview, versions, release notes

The authoritative description and release notes are on the [dataset card](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal). This page keeps what an API user needs.

**Contents:** [About the dataset](#about-the-dataset) · [Versioning](#versioning) · [Which API reads which release](#which-api-reads-which-release) · [Release notes](#release-notes) · [Legacy: dataset v1.x](#legacy-dataset-v1x)

## About the dataset

KITScenes Multimodal is a European urban autonomous driving dataset targeting Level 4 robotaxi requirements, recorded across Karlsruhe, Frankfurt, and Sindelfingen by the Institute of Measurement and Control Systems (MRT) at Karlsruhe Institute of Technology (KIT). The dataset is built around a **high-fidelity sensor suite**: nine high-resolution global-shutter cameras provide full 360° surround coverage at 72.5 MPix per frame, enabling novel view synthesis and holistic HD map perception, while seven highly dense long-range LiDARs with an effective range beyond 400 m push the limits of what current perception methods can achieve.

KITScenes Multimodal provides what we believe to be **the most complete HD maps of any public autonomous driving dataset**, annotated in Lanelet2 format across 62 km² with full topological connectivity between lanes, signs, and traffic lights. The maps have been validated in closed-loop autonomous driving trials using the open-source Autoware stack — meaning they are ready to drive on directly, while simultaneously enabling research to close the gap between the current state of the art and the actual requirements of L4 robotaxi deployment.

| | | | |
|---|---|---|---|
| **2,079** driving scenarios · 11.8 h | **72.5 MPix** per frame · global shutter | **400+ m** effective LiDAR range | **62 km²** Lanelet2 HD maps |

- **European urban focus** — recordings from Karlsruhe (1,001 scenarios), Frankfurt (682), and Sindelfingen (284), plus 112 end-to-end test scenarios; 424,460 frames at 10 Hz recorded, 406,667 shipped (the end-to-end test scenarios ship their first 4 s)
- **High-fidelity sensor suite** — 72.5 MPix global-shutter surround cameras, 7 LiDARs (~906k points/frame on average) in two generations (2025: Velodyne top LiDAR, 979 scenarios; 2026: Hesai OT128, 1,100 scenarios), 3 Continental ARS548 4D imaging radars, and redundant GNSS/INS with a 50 Hz IMU
- **Long-range sensing** — effective LiDAR range beyond 400 m with substantially higher return density than common public driving datasets
- **Production-grade HD maps** — Lanelet2 maps with lane topology, regulatory elements, 29 road-feature classes, 120 traffic-sign classes (GTSIGN-220 taxonomy), and 3D-localized signs, traffic lights, and poles
- **Research benchmarks** — relational HD map perception, long-range monocular depth estimation, novel view synthesis, and end-to-end / world-model research

## Versioning

The [HuggingFace dataset](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal) follows [Semantic Versioning](https://semver.org) (`MAJOR.MINOR.PATCH`) for **data and on-disk layout**:

| Bump | Version | Meaning |
|------|---------|---------|
| **Major** | `X.0.0` | New dataset schema or file format — breaks backwards compatibility |
| **Minor** | `2.X.0` | New driving data — additional scenarios or recordings |
| **Patch** | `2.0.X` | Updated maps, labels, metadata, or calibrations within the existing schema — fully compatible, no new scenarios |

Releases are tagged on HuggingFace (e.g. `v2.0.0`). Pin a tag or commit for benchmark work (`kitscenes download --revision v2.0.0`, [Download](download.md#versions)).

The Python API versions its code separately ([CHANGELOG.md](../CHANGELOG.md)); this is `kitscenes` 0.2.0.

## Which API reads which release

| Dataset | Format | API |
|---|---|---|
| 2.0.0 | one zip per sensor and scene | `kitscenes` 0.2.0 (this package) |
| 1.0.0, 1.0.1 | one tar per scene | legacy `kitscenes` 0.1.0 |

Results on v1.0.x and v2.0.0 are not comparable: scenes changed split.

## Release notes

### v2.0.0

**Breaking: new release format and split definition.** Results on v1.0.x and v2.0.0 are not comparable.

- **2,079 driving scenarios** (was 1,006), recorded 2025 (Velodyne top lidar) and 2026 (Hesai OT128).
- **Default split `v2.0.0_default_geosplit`**: `train` 1,209 · `val` 288 · `test` 290 · `test-e2e` 107 · `overlap_train_val` 55, and the new **map-change splits** (scenes where the road changed after mapping; the map shows the old state): `map_change_train` 91 · `map_change_val` 15 · `map_change_test` 17 · `map_change_test-e2e` 5 · `map_change_overlap_train_val` 2. See [Splits](splits.md).
- **Scenes moved to their new split**: 118 of the 1,006 v1.0.1 scenes changed split (40 `test`/`test-e2e` → `train`, 32 `train` → `val`, 17 `train` → `overlap_train_val`, 16 `overlap_train_val` → `val`, 13 others). 17 left the release: 11 `test-e2e` scenes (excluded) and 6 that overlap the test regions.
- **One zip per sensor and scene** instead of one tar per scene, uncompressed (camera and lidar zips with 4 KB-aligned members): the API reads them in place, NVIDIA DALI reads the cameras straight from them. Timestamps, poses, calibration and the scene's HD map are loose files. See [Download](download.md#layout).
- **Manifest** `data/sequence_archives.csv`: one row per file (with `component`, SHA-256 and size). `index.parquet` per scene lists every member of its zips.
- **Partial downloads by sensor**: choose cameras and lidars; metadata, maps, labels, radar and GNSS/INS always come along (~20 MB per scene). See [Download](download.md#partial-downloads).
- **GNSS/INS as Parquet**: one file per message topic, `<device>/<topic>.parquet`, one row per message at the receiver's rate, with the frame it is assigned to. Replaces one YAML file per message. The `async/` folder is gone (its messages are the rows); the per-frame copies in `gnss/<topic>/` and `gnss_ins/<topic>/` are the rows with a `frame_index`. About 19,000 files per scene (median) become 45; reading all topics of a scene takes well under a second. See [GNSS/INS](gnss_ins.md).
- **Exact timestamps**: frame and sync times are stored exactly (decimal seconds, 9 decimals; v1.0.x rounded them to 0.1 ms), and every scene starts at its earliest measurement (time 0).
- **Release calibration** per lidar generation and rig state (2025 Velodyne; 2026 Hesai before and after a recalibration), with a valid IMU rotation (`gnss_ins_imu`) in every scene that has IMU data: the API refines the ego motion with the gyro there ([Sensors](sensors.md#calibration-and-fusion)).
- **Image corrections**: `calibration/image_corrections.json` in every scene, a small warp of `camera_base_front_right_rect` onto the rectified stereo pair that `calib.json` describes. The API applies it wherever it decodes the camera ([Sensors](sensors.md#calibration-and-fusion)).
- **City maps**: the lifted HD maps of Karlsruhe, Frankfurt and Sindelfingen in `lanelet2_hd_maps/<City>.osm`, with the test regions cut out (a 60 m × 60 m box around every pose of the test scenes) and the mapping tools' bookkeeping tags removed. The scene maps are cut from them.
- **SD maps**: OpenStreetMap snapshots of the three cities in `openstreetmap_sd_maps/` (contributor metadata removed; `ds.sd_maps()`), the SD-map prior for map perception: one per city and recording week, `<City>_<YYYY>-W<ww>.osm.pbf` (the state on Monday 00:00 UTC of the week, covering every scene with a margin of at least 2.9 km), and the 2026-05-03 snapshots used in the paper, `paper_2026-05-03/<City>.osm`. They are OpenStreetMap data, © OpenStreetMap contributors, under the ODbL 1.0 (`openstreetmap_sd_maps/ATTRIBUTION.md`), not under the dataset licence. See [SD maps](sdmap.md).
- **KITScenes Aerial**: DOP20 orthophotos of the three cities (20 cm/px RGB tiles, ~2 GB, third-party open data, `aerial/ATTRIBUTION.md`) and DINOv3 feature tiles computed from 8 cm/px municipal orthophotos (~657 GB, terms in `aerial/features/<set>/TERMS.md`), for the areas of the selected scenes: the tiles in every download (`--no-aerial` leaves them out), the features in the full download (`--no-aerial-features` leaves them out) and in a filtered one with `--aerial-features`. See [Aerial imagery](aerial.md).
- **Scene table** in the Python API: split, city, lidar generation and frame count per scene, and recording metadata (week, time of day, light, recording order and anonymous drive, rig, driving statistics, road types, weather; see [Scene table](scene_table.md)). The `test-e2e` splits withhold the recording order and the road types.
- **Data fixes**: ground labels aligned with their lidar sweep (one frame late for some lidars before); 68 lidar sweeps missing in 38 scenes restored from the recordings; radar timestamps that stayed absolute repaired; the native GNSS/INS messages of 39 scenes recovered from the raw recordings.
- **Otherwise unchanged per scene**: extracted, a scene has the same sensor files as in v1.0.1 (plus `index.parquet` and `calibration/image_corrections.json`) except the GNSS/INS messages, so code that reads the sensor folders keeps working.

### v1.0.1

- Small map update.
- In 8 scenes one radar has frames with timestamps that were not converted to the scene's time (absolute Unix-like values instead of seconds since the start; train `77945790`, `bbab8b1e`, `c562bd58`; test-e2e `0f59465a`, `4e0cc196`, `619bdca8`, `8e4102c5`, `a8085b84`). Fixed in v2.0.0.

### v1.0.0

- Initial release on HuggingFace.
- Established data layout, splits, and release structure.

## Legacy: dataset v1.x

This package reads dataset release 2.0 and later; `kitscenes download` refuses the 1.x releases. The 1.x releases (`v1.0.0`, `v1.0.1`: one tar per scene, still on HuggingFace under their tags) are read by the legacy **kitscenes 0.1.0** API: check out the `v0.1.0` tag of this repository and follow the README there (downloading and extracting the tars, mounting them with ratarmount or as SquashFS images).
