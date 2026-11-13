# The scene table

One row of recording metadata per scene (split, city, lidar, week, time of day, light, weather, road types, speeds), shipped with the API. Use it to pick scenes before you download them, or to filter a dataset.

```python
kitscenes.find_scenes(weather="rain", dark=False)          # SceneInfo rows; no data needed
ds.filter(city="Frankfurt", time_of_day="night")           # a sub-dataset
```

```bash
kitscenes download "$KITSCENES_ROOT" --where weather=rain --where dark=true
```

`kitscenes/_release_tables/v2.0.0/scenes.csv` describes every scene of dataset release v2.0.0 (2,079 scenes,
every release folder; `kitscenes.splits.RELEASE`). It ships with the API package, not with the dataset: `kitscenes.scene_table()`
returns it, `scene.info` is a scene's row (`kitscenes.splits.SceneInfo`), and `Dataset.filter(**where)`,
`kitscenes.find_scenes(**where)` and `kitscenes download --where COLUMN=VALUE` select scenes by it (see
[Filters](#filters)).

**Versions.** The table's version is the API's: corrections of its metadata (e.g. a weather label) come
as API patch releases, not as dataset releases.

**No absolute times.** Only categories and rounded statistics: nothing finer than the ISO week and a
time-of-day bucket says when a scene was recorded.

## Columns

Missing values are empty in the CSV and `None` in `SceneInfo`. Counts are over the 2,079 scenes.

### Identity and split

| Column | Type | Meaning |
|---|---|---|
| `scene_id` | str | The scene's UUID (its folder name under `data/<split>/`). |
| `split` | category | The release folder (`data/<split>/`), the scene's split in the default split definition `v2.0.0_default_geosplit`: `train`, `val`, `test`, `test-e2e`, `overlap_train_val`, `map_change_train`, `map_change_val`, `map_change_test`, `map_change_test-e2e`, `map_change_overlap_train_val`. |
| `city` | category | `Karlsruhe`, `Frankfurt`, `Sindelfingen`; filled for every split, test-e2e included. |
| `lidar` | category | Lidar generation: `velodyne` (2025, 979 scenes) or `hesai` (2026, 1,100). |
| `num_frames` | int | Frames of the scene as defined by the split (10 Hz). For the test-e2e splits this is the length of the recorded scene; 41 frames (4 s) of it ship. |

### When it was recorded

| Column | Type | Meaning |
|---|---|---|
| `recording_week` | category | ISO week of the recording, e.g. `2026-W06`. |
| `time_of_day` | category | Local time (Europe/Berlin) at the middle of the scene: `morning` 05–11, `midday` 11–14, `afternoon` 14–18, `evening` 18–22, `night` 22–05. |
| `light` | category | Sun elevation at the middle of the scene: `day` ≥ 6°, `low_sun` 0°..6°, `twilight` −6°..0° (civil twilight), `night` < −6°. |
| `sun_below_horizon` | bool | Sun elevation < 0° at the middle of the scene. |
| `dark` | bool | Sun elevation < −6° during the whole scene: a real night scene (425 scenes). |
| `recording_order` | int | Global order of the scenes by start time, 1..1967, **numbered over the public scenes only** (every split except the test-e2e splits), without gaps. |
| `drive` | int | Anonymous number of the recording drive (1..43, in recording order), numbered over the public scenes only. |
| `order_in_drive` | int | Order of the scene within its drive (1..n without gaps), numbered over the public scenes only. |

### Sensors and maps

| Column | Type | Meaning |
|---|---|---|
| `rig` | category | Sensor and calibration state: `velodyne_2025`, `hesai_2026_before_incident`, `hesai_2026_after_incident` (the 2026 rig was recalibrated after an incident). |
| `sd_snapshot` | str | File name of the scene's OpenStreetMap SD-map snapshot (`openstreetmap_sd_maps/<City>_<YYYY>-W<ww>.osm.pbf`, the city on Monday 00:00 UTC of the recording week); `scene.sd_map` loads it. |

### Motion (from the shipped poses)

| Column | Type | Meaning |
|---|---|---|
| `duration_s` | float | Duration of the shipped poses (4.0 for the test-e2e splits). |
| `distance_m` | float | Distance driven (horizontal). |
| `mean_speed_mps` | float | `distance_m / duration_s`. |
| `max_speed_mps` | float | Highest speed averaged over 1 s windows. |

### Road types (from the scene's SD snapshot)

Every 5th pose is matched to the nearest drivable OpenStreetMap way segment within 15 m whose direction
agrees with the driving direction (|cos| ≥ 0.8, which keeps crossing roads and bridges out), and the
distance driven between poses is summed per road class. A standing scene gets the class of the road under
its first pose.

| Column | Type | Meaning |
|---|---|---|
| `road_motorway_share` | float | Share of the distance on `motorway`, `trunk` (+ `_link`). |
| `road_arterial_share` | float | On `primary`, `secondary` (+ `_link`). |
| `road_local_share` | float | On `tertiary` (+ `_link`), `unclassified`, `residential`, `living_street`, `road`. |
| `road_service_share` | float | On `service`. |
| `road_unmatched_share` | float | Off the mapped drivable roads (no way within 15 m: yards, campus and parking areas). The five shares sum to 1 (two decimals). |
| `road_dominant` | category | The class with the largest share: `motorway`, `arterial`, `local`, `service`, `unmatched`. |
| `speed_limit_kph_median` | float | Median of the matched ways' `maxspeed` over the driven distance, in km/h (numbers; `DE:urban` 50, `DE:rural` 100). `inf` (`none` in the CSV): an unlimited motorway stretch is the median. Empty where no matched way carries a speed limit (135 scenes, the test-e2e ones included). |

### Weather (from the cameras)

Classified per ~10-minute segment of a drive: one frame of four ring cameras (front, front-left,
front-right, rear) per segment, classified by an image model; every scene gets the values of its segment.

| Column | Type | Meaning |
|---|---|---|
| `weather` | category | `clear`, `partly_cloudy`, `overcast`, `rain`, `fog`, `snow`, `unknown` (mostly night without visible sky). |
| `road_surface` | category | `dry`, `wet`, `snow_ice`, `unknown`. |
| `weather_confidence` | category | The classifier's confidence in both: `high`, `medium`, `low` (low is mostly night). |

## The test-e2e splits

The test-e2e splits (`test-e2e`, `map_change_test-e2e`) ship only the first 4 s of a scene, with poses
re-based to the scene start, so that where they are stays hidden. Their rows therefore

- describe the shipped 4 s only (motion, time of day, light, weather of the shipped frames);
- leave these columns empty on purpose (`kitscenes.splits.withheld_columns(split)`, the data file
  `_release_tables/v2.0.0/withheld_columns.json`):
  - `recording_order`, `drive`, `order_in_drive`: their neighbours in a drive are often public scenes a
    few seconds and a few hundred metres away, so the order would place them. The public scenes are
    numbered without them, so no gap in the numbering shows where one sat;
  - the road types (`road_*_share`, `road_dominant`, `speed_limit_kph_median`): together with city, week
    and speed they would narrow a scene down to a few road stretches.

A filter on a withheld column never matches a test-e2e scene (its value is `None`).

## Filters

Missing values are `None`, and a criterion never matches them: a filter on a withheld column leaves out the test-e2e scenes ([above](#the-test-e2e-splits)).

```python
kitscenes.find_scenes(weather="rain", dark=False)              # SceneInfo rows, table order
ds.filter(splits="train", time_of_day=["evening", "night"])    # categories: a value or a list (any case)
ds.filter(max_speed_mps=(None, 15), road_motorway_share=(0.5, None))   # numbers: a value, a list or a range (low, high)
ds.filter(dark=True, city="Karlsruhe", lidar="hesai")          # booleans: True / False
```

The download takes the same criteria: `kitscenes download ROOT --where weather=rain,overcast --where max_speed_mps=..15` (see `kitscenes download --help`).
