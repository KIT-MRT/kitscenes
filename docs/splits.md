# Splits

`$KITSCENES_ROOT` is always the dataset root. Scenes are in `data/<split>/<scene_id>/`, and by default a scene belongs to the split whose folder it is in. Pass `splits=` (one name or a list) to pick splits; omit it to load all scenes you have on disk, across splits. To use another split without moving files, see [Split definitions](#split-definitions).

```python
ds_train = kitscenes.open(splits="train")
ds_test = kitscenes.open(splits=["test", "test-e2e"])
ds_all = kitscenes.open()                # all downloaded scenes under $KITSCENES_ROOT
ds_all.splits                            # split folders found, e.g. ['train', 'val', 'map_change_train']
ds_all[0].split_info                     # has_maps, has_labels, pose_frame, base split, family
ds_all.filter(splits="val")[:10]         # sub-datasets
```

## Release folders

Default split `v2.0.0_default_geosplit` (dataset v2.0.0, 2,079 scenes, a geographic split):

| Split | Scenes | Maps and labels | Poses |
|---|---|---|---|
| `train` | 1,209 | yes | map frame |
| `val` | 288 | yes | map frame |
| `test` | 290 | withheld | map frame |
| `test-e2e` | 107 | withheld; no GNSS/INS | relative to the first pose ([End-to-end driving](e2e.md)) |
| `overlap_train_val` | 55 | yes | map frame |
| `map_change_train` · `_val` · `_test` · `_test-e2e` · `_overlap_train_val` | 91 · 15 · 17 · 5 · 2 | as above | as above |

- The `map_change_` splits hold scenes where the road changed after mapping; the map shows the old state (`labels.map_changed`).
- `test_e2e` is accepted as an alias of `test-e2e`. The release folders and their conventions are in `kitscenes.splits.SPLITS`.
- The test regions are cut out of the city maps, so no test road is in a training map.
- `scene.split_info` describes the scene's release folder (`scene.release_split`), whatever split definition the dataset follows.

## The scene table

The API ships the scene table of the default split `v2.0.0_default_geosplit` (`kitscenes.scene_table()`): split, city, lidar generation, frame count and recording metadata per scene ([Scene table](scene_table.md)). The test-e2e scenes withhold the recording order and the road types.

```python
scene.info                                   # SceneInfo: split, city, lidar, frames, week, light, weather, road types, ... (scene table)
ds.filter(city="Frankfurt", lidar="hesai")   # any case
```

`kitscenes download --splits/--cities/--lidar` select from the same table ([Download](download.md)).

## Split definitions

A split definition says which split each scene is in: a table `scene_id → split`. The files stay where the release puts them.

**The default split.** `v2.0.0_default_geosplit` (`kitscenes.splits.DEFAULT_SPLIT`) is the release folders. It is a geographic split:

- the test regions are cut out of the city maps, so no test road is in a training map;
- test scenes are at least 50 m from train and val scenes;
- train and val scenes are at least 40 m apart.

The benchmark and all reported results use it.

**Your own split: only with care.** You can define another split. Do it only with great care and a clear scientific reason why your split is interesting, helpful or better, and state that reason in the work that uses it. Results on another split are not comparable to the benchmark. Only the default split guarantees that no test road is in a training map; your split has to take care of overlap between its splits itself.

**How.**

```bash
kitscenes splits export my_split.csv     # the default as a CSV (scene_id,split) to edit
kitscenes splits check my_split.csv      # the rules below; every problem is listed (exit code 2)
```

```python
ds = kitscenes.open(split_definition="my_split.csv")   # splits=, filter, split_of and scene.split follow it
scene.split, scene.release_split                      # e.g. 'holdout', 'val': the data keeps its release folder's conventions
```

The map converter takes the same file (`kitscenes-map-converter --split-definition my_split.csv`) and writes the definition's name and `id` (a hash of its content) into its outputs; kitscenes-map-eval refuses a submission made for another definition.

Rules: every scene is a released scene and is listed once; split names are `[A-Za-z0-9_.-]+`. The scenes of a split are of one kind: labelled, label-free (test) or test-e2e. A definition may leave scenes out; `# name: <name>` names it, else it is named after the file.
