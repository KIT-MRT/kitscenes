# Lidar maps

`kitscenes.mapping` builds a voxel map on a GPU, for visualization and mapping over many frames or several overlapping scenes. Needs `torch` ([Training](training.md#install)).

Per voxel it keeps:

- the centroid and the number of points;
- how many frames hit it and the longest run of *consecutive* frames that did (a passing car occupies a voxel for an instant, a wall frame after frame);
- the mean reflectivity;
- if asked, the mean camera colour and the majority seamseg class. Every point the front or rear camera saw in the voxel votes, for its class or, where no mask covers it, for "unlabelled" (the masks label objects, not road or buildings); the class is the plain majority, ties go to "unlabelled". Colour and class come from [`kitscenes.painting`](sensors.md#more-per-frame-and-scene): a point hidden from a camera by something nearer takes the colour and class of what is in front of it, so the road behind a parked car votes "Car".

Memory grows with the mapped surface, not with the number of frames.

```python
from kitscenes.mapping import VoxelMap, keyframes

vmap = VoxelMap(voxel_size=0.05, device="cuda", colour=True, classes=True)
for scene in scenes:                                    # scenes with maps/origin.json, one UTM zone
    vmap.add_scene(scene, keyframes(scene, min_distance=0.5), max_range=60)
cloud = vmap.to_cloud(min_run=3)                        # centroids in the first scene's map frame; traffic removed
write_pcd("map.pcd", cloud)                             # with count, frames, run, reflectivity, rgb, class, class_votes
tiles = vmap.to_tiles(50.0, min_run=3)                  # {(utm_x // 50, utm_y // 50): cloud}: tiles of all maps line up

vmap.save("scene_a.npz")                                # per-scene maps merge exactly later:
VoxelMap.load("scene_a.npz").merge(VoxelMap.load("scene_b.npz"))
```

- Each point is moved with the pose at its own time, from a table on a 0.1 ms grid (deskew error at most half a bin of motion, under 1 mm in practice).
- Reading decodes only `x, y, z, timestamp, reflectivity`.
- Voxels lie on a grid fixed in UTM.
- Device memory stays at a few GB however large the map gets. Finished scenes are kept in host memory, 68 bytes per voxel (84 with colour).
- test-e2e scenes have no map frame, so they cannot join a map of several scenes.

For a few frames of one scene, `scene.accumulate(...)` is simpler ([Sensors](sensors.md#more-per-frame-and-scene)).
