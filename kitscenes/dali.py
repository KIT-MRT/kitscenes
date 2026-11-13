"""NVIDIA DALI input: index files so ``fn.readers.webdataset`` reads camera zips directly.

The release archives store every member uncompressed at a 4096-byte aligned
offset (:mod:`kitscenes._archive`), which is what DALI's webdataset reader
needs. :func:`write_camera_indices` writes its index files (format v1.2)
from ``index.parquet``, without touching the zips. Write the indices of all
cameras of a pipeline over the same frame list: readers with the same
``seed``, ``random_shuffle`` and ``initial_fill`` then stay in lockstep.

Example (one reader per camera, decoded on the GPU (nvJPEG) and resized to
the model's input size there)::

    from nvidia.dali import fn, pipeline_def, types
    from kitscenes.dali import write_camera_indices
    from kitscenes.decode import scale_intrinsics

    cams = ("camera_ring_front", "camera_ring_rear")
    frames = ds.frames(require=cams)
    sources = write_camera_indices(frames, cams, "/tmp/idx")
    W, H = 800, 520                                  # the model's input (width, height)

    @pipeline_def(batch_size=8, num_threads=4, device_id=0)
    def pipe():
        outs = []
        for cam, (paths, index_paths) in sources.items():
            jpg = fn.readers.webdataset(paths=paths, index_paths=index_paths, ext=["jpg"],
                                        random_shuffle=True, initial_fill=64, seed=1, name=cam)
            image = fn.decoders.image(jpg, device="mixed", output_type=types.RGB)
            outs.append(fn.resize(image, resize_x=W, resize_y=H))   # linear, antialiased
        return tuple(outs)

    # K of the resized images: the same formula as kitscenes.Camera.scaled, per axis
    camera = frames[0].scene.rig.cameras["camera_ring_front"]   # per scene: the rig can differ
    w, h = camera.image_size
    K = scale_intrinsics(camera.K, (W / w, H / h))               # = camera.scaled(W / w).K if W / w == H / h

``fn.resize`` keeps pixel centres where :func:`kitscenes.decode.decode_images`
does (same images up to rounding), so both use :func:`~kitscenes.decode.scale_intrinsics`.
The readers yield images only; ``K`` comes from the calibration of each
image's scene (with ``random_shuffle=False`` the images come in the order of
*frames*).

DALI yields the stored JPEGs; apply
:func:`kitscenes.decode.apply_image_corrections` yourself for cameras in
``scene.files.image_corrections``.
"""

from __future__ import annotations

from itertools import groupby
from pathlib import Path
from typing import Sequence

from kitscenes.constants import CameraNameLike, sensor_names
from kitscenes.errors import Availability
from kitscenes.frame import Frame
from kitscenes._storage import ZipStorage


def write_camera_indices(
    frames: Sequence[Frame], cameras: Sequence[CameraNameLike], out_dir: Path | str,
) -> dict[str, tuple[list[str], list[str]]]:
    """Write DALI webdataset index files for *cameras* over *frames*, in this order.

    Frames of one scene must be contiguous in *frames* (as ``Dataset.frames()``
    returns them); each run becomes one (zip, index) pair.

    Returns:
        ``{camera: (zip_paths, index_paths)}`` for ``fn.readers.webdataset``.

    Raises:
        ValueError: If a scene is not in the archive format, a frame has no image, or a camera's zip is
            not in the scene folder (extracted or not downloaded).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = sensor_names(cameras)
    result: dict[str, tuple[list[str], list[str]]] = {cam: ([], []) for cam in cameras}
    seen: set[str] = set()
    for run, (scene_id, group) in enumerate(groupby(frames, key=lambda f: f.scene.id)):
        group = list(group)
        if scene_id in seen:
            raise ValueError(f"Frames of scene {scene_id} are not contiguous.")
        seen.add(scene_id)
        storage = group[0].scene.files.storage
        if not isinstance(storage, ZipStorage):
            raise ValueError(f"Scene {scene_id} is not in the release archive format (no index.parquet).")
        for cam in cameras:
            lines = []
            for frame in group:
                where = frame.sensor_file(cam, missing="none")
                if where is None or not where.in_zip:  # dropped, extracted or not downloaded: DALI reads only the zip
                    raise ValueError(_not_in_zip(frame, cam, storage))
                lines.append(f"jpg {where.offset} {where.size} {where.zip_member}")
            index_path = out_dir / f"{run:05d}_{scene_id}_{cam}.idx"
            index_path.write_text(f"v1.2 {len(lines)}\n" + "\n".join(lines) + "\n")
            result[cam][0].append(str(where.path))
            result[cam][1].append(str(index_path))
    return result


def _not_in_zip(frame: Frame, camera: str, storage: ZipStorage) -> str:
    """Why DALI cannot read *camera* at *frame* from its zip."""
    scene_id = frame.scene.id
    if frame.availability(camera) is Availability.DROPPED:
        return f"Scene {scene_id} has no {camera} image at frame {frame.index}."
    zip_path = storage.root / f"{camera}.zip"  # a camera's component is the camera
    if frame.availability(camera) is Availability.PRESENT:
        how = (f"it was extracted; remove {storage.root / camera} and "
               f"{storage.root / '.extracted' / (camera + '.zip.sha256')}, then download it")
    else:
        how = "it is not downloaded"
    return (f"Scene {scene_id}: {zip_path} is not there ({how}: kitscenes download "
            f"{storage.root.parent.parent} --scenes {scene_id} --components {camera}). DALI reads the zips.")
