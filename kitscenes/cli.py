"""The kitscenes command line.

    kitscenes download ROOT --scenes ID                      # one scene
    kitscenes download ROOT --splits val --components ring-cameras lidar-top
    kitscenes download ROOT --splits train --images half --dry-run   # cameras at 1/2 size; what it would fetch
    kitscenes download ROOT                                  # everything (12.1 TB)
    kitscenes info ROOT -v                                   # splits, scenes and downloaded components
    kitscenes transcode ROOT --images quarter                # downscale the cameras of a root
    kitscenes extract ROOT                                   # unzip (optional; the API reads the zips)
    kitscenes splits export my_split.csv                     # the default split definition as a CSV to edit
    kitscenes video ROOT --splits val --out val.mp4          # cameras, map labels and a BEV panel as MP4

Every command has its own --help. Exit codes: 0 done, 1 something failed
(the errors are listed), 2 a version conflict or a usage error.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from kitscenes.download import archives
from kitscenes.splits import DEFAULT_SPLIT, _split_name

# The command line is the interface; this module itself is internal.
_DEFAULT_REPO_ID = "KIT-MRT/KITScenes-Multimodal"
_MANIFEST = "data/sequence_archives.csv"


class VersionConflict(ValueError):
    """The root holds another version of the dataset than asked for."""


class Hub:
    """The HuggingFace dataset repo (tests stand in for it)."""

    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id

    def resolve(self, revision: Optional[str]) -> str:
        """The commit SHA of *revision* (a branch, tag or SHA; None: the default branch)."""
        import huggingface_hub

        return huggingface_hub.HfApi().dataset_info(self.repo_id, revision=revision).sha

    def download(self, filename: str, revision: Optional[str], local_dir: Path) -> Path:
        import huggingface_hub

        return Path(huggingface_hub.hf_hub_download(
            repo_id=self.repo_id, repo_type="dataset", filename=filename, revision=revision, local_dir=local_dir,
        ))


@dataclasses.dataclass
class _Manifest:
    """The manifest a download works with, and what it changes in the root (see :func:`_commit`)."""

    rows: list[dict]
    files: list[archives.ArchiveFile]
    revision: Optional[str]
    #: A manifest fetched from the hub, not yet in the root.
    staged: Optional[Path] = None
    #: Files of the new manifest whose SHA-256 differs from the manifest the root has.
    changed: list[archives.ArchiveFile] = dataclasses.field(default_factory=list)
    #: Write ``dataset_info.json``.
    record: bool = False


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _manifest(root: Path, repo_id: str, *, revision: Optional[str] = None, update: bool = False,
              hub: Optional[Hub] = None, workdir: Path) -> _Manifest:
    """The manifest for *root*, and the revision its files come from; nothing in *root* changes.

    The first download pins a revision in ``dataset_info.json``; later ones
    use it. Asking for another revision raises :class:`VersionConflict`
    unless *update*, which fetches that revision's manifest. A local
    manifest that no longer matches the recorded SHA-256 is fetched again.
    A root with a manifest but no pinned revision (a local copy)
    keeps its manifest, unpinned, unless *update* pins it. A fetched
    manifest goes to *workdir*, is checked (format, paths), and is compared
    with the root's: :func:`_commit` puts it in place.
    """
    hub = hub or Hub(repo_id)
    local = root / _MANIFEST
    info = archives.read_dataset_info(root)
    if info is not None and info.get("repo_id") not in (None, repo_id):
        raise VersionConflict(f"{root} holds {info['repo_id']}, not {repo_id}: use another root.")
    pinned = info.get("revision") if info else None
    want = pinned
    if revision is not None or update or (pinned is None and not local.is_file()):
        want = hub.resolve(revision)
        if pinned is not None and want != pinned and not update:
            raise VersionConflict(
                f"{root} holds revision {pinned[:12]} of {repo_id}; you asked for {want[:12]}. "
                "Add --update to move the whole root to it, or use another root.")
        if pinned is None and local.is_file() and not update:
            raise VersionConflict(f"{root} has an unpinned manifest; add --update to pin it to {want[:12]}.")
    local_sha = archives.sha256_file(local) if local.is_file() else None
    fresh = want is not None and (
        local_sha is None or want != pinned or (info is not None and info.get("manifest_sha256") != local_sha))
    staged = Path(hub.download(_MANIFEST, want, workdir)) if fresh else None
    rows = _read_rows(staged or local)
    try:
        archives.check_manifest(rows)
    except archives.UnsupportedRelease as err:
        raise VersionConflict(f"revision {str(want)[:12]}: {err}") from None
    files = archives.parse_manifest(rows)
    changed = []
    if staged is not None and local_sha is not None:
        try:
            old = {r["archive_path"]: r["archive_sha256"] for r in _read_rows(local)}
        except (KeyError, csv.Error):
            old = {}
        changed = [f for f in files if old.get(f.path, f.sha256) != f.sha256]
    return _Manifest(rows, files, want, staged, changed, record=bool(rows) and (fresh or info is None))


def _commit(root: Path, manifest: _Manifest, repo_id: str) -> list[archives.ArchiveFile]:
    """Move *root* to *manifest*: remove the files that changed, put the manifest in place, record it.

    Returns:
        The changed files that had been extracted (to extract again).
    """
    extracted = []
    if manifest.staged is not None:
        extracted = archives.remove_stale(root, manifest.changed)  # first: an interrupted run keeps the old manifest
        local = root / _MANIFEST
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(local.name + ".partial")
        shutil.copyfile(manifest.staged, tmp)
        os.replace(tmp, local)
    if manifest.record:
        archives.write_dataset_info(root, repo_id=repo_id, revision=manifest.revision, manifest=root / _MANIFEST,
                                    rows=manifest.rows)
    return extracted


def _hub_fetcher(hub: Hub, revision: Optional[str]):
    def download_one(f: archives.ArchiveFile, root: Path) -> Path:
        return hub.download(f.path, revision, root)  # straight to root / f.path (HF renames when complete)
    return download_one


def _gb(files) -> float:
    return sum(f.size for f in files) / 1000 ** 3


def cmd_download(args: argparse.Namespace, download_one=None, hub: Optional[Hub] = None) -> int:
    root = Path(args.root)
    if args.dry_run:  # reads only
        return _download(args, root, download_one, hub or Hub(args.repo_id))
    root.mkdir(parents=True, exist_ok=True)
    with archives.root_lock(root):
        return _download(args, root, download_one, hub or Hub(args.repo_id))


def _area_index(root: Path, manifest: _Manifest, hub: Hub, download_one, workdir: Path) -> Optional[dict]:
    """The release's aerial area index (scene -> area): the root's copy if it is current, else fetched to *workdir*."""
    import json

    entry = next((f for f in manifest.files if f.path == archives.AREA_INDEX_PATH), None)
    if entry is None:
        return None  # a release without aerial imagery
    changed = {f.path for f in manifest.changed}
    if entry.path not in changed and archives.is_complete(root, entry):
        path = root / entry.path
    elif download_one is not None:
        path = Path(download_one(entry, workdir))
    else:
        path = hub.download(entry.path, manifest.revision, workdir)
    return json.loads(Path(path).read_text())


def _download(args: argparse.Namespace, root: Path, download_one, hub: Hub) -> int:
    with tempfile.TemporaryDirectory(prefix="kitscenes-manifest-") as workdir:
        try:
            manifest = _manifest(root, args.repo_id, revision=args.revision, update=args.update, hub=hub,
                                 workdir=Path(workdir))
        except VersionConflict as err:
            print(f"kitscenes download: {err}", file=sys.stderr)
            return 2
        selection = dict(splits=args.splits, scene_ids=args.scenes, cities=args.cities, lidar=args.lidar,
                         max_gb=args.max_gb, where=_where(args))
        aerial = _aerial_flags(args, archives.scene_filtered(**selection))
        area_index = None
        if archives.needs_area_index(manifest.files, **aerial, **selection):
            area_index = _area_index(root, manifest, hub, download_one, Path(workdir))
        files = archives.select_files(manifest.files, components=args.components, area_index=area_index,
                                      **selection, **aerial)
        if args.max_gb is not None:
            left_out = archives.aerial_left_out(manifest.files, files, area_index, **aerial)
            if left_out:
                print(f"--max-gb {args.max_gb:g}: KITScenes Aerial of the selected scenes' areas that does not fit "
                      f"is left out: {len(left_out)} zip(s), {_gb(left_out):.2f} GB "
                      f"({', '.join(sorted({f.component for f in left_out}))})")
        if aerial["aerial_tiles"] and not aerial["aerial_features"] and _features_default(args):
            features = archives.aerial_left_out(manifest.files, files, area_index, aerial_tiles=False)
            if features:
                print(f"Filtered download: the DINOv3 aerial features of the selected areas are left out; "
                      f"--aerial-features adds them ({len(features)} zip(s), {_gb(features):.2f} GB)")
        changed = {f.path for f in manifest.changed}
        if changed:
            print(f"{len(changed)} file(s) changed in revision {str(manifest.revision)[:12]}: "
                  f"{'would be ' if args.dry_run else ''}removed from {root}; the selected ones are downloaded again")
        try:
            recipes, policy_changes = _image_recipes(root, files, args.images)
        except VersionConflict as err:
            print(f"kitscenes download: {err}", file=sys.stderr)
            return 2
        extracted = [] if args.dry_run else _commit(root, manifest, args.repo_id)
    hashes = archives.HashCache(root) if args.verify else None
    todo = [f for f in files
            if f.path in changed or not archives.is_complete(root, f, verify=args.verify, hashes=hashes,
                                                             recipes=recipes)]
    if hashes is not None and not args.dry_run:
        hashes.save()
    scenes = {(f.split, f.scene_id) for f in files if f.scene_id}
    print(f"{len(scenes)} scene(s), {len(files)} file(s), {_gb(files):.2f} GB selected; "
          f"{len(todo)} file(s), {_gb(todo):.2f} GB to download into {root}")
    if recipes:
        _print_image_plan(files, todo, recipes, args.staging_gb)
    if args.dry_run:
        for f in todo:
            print(f"  {f.path}  ({f.size / 1e6:.1f} MB)")
        return 0
    import time

    transcoding = None
    if policy_changes:
        from kitscenes.transcode import update_policy

        update_policy(root, policy_changes, create=True)  # before fetching: an interrupted run continues with the recipes
    if recipes:
        from kitscenes.transcode import DownloadTranscoding

        transcoding = DownloadTranscoding(root, recipes, staging_bytes=args.staging_gb * 1e9,
                                          backend=args.transcode_backend, gpus=args.transcode_gpus, progress=print)
    started = time.perf_counter()
    try:
        failed = archives.fetch(
            root, files, download_one or _hub_fetcher(hub, manifest.revision),
            jobs=args.jobs, verify=args.verify, extract=args.extract, hashes=hashes,  # hashed above: not again
            transcoding=transcoding,
        )
    finally:
        if transcoding is not None:
            transcoding.close(cancel=True)
            transcoding.clean_staging()
    if extracted and not args.extract:  # extracted before the update: extract the new version too
        bad = {f.path for f, _ in failed}
        for f in extracted:
            if f.path not in bad and (root / f.path).is_file():
                try:
                    archives.extract_archive(root, f, hashes=hashes)
                except Exception as err:  # collected and reported
                    failed.append((f, err))
    for f, err in failed:
        print(f"  FAILED {f.path}: {err}", file=sys.stderr)
    seconds = time.perf_counter() - started
    rate = f", {_gb(todo) * 1000 / seconds:.0f} MB/s" if todo and seconds > 0 else ""
    check = "SHA-256" if args.verify else "size"
    print(f"{'Done' if not failed else f'{len(failed)} error(s)'} in {seconds:.0f} s{rate} ({check} checked).")
    return 1 if failed else 0


def _aerial_conflict(args: argparse.Namespace) -> Optional[str]:
    """The error for ``--aerial-features`` with ``--no-aerial-features`` or ``--no-aerial``, else None."""
    off = [f for f, on in (("--no-aerial-features", getattr(args, "no_aerial_features", False)),
                           ("--no-aerial", getattr(args, "no_aerial", False))) if on]
    if getattr(args, "aerial_features", False) and off:
        return f"--aerial-features asks for the aerial features, {' and '.join(off)} leaves them out: give one"
    return None


def _features_default(args: argparse.Namespace) -> bool:
    """Whether no flag decides about the aerial features (the rule does: full download yes, filtered no)."""
    return not (args.aerial_features or args.no_aerial_features or args.no_aerial)


def _aerial_flags(args: argparse.Namespace, filtered: bool) -> dict:
    """``--no-aerial`` / ``--aerial-features`` / ``--no-aerial-features`` and whether the selection filters scenes
    -> the ``aerial_tiles`` / ``aerial_features`` of :func:`kitscenes.download.archives.select_files`.

    Raises:
        ValueError: For contradicting flags.
    """
    conflict = _aerial_conflict(args)
    if conflict:
        raise ValueError(conflict)
    features = not (args.no_aerial or args.no_aerial_features) and (args.aerial_features or not filtered)
    return {"aerial_tiles": not args.no_aerial, "aerial_features": features}


def _image_recipes(root: Path, files, images: Optional[str]):
    """``({camera: recipe} to apply, {camera: recipe or None} to record)`` for the selected cameras.

    The root's recipes (``dataset_info.json``) apply to every download into
    it. ``--images RECIPE`` sets one for selected cameras that have none,
    ``--images original`` removes theirs (the release's images are fetched
    again). Another recipe than a camera already has is refused: change it
    with ``kitscenes transcode`` (smaller only), or go back to the release's
    images first.
    """
    from kitscenes.constants import CAMERA_NAMES
    from kitscenes.transcode import ORIGINAL, ImageRecipe, read_policy

    policy = read_policy(root)
    selected = sorted({f.component for f in files if f.component in CAMERA_NAMES})
    changes: dict = {}
    if images is not None:
        if images == ORIGINAL:
            changes = {c: None for c in selected if c in policy}
        else:
            recipe = ImageRecipe.parse(images)
            clash = sorted(c for c in selected if c in policy and policy[c] != recipe)
            if clash:
                raise VersionConflict(
                    f"{root} keeps {', '.join(clash)} at {policy[clash[0]]}, not {recipe}. Smaller: kitscenes "
                    f"transcode {root} --images {images} --cameras {' '.join(clash)}; back to the release's images: "
                    f"--images {ORIGINAL}")
            changes = {c: recipe for c in selected if c not in policy}
    recipes = {c: r for c, r in {**policy, **changes}.items() if r is not None and c in selected}
    return recipes, changes


def _print_image_plan(files, todo, recipes, staging_gb: float) -> None:
    from kitscenes.transcode import estimate_size

    cams = [f for f in files if f.component in recipes]
    before = sum(f.size for f in cams)
    after = sum(estimate_size(f.size, recipes[f.component]) for f in cams)
    kinds = sorted({str(r) for r in recipes.values()})
    rest = sum(f.size for f in files) - before
    print(f"images: {len(recipes)} camera(s) at {' / '.join(kinds)}: {before / 1e9:.2f} GB as released -> "
          f"~{after / 1e9:.2f} GB (estimate); with the rest ~{(rest + after) / 1e9:.2f} GB on disk, "
          f"plus a staging buffer of up to {staging_gb:g} GB while downloading")


def cmd_extract(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not (root / "data").is_dir() and not (root / "aerial").is_dir():
        raise ValueError(f"{root} holds no downloaded scenes (no data/ folder).")
    with archives.root_lock(root):
        return _extract(args, root)


def _extract(args: argparse.Namespace, root: Path) -> int:
    from kitscenes.constants import CAMERA_NAMES

    rows = _read_rows(root / _MANIFEST) if (root / _MANIFEST).is_file() else []
    archives.check_manifest(rows)  # refuses a 1.x root
    wanted = archives.resolve_components(args.components)
    known_sha = {r["archive_path"]: r["archive_sha256"] for r in rows if r.get("archive_path", "").endswith(".zip")}
    scene_dirs = [p for p in (root / "data").glob("*/*") if p.is_dir() and not p.name.startswith(".")]
    known_sha.update({r["archive_path"]: r["archive_sha256"] for r in rows
                      if r.get("component") in (archives.AERIAL_TILES, archives.AERIAL_FEATURES)})
    splits = scenes = None
    if args.splits:
        splits = {_split_name(s) for s in args.splits}
        present = {p.parent.name for p in scene_dirs}
        unknown = [s for s in args.splits if _split_name(s) not in present]
        if unknown:
            raise ValueError(f"No split(s) {unknown} in {root}. Present: {sorted(present)}")
    if args.scenes:
        scenes = set(args.scenes)
        unknown = sorted(scenes - {p.name for p in scene_dirs})
        if unknown:
            raise ValueError(f"Scene ID(s) not in {root}: {unknown}")
    where = archives.scenes_where(args.cities, args.lidar, _where(args))
    failed = done = 0

    def extract(how, zip_path: Path, expected: Optional[str]) -> None:
        nonlocal failed, done
        try:
            how(zip_path, sha256=expected)
            done += 1
            print(f"  extracted {zip_path.relative_to(root)}")
        except Exception as err:  # collected and reported
            failed += 1
            print(f"  FAILED {zip_path}: {err}", file=sys.stderr)

    for zip_path in sorted((root / "data").glob("*/*/*.zip")):
        split, scene = zip_path.parent.parent.name, zip_path.parent.name
        if (splits is not None and split not in splits) or (scenes is not None and scene not in scenes):
            continue
        if where is not None and scene not in where:
            continue
        if wanted is not None and zip_path.stem in archives.OPTIONAL_COMPONENTS and zip_path.stem not in wanted:
            continue
        expected = known_sha.get(zip_path.relative_to(root).as_posix())
        if zip_path.stem in CAMERA_NAMES:
            from kitscenes.transcode import read_variant

            variant = read_variant(zip_path.parent, zip_path.stem)
            expected = variant.sha256 if variant is not None and variant.sha256 else expected
        extract(archives.extract_component, zip_path, expected)
    area_zips = []  # KITScenes Aerial, unless left out: all, or the areas of the selection
    aerial = _aerial_flags(args, any(x is not None for x in (splits, scenes, where)))
    if aerial["aerial_tiles"]:
        area_zips += sorted((root / "aerial" / "raw_tiles").glob("*/*.zip"))
    if aerial["aerial_features"]:
        area_zips += sorted((root / "aerial" / "features").glob("*/*.zip"))
    if area_zips:
        areas = None
        if any(x is not None for x in (splits, scenes, where)):
            import json

            index_path = root / archives.AREA_INDEX_PATH
            index = json.loads(index_path.read_text()) if index_path.is_file() else {}
            selected = {(p.parent.name, p.name) for p in scene_dirs if (splits is None or p.parent.name in splits)
                        and (scenes is None or p.name in scenes) and (where is None or p.name in where)}
            areas = archives.scene_areas(selected, index)
        for zip_path in area_zips:
            if areas is None or zip_path.stem in areas:
                extract(archives.extract_aerial_tiles, zip_path, known_sha.get(zip_path.relative_to(root).as_posix()))
    print(f"{done} archive(s) extracted" + (f", {failed} failed" if failed else "") + ".")
    return 1 if failed else 0


def _scene_filter(args: argparse.Namespace) -> Optional[list[str]]:
    """Scene IDs of ``--scenes`` / ``--cities`` / ``--lidar`` / ``--where`` (None: no filter)."""
    where = archives.scenes_where(args.cities, args.lidar, _where(args))
    if args.scenes and where is not None:
        return sorted(set(args.scenes) & where)
    if args.scenes:
        return list(args.scenes)
    return sorted(where) if where is not None else None


def cmd_transcode(args: argparse.Namespace) -> int:
    from kitscenes import transcode as tc

    root = Path(args.root)
    target = Path(args.out) if args.out else root
    if not (root / "data").is_dir():
        raise ValueError(f"{root} has no data/ folder (scenes are expected in data/<split>/<scene_id>/)")
    recipe = tc.ImageRecipe.parse(args.images)
    scenes = _scene_filter(args)
    if scenes == []:
        print("kitscenes transcode: no scene matches the selection", file=sys.stderr)
        return 1
    options = dict(out=args.out, splits=args.splits, scenes=scenes, cameras=args.cameras, progress=print)
    gb = lambda n: n / 1e9  # noqa: E731
    if args.dry_run:  # reads only
        plan = tc.transcode(root, recipe, dry_run=True, **options)
        print(f"{len(plan.done)} camera(s) would be transcoded ({gb(plan.bytes_before):.2f} GB as stored), "
              f"{len(plan.skipped)} already {recipe}; nothing was changed.")
        return 0
    target.mkdir(parents=True, exist_ok=True)
    import time

    started = time.perf_counter()
    with archives.root_lock(target):
        result = tc.transcode(root, recipe, backend=args.backend, jobs=args.jobs, gpus=args.gpus, **options)
    print(f"{len(result.done)} camera(s) transcoded ({gb(result.bytes_before):.2f} GB -> {gb(result.bytes_after):.2f} GB), "
          f"{len(result.skipped)} already {recipe}, {len(result.failed)} failed, in {time.perf_counter() - started:.0f} s.")
    for scene, camera, err in result.failed:
        print(f"  FAILED {scene}/{camera}: {err}", file=sys.stderr)
    return 1 if result.failed else 0


def cmd_pack(args: argparse.Namespace) -> int:
    from kitscenes.pack import pack_scene
    from kitscenes.transcode import _copy_top_level, find_scenes

    src, out = Path(args.root), Path(args.out)
    if out.resolve() == src.resolve():
        raise ValueError("--out must be another root (the source is only read)")
    scenes = find_scenes(src, args.splits, _scene_filter(args))
    if not args.dry_run:
        _copy_top_level(src, out)
    failed = 0
    for split, scene in scenes:
        dst = out / "data" / split / scene.name
        if dst.exists() and not args.overwrite:
            print(f"  {split}/{scene.name}: exists, skipped (--overwrite)")
            continue
        if args.dry_run:
            print(f"  {split}/{scene.name}: would be packed")
            continue
        try:
            files = pack_scene(scene, dst, overwrite=args.overwrite)
        except Exception as err:  # collected and reported
            failed += 1
            print(f"  FAILED {split}/{scene.name}: {err}", file=sys.stderr)
            continue
        zips = [f for f in files if f.component != "metadata"]
        print(f"  {split}/{scene.name}: {len(zips)} archive(s), {sum(f.size for f in files) / 1e9:.2f} GB")
    if args.dry_run:
        print(f"{len(scenes)} scene(s) selected; nothing was written to {out}.")
        return 0
    print(f"{len(scenes) - failed} of {len(scenes)} scene(s) packed into {out}.")
    return 1 if failed else 0


def cmd_info(args: argparse.Namespace) -> int:
    import kitscenes
    from kitscenes._archive import COMPONENTS
    from kitscenes._storage import ZipStorage

    ds = kitscenes.open(args.root)
    print(ds)
    if (Path(args.root) / "aerial").is_dir():
        from kitscenes.aerial import AerialTiles

        try:
            print(f"aerial: {AerialTiles(args.root)}")
        except (OSError, ValueError) as err:
            print(f"aerial: {err}")
        from kitscenes.aerial import AerialFeatures, feature_set_ids

        for feature_set in feature_set_ids(args.root):
            try:
                print(f"aerial features: {AerialFeatures(args.root, feature_set)}")
            except (OSError, ValueError, KeyError) as err:
                print(f"aerial features {feature_set}: {err}")
    from collections import Counter

    variants: Counter = Counter()
    for scene in ds:
        for camera, variant in scene.files.image_variants.items():
            w, h = variant.image_size
            ow, oh = variant.original_size
            variants[(camera, f"{w}x{h} = {variant.recipe} of {ow}x{oh}")] += 1
    if variants:
        print("downscaled images (kitscenes transcode):")
        for (camera, what), n in sorted(variants.items()):
            print(f"  {camera}: {what} in {n} scene(s)")
    for split in ds.splits or [None]:
        sub = ds.filter(splits=split) if split else ds
        print(f"{split or '(no split)'}: {len(sub)} scene(s)")
        if args.verbose:
            for scene in sub:
                if isinstance(scene.files.storage, ZipStorage):
                    shipped = scene.files.storage.index.components
                    here = {c for c in shipped if any(scene.files.storage.is_dir(f) for f in COMPONENTS[c])}
                    missing = sorted(shipped - here)
                    print(f"  {scene.id}  archives, {len(here)}/{len(shipped)} components" + (f", missing {missing}" if missing else ""))
                else:
                    print(f"  {scene.id}  folder, {len(scene.sensors)} sensors")
    return 0


def cmd_splits_export(args: argparse.Namespace) -> int:
    from kitscenes.splits import SplitDefinition

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise ValueError(f"{out} exists (--overwrite replaces it)")
    definition = SplitDefinition.load(args.name)
    name = args.rename or (out.stem if definition.name == DEFAULT_SPLIT else definition.name)
    SplitDefinition(name, dict(definition.assignments)).to_csv(out)
    print(f"{out}: split definition {name} ({len(definition)} scenes, splits {', '.join(definition.splits)}; "
          f"id {definition.id})")
    return 0


def cmd_splits_check(args: argparse.Namespace) -> int:
    from kitscenes.splits import SplitDefinition

    try:
        definition = SplitDefinition.load(args.definition)  # every rule it breaks, at once
    except ValueError as err:
        print(f"kitscenes splits check: {err}", file=sys.stderr)
        return 2
    print(f"split definition {definition.name}: {len(definition)} scenes, id {definition.id}; the rules hold")
    for split in definition.splits:
        print(f"  {split}: {len(definition.scene_ids(split))} scene(s)")
    return 0


_TRUE, _FALSE = ("true", "yes", "1"), ("false", "no", "0")


def where_criterion(text: str) -> tuple[str, object]:
    """``--where COLUMN=VALUE`` -> ``(column, criterion)`` for :func:`kitscenes.find_scenes`.

    Categories take one value or several separated by commas (any of them),
    ``dark`` / ``sun_below_horizon`` true or false, numbers a value, several
    values, or a range ``LOW..HIGH`` with both ends included and either end
    left out for an open one (``max_speed_mps=..15``, ``road_motorway_share=0.5..``).
    """
    from kitscenes.splits import FILTER_COLUMNS

    column, sep, value = text.partition("=")
    column = column.strip().replace("-", "_")
    if not sep or not value:
        raise argparse.ArgumentTypeError(f"expected COLUMN=VALUE, got {text!r}")
    if column in FILTER_COLUMNS["category"]:
        return column, [v.strip() for v in value.split(",") if v.strip()]
    if column in FILTER_COLUMNS["bool"]:
        if value.lower() not in _TRUE + _FALSE:
            raise argparse.ArgumentTypeError(f"{column} is true or false, got {value!r}")
        return column, value.lower() in _TRUE
    if column in FILTER_COLUMNS["number"]:
        try:
            if ".." in value:
                lo, hi = value.split("..", 1)
                return column, (float(lo) if lo.strip() else None, float(hi) if hi.strip() else None)
            numbers = [float(v) for v in value.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError(f"{column} takes numbers or a range LOW..HIGH, got {value!r}") from None
        return column, numbers if len(numbers) > 1 else numbers[0]
    columns = [c for kind in FILTER_COLUMNS.values() for c in kind]
    raise argparse.ArgumentTypeError(f"unknown column {column!r}; columns: {', '.join(columns)}")


def _where(args: argparse.Namespace) -> Optional[dict]:
    out: dict = {}
    for column, criterion in args.where or []:
        if column in out:
            raise ValueError(f"--where {column} is given twice; list the values in one: {column}=a,b")
        out[column] = criterion
    return out or None


DOWNLOAD_HELP = """Download scenes from HuggingFace into ROOT; only what is missing is fetched.

    kitscenes download ROOT --splits val --components ring-cameras lidar-top
    kitscenes download ROOT --cities Karlsruhe --lidar hesai
    kitscenes download ROOT --where weather=rain,overcast --where max_speed_mps=..15
    kitscenes download ROOT --splits train --max-gb 50 --dry-run
    kitscenes download ROOT --update                  # move ROOT to the latest version

--components chooses cameras and lidars. Everything else always comes along:
metadata, maps, labels, radar and GNSS/INS.

KITScenes Aerial comes by area. The DOP20 tiles of the selected scenes' areas
come with every download (~2 GB for all). The DINOv3 feature tiles (~657 GB for
all) come with a full download, and with a filtered one (--splits, --scenes,
--cities, --lidar, --where, --max-gb) only with --aerial-features.
--components and --images do not filter scenes.

A root holds one version of the dataset: the first download pins the
HuggingFace revision in ROOT/dataset_info.json, later downloads use it, and
another --revision is refused unless --update moves the whole root.

The HuggingFace token comes from 'hf auth login' or $HF_TOKEN.
"""


TRANSCODE_HELP = """Downscale (re-encode) the camera images of a root, to save disk.

    kitscenes transcode ROOT --images half                      # in place, every camera
    kitscenes transcode ROOT --images quarter --cameras ring-cameras --splits train
    kitscenes transcode SRC --out DST --images half             # a new root; SRC is only read

With --out, DST gets the selected scenes in the layout of SRC (archives or
extracted folders); everything but the transcoded cameras is copied as it
is. 'kitscenes pack' and 'kitscenes extract' change the layout.

Your code does not change: scene.rig gives the cameras of the
stored images (K, image_size scaled), scene.files.image_variants[camera] what was
done; calib.json and the release's other files are not changed. A download
root remembers the recipe, so later downloads into it are transcoded too.
Only power-of-two reductions (1/2, 1/4, 1/8): exact pixel grids and
intrinsics. Reducing further from stored images works (half -> quarter);
larger needs the release's images (kitscenes download ROOT --images original).

Backends: gpu (NVIDIA DALI + nvImageCodec: pip install nvidia-dali-cuda120;
~180-450 images/s at 1/2 per GPU, depending on the camera mix) or cpu
(Pillow, one process per --jobs; ~19 ring images/s per core); auto takes
the GPU if it works. On a machine with many free cores, --backend cpu
--jobs 32 can be as fast.
"""


SPLITS_HELP = """Split definitions: which scene is in which split, without moving files.

    kitscenes splits export my_split.csv     # the default (v2.0.0_default_geosplit), to edit
    kitscenes splits check my_split.csv      # the rules; every problem is listed (exit 2)

The benchmark and all reported results use the default split, a geographic
split: no test road is in a training map, test scenes are at least 50 m
from train and val scenes, and train and val scenes are at least 40 m apart.
Make another one only with a clear scientific reason; results on it are not
comparable to the benchmark, and it has to take care of overlap between its
splits itself. See docs/splits.md.

A definition is a CSV with the header scene_id,split ('#' comments; '# name:
NAME' names it, else the file's stem). Every scene is a released scene,
listed once; the scenes of a split are of one kind: labelled, label-free, or
test-e2e.
"""


def build_parser() -> argparse.ArgumentParser:
    from kitscenes.transcode import DEFAULT_STAGING_GB

    parser = argparse.ArgumentParser(prog="kitscenes", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def selection(p: argparse.ArgumentParser) -> None:
        p.add_argument("root", metavar="ROOT", help="dataset root (holds data/<split>/<scene>/)")
        p.add_argument("--splits", nargs="+", metavar="SPLIT", help="split folders, e.g. train val test-e2e")
        p.add_argument("--scenes", nargs="+", metavar="ID", help="scene IDs")
        p.add_argument("--cities", nargs="+", metavar="CITY", help="scenes recorded in these cities: Karlsruhe, "
                       "Frankfurt, Sindelfingen")
        p.add_argument("--lidar", metavar="TYPE", help="scenes recorded with this lidar generation: velodyne or hesai")
        p.add_argument("--where", action="append", type=where_criterion, metavar="COLUMN=VALUE",
                       help="scenes by a column of the scene table, e.g. weather=rain,overcast, dark=true, "
                            "max_speed_mps=..15 (a range LOW..HIGH); repeat it and all must hold")

    def components(p: argparse.ArgumentParser, verb: str) -> None:
        p.add_argument("--components", nargs="+", metavar="NAME",
                       help=f"cameras and lidars to {verb}, by name or preset ({', '.join(archives.PRESETS)}); "
                            "default all. Everything else always comes along")

    def aerial_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--aerial-features", action="store_true",
                       help="the DINOv3 aerial features of the selected scenes' areas, also with a scene filter")
        p.add_argument("--no-aerial-features", action="store_true", help="no DINOv3 aerial features")
        p.add_argument("--no-aerial", action="store_true",
                       help="no KITScenes Aerial: neither the DOP20 tiles nor the DINOv3 features")

    backend_help = "where to transcode (default auto: the GPU if DALI and nvImageCodec work, else the CPU)"
    gpus_help = "CUDA devices for the GPU backend, two workers each (default 0)"

    d = sub.add_parser("download", help="download scenes from the hub (only what is missing)",
                       description=DOWNLOAD_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection(d)
    components(d, "get")
    aerial_options(d)
    d.add_argument("--jobs", type=int, default=8, help="files at a time (default 8)")
    d.add_argument("--repo-id", default=_DEFAULT_REPO_ID, help=f"HuggingFace dataset (default {_DEFAULT_REPO_ID})")
    d.add_argument("--max-gb", type=float, help="whole scenes, in order, up to this size in total (aerial included)")
    d.add_argument("--extract", action="store_true", help="unzip the components after downloading")
    d.add_argument("--verify", action="store_true",
                   help="check SHA-256 of downloaded and present files (slower; by default the size is checked)")
    d.add_argument("--revision", help="HF revision (branch, tag or commit) for a new root; a root keeps the one it has")
    d.add_argument("--update", action="store_true",
                   help="move the root to --revision (default: the latest); changed files are fetched again")
    d.add_argument("--dry-run", action="store_true", help="list what would be downloaded; change nothing")
    d.add_argument("--images", metavar="RECIPE",
                   help="downscale the cameras while downloading: half, quarter, eighth, full, or 'original' for "
                        "the release's images again (see 'kitscenes transcode --help')")
    d.add_argument("--staging-gb", type=float, default=DEFAULT_STAGING_GB,
                   help=f"with --images: disk for camera zips waiting to be transcoded (default {DEFAULT_STAGING_GB:g})")
    d.add_argument("--transcode-backend", choices=("auto", "gpu", "cpu"), default="auto",
                   help="with --images: " + backend_help)
    d.add_argument("--transcode-gpus", type=int, nargs="+", metavar="ID", help="with --images: " + gpus_help)
    d.set_defaults(func=cmd_download)

    e = sub.add_parser("extract", help="unzip downloaded components",
                       description="Unzip downloaded components (optional: the API reads the zips). KITScenes "
                                   "Aerial is chosen as in 'kitscenes download'.")
    selection(e)
    components(e, "extract")
    aerial_options(e)
    e.set_defaults(func=cmd_extract)

    t = sub.add_parser("transcode", help="downscale camera images (in place, or into a new root with --out)",
                       description=TRANSCODE_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection(t)
    t.add_argument("--images", required=True, metavar="RECIPE",
                   help="half (1/2, q90, 4:2:0), quarter, eighth, full (re-encode only), or "
                        "scale=1/4,quality=85,subsampling=4:4:4")
    t.add_argument("--cameras", nargs="+", metavar="NAME", help="cameras to transcode (names or presets such as "
                   "ring-cameras); default all")
    t.add_argument("--out", metavar="DST", help="write a new root (the source is only read); default: in place")
    t.add_argument("--backend", choices=("auto", "gpu", "cpu"), default="auto", help=backend_help)
    t.add_argument("--jobs", type=int, metavar="N", help="CPU worker processes (default: up to 8)")
    t.add_argument("--gpus", type=int, nargs="+", metavar="ID", help=gpus_help)
    t.add_argument("--dry-run", action="store_true", help="list what would be transcoded; change nothing")
    t.set_defaults(func=cmd_transcode)

    pk = sub.add_parser("pack", help="pack extracted scenes into the release archive format (into a new root)")
    selection(pk)
    pk.add_argument("--out", required=True, metavar="DST", help="the new root (the source is only read)")
    pk.add_argument("--overwrite", action="store_true", help="re-pack scenes that are in DST already")
    pk.add_argument("--dry-run", action="store_true", help="list what would be packed; write nothing")
    pk.set_defaults(func=cmd_pack)

    i = sub.add_parser("info", help="list splits, scenes and downloaded components")
    i.add_argument("root", metavar="ROOT")
    i.add_argument("-v", "--verbose", action="store_true")
    i.set_defaults(func=cmd_info)

    sp = sub.add_parser("splits", help="split definitions: export the default, check one",
                        description=SPLITS_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    spsub = sp.add_subparsers(dest="splits_command", required=True)
    se = spsub.add_parser("export", help="write a split definition as a CSV (scene_id,split) to edit")
    se.add_argument("name", nargs="?", default=DEFAULT_SPLIT, metavar="NAME",
                    help=f"the definition (its name or a CSV file; default {DEFAULT_SPLIT})")
    se.add_argument("out", metavar="OUT.csv")
    se.add_argument("--rename", metavar="NAME", help="the name written into the file (default: the file's stem for "
                    "the default definition, else its name)")
    se.add_argument("--overwrite", action="store_true", help="replace OUT.csv if it exists")
    se.set_defaults(func=cmd_splits_export)
    sc = spsub.add_parser("check", help="check the rules of a split definition (exit 2 if it breaks any)")
    sc.add_argument("definition", metavar="DEF", help="the definition (its name or a CSV file)")
    sc.set_defaults(func=cmd_splits_check)

    sub.add_parser("view", help="show a scene or a city of scenes in the 3D viewer (kitscenes-viewer)", add_help=False)
    sub.add_parser("aerial", help="aerial imagery: tile stores, coverage, crop caches (see --help)", add_help=False)
    sub.add_parser("video", help="an MP4 of the ring cameras with the map labels and a BEV panel, on the GPU "
                   "(see --help)", add_help=False)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["aerial"]:
        from kitscenes.aerial.__main__ import main as aerial_main

        return aerial_main(argv[1:])
    if argv[:1] == ["video"]:
        from kitscenes.visualization.gpu_video import main as video_main

        return video_main(argv[1:])
    if argv[:1] == ["view"]:
        try:
            from kitscenes_viewer.__main__ import main as view_main
        except ImportError:
            print("kitscenes view: needs the viewer: pip install -e viewer (in a clone of the repository)", file=sys.stderr)
            return 1
        return view_main(argv[1:]) or 0
    parser = build_parser()
    args = parser.parse_args(argv)
    conflict = _aerial_conflict(args)
    if conflict:
        parser.error(conflict)  # exit code 2
    from kitscenes.errors import KITScenesError

    try:
        return args.func(args)
    except (ValueError, OSError, KITScenesError) as err:  # bad selection, I/O and hub errors (HTTP errors are OSErrors)
        print(f"kitscenes {args.command}: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"kitscenes {args.command}: interrupted; run it again to continue", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
