"""Split definitions: the default, CSV files, the rules, a dataset opened with one and the CLI."""

from __future__ import annotations

import hashlib
import logging
import pickle
import random
from pathlib import Path

import pytest

import kitscenes
from kitscenes.cli import main
from kitscenes.splits import DEFAULT_SPLIT, SPLITS, SplitDefinition, default_split, scene_table
from release_fixture import make_scene


def _ids(folder: str, n: int) -> list[str]:
    return [sid for sid, info in scene_table().items() if info.split == folder][:n]


TRAIN, VAL, TEST, E2E = _ids("train", 3), _ids("val", 2), _ids("test", 1), _ids("test-e2e", 1)
MAP_CHANGE_VAL = _ids("map_change_val", 1)


def test_default_definition() -> None:
    d = default_split()
    assert d.name == DEFAULT_SPLIT and len(d) == len(scene_table()) == 2079
    assert d.splits == tuple(SPLITS)  # every release folder has scenes, in the order of SPLITS
    assert d.assignments == {sid: info.split for sid, info in scene_table().items()}
    pairs = list(d.assignments.items())
    assert pairs == sorted(pairs, key=lambda p: (d.splits.index(p[1]), p[0]))  # by split, then ID
    text = "".join(f"{sid},{split}\n" for sid, split in sorted(pairs))
    assert d.id == hashlib.sha256(text.encode()).hexdigest()[:16] == "d863fd7751539f2f"  # content only
    assert SplitDefinition.load(DEFAULT_SPLIT) is d
    assert SplitDefinition.load(d) is d
    assert d.scene_ids("val") == sorted(sid for sid, i in scene_table().items() if i.split == "val")
    assert d.split_of(VAL[0]) == "val" and d.split_of("nope") is None and VAL[0] in d
    assert pickle.loads(pickle.dumps(d)) == d
    with pytest.raises(ValueError, match="no split.*'holdout'"):
        d.scene_ids("holdout")


def test_csv_round_trip(tmp_path: Path) -> None:
    mine = SplitDefinition("mine", {VAL[0]: "holdout", TRAIN[0]: "train", TRAIN[1]: "holdout",
                                                 MAP_CHANGE_VAL[0]: "train"})
    assert mine.splits == ("holdout", "train")  # first seen
    assert list(mine.assignments) == sorted([VAL[0], TRAIN[1]]) + sorted([TRAIN[0], MAP_CHANGE_VAL[0]])
    mine.to_csv(tmp_path / "out.csv")
    back = SplitDefinition.load(tmp_path / "out.csv")
    assert back == mine and back.name == "mine" and back.splits == mine.splits
    rows = [f"{sid},{split}" for sid, split in mine.assignments.items()]
    random.Random(0).shuffle(rows)
    (tmp_path / "other_name.csv").write_text("# a comment\n\nscene_id,split\n" + "\n".join(rows) + "\n# end\n")
    other = SplitDefinition.load(str(tmp_path / "other_name.csv"))
    assert other.name == "other_name" and other.id == mine.id and other != mine  # the id is the content's
    with pytest.raises(ValueError, match="No split definition"):
        SplitDefinition.load(tmp_path / "missing.csv")


def test_rules(tmp_path: Path) -> None:
    def problems(text: str) -> str:
        (tmp_path / "d.csv").write_text(text)
        with pytest.raises(ValueError) as err:
            SplitDefinition.load(tmp_path / "d.csv")
        return str(err.value)

    message = problems(f"scene_id,split\n{TRAIN[0]},a\nnot-a-scene,a\n{TEST[0]},a\n{E2E[0]},b c\n")
    assert "1 scene(s) that are not scenes of release v2.0.0: not-a-scene" in message
    assert "split a mixes kinds of scenes: 1 label-free scenes" in message and "1 labelled scenes" in message
    assert "split names that are not [A-Za-z0-9_.-]+: ['b c']" in message  # every problem at once
    message = problems(f"scene_id,split\n{TRAIN[0]},a\n{TRAIN[0]},b\nnot-a-scene,a\n{TEST[0]},a\n")
    assert "1 scene(s) listed more than once: " + TRAIN[0] in message and "not-a-scene" in message
    assert "split a mixes kinds of scenes" in message  # with the duplicate: every problem at once
    assert "header must be scene_id,split" in problems(f"id,split\n{TRAIN[0]},a\n")
    assert "with poses relative to the first pose" in problems(f"scene_id,split\n{TEST[0]},t\n{E2E[0]},t\n")
    many = _ids("train", 25)
    message = problems("scene_id,split\n" + "".join(f"{sid},a\n" for sid in many) + f"{TEST[0]},a\n")
    assert "and 5 more" in message  # at most 20 IDs shown
    with pytest.raises(ValueError, match="default split definition's; give yours another name"):
        SplitDefinition(DEFAULT_SPLIT, {TRAIN[0]: "train"})
    with pytest.raises(ValueError, match="no scenes"):
        SplitDefinition("empty", {})
    for folder in ("test", "train", "map_change_test-e2e"):  # release folder names are just names
        SplitDefinition("ok", {VAL[0]: folder})


def _root(tmp_path: Path, **kwargs) -> Path:
    for folder, ids in (("train", TRAIN), ("val", VAL), ("test", TEST)):
        for sid in ids:
            make_scene(tmp_path / "data" / folder / sid, n=3, cameras=(), lidars=(), radars=(),
                       maps=folder != "test", **kwargs)
    return tmp_path


def _mine() -> SplitDefinition:
    # val goes into train, one train scene and a scene not on disk into holdout; TRAIN[2] is left out
    mapping = {TRAIN[0]: "train", VAL[0]: "train", VAL[1]: "train", TRAIN[1]: "holdout", _ids("train", 4)[3]: "holdout",
               TEST[0]: "test"}
    return SplitDefinition("mine", mapping)


def test_dataset_with_a_definition(tmp_path: Path, caplog) -> None:
    root = _root(tmp_path)
    folders = kitscenes.open(root)
    same = kitscenes.open(root, split_definition=DEFAULT_SPLIT)
    assert same.split_definition is default_split() and folders.split_definition is None
    assert same.splits == folders.splits and {s: same.split_of(s) for s in same.scene_ids} == \
        {s: folders.split_of(s) for s in folders.scene_ids}

    with caplog.at_level(logging.INFO, logger="kitscenes.catalog"):
        ds = kitscenes.open(root, split_definition=_mine())
    assert "1 scene(s) under" in caplog.text and "not in split definition mine" in caplog.text
    assert ds.splits == ["train", "holdout", "test"] and TRAIN[2] not in ds and len(ds) == 5
    assert ds.scene_ids == [*sorted([TRAIN[0], *VAL]), TRAIN[1], TEST[0]]  # definition order
    scene = ds[VAL[0]]
    assert (scene.split, scene.release_split, scene.split_info.name, ds.split_of(VAL[0])) == ("train", "val", "val",
                                                                                              "train")
    assert scene.files.path == root / "data" / "val" / VAL[0]  # the files stay where the release puts them
    assert scene.pose_frame == "local_utm" and scene.track.split == "train"
    assert ds.filter(splits="holdout").scene_ids == [TRAIN[1]]
    assert kitscenes.open(root, "holdout", split_definition=_mine()).scene_ids == [TRAIN[1]]
    with pytest.raises(ValueError, match="no split.*'val'"):
        ds.filter(splits="val")
    with pytest.raises(kitscenes.DatasetNotFound, match=r"Splits found: \[.train.\] \(split definition x\)"):
        kitscenes.open(root, "holdout", split_definition=SplitDefinition("x", {TRAIN[2]: "train",
                                                                                            _ids("val", 3)[2]: "holdout"}))
    clone = pickle.loads(pickle.dumps(ds))
    assert clone.split_definition == ds.split_definition and clone[VAL[0]].split == "train"
    assert pickle.loads(pickle.dumps(scene)).release_split == "val"
    assert "split_definition='mine'" in repr(ds)

    csv_path = tmp_path / "mine.csv"
    _mine().to_csv(csv_path)
    assert kitscenes.open(root, split_definition=csv_path).splits == ds.splits


def test_scene_release_split(tmp_path: Path) -> None:
    path = make_scene(tmp_path / "data" / "val" / VAL[0], n=3, cameras=(), lidars=(), radars=())
    assert (kitscenes.Scene(path).split, kitscenes.Scene(path).release_split) == ("val", "val")
    scene = kitscenes.Scene(path, split="mine")
    assert (scene.split, scene.release_split, scene.split_info.name) == ("mine", "val", "val")
    assert kitscenes.Scene(path, split="train").release_split == "train"  # a scene opened alone: a folder name wins
    loose = make_scene(tmp_path / "loose" / TEST[0], n=3, cameras=(), lidars=(), radars=())
    assert kitscenes.Scene(loose).release_split == "test"  # from the scene table
    assert kitscenes.Scene(loose, split="test-e2e").split_info.name == "test-e2e"  # a release folder name


def test_cli_export_and_check(tmp_path: Path, capsys) -> None:
    out = tmp_path / "mine.csv"
    assert main(["splits", "export", str(out)]) == 0
    exported = SplitDefinition.load(out)
    assert exported.name == "mine" and exported.id == default_split().id  # renamed to the file's stem
    assert main(["splits", "export", str(out)]) == 1 and "exists" in capsys.readouterr().err
    assert main(["splits", "export", str(out), str(tmp_path / "again.csv"), "--rename", "kept"]) == 0
    assert SplitDefinition.load(tmp_path / "again.csv").name == "kept"

    capsys.readouterr()
    assert main(["splits", "check", str(out)]) == 0
    assert "the rules hold" in capsys.readouterr().out
    bad = tmp_path / "bad.csv"
    bad.write_text(f"scene_id,split\n{TRAIN[0]},a\n{TEST[0]},a\nnot-a-scene,b\n{VAL[0]},b\n{VAL[0]},c\n")
    assert main(["splits", "check", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "mixes kinds of scenes" in err and "not-a-scene" in err and "listed more than once" in err

    _mine().to_csv(tmp_path / "custom.csv")
    assert main(["splits", "check", str(tmp_path / "custom.csv")]) == 0
    text = capsys.readouterr().out
    assert f"split definition mine: 6 scenes, id {_mine().id}" in text
    assert "  train: 3 scene(s)" in text and "  holdout: 2 scene(s)" in text and "  test: 1 scene(s)" in text
    with pytest.raises(SystemExit):  # it takes the definition only
        main(["splits", "check", str(out), str(tmp_path)])
