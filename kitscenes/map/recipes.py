"""Label recipes: named settings that fix how map labels are made.

- ``"default"``: the labels of this API. The ego motion is IMU-refined where
  the scene allows it (``kitscenes.open(use_imu=True)``, the default), lines,
  crosswalk polygons and lane segments are cut exactly to the region in the
  label frame (pieces under 0.1 m dropped), with the lanelet2 build that
  ``kitscenes[map]`` pins (:data:`kitscenes.map.LANELET2_BUILD`,
  ``1.2.3+kitscenes4``: bicycle lane markings are ``BikeMarkingDashed`` /
  ``BikeMarkingSolid``); it accepts any build with the ML converter.
- ``"paper-v2.0.0"``: the labels of the v2.0.0 paper runs, made with the code
  of that time: the poses alone (``use_imu=False``), the lanelet2 converter's
  own cut at the region (``exact_cut=False``; lines can reach a few cm past
  the region on slopes), and lanelet2 ``1.2.3+kitscenes3`` exactly
  (:data:`kitscenes.map.LANELET2_PAPER_BUILD`).

:func:`check_label_recipe` checks a recipe against the installed lanelet2 and
a scene; :meth:`kitscenes.map.MLConverter.for_scene` and
``kitscenes.MapLabelSpec(recipe=...)`` apply it. See ``docs/maps.md``, Label recipes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from kitscenes.map import _lanelet2

if TYPE_CHECKING:
    from kitscenes.scene import Scene

#: Recipe name -> settings: ``use_imu`` (the ego motion the labels are made with), ``exact_cut``
#: (``MLConverter(exact_cut=...)``) and ``lanelet2`` (the build the recipe needs exactly; None: any with the ML
#: converter, normally the one ``kitscenes[map]`` pins, :data:`~kitscenes.map.LANELET2_BUILD`).
LABEL_RECIPES: dict[str, dict] = {
    "default": {"use_imu": True, "exact_cut": True, "lanelet2": None},
    "paper-v2.0.0": {"use_imu": False, "exact_cut": False, "lanelet2": _lanelet2.LANELET2_PAPER_BUILD},
}

#: The release page of the KITScenes lanelet2 builds (one tag per build, with its wheels).
LANELET2_RELEASES = "https://github.com/KITScenes/Lanelet2/releases"


def check_label_recipe(name: str, scene: Optional["Scene"] = None) -> dict:
    """The settings of label recipe *name* (:data:`LABEL_RECIPES`), checked against the installed lanelet2 and *scene*.

    Raises:
        ValueError: For an unknown recipe, or a scene whose ego motion the
            recipe does not allow (``"paper-v2.0.0"`` with IMU-refined ego
            motion, ``scene.ego_motion.source == "imu"``).
        MissingDependencyError: If the recipe needs another lanelet2 build than
            the installed one; the message names the release page with its wheels.
    """
    if name not in LABEL_RECIPES:
        raise ValueError(f"unknown label recipe {name!r}; one of {tuple(LABEL_RECIPES)}")
    settings = dict(LABEL_RECIPES[name])
    build = settings["lanelet2"]
    if build is not None:
        found = _lanelet2._installed_version()
        if found != build:
            what = "no lanelet2" if found is None else f"lanelet2 {found}"
            raise _lanelet2.MissingDependencyError(
                f"label recipe {name!r} needs lanelet2 {build} exactly (the build its labels were made with); "
                f"{what} is installed. Install a wheel of {LANELET2_RELEASES}/tag/{build.replace('+', '-')}")
    if scene is not None and not settings["use_imu"] and scene.use_imu and scene.ego_motion.source == "imu":
        raise ValueError(
            f"label recipe {name!r} is made with the poses alone, but scene {scene.id} has IMU-refined ego motion: "
            "open the dataset with kitscenes.open(..., use_imu=False) (or Scene(..., use_imu=False))")
    return settings
