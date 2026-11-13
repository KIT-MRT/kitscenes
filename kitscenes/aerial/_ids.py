"""Deterministic cache/manifest ID helpers."""

from __future__ import annotations


def _format_resolution(resolution_m: float) -> str:
    """0.20 -> '020', 0.10 -> '010'."""
    return f"{round(resolution_m * 100):03d}"


def _format_roi_component(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _format_roi(roi_m: tuple[float, float]) -> str:
    longitudinal_m, lateral_m = roi_m
    return f"{_format_roi_component(longitudinal_m)}x{_format_roi_component(lateral_m)}"


def make_aerial_cache_id(
    provider: str,
    resolution_m: float,
    roi_m: tuple[float, float],
    orientation: str,
    version: str,
) -> str:
    """e.g. make_aerial_cache_id('dop20', 0.20, (60, 30), 'ego', 'v1')
    -> 'dop20_res020_roi60x30_ego_v1'.
    """
    return (
        f"{provider}_res{_format_resolution(resolution_m)}"
        f"_roi{_format_roi(roi_m)}_{orientation}_{version}"
    )
