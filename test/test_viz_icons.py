"""The traffic element icons of kitscenes.visualization (no OpenCV or matplotlib needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import kitscenes
from kitscenes.visualization import style


def test_icons_ship_with_kitscenes_as_png(monkeypatch) -> None:
    """Every icon is in kitscenes itself (not the viewer), as a PNG."""
    monkeypatch.delenv("KITSCENES_TE_ICON_PATH", raising=False)
    assert style.icon_dir() == Path(kitscenes.__file__).parent / "visualization" / "icons"
    style.icon_rgba.cache_clear()
    for te_type, name in style.ICONS.items():
        assert name.endswith(".png") and style.icon_path(te_type).name == name, te_type
        icon = style.icon_rgba(te_type)
        assert icon.ndim == 3 and icon.shape[2] == 4 and icon.dtype == np.uint8
    tl = style.icon_rgba("TLBike")
    assert tl.shape[0] > tl.shape[1] and tl[..., 3].any()  # upright, not blank


def test_icon_dir_override(monkeypatch, tmp_path) -> None:
    from PIL import Image

    Image.new("RGBA", (4, 8), (255, 0, 0, 255)).save(tmp_path / "traffic_sign.png")
    monkeypatch.setenv("KITSCENES_TE_ICON_PATH", str(tmp_path))
    style.icon_rgba.cache_clear()
    try:
        assert style.icon_path("TSStop") == tmp_path / "traffic_sign.png"  # falls back to the generic sign
        assert style.icon_rgba("TSStop").shape == (8, 4, 4)
    finally:
        style.icon_rgba.cache_clear()
