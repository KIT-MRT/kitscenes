"""GPU (PyTorch) application of LiDAR motion compensation.

Optional: needs ``torch``; nothing else in kitscenes imports this module.

The expensive part of deskewing is applying one transform per point. The
transforms themselves are few (one per 0.02 ms time bin) and cheap to compute
on the CPU. So a data pipeline can build :class:`~kitscenes.motion.DeskewInputs`
in its workers (``Frame.lidar_deskew_inputs()``, no torch needed there), ship
the raw float32 points + int32 bin index + small transform table, and apply
everything on the GPU in one batched gather::

    from kitscenes.motion_torch import deskew_on_device

    inputs = [frame.lidar_deskew_inputs(n) for n in names]  # e.g. in DataLoader workers
    clouds = deskew_on_device(inputs, device="cuda")       # list of (N_i, 3) tensors, base frame

Results match :func:`kitscenes.motion.apply_deskew_inputs` (float32).
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

from kitscenes.motion import DeskewInputs

__all__ = ["collate_deskew_inputs", "deskew_on_device"]


def _apply(xyz: torch.Tensor, index: torch.Tensor, measured: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    rows = table.index_select(0, index)  # (N, 12)
    x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
    out = rows[:, 0:9:3] * x + rows[:, 1:9:3] * y + rows[:, 2:9:3] * z + rows[:, 9:12]
    return out * measured.unsqueeze(1)  # rows without a measurement stay at the origin


def _deskew_tensors(
    xyz: torch.Tensor, index: torch.Tensor, measured: torch.Tensor, table: torch.Tensor,
) -> torch.Tensor:
    """Apply the (M, 12) *table* rows ``[R (row-major), t]`` to the points, on the device of the tensors."""
    return _apply(xyz, index.long(), measured.to(xyz.dtype), table)


def collate_deskew_inputs(
    inputs: Sequence[DeskewInputs],
) -> dict[str, Union[torch.Tensor, list[int]]]:
    """Concatenate several sweeps into one batch of CPU tensors (e.g. a DataLoader ``collate_fn``).

    Table indices are offset per sweep, so the batch can be applied with one
    gather (:func:`deskew_on_device`). ``lengths`` splits the result again.
    """
    offsets = np.cumsum([0] + [len(i.table) for i in inputs[:-1]])
    return {
        "xyz": torch.from_numpy(np.concatenate([i.xyz for i in inputs])),
        "index": torch.from_numpy(np.concatenate([i.index + o for i, o in zip(inputs, offsets)]).astype(np.int32)),
        "measured": torch.from_numpy(np.concatenate([i.measured for i in inputs])),
        "table": torch.from_numpy(np.concatenate([i.table for i in inputs])),
        "lengths": [len(i.xyz) for i in inputs],
    }


def deskew_on_device(
    inputs: Union[Sequence[DeskewInputs], dict],
    device: Union[str, torch.device] = "cuda",
    *,
    non_blocking: bool = True,
) -> list[torch.Tensor]:
    """Deskew a batch of sweeps on ``device``; returns one (N_i, 3) float32 tensor per sweep.

    ``inputs`` is a sequence of :class:`DeskewInputs` or the dict from
    :func:`collate_deskew_inputs` (pin its tensors for faster transfers).
    """
    dev = torch.device(device)
    if isinstance(inputs, dict):
        batch = inputs
        out = _deskew_tensors(
            batch["xyz"].to(dev, non_blocking=non_blocking),
            batch["index"].to(dev, non_blocking=non_blocking),
            batch["measured"].to(dev, non_blocking=non_blocking),
            batch["table"].to(dev, non_blocking=non_blocking),
        )
        return list(torch.split(out, batch["lengths"]))

    # Transfer sweep by sweep and concatenate on the device: avoids a CPU-side copy.
    offset = 0
    xyz, index, measured, table = [], [], [], []
    for inp in inputs:
        xyz.append(torch.from_numpy(inp.xyz).to(dev, non_blocking=non_blocking))
        index.append(torch.from_numpy(inp.index).to(dev, non_blocking=non_blocking).long() + offset)
        measured.append(torch.from_numpy(inp.measured).to(dev, non_blocking=non_blocking))
        table.append(torch.from_numpy(inp.table).to(dev, non_blocking=non_blocking))
        offset += len(inp.table)
    out = _deskew_tensors(torch.cat(xyz), torch.cat(index), torch.cat(measured), torch.cat(table))
    return list(torch.split(out, [len(i.xyz) for i in inputs]))

