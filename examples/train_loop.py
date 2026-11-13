"""A minimal PyTorch training loop on KITScenes: front camera -> the next 3 s of the ego trajectory.

Shows the training path of the API end to end: a ``SampleSpec`` (what a sample holds), ``SampleDataset``
(frames as dicts of arrays, safe in DataLoader workers), ``kitscenes.collate`` (batches), and how to turn a
batch into model inputs and a masked loss. The model is a toy; replace it with yours.

Run (needs ``kitscenes[torch]``, see "Install options" in the README; torch for your CUDA from pytorch.org if needed)::

    KITSCENES_ROOT=/data/kitscenes python examples/train_loop.py --split train --steps 50
    python examples/train_loop.py --root /data/kitscenes --split val --steps 5 --workers 2 --device cpu
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

import kitscenes

HORIZON_S = 3.0  # predict this far ahead, a pose every 0.1 s
SLOTS = round(HORIZON_S / 0.1)


class TinyPlanner(nn.Module):
    """Image (B, 3, H, W) + current speed (B, 1) -> future x, y in the base frame (B, SLOTS, 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=4), nn.ReLU(), nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(32 + 1, 64), nn.ReLU(), nn.Linear(64, SLOTS * 2))

    def forward(self, image: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.encoder(image), speed], 1)).view(-1, SLOTS, 2)


def to_inputs(batch: dict, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A collated batch -> (image, speed, target xy, valid mask) on *device*."""
    image = batch["camera"]["camera_ring_front"]["image"].to(device).float() / 255  # (B, 3, H, W)
    past = batch["past"]["xyz_ego"].to(device).float()  # (B, 1, 3): the pose 0.1 s ago, in the current base frame
    speed = past[:, -1, :1].abs() / 0.1  # rough speed from the last 0.1 s (m/s)
    target = batch["future"]["xyz_ego"][..., :2].to(device).float()  # (B, SLOTS, 2)
    valid = batch["future"]["valid"].to(device)  # slots past the end of the scene's poses are not valid
    return image, speed, target, valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", help="dataset root (default: $KITSCENES_ROOT)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ds = kitscenes.open(args.root, splits=args.split, scene_cache=256)  # shuffling over many scenes: keep them open
    spec = kitscenes.SampleSpec(cameras=("camera_ring_front",), reduce=8, future_s=HORIZON_S, past_s=0.1)
    samples = kitscenes.SampleDataset(ds, spec)  # only frames with the front camera
    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=kitscenes.collate, drop_last=True, persistent_workers=args.workers > 0)
    print(f"{samples}\n{len(loader)} batches per epoch on {args.device}")

    model = TinyPlanner().to(args.device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    step, start = 0, time.perf_counter()
    while step < args.steps:
        for batch in loader:
            image, speed, target, valid = to_inputs(batch, args.device)
            error = (model(image, speed) - target).abs().sum(-1)  # L1 per slot
            loss = (error * valid).sum() / valid.sum().clamp(min=1)  # only the slots the poses cover
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            step += 1
            if step % 5 == 0 or step == 1:
                print(f"step {step:4d}  loss {loss.item():7.3f} m  ({(time.perf_counter() - start) / step:.2f} s/step)")
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
