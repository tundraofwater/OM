"""Patch-grid geometry and normalized overlap blending.

Derived from and substantially modified relative to the Apache-2.0 RDED
implementation; see the repository NOTICE file.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


BLEND_MODES = {"hard", "average", "linear", "cosine"}


def _layout(canvas_size: int, grid_size: int, overlap: float) -> tuple[int, list[int]]:
    if canvas_size < 1 or not 1 <= grid_size <= canvas_size:
        raise ValueError("require canvas_size >= grid_size >= 1")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if grid_size == 1:
        return canvas_size, [0]
    width = math.ceil(canvas_size / (grid_size - (grid_size - 1) * overlap))
    starts = [round(i * (canvas_size - width) / (grid_size - 1)) for i in range(grid_size)]
    if len(set(starts)) != grid_size:
        raise ValueError("overlap is too large for distinct patch positions")
    return width, starts


def _axis_weight(index: int, width: int, starts: list[int], mode: str, ref: torch.Tensor) -> torch.Tensor:
    weight = ref.new_ones(width)
    if mode == "average":
        return weight
    for side, neighbor in (("left", index - 1), ("right", index + 1)):
        if not 0 <= neighbor < len(starts):
            continue
        count = width - abs(starts[index] - starts[neighbor])
        if count <= 0:
            continue
        if mode == "hard":
            split = (count + 1) // 2
            if side == "left":
                weight[:split] = 0
            elif count - split:
                weight[-(count - split):] = 0
            continue
        t = torch.arange(1, count + 1, device=ref.device, dtype=ref.dtype) / (count + 1)
        ramp = t if mode == "linear" else 0.5 - 0.5 * torch.cos(math.pi * t)
        if side == "left":
            weight[:count] *= ramp
        else:
            weight[-count:] *= ramp.flip(0)
    return weight


def compose_grid(
    patches: torch.Tensor,
    *,
    images: int,
    canvas_size: int,
    grid_size: int = 2,
    overlap: float = 0.0,
    blend: str = "cosine",
) -> torch.Tensor:
    """Compose slot-major NCHW patches into ``images`` square images.

    ``overlap`` is adjacent-overlap width divided by resized patch width.
    Patch order is slot-major: all output images for slot 0, then slot 1, etc.
    """
    if blend not in BLEND_MODES:
        raise ValueError(f"blend must be one of {sorted(BLEND_MODES)}")
    slots = grid_size**2
    if patches.ndim != 4 or patches.shape[0] != images * slots:
        raise ValueError("expected images * grid_size**2 NCHW patches")
    if not patches.is_floating_point():
        raise TypeError("patches must be floating point")

    width, starts = _layout(canvas_size, grid_size, overlap)
    output = patches.new_zeros(images, patches.shape[1], canvas_size, canvas_size)
    normalizer = patches.new_zeros(canvas_size, canvas_size)
    for row, y in enumerate(starts):
        for col, x in enumerate(starts):
            slot = row * grid_size + col
            patch = F.interpolate(patches[slot * images:(slot + 1) * images], (width, width), mode="nearest")
            wy = _axis_weight(row, width, starts, blend, patch)
            wx = _axis_weight(col, width, starts, blend, patch)
            weight = wy[:, None] * wx[None, :]
            output[..., y:y + width, x:x + width] += patch * weight
            normalizer[y:y + width, x:x + width] += weight
    return output / normalizer.clamp_min(torch.finfo(patches.dtype).tiny)
