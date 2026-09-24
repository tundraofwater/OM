"""Teacher-guided candidate crop selection.

Derived from and substantially modified relative to the Apache-2.0 RDED
implementation; see the repository NOTICE file.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _center_pad(images: torch.Tensor, size: int) -> torch.Tensor:
    dh, dw = size - images.shape[-2], size - images.shape[-1]
    if dh < 0 or dw < 0:
        raise ValueError("candidate patches cannot exceed teacher input size")
    return F.pad(images, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))


@torch.inference_mode()
def select_teacher_patches(
    candidates: torch.Tensor,
    labels: torch.Tensor,
    teacher: torch.nn.Module,
    *,
    count: int,
    teacher_size: int,
    forward_batch: int = 256,
) -> torch.Tensor:
    """Select patches by ground-truth-class probability.

    Candidates have shape ``[sources, crops, channels, height, width]``. The
    best crop from each source is retained before sources are ranked, avoiding
    one source image monopolizing the selected set.
    """
    if candidates.ndim != 5 or labels.shape != (candidates.shape[0],):
        raise ValueError("expected candidates [sources,crops,C,H,W] and one label per source")
    sources, crops = candidates.shape[:2]
    if not 1 <= count <= sources:
        raise ValueError("count must not exceed the number of source images")
    flat = candidates.transpose(0, 1).reshape(sources * crops, *candidates.shape[2:]).to(next(teacher.parameters()).device)
    repeated_labels = labels.repeat(crops).to(flat.device)
    logits = torch.cat([teacher(_center_pad(flat[i:i + forward_batch], teacher_size)) for i in range(0, len(flat), forward_batch)])
    losses = -F.log_softmax(logits, dim=1).gather(1, repeated_labels[:, None]).squeeze(1)
    losses = losses.reshape(crops, sources)
    crop_ids = losses.argmin(dim=0)
    source_ids = torch.arange(sources, device=flat.device)
    best_losses = losses[crop_ids, source_ids]
    ranked_sources = best_losses.argsort()[:count]
    shaped = flat.reshape(crops, sources, *flat.shape[1:])
    return shaped[crop_ids[ranked_sources], ranked_sources].detach()
