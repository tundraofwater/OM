"""Small model factory with explicit checkpoint handling."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from torchvision import models


def build_model(name: str, classes: int, *, checkpoint: str | None = None, imagenet_weights: bool = False, small_input: bool = False) -> nn.Module:
    weights = "DEFAULT" if imagenet_weights and checkpoint is None else None
    model = models.get_model(name, weights=weights)
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, classes)
    else:
        raise ValueError(f"unsupported classifier layout for {name}")
    if small_input and hasattr(model, "conv1"):
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    if checkpoint:
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        state = state.get("model", state.get("state_dict", state))
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state)
    return model

