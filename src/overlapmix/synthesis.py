"""End-to-end construction of an overlap-composed dataset."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm

from .compose import compose_grid
from .selection import select_teacher_patches

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _save_batch(images: torch.Tensor, root: Path, class_id: int) -> None:
    folder = root / f"{class_id:05d}"
    folder.mkdir(parents=True, exist_ok=True)
    mean = images.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = images.new_tensor(IMAGENET_STD)[None, :, None, None]
    images = (images * std + mean).clamp(0, 1).cpu()
    for index, image in enumerate(images):
        array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(array).save(folder / f"image_{index:05d}.jpg", quality=95)


def synthesize(
    teacher: torch.nn.Module,
    *,
    source: str,
    output: str,
    ipc: int,
    pool_per_class: int,
    crops_per_source: int,
    image_size: int,
    grid_size: int,
    overlap: float,
    blend: str,
    seed: int,
) -> None:
    required = ipc * grid_size**2
    if pool_per_class < required:
        raise ValueError("pool_per_class must be at least ipc * grid_size**2")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    dataset = datasets.ImageFolder(source)
    by_class: dict[int, list[str]] = {index: [] for index in range(len(dataset.classes))}
    for path, label in dataset.samples:
        by_class[label].append(path)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.mkdir(parents=True)
    crop = transforms.Compose([
        transforms.RandomResizedCrop(image_size // grid_size, ratio=(1, 1), antialias=True),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    teacher.eval()
    for class_id, paths in tqdm(by_class.items(), desc="classes"):
        if len(paths) < pool_per_class:
            raise RuntimeError(f"class {class_id} has {len(paths)} images; need {pool_per_class}")
        paths = rng.sample(paths, pool_per_class)
        candidates = torch.stack([
            torch.stack([crop(Image.open(path).convert("RGB")) for _ in range(crops_per_source)])
            for path in paths
        ])
        labels = torch.full((pool_per_class,), class_id, dtype=torch.long)
        patches = select_teacher_patches(candidates, labels, teacher, count=required, teacher_size=image_size)
        images = compose_grid(patches, images=ipc, canvas_size=image_size, grid_size=grid_size, overlap=overlap, blend=blend)
        _save_batch(images, destination, class_id)

