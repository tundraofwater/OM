"""Online teacher-label evaluation for a synthesized ImageFolder."""

from __future__ import annotations

import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .synthesis import IMAGENET_MEAN, IMAGENET_STD


def _cutmix(images: torch.Tensor, alpha: float) -> torch.Tensor:
    order = torch.randperm(len(images), device=images.device)
    lam = np.random.beta(alpha, alpha)
    ratio = math.sqrt(1 - lam)
    height, width = images.shape[-2:]
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    cy, cx = np.random.randint(height), np.random.randint(width)
    y1, y2 = max(0, cy - cut_h // 2), min(height, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(width, cx + cut_w // 2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[order, :, y1:y2, x1:x2]
    return mixed


def train_student(teacher: nn.Module, student: nn.Module, *, train_dir: str, val_dir: str, image_size: int, epochs: int, batch_size: int, temperature: float, lr: float, workers: int, seed: int) -> float:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = next(teacher.parameters()).device
    train_tf = transforms.Compose([transforms.RandomResizedCrop(image_size, scale=(0.25, 1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    val_tf = transforms.Compose([transforms.Resize(image_size * 8 // 7), transforms.CenterCrop(image_size), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    train_loader = DataLoader(datasets.ImageFolder(train_dir, transform=train_tf), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(datasets.ImageFolder(val_dir, transform=val_tf), batch_size=batch_size, num_workers=workers, pin_memory=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    teacher.eval(); student.train()
    for _ in range(epochs):
        for images, _ in train_loader:
            mixed = _cutmix(images.to(device), 1.0)
            with torch.no_grad():
                targets = F.softmax(teacher(mixed) / temperature, dim=1)
            prediction = F.log_softmax(student(mixed) / temperature, dim=1)
            loss = F.kl_div(prediction, targets, reduction="batchmean")
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
    student.eval(); correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            predicted = student(images.to(device)).argmax(1).cpu()
            correct += (predicted == labels).sum().item(); total += len(labels)
    return 100 * correct / total

