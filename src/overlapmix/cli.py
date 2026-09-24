"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from .evaluate import train_student
from .models import build_model
from .synthesis import synthesize


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="overlapmix")
    commands = root.add_subparsers(dest="command", required=True)
    make = commands.add_parser("synthesize", help="select patches and compose a dataset")
    make.add_argument("--source", required=True); make.add_argument("--output", required=True)
    make.add_argument("--classes", type=int, required=True); make.add_argument("--teacher", default="resnet18")
    make.add_argument("--teacher-checkpoint"); make.add_argument("--imagenet-weights", action="store_true")
    make.add_argument("--ipc", type=int, default=10); make.add_argument("--pool-per-class", type=int, default=300)
    make.add_argument("--crops-per-source", type=int, default=5); make.add_argument("--image-size", type=int, default=224)
    make.add_argument("--grid-size", type=int, default=2); make.add_argument("--overlap", type=float, default=0.15)
    make.add_argument("--blend", choices=["hard", "average", "linear", "cosine"], default="cosine")
    make.add_argument("--small-input", action="store_true"); make.add_argument("--seed", type=int, default=42)
    make.add_argument("--device", default="auto")

    test = commands.add_parser("evaluate", help="train a student with online teacher labels")
    test.add_argument("--train-dir", required=True); test.add_argument("--val-dir", required=True)
    test.add_argument("--classes", type=int, required=True); test.add_argument("--teacher", default="resnet18")
    test.add_argument("--teacher-checkpoint"); test.add_argument("--imagenet-weights", action="store_true")
    test.add_argument("--student", default="resnet18"); test.add_argument("--small-input", action="store_true")
    test.add_argument("--image-size", type=int, default=224); test.add_argument("--epochs", type=int, default=300)
    test.add_argument("--batch-size", type=int, default=100); test.add_argument("--temperature", type=float, default=20)
    test.add_argument("--lr", type=float, default=1e-3); test.add_argument("--workers", type=int, default=4)
    test.add_argument("--seed", type=int, default=42); test.add_argument("--device", default="auto")
    test.add_argument("--result")
    return root


def main() -> None:
    args = parser().parse_args()
    device = _device(args.device)
    teacher = build_model(args.teacher, args.classes, checkpoint=args.teacher_checkpoint, imagenet_weights=args.imagenet_weights, small_input=args.small_input).to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if args.command == "synthesize":
        synthesize(teacher, source=args.source, output=args.output, ipc=args.ipc,
                   pool_per_class=args.pool_per_class, crops_per_source=args.crops_per_source,
                   image_size=args.image_size, grid_size=args.grid_size, overlap=args.overlap,
                   blend=args.blend, seed=args.seed)
        return
    student = build_model(args.student, args.classes, small_input=args.small_input).to(device)
    top1 = train_student(teacher, student, train_dir=args.train_dir, val_dir=args.val_dir,
                         image_size=args.image_size, epochs=args.epochs, batch_size=args.batch_size,
                         temperature=args.temperature, lr=args.lr, workers=args.workers, seed=args.seed)
    result = {"top1": top1, "epochs": args.epochs, "seed": args.seed}
    print(json.dumps(result, indent=2))
    if args.result:
        path = Path(args.result); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

