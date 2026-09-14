# OverlapMix-DD

OverlapMix-DD constructs compact image-classification datasets by selecting
teacher-confident local crops and composing them with normalized spatial
overlap. It also provides a small evaluation command that generates teacher
soft labels online after augmentation and CutMix.

The repository contains code only. Datasets, generated images, checkpoints,
logs, and experiment outputs are intentionally excluded.

## Method

For every class, the pipeline samples a candidate pool and creates several
random square crops per source image. The teacher scores each crop using the
probability assigned to the source image's ground-truth class. At most one crop
per source image enters the final ranking. The selected crops are resized and
placed on a fixed grid.

`--overlap` means adjacent overlap width divided by the resized patch width.
For example, with a 224-pixel canvas, a 2x2 grid and overlap 0.15, patches are
122 pixels wide and share a 20-pixel band after integer rounding. Available
blend modes are:

- `average`: equal contribution in covered regions;
- `linear`: complementary linear ramps;
- `cosine`: complementary cosine ramps;
- `hard`: enlarged geometry with a hard ownership boundary.

All modes normalize accumulated weights at every output pixel.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Input datasets use the standard `torchvision.datasets.ImageFolder` layout.

## Synthesize

```bash
overlapmix synthesize \
  --source /path/to/train \
  --output outputs/imagenette_ipc10 \
  --classes 10 \
  --teacher resnet18 \
  --teacher-checkpoint checkpoints/imagenette_resnet18.pth \
  --ipc 10 --pool-per-class 300 --crops-per-source 5 \
  --image-size 224 --grid-size 2 \
  --overlap 0.15 --blend cosine --device cuda:0
```

The output path must not already exist; this protects existing generated data
from accidental deletion.

For an ImageNet-1K torchvision teacher, replace the checkpoint argument with
`--imagenet-weights` and set `--classes 1000`.

## Evaluate

```bash
overlapmix evaluate \
  --train-dir outputs/imagenette_ipc10 \
  --val-dir /path/to/val \
  --classes 10 \
  --teacher resnet18 \
  --teacher-checkpoint checkpoints/imagenette_resnet18.pth \
  --student resnet18 --epochs 300 \
  --result outputs/evaluation.json --device cuda:0
```

The evaluator does not average stored source-image predictions. It applies the
teacher directly to the current augmented and CutMix-composed images, so soft
targets are regenerated online throughout student training.

## Tests

```bash
pip install -e . pytest
pytest -q
```

## Acknowledgements and license

This implementation builds on the patch-selection and efficient dataset
distillation ideas introduced by RDED. The code has been reorganized into a
package with a separate compositor, explicit CLI, overwrite protection, tests,
and online-evaluation API. See [NOTICE](NOTICE) for attribution and modification
details.

Distributed under the Apache License 2.0. If you use the method, please cite
the original RDED paper as well as any publication associated with your
modifications.

