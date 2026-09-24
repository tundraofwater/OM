#!/usr/bin/env bash
set -euo pipefail

overlapmix synthesize \
  --source "${TRAIN_DIR:?set TRAIN_DIR}" \
  --output "${OUTPUT_DIR:-outputs/imagenette_ipc10}" \
  --classes 10 \
  --teacher resnet18 \
  --teacher-checkpoint "${TEACHER_CHECKPOINT:?set TEACHER_CHECKPOINT}" \
  --ipc 10 \
  --pool-per-class 300 \
  --crops-per-source 5 \
  --image-size 224 \
  --grid-size 2 \
  --overlap 0.15 \
  --blend cosine \
  --seed 42 \
  --device cuda:0

