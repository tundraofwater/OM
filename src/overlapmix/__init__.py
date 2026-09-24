"""Overlap-aware, teacher-guided patch dataset construction."""

from .compose import compose_grid
from .selection import select_teacher_patches

__all__ = ["compose_grid", "select_teacher_patches"]
__version__ = "0.1.0"

