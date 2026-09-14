import torch
from torch import nn
from overlapmix.selection import select_teacher_patches


class MeanTeacher(nn.Module):
    def __init__(self):
        super().__init__(); self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, images):
        score = images.mean((1, 2, 3))
        return torch.stack((-score, score), dim=1)


def test_uses_true_class_probability():
    candidates = torch.tensor([0.1, 0.9, 0.8, 0.2]).view(2, 2, 1, 1, 1).expand(2, 2, 1, 4, 4)
    labels = torch.tensor([1, 0])
    picked = select_teacher_patches(candidates, labels, MeanTeacher(), count=2, teacher_size=4)
    assert torch.isclose(picked[0].mean(), torch.tensor(0.9))
    assert torch.isclose(picked[1].mean(), torch.tensor(0.2))

