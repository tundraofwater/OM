import torch
from overlapmix.compose import compose_grid


def test_shape_and_convex_range():
    patches = torch.arange(4.0).view(4, 1, 1, 1).expand(4, 1, 8, 8)
    output = compose_grid(patches, images=1, canvas_size=32, grid_size=2, overlap=0.15, blend="cosine")
    assert output.shape == (1, 1, 32, 32)
    assert 0 <= output.min() <= output.max() <= 3


def test_average_blends_two_neighbors_equally():
    patches = torch.tensor([0.0, 2.0, 0.0, 2.0]).view(4, 1, 1, 1)
    output = compose_grid(patches, images=1, canvas_size=20, grid_size=2, overlap=0.5, blend="average")
    assert torch.isclose(output[0, 0, 10, 10], torch.tensor(1.0))


def test_rejects_wrong_patch_count():
    try:
        compose_grid(torch.zeros(3, 1, 2, 2), images=1, canvas_size=16, grid_size=2)
    except ValueError:
        return
    raise AssertionError("wrong patch count was accepted")

