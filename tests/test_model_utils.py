import pytest
pytest.importorskip('matplotlib')
pytest.importorskip('torch')
import torch
from src.utils.model import generate_cos_pos_encoding


def test_generate_cos_pos_encoding_basic():
    enc = generate_cos_pos_encoding(2, 4, device='cpu')
    assert enc.shape == (2, 4)
    assert torch.allclose(enc[0, ::2], torch.zeros(2))
    assert torch.allclose(enc[0, 1::2], torch.ones(2))
