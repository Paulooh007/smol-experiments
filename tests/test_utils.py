import torch

from src.utils import amp_enabled


def test_amp_enabled_only_for_cuda_device():
    assert amp_enabled(torch.device("cuda")) is True
    assert amp_enabled(torch.device("mps")) is False
    assert amp_enabled(torch.device("cpu")) is False
