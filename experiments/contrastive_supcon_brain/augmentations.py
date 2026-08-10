"""Stochastic augmentations for generating positive-pair views of scRNA expression vectors."""

from __future__ import annotations

import torch


class GeneMask:
    """Randomly zeroes out a fraction of genes (dropout-style augmentation)."""

    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask


class GaussianNoise:
    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class RandomScale:
    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + torch.randn(x.shape[0], 1, device=x.device) * self.std
        return x * scale


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


def default_augmentation(mask_p: float = 0.2, noise_std: float = 0.1, scale_std: float = 0.1) -> Compose:
    return Compose([GeneMask(mask_p), RandomScale(scale_std), GaussianNoise(noise_std)])
