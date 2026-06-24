"""SHARP CSI classification network."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader


class ConvActivation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding="same",
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SharpReductionSmall(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.pool_branch = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_branch = ConvActivation(in_channels, 5, kernel_size=2, stride=2, padding=0)
        self.stacked_branch = nn.Sequential(
            ConvActivation(in_channels, 3, kernel_size=1),
            ConvActivation(3, 6, kernel_size=2),
            ConvActivation(6, 9, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.pool_branch(x), self.conv_branch(x), self.stacked_branch(x)],
            dim=1,
        )


class SharpCsiNetwork(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            SharpReductionSmall(in_channels=1),
            ConvActivation(15, 3, kernel_size=1),
            nn.Dropout(p=0.2),
        )
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def has_uninitialized_parameters(model: nn.Module) -> bool:
    return any(isinstance(p, UninitializedParameter) for p in model.parameters())


@torch.no_grad()
def initialize_lazy_modules(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    """Run one forward pass to materialise LazyLinear before optimiser/AMP setup."""
    if not has_uninitialized_parameters(model):
        return
    if len(loader.dataset) == 0:
        raise RuntimeError("Cannot initialize lazy SHARP model with an empty training dataset.")
    was_training = model.training
    model.eval()
    features, _ = next(iter(loader))
    model(features.to(device))
    model.train(was_training)
    if has_uninitialized_parameters(model):
        raise RuntimeError("SHARP model still has uninitialized parameters after warm-up pass.")
