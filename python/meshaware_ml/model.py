from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class PaperCNNConfig:
    input_channels: int = 3
    view_size: int = 100
    conv_width: int = 40
    conv_depth: int = 3
    kernel_size: int = 3
    pool_size: int = 2
    dropout: float = 0.25
    embedding_width: int = 128
    scalar_features: int = 2
    dense_width: int = 128
    dense_depth: int = 5

    def validate(self) -> None:
        integer_fields = (
            self.input_channels,
            self.view_size,
            self.conv_width,
            self.conv_depth,
            self.kernel_size,
            self.pool_size,
            self.embedding_width,
            self.scalar_features,
            self.dense_width,
            self.dense_depth,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("all model dimensions and depths must be positive")
        if self.conv_depth < 1:
            raise ValueError("conv_depth must include at least the padded layer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        unpadded_reduction = (
            self.conv_depth - 1
        ) * (self.kernel_size - 1)
        if self.view_size - unpadded_reduction < self.pool_size:
            raise ValueError("convolution stack collapses the input view")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaperRhoCNN(nn.Module):
    """Paper-selected CNN with mesh scale and theta conditioning."""

    def __init__(self, config: PaperCNNConfig | None = None):
        super().__init__()
        self.config = config or PaperCNNConfig()
        self.config.validate()

        convolutions: list[nn.Module] = []
        in_channels = self.config.input_channels
        for layer_index in range(self.config.conv_depth):
            padding = self.config.kernel_size // 2 if layer_index == 0 else 0
            convolutions.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        self.config.conv_width,
                        kernel_size=self.config.kernel_size,
                        padding=padding,
                    ),
                    nn.ReLU(),
                ]
            )
            in_channels = self.config.conv_width
        self.convolutions = nn.Sequential(*convolutions)
        self.pool = nn.MaxPool2d(self.config.pool_size)
        self.dropout = nn.Dropout(self.config.dropout)

        flattened = self._flattened_size()
        self.view_embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, self.config.embedding_width),
            nn.ReLU(),
        )

        dense_layers: list[nn.Module] = []
        dense_input = (
            self.config.embedding_width + self.config.scalar_features
        )
        for _ in range(self.config.dense_depth):
            dense_layers.extend(
                [
                    nn.Linear(dense_input, self.config.dense_width),
                    nn.ReLU(),
                ]
            )
            dense_input = self.config.dense_width
        self.conditioned_dense = nn.Sequential(*dense_layers)
        self.output = nn.Linear(dense_input, 1)

    def _flattened_size(self) -> int:
        size = self.config.view_size
        size -= (self.config.conv_depth - 1) * (
            self.config.kernel_size - 1
        )
        size //= self.config.pool_size
        return self.config.conv_width * size * size

    def forward(
        self,
        view: torch.Tensor,
        scalars: torch.Tensor,
        view_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if view.ndim != 4:
            raise ValueError("view must have shape [batch, channels, H, W]")
        if scalars.ndim != 2:
            raise ValueError("scalars must have shape [batch, features]")
        if view_indices is None and view.shape[0] != scalars.shape[0]:
            raise ValueError("view and scalar batch sizes differ")
        if view.shape[1:] != (
            self.config.input_channels,
            self.config.view_size,
            self.config.view_size,
        ):
            raise ValueError(
                "unexpected view shape: "
                f"{tuple(view.shape[1:])}"
            )
        if scalars.shape[1] != self.config.scalar_features:
            raise ValueError(
                f"expected {self.config.scalar_features} scalar features"
            )
        features = self.dropout(self.pool(self.convolutions(view)))
        embedding = self.view_embedding(features)
        if view_indices is not None:
            if view_indices.ndim != 1 or len(view_indices) != len(scalars):
                raise ValueError(
                    "view_indices must map every scalar row to one view"
                )
            if view_indices.dtype != torch.int64:
                raise ValueError("view_indices must use int64")
            if len(view_indices) and (
                int(view_indices.min()) < 0
                or int(view_indices.max()) >= len(embedding)
            ):
                raise ValueError("view_indices contains an invalid view index")
            embedding = embedding.index_select(0, view_indices)
        conditioned = torch.cat((embedding, scalars), dim=1)
        hidden = self.conditioned_dense(conditioned)
        return self.output(hidden).squeeze(1)
