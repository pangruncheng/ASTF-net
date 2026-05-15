"""RNN factory utilities for ASTF-net models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

import torch.nn as nn

RNN_REGISTRY: Dict[str, Type[nn.RNNBase]] = {
    "GRU": nn.GRU,
    "LSTM": nn.LSTM,
}


@dataclass
class RNNFactory:
    """Factory for constructing recurrent layers.

    Args:
        name: Recurrent layer type. Supported values are ``"GRU"`` and
            ``"LSTM"``.
        hidden_size: Number of hidden units per recurrent direction.
        num_layers: Number of stacked recurrent layers.
        bidirectional: Whether to use bidirectional recurrence.
        dropout: Dropout between recurrent layers. PyTorch only applies this
            when ``num_layers > 1``.
    """

    name: str = "GRU"
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate factory settings."""
        self.name = self.name.upper()
        if self.name not in RNN_REGISTRY:
            raise ValueError(f"Unknown RNN {self.name!r}. Supported: {list(RNN_REGISTRY)}.")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size!r}.")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers!r}.")
        if self.dropout < 0:
            raise ValueError(f"dropout must be non-negative, got {self.dropout!r}.")

    @property
    def output_size(self) -> int:
        """Return the feature size produced by the recurrent layer."""
        directions = 2 if self.bidirectional else 1
        return self.hidden_size * directions

    def build(self, input_size: int) -> nn.RNNBase:
        """Instantiate the configured recurrent layer.

        Args:
            input_size: Number of input features per time step.

        Returns:
            A configured PyTorch recurrent layer.
        """
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size!r}.")

        rnn_dropout = self.dropout if self.num_layers > 1 else 0.0
        return RNN_REGISTRY[self.name](
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=rnn_dropout,
        )
