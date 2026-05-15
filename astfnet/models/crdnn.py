"""CRDNN backbone models for ASTF-net."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from astfnet.models.backbone import register_backbone
from astfnet.models.rnn import RNNFactory


class ConvBlock1D(nn.Module):
    """Apply a Conv1d, BatchNorm1d, ReLU, pooling, and dropout block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
        dropout: float,
    ) -> None:
        """Initialize a convolutional block."""
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the convolutional block."""
        return self.block(x)


@register_backbone("crdnn")
class CRDNN(nn.Module):
    """Convolutional recurrent DNN backbone for ASTF prediction.

    This follows the same high-level structure as SpeechBrain's CRDNN:
    convolutional feature extraction, recurrent sequence modeling, and a dense
    regression head. The implementation is adapted to ASTF-net's 1D waveform
    inputs, where target waveform and EGF are treated as input channels.

    Args:
        in_channels: Number of input waveform channels. Must be 2 because
            ``forward`` receives target waveform and EGF as separate inputs.
        output_length: Length of the predicted ASTF sequence.
        cnn_channels: Output channels for each convolutional block.
        cnn_kernel_size: Kernel size used by each convolutional block.
        cnn_pool_size: Pooling size used by each convolutional block.
        rnn_name: Recurrent layer type, either ``"GRU"`` or ``"LSTM"``.
        rnn_hidden_size: Number of hidden units per recurrent direction.
        rnn_layers: Number of recurrent layers.
        rnn_bidirectional: Whether to use bidirectional recurrence.
        dnn_hidden_size: Hidden size of dense regression blocks.
        dnn_layers: Number of dense blocks before the output layer.
        dropout: Dropout probability used in CNN, RNN, and DNN blocks.
        rnn_factory: Optional preconfigured RNN factory. When supplied, it
            takes precedence over the ``rnn_*`` arguments.
    """

    def __init__(
        self,
        in_channels: int = 2,
        output_length: int = 501,
        cnn_channels: Sequence[int] = (64, 128),
        cnn_kernel_size: int = 5,
        cnn_pool_size: int = 2,
        rnn_name: str = "GRU",
        rnn_hidden_size: int = 128,
        rnn_layers: int = 2,
        rnn_bidirectional: bool = True,
        dnn_hidden_size: int = 256,
        dnn_layers: int = 1,
        dropout: float = 0.1,
        rnn_factory: Optional[RNNFactory] = None,
    ) -> None:
        """Initialize the CRDNN backbone."""
        super().__init__()
        if in_channels != 2:
            raise ValueError(f"CRDNN expects exactly 2 input channels: target waveform and EGF; got {in_channels!r}.")
        if output_length <= 0:
            raise ValueError(f"output_length must be positive, got {output_length!r}.")
        if cnn_kernel_size <= 0:
            raise ValueError(f"cnn_kernel_size must be positive, got {cnn_kernel_size!r}.")
        if cnn_pool_size <= 0:
            raise ValueError(f"cnn_pool_size must be positive, got {cnn_pool_size!r}.")
        if dnn_layers < 0:
            raise ValueError(f"dnn_layers must be non-negative, got {dnn_layers!r}.")

        channels = list(cnn_channels)
        if not channels:
            raise ValueError("cnn_channels must contain at least one channel size.")
        if any(channel <= 0 for channel in channels):
            raise ValueError("cnn_channels must contain only positive channel sizes.")

        cnn_blocks = []
        prev_channels = in_channels
        for channels_out in channels:
            cnn_blocks.append(
                ConvBlock1D(
                    in_channels=prev_channels,
                    out_channels=channels_out,
                    kernel_size=cnn_kernel_size,
                    pool_size=cnn_pool_size,
                    dropout=dropout,
                )
            )
            prev_channels = channels_out
        self.cnn = nn.Sequential(*cnn_blocks)

        if rnn_factory is None:
            rnn_factory = RNNFactory(
                name=rnn_name,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_layers,
                bidirectional=rnn_bidirectional,
                dropout=dropout,
            )
        self.rnn_factory = rnn_factory
        self.rnn = self.rnn_factory.build(input_size=prev_channels)

        dnn_blocks = []
        input_features = self.rnn_factory.output_size
        for _ in range(dnn_layers):
            dnn_blocks.extend(
                [
                    nn.Linear(input_features, dnn_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_features = dnn_hidden_size

        dnn_blocks.extend([nn.Linear(input_features, output_length), nn.Softplus()])
        self.regressor = nn.Sequential(*dnn_blocks)

    def forward(self, target_waveform: torch.Tensor, egf: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            target_waveform: Target waveform with shape ``(batch_size, seq_len)``.
            egf: Empirical Green's function with shape ``(batch_size, seq_len)``.

        Returns:
            Predicted ASTF with shape ``(batch_size, output_length)``.
        """
        x = torch.stack([target_waveform, egf], dim=1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = x.mean(dim=1)
        return self.regressor(x)
