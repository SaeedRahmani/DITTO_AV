"""Stacked GRU cell — used inside RSSM."""

import torch
import torch.nn as nn
from torch import Tensor


class GRUCellStack(nn.Module):
    """Stack of GRU cells for multi-layer recurrence.

    When num_layers > 1, the hidden state `h` should be (num_layers, B, hidden_size)
    with each layer maintaining its own hidden state. For num_layers == 1, `h` can
    be (B, hidden_size) for backward compatibility.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList(
            [nn.GRUCell(input_size if i == 0 else hidden_size, hidden_size)
             for i in range(num_layers)]
        )

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        if self.num_layers == 1:
            return self.cells[0](x, h)

        # h: (num_layers, B, hidden_size) — per-layer hidden states
        h_layers = h.unbind(0)  # tuple of (B, hidden_size)
        h_out = []
        for i, cell in enumerate(self.cells):
            x = cell(x, h_layers[i])
            h_out.append(x)
        return x
