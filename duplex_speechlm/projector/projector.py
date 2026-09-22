"""Projector: WavLM features -> Conv1D -> MLP -> RMSNorm -> Qwen-compatible width.

This is simple enough to implement for real now rather than stub it -- it has
no dependency on Qwen actually being trained yet. It is untrained (random
init) until Stage 1; this file only defines the architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(dtype)


class Projector(nn.Module):
    def __init__(self, d_in: int = 768, d_out: int = 1536, conv_kernel: int = 3, mlp_mult: float = 4.0) -> None:
        """d_in: WavLM's hidden size (768 for wavlm-base-plus).
        d_out: Qwen2.5-1.5B-Instruct's hidden size (1536) -- confirm against the
        actual loaded model's config.hidden_size before training, don't assume."""
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_in, kernel_size=conv_kernel, padding=conv_kernel // 2)
        d_hidden = int(d_out * mlp_mult)
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )
        self.norm = RMSNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in) WavLM features -> (B, T, d_out)."""
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.mlp(h)
        return self.norm(h)
