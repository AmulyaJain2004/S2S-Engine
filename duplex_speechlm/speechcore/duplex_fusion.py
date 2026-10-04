"""Duplex fusion: concatenate the incoming user-frame embedding with the
speech core's own previous-frame output embedding, then project down to the
speech core's hidden size with a single learned linear layer.

Per spec section 2, this is deliberately the simplest possible fusion
function -- a concrete first pass for Stage 2 to run and iterate from, not
a final design.

Training-time note: "the speech core's own previously generated frame
embedding" is replaced by the ground-truth agent-channel embedding, shifted
by one frame (standard teacher forcing). At inference time this would
instead be the model's own last output, fed back in.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DuplexFusion(nn.Module):
    def __init__(self, d_in: int, d_hidden: int) -> None:
        """d_in: the per-stream embedding width (projector output width,
        i.e. the speech core's hidden size -- user and prior-output frames
        are both already projected to this width before fusion).
        d_hidden: the speech core's hidden size (fusion output width)."""
        super().__init__()
        self.proj = nn.Linear(2 * d_in, d_hidden)
        self.start_token = nn.Parameter(torch.zeros(d_in))

    def forward(self, user_emb: torch.Tensor, prev_output_emb: torch.Tensor) -> torch.Tensor:
        """user_emb, prev_output_emb: (B, T, d_in), already time-aligned and
        already shifted by one frame by the caller (prev_output_emb[:, t] is
        the agent-channel embedding at t-1; the caller is responsible for
        that shift and for filling position 0 with self.start_token).
        Returns: (B, T, d_hidden) fused embeddings ready to feed into the
        speech core as inputs_embeds."""
        fused = torch.cat([user_emb, prev_output_emb], dim=-1)
        return self.proj(fused)

    def shift_with_start_token(self, agent_emb: torch.Tensor) -> torch.Tensor:
        """agent_emb: (B, T, d_in) ground-truth agent-channel embedding.
        Returns the teacher-forcing input: agent_emb shifted right by one
        frame, with position 0 filled by the learned start token."""
        b, t, d = agent_emb.shape
        start = self.start_token.view(1, 1, d).expand(b, 1, d).to(agent_emb.dtype)
        return torch.cat([start, agent_emb[:, :-1, :]], dim=1)
