"""Duplex fusion: projects the incoming user-frame embedding to the speech
core's hidden size.

CORRECTED DESIGN (previous version concatenated the user-frame embedding
with the real ground-truth agent audio's embedding, shifted by one 20ms
frame, per an early reading of spec section 2). That created a training
shortcut: adjacent 20ms frames of real speech are highly autocorrelated,
so the model could minimize training loss by mostly echoing the
ground-truth "previous frame" forward, instead of actually learning to
generate agent content from the user audio + the speech core's own
understanding. It also meant training and inference used DIFFERENT
mechanisms: training always had real ground-truth audio to lean on,
while real inference never does (there is no ground-truth agent audio at
deployment time), forcing a fragile synthesize-then-re-encode self-feedback
loop at inference that kept producing noise no amount of signal-processing
patching (clamping, loudness-matching, crossfading) fixed -- because the
problem was never signal corruption, it was that the model had never
learned genuine content generation in the first place.

This version drops the explicit "previous output" input entirely. The
speech core (Qwen, causal self-attention) already carries forward its own
history through its own hidden states at every earlier frame position --
that is what an autoregressive transformer's hidden state IS -- so an
explicit re-injection of (re-encoded) past audio is both unnecessary and,
per the above, actively harmful. Training and inference now use the
IDENTICAL mechanism: only the user stream is ever fed in as external
input; whatever "memory of what I've been saying" the model needs comes
from its own hidden states via self-attention / the KV cache, exactly the
way any autoregressive LM conditions on its own past outputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DuplexFusion(nn.Module):
    def __init__(self, d_in: int, d_hidden: int) -> None:
        """d_in: the projector's output width (already the speech core's
        hidden size). d_hidden: the speech core's hidden size. Kept as a
        separate learned layer (rather than folding into the projector)
        so this adaptation can specialize for the speech core's input
        distribution independently of the projector's acoustic-feature role."""
        super().__init__()
        self.proj = nn.Linear(d_in, d_hidden)

    def forward(self, user_emb: torch.Tensor) -> torch.Tensor:
        """user_emb: (B, T, d_in) -> (B, T, d_hidden), ready to feed into
        the speech core as inputs_embeds."""
        return self.proj(user_emb)
