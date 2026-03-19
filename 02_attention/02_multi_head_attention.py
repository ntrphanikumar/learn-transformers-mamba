"""
Module 2.2: Multi-Head Attention
=================================

WHY THIS MATTERS:
Single-head attention computes ONE set of attention weights. But language
has many simultaneous relationships:
  - Syntactic: "cat" is the subject of "sat"
  - Semantic: "mat" relates to "floor", "room", "sitting"
  - Positional: "the" before "cat" vs "the" before "mat"

Multi-head attention runs MULTIPLE attention functions in parallel,
each learning to capture different types of relationships.

Think of it like looking at a scene with multiple cameras at different
angles — each camera captures different details, and combining them
gives a fuller picture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention from "Attention Is All You Need".

    Architecture:
    1. Project input into num_heads separate Q, K, V spaces
    2. Run scaled dot-product attention on each head in parallel
    3. Concatenate all heads
    4. Project back to d_model dimensions

    Parameters:
        d_model: total dimension of the model (e.g., 512)
        num_heads: number of parallel attention heads (e.g., 8)
        → each head has dimension d_k = d_model / num_heads (e.g., 64)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Linear projections for Q, K, V, and output
        # These are the LEARNABLE parameters — the model learns what to
        # query for, what keys to expose, what values to return
        self.W_q = nn.Linear(d_model, d_model)  # projects to all heads at once
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # combines heads back

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len, d_model)
            key:   (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask:  optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # Step 1: Linear projections
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # Step 2: Reshape to separate heads
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        # -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: Scaled dot-product attention (per head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(weights, V)
        # attn_output: (batch, num_heads, seq_len, d_k)

        # Step 4: Concatenate heads
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Step 5: Final linear projection
        output = self.W_o(attn_output)

        return output, weights


# =============================================================================
# DEMO
# =============================================================================

print("Multi-Head Attention Demo")
print("=" * 50)

d_model = 64
num_heads = 8
d_k = d_model // num_heads
print(f"d_model={d_model}, num_heads={num_heads}, d_k per head={d_k}\n")

mha = MultiHeadAttention(d_model, num_heads)

# Input: batch of 2 sequences, each 10 tokens, 64-dimensional
x = torch.randn(2, 10, d_model)

# Self-attention: Q = K = V = x
output, weights = mha(x, x, x)

print(f"Input shape:   {x.shape}")
print(f"Output shape:  {output.shape}")
print(f"Weights shape: {weights.shape}")
print(f"  = (batch={weights.shape[0]}, heads={weights.shape[1]}, "
      f"query_len={weights.shape[2]}, key_len={weights.shape[3]})")

# Show that different heads learn different patterns
print(f"\nHead attention patterns for first sequence, word 0:")
for h in range(min(4, num_heads)):
    w = weights[0, h, 0].detach().numpy().round(3)
    print(f"  Head {h}: {w[:6]}...")
print("→ Each head produces a DIFFERENT attention pattern!")

# Parameter count
total_params = sum(p.numel() for p in mha.parameters())
print(f"\nTotal parameters: {total_params:,}")
print(f"  W_q: {d_model}×{d_model} = {d_model*d_model}")
print(f"  W_k: {d_model}×{d_model} = {d_model*d_model}")
print(f"  W_v: {d_model}×{d_model} = {d_model*d_model}")
print(f"  W_o: {d_model}×{d_model} = {d_model*d_model}")
print(f"  + 4 biases")


# =============================================================================
# WITH CAUSAL MASK (for GPT-style generation)
# =============================================================================

print(f"\n{'='*50}")
print("With Causal Mask (GPT-style)")
print(f"{'='*50}")

seq_len = 10
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
# Expand for batch and heads: (1, 1, seq_len, seq_len) broadcasts
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

output_causal, weights_causal = mha(x, x, x, mask=causal_mask)
print(f"Causal attention for word 3:")
print(f"  {weights_causal[0, 0, 3].detach().numpy().round(3)}")
print(f"  Positions 4-9 are zero (future is masked)")


# =============================================================================
# SELF-ATTENTION vs CROSS-ATTENTION
# =============================================================================

print(f"\n{'='*50}")
print("Self-Attention vs Cross-Attention")
print(f"{'='*50}")

print("""
Self-Attention (Q=K=V=same input):
  - Each word attends to other words in the SAME sequence
  - Used in: GPT decoder, BERT encoder

Cross-Attention (Q=one input, K=V=different input):
  - Words from sequence A attend to words in sequence B
  - Used in: machine translation (decoder attends to encoder)
  - Also how image captioning works (text attends to image features)
""")

# Cross-attention demo
encoder_output = torch.randn(2, 15, d_model)  # 15 source words
decoder_input = torch.randn(2, 10, d_model)   # 10 target words

# Q from decoder, K and V from encoder
cross_output, cross_weights = mha(decoder_input, encoder_output, encoder_output)
print(f"Cross-attention:")
print(f"  Decoder input: {decoder_input.shape}")
print(f"  Encoder output: {encoder_output.shape}")
print(f"  Result: {cross_output.shape}")
print(f"  Weights: {cross_weights.shape}")
print(f"  = Each of 10 target words attends to all 15 source words")
