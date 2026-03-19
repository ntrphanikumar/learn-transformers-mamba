"""
Module 1.3: Positional Encoding — How Models Know Word Order
=============================================================

WHY THIS MATTERS:
Unlike RNNs that process words one-by-one (and thus know order naturally),
Transformers process ALL words simultaneously. Without positional encoding,
"dog bites man" and "man bites dog" would look identical to the model!

Mamba handles this differently (via its sequential state), but understanding
positional encoding is essential for Transformers.

KEY CONCEPTS:
- Sinusoidal positional encoding (original Transformer)
- Learned positional encoding (GPT-2, BERT)
- Rotary positional encoding / RoPE (modern models like Llama)
"""

import torch
import torch.nn as nn
import math

# =============================================================================
# 1. THE PROBLEM: Transformers have no sense of order
# =============================================================================

print("THE PROBLEM:")
print("Embeddings for 'dog bites man' and 'man bites dog' are the same")
print("(just shuffled). The model needs position information.\n")

# =============================================================================
# 2. SINUSOIDAL POSITIONAL ENCODING (Original Transformer, 2017)
# =============================================================================
# Uses sine and cosine waves of different frequencies.
# Intuition: it's like giving each position a unique "fingerprint"
# made of waves at different frequencies — similar to how radio stations
# use different frequencies so you can tell them apart.


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    Creates position encodings using sine/cosine functions.

    Each position gets a d_model-dimensional vector where:
    - Even indices use sin(pos / 10000^(2i/d_model))
    - Odd indices use cos(pos / 10000^(2i/d_model))

    The 10000 base creates waves from very fast (index 0) to very slow
    (index d_model), so each position has a unique combination.
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

    # Create the division term: 10000^(2i/d_model)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)  # even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

    return pe


# Generate and inspect
pe = sinusoidal_positional_encoding(max_len=10, d_model=16)
print("Sinusoidal Positional Encoding:")
print(f"Shape: {pe.shape}  (10 positions × 16 dimensions)")
print(f"\nPosition 0: {pe[0][:8].numpy().round(3)}...")
print(f"Position 1: {pe[1][:8].numpy().round(3)}...")
print(f"Position 9: {pe[9][:8].numpy().round(3)}...")

# Key property: nearby positions have similar encodings
dist_01 = (pe[0] - pe[1]).norm()
dist_09 = (pe[0] - pe[9]).norm()
print(f"\nDistance pos 0 ↔ pos 1: {dist_01:.3f}")
print(f"Distance pos 0 ↔ pos 9: {dist_09:.3f}")
print("→ Nearby positions are closer in encoding space\n")


# =============================================================================
# 3. LEARNED POSITIONAL ENCODING (GPT-2, BERT)
# =============================================================================
# Instead of fixed math, just let the model LEARN a vector for each position.
# Simple, effective, but limited to a maximum sequence length.

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # A simple lookup table — position i → learned vector
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)


# Demo
batch_size, seq_len, d_model = 2, 5, 16
x = torch.randn(batch_size, seq_len, d_model)

pos_enc = LearnedPositionalEncoding(max_len=512, d_model=d_model)
x_with_pos = pos_enc(x)

print("Learned Positional Encoding:")
print(f"Input shape:  {x.shape}")
print(f"Output shape: {x_with_pos.shape}")
print(f"The position info is ADDED to the embeddings")
print(f"Parameters: {512 * d_model:,} (one vector per position)\n")


# =============================================================================
# 4. HOW POSITION + EMBEDDING COMBINE
# =============================================================================

print("FULL PICTURE: Token Embedding + Position Embedding")
print("=" * 50)

vocab_size = 100
d_model = 16
max_len = 50

# The two embedding layers
token_emb = nn.Embedding(vocab_size, d_model)
pos_emb = nn.Embedding(max_len, d_model)

# Input: a sequence of token IDs
token_ids = torch.tensor([[4, 12, 7, 42, 4]])  # Note: token 4 appears twice
print(f"Token IDs: {token_ids}")

# Get embeddings
tok_vectors = token_emb(token_ids)         # (1, 5, 16)
positions = torch.arange(5).unsqueeze(0)    # (1, 5)
pos_vectors = pos_emb(positions)            # (1, 5, 16)

# Combine by addition
combined = tok_vectors + pos_vectors        # (1, 5, 16)

print(f"\nToken embeddings shape: {tok_vectors.shape}")
print(f"Position embeddings shape: {pos_vectors.shape}")
print(f"Combined shape: {combined.shape}")

# The same token (4) at positions 0 and 4 now has DIFFERENT vectors
vec_pos0 = combined[0, 0]  # token 4 at position 0
vec_pos4 = combined[0, 4]  # token 4 at position 4
print(f"\nSame token '4' at position 0 vs position 4:")
print(f"  Distance: {(vec_pos0 - vec_pos4).norm():.3f}")
print(f"  → They're different! The model can distinguish them.\n")


# =============================================================================
# 5. SUMMARY
# =============================================================================

print("""
SUMMARY:
┌──────────────────────────────────────────────────────────┐
│ Method          │ Used By        │ Pros/Cons              │
├──────────────────────────────────────────────────────────┤
│ Sinusoidal      │ Original       │ No parameters,         │
│                 │ Transformer    │ generalizes to any len  │
│ Learned         │ GPT-2, BERT    │ More flexible,          │
│                 │                │ limited to max_len      │
│ RoPE            │ Llama, modern  │ Best of both worlds,    │
│                 │ models         │ good length extension   │
└──────────────────────────────────────────────────────────┘

In Mamba: positions are implicit because the model processes tokens
sequentially through its state — no explicit positional encoding needed!
""")
