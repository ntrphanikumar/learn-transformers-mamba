"""
Module 2.1: Scaled Dot-Product Attention — The Core of Transformers
====================================================================

WHY THIS MATTERS:
Attention is THE breakthrough idea that makes Transformers work. It lets
each word "look at" every other word and decide what's relevant.

Before attention: models processed text sequentially (RNNs) and struggled
with long-range dependencies ("The cat that the dog chased ... was black").

With attention: every word can directly attend to every other word,
regardless of distance. This is why Transformers are so powerful at
understanding context.

THE CORE IDEA:
"Given a query, find the most relevant keys, and return their values."
  - Query: "What am I looking for?"
  - Key: "What do I contain?"
  - Value: "What information do I provide?"

Think of it like a search engine:
  - Query = your search terms
  - Key = the title/tags of each document
  - Value = the content of each document
  - Attention weights = relevance scores
"""

import torch
import torch.nn.functional as F
import math

# =============================================================================
# 1. DOT-PRODUCT ATTENTION: The simplest version
# =============================================================================

def simple_attention(query, key, value):
    """
    The most basic attention mechanism.

    Args:
        query: (seq_len, d_k) — what we're looking for
        key:   (seq_len, d_k) — what each position offers
        value: (seq_len, d_v) — the actual content at each position

    Returns:
        output: (seq_len, d_v) — weighted sum of values
        weights: (seq_len, seq_len) — attention pattern
    """
    # Step 1: Compute similarity between each query and each key
    # query @ key^T gives a (seq_len × seq_len) matrix of scores
    scores = query @ key.transpose(-2, -1)
    print(f"  Scores shape: {scores.shape} (each word scores against every other)")

    # Step 2: Normalize scores to probabilities (attention weights)
    weights = F.softmax(scores, dim=-1)
    print(f"  Weights shape: {weights.shape} (rows sum to 1)")

    # Step 3: Weighted sum of values
    output = weights @ value
    print(f"  Output shape: {output.shape}")

    return output, weights


# Demo with a 4-word sequence, 8-dimensional
seq_len, d_k = 4, 8
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_k)

print("Simple Dot-Product Attention:")
output, weights = simple_attention(Q, K, V)

print(f"\nAttention weights (who attends to whom):")
print(f"  Rows = query positions, Cols = key positions")
for i in range(seq_len):
    print(f"  Word {i}: {weights[i].detach().numpy().round(3)}")


# =============================================================================
# 2. SCALED DOT-PRODUCT ATTENTION (what Transformers actually use)
# =============================================================================

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    The attention mechanism from "Attention Is All You Need" (2017).

    The only difference from simple attention: we SCALE the scores by
    sqrt(d_k). Why? Without scaling, when d_k is large, dot products
    become large, pushing softmax into regions with tiny gradients.
    Scaling keeps the variance at 1, making training stable.

    Args:
        query: (..., seq_len, d_k)
        key:   (..., seq_len, d_k)
        value: (..., seq_len, d_v)
        mask:  optional mask to prevent attending to certain positions

    Returns:
        output: (..., seq_len, d_v)
        weights: (..., seq_len, seq_len)
    """
    d_k = query.shape[-1]

    # Score = Q·K^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Optional: apply mask (used in decoder to prevent looking ahead)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, value)

    return output, weights


# =============================================================================
# 3. WHY SCALING MATTERS
# =============================================================================

print(f"\n{'='*60}")
print("WHY SCALING MATTERS")
print(f"{'='*60}")

for d_k in [8, 64, 512]:
    q = torch.randn(4, d_k)
    k = torch.randn(4, d_k)

    raw_scores = q @ k.transpose(-2, -1)
    scaled_scores = raw_scores / math.sqrt(d_k)

    raw_weights = F.softmax(raw_scores, dim=-1)
    scaled_weights = F.softmax(scaled_scores, dim=-1)

    print(f"\nd_k={d_k:3d}:")
    print(f"  Raw score std:    {raw_scores.std():.2f}")
    print(f"  Scaled score std: {scaled_scores.std():.2f}")
    print(f"  Raw weights max:  {raw_weights.max():.4f} (peaky = bad gradients)")
    print(f"  Scaled weights max: {scaled_weights.max():.4f}")


# =============================================================================
# 4. CAUSAL MASK: Preventing the model from "cheating"
# =============================================================================

print(f"\n{'='*60}")
print("CAUSAL MASKING (for language generation)")
print(f"{'='*60}")

# In language generation, word at position t should only see words 0..t
# (can't peek at the future!). We achieve this with a causal mask.

seq_len = 5
# Lower-triangular mask: position i can attend to positions 0..i
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
print(f"\nCausal mask (1 = can attend, 0 = blocked):")
print(causal_mask.int())

Q = torch.randn(seq_len, 8)
K = torch.randn(seq_len, 8)
V = torch.randn(seq_len, 8)

output, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print(f"\nCausal attention weights:")
for i in range(seq_len):
    print(f"  Position {i}: {weights[i].detach().numpy().round(3)}")
print("→ Each position only attends to itself and earlier positions")
print("→ The upper triangle is all zeros (future is blocked)")


# =============================================================================
# 5. INTUITIVE EXAMPLE: What attention actually does
# =============================================================================

print(f"\n{'='*60}")
print("INTUITIVE EXAMPLE")
print(f"{'='*60}")

print("""
Sentence: "The cat sat on the mat"

When computing attention for "sat":
  Query("sat") asks: "Who did the action? What was involved?"
  Key("cat") responds: "I'm the subject/actor"
  Key("mat") responds: "I'm the location"

  Attention weights might be: [The=0.05, cat=0.60, sat=0.10, on=0.05, the=0.05, mat=0.15]

  → "sat" attends most to "cat" (who sat) and "mat" (where)
  → This is how the model builds understanding of relationships

In REAL models, this happens in PARALLEL for all words simultaneously.
That's why Transformers are fast — no sequential bottleneck like RNNs.
""")

# =============================================================================
# 6. EXERCISE
# =============================================================================

print("""
EXERCISES:
1. Make Q = K (self-attention). What pattern do you see in the weights?
   (Hint: each word attends most to itself — why?)

2. Remove the scaling (divide by 1 instead of sqrt(d_k)) with d_k=512.
   What happens to the attention weights?

3. Try a padding mask: set the last 2 positions as "padding" and mask
   them out. The model should ignore them completely.
""")
