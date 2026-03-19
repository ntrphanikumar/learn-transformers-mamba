"""
Module 1.3: Positional Encoding — How Models Know Word Order
=============================================================

WHY THIS MATTERS:
Unlike RNNs that process words one-by-one (and thus know order naturally),
Transformers process ALL words simultaneously. Without positional encoding,
"dog bites man" and "man bites dog" would look identical to the model!

WHEN IS THIS USED:
- In EVERY Transformer model (GPT, BERT, Llama, T5, etc.)
- Right after the embedding layer, before the first attention layer
- NOT needed in Mamba (position is implicit from sequential processing)

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
# WHAT: Transformers process all tokens in PARALLEL — they see the whole
#       sentence at once, like looking at all words on a page simultaneously.
# WHEN: This becomes a problem the moment you use attention — the attention
#       mechanism treats input as a SET (unordered), not a SEQUENCE (ordered).
# WHY IT'S BAD: Without position info, the model literally cannot tell
#       "dog bites man" from "man bites dog".

print("THE PROBLEM: Order matters, but Transformers don't know order")
print("=" * 60)

# ── REAL-WORLD: Why order matters ────────────────────────────────────────
print("""
Consider these sentences — same words, different meanings:
  1. "The doctor treated the patient"    → doctor is the healer
  2. "The patient treated the doctor"    → patient is the healer

  3. "Only I love you"                   → nobody else loves you
  4. "I only love you"                   → I love nobody else
  5. "I love only you"                   → same as 4, emphasis on "you"

A model without position info would see the SAME bag of words for all of
sentences 3, 4, 5. It needs to know WHERE each word is to understand meaning.

RNNs solved this naturally — they process words left-to-right, so position
is built in. But Transformers see everything at once (which is faster) and
must ADD position information explicitly.
""")


# =============================================================================
# 2. SINUSOIDAL POSITIONAL ENCODING (Original Transformer, 2017)
# =============================================================================
# WHAT: Uses sine and cosine waves at different frequencies to create a
#       unique "fingerprint" for each position.
# WHEN: Used in the original "Attention Is All You Need" paper. Not commonly
#       used in modern models, but important to understand the concept.
# WHY SINE/COSINE: Two key properties:
#   1. Each position gets a unique vector (can tell positions apart)
#   2. Relative positions are encoded: PE(pos+k) can be expressed as a
#      linear function of PE(pos), so the model can learn "3 words apart"

print("SINUSOIDAL POSITIONAL ENCODING")
print("=" * 60)

# ── REAL-WORLD ANALOGY: Clock encoding ───────────────────────────────────
print("""
ANALOGY — Think of how a clock encodes time:
  - The second hand spins fast (high frequency)
  - The minute hand spins slower (medium frequency)
  - The hour hand spins slowest (low frequency)

  Time 3:15:30 has a unique combination of hand positions.
  Time 3:15:31 is very SIMILAR (only seconds changed).
  Time 9:45:00 is very DIFFERENT (all hands moved).

Sinusoidal encoding does the same thing with sine waves:
  - Some dimensions oscillate fast → distinguish nearby positions
  - Some dimensions oscillate slow → distinguish distant positions
  - The combination is unique for each position.
""")


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
# WHAT: Instead of fixed math, let the model LEARN a vector for each position.
# WHEN: GPT-2, BERT, and most 2018-2022 era models use this.
# WHY: Simpler, more flexible — the model figures out the best position
#      representation on its own. Downside: limited to a max sequence length.

print("LEARNED POSITIONAL ENCODING")
print("=" * 60)

# ── REAL-WORLD ANALOGY: Assigned seating ─────────────────────────────────
print("""
ANALOGY — Sinusoidal vs Learned, like addressing systems:

  Sinusoidal = GPS coordinates
    - Mathematical formula gives every location a unique address
    - Works for ANY location, even ones you've never visited
    - But the coordinates don't "mean" anything to humans

  Learned = Named street addresses
    - "221B Baker Street" is a learned label for a specific location
    - More meaningful (the model learns what each position "means")
    - But only works for addresses that exist (max_len limit)
    - GPT-2 learns 1024 position vectors → can't handle position 1025
""")


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
# 4. HOW POSITION + EMBEDDING COMBINE — The full first layer
# =============================================================================
# WHAT: The input to a Transformer is: token_embedding + position_embedding
# WHEN: This is literally the first computation in GPT, BERT, etc.
# WHY:  The model needs to know BOTH what the token is AND where it is.

print("FULL PICTURE: Token Embedding + Position Embedding")
print("=" * 60)

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
print(f"  → They're different! The model can distinguish them.")

# ── REAL-WORLD: What this looks like in GPT ──────────────────────────────
print(f"""
REAL-WORLD: First layer of GPT-2 (simplified pseudocode):

  def gpt2_embed(token_ids):
      # token_ids: [128, 256, 42, 7891, ...]  (from BPE tokenizer)

      tok = token_embedding_table[token_ids]    # (seq_len, 768)
      pos = position_embedding_table[0, 1, 2, ...seq_len]  # (seq_len, 768)

      return tok + pos   # (seq_len, 768) → this goes into the first attention layer

  That's it. The ENTIRE input processing is just two table lookups + addition.
  Everything after this is transformer blocks (attention + FFN).
""")


# =============================================================================
# 5. SUMMARY
# =============================================================================

print("SUMMARY")
print("=" * 60)
print("""
┌──────────────────────────────────────────────────────────────────┐
│ Method          │ Used By          │ Pros             │ Cons      │
├──────────────────────────────────────────────────────────────────┤
│ Sinusoidal      │ Original         │ No parameters,   │ Less      │
│                 │ Transformer      │ any length works  │ flexible  │
│                 │                  │                   │           │
│ Learned         │ GPT-2, BERT      │ More flexible,    │ Fixed max │
│                 │                  │ model optimizes   │ length    │
│                 │                  │                   │           │
│ RoPE            │ Llama, Mistral,  │ Best of both:     │ More      │
│                 │ modern models    │ flexible + can    │ complex   │
│                 │                  │ extend to new     │ math      │
│                 │                  │ lengths           │           │
└──────────────────────────────────────────────────────────────────┘

In Mamba: positions are implicit because the model processes tokens
sequentially through its state — no explicit positional encoding needed!
""")


# =============================================================================
# 6. EXERCISES
# =============================================================================

print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. POSITION SIMILARITY:
   Compute the distance between all pairs of positions (0-9) in the
   sinusoidal encoding. Do positions 3 and 4 have a similar distance
   as positions 7 and 8? Why does this matter?

2. MAX LENGTH PROBLEM:
   GPT-2 was trained with max_len=1024. What happens if you give it
   a sequence of 1025 tokens? What would you do to fix this?

3. ADDITION vs CONCATENATION:
   We ADD position embeddings to token embeddings. Why not concatenate
   them (making vectors twice as long)? What are the tradeoffs?

4. NO POSITION ENCODING:
   Remove positional encoding entirely. For which tasks would the model
   still work fine? For which would it fail completely?
""")


# =============================================================================
# 7. ANSWERS
# =============================================================================

print("\n\n")
print("=" * 60)
print("ANSWERS (try the exercises first!)")
print("=" * 60)

# ── ANSWER 1: Position Similarity ─────────────────────────────────────────
print("\n--- Answer 1: Position Similarity ---")

pe = sinusoidal_positional_encoding(max_len=10, d_model=16)

dist_34 = (pe[3] - pe[4]).norm()
dist_78 = (pe[7] - pe[8]).norm()
dist_05 = (pe[0] - pe[5]).norm()
print(f"Distance between adjacent positions:")
print(f"  pos 3 ↔ pos 4: {dist_34:.4f}")
print(f"  pos 7 ↔ pos 8: {dist_78:.4f}")
print(f"  pos 0 ↔ pos 5: {dist_05:.4f}")
print(f"""
→ Adjacent positions (3↔4 and 7↔8) have SIMILAR distances!
  This is a key property: the encoding captures RELATIVE position.
  "3 steps apart" looks the same regardless of where you start.

  This matters because the model learns patterns like:
  "the word 2 positions before a verb is likely the subject"
  This pattern should work at position 5 the same as position 50.

  (In practice, sinusoidal encoding achieves this because
  PE(pos+k) is a linear transformation of PE(pos) — the
  relationship between positions is consistent everywhere.)
""")

# ── ANSWER 2: Max Length Problem ──────────────────────────────────────────
print("--- Answer 2: Max Length Problem ---")
print("""
If GPT-2 gets 1025 tokens (but max_len=1024):
  → It crashes! Index out of bounds for the position embedding table.
  → There is no position vector for position 1024.

Solutions used in practice:
  1. Truncation: Just cut the input to 1024 tokens (loses information)
  2. Sliding window: Process 1024 tokens at a time, slide forward
  3. RoPE (Rotary Position Embeddings): Used by Llama/Mistral
     - Encodes position in the rotation of Q/K vectors
     - Can extrapolate to longer sequences than trained on
     - This is why Llama can be "extended" from 4K to 128K tokens
  4. ALiBi (Attention with Linear Biases): Used by some models
     - Adds a bias based on distance instead of position embeddings
     - Naturally handles any length

This max_len limitation is one reason modern models moved away
from simple learned positional embeddings to RoPE.
""")

# ── ANSWER 3: Addition vs Concatenation ──────────────────────────────────
print("--- Answer 3: Addition vs Concatenation ---")

d_model_demo = 8
tok_vec = torch.randn(d_model_demo)
pos_vec = torch.randn(d_model_demo)

added = tok_vec + pos_vec       # still 8-dim
concatenated = torch.cat([tok_vec, pos_vec])  # 16-dim

print(f"Token vector:   {tok_vec.shape}  (8 dimensions)")
print(f"Position vector:{pos_vec.shape}  (8 dimensions)")
print(f"After addition:      {added.shape}  (still 8 dimensions)")
print(f"After concatenation: {concatenated.shape}  (now 16 dimensions)")
print(f"""
Addition (what models use):
  ✓ Dimension stays the same → all layers can stay the same size
  ✓ Fewer parameters (no need to double all weight matrices)
  ✓ Works well in practice — information mixes during attention
  ✗ Token and position info are "mixed" — can't cleanly separate them

Concatenation (rarely used):
  ✓ Token and position info stay separate — cleaner in theory
  ✗ Doubles the dimension → doubles ALL weight matrices in the model
  ✗ For GPT-3 (d_model=12288), this would mean d=24576 — huge!
  ✗ Doesn't actually work better in practice

Real-world: EVERY major model (GPT, BERT, Llama) uses addition.
The mixing of token + position info is fine because the model
learns to separate them through its attention layers.
""")

# ── ANSWER 4: No Position Encoding ──────────────────────────────────────
print("--- Answer 4: No Position Encoding ---")
print("""
Tasks that would STILL WORK without position encoding:

  ✓ Sentiment analysis: "This movie was terrible" → negative
    (Word order matters less; "terrible" alone signals negative)

  ✓ Keyword detection: "Does this email mention 'refund'?"
    (Just checking presence, not position)

  ✓ Bag-of-words tasks: Topic classification, spam detection
    (These mostly depend on WHICH words appear, not their order)

Tasks that would FAIL without position encoding:

  ✗ "Who bit whom?" in "dog bites man" vs "man bites dog"
    (Without position: identical inputs → identical outputs)

  ✗ Translation: Word order differs between languages
    "I eat fish" (English) → "fish eat I" (Japanese SOV order)

  ✗ Code generation: "x = y + 1" vs "y = x + 1"
    (Completely different programs!)

  ✗ Any task where ORDER changes MEANING — which is most of language.

Key insight: Position encoding is what makes a Transformer a SEQUENCE
model instead of just a BAG-OF-WORDS model. Without it, attention
treats input as an unordered set.
""")
