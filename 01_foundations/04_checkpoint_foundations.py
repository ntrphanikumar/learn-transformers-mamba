"""
Module 1 CHECKPOINT: Putting Foundations Together
==================================================

You've learned 3 things:
  1. Tensors & operations — the math (matmul, softmax, layernorm)
  2. Embeddings — how words become vectors
  3. Positional encoding — how models know word order

Now let's see them work TOGETHER as the input pipeline of a real model.
This is exactly what happens in the first layer of GPT before any
attention happens.

After this checkpoint, you should feel confident about:
  - What goes IN to a Transformer (token IDs)
  - What comes OUT of the first layer (position-aware embedding vectors)
  - Why each step is necessary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# THE COMPLETE INPUT PIPELINE
# =============================================================================
# This is what happens BEFORE the first attention layer in GPT.
# After this, the Transformer blocks take over.

print("=" * 60)
print("THE COMPLETE INPUT PIPELINE (what GPT does before attention)")
print("=" * 60)

# ── Step 0: Raw text ─────────────────────────────────────────────────────
raw_text = "the cat sat on the mat"
print(f"\nStep 0 — Raw text: '{raw_text}'")
print(f"  → This is what the user types. Can't do math on strings.\n")

# ── Step 1: Tokenization (text → token IDs) ──────────────────────────────
# From Module 1.2: Build vocab and convert to integers
words = raw_text.split()
vocab = {word: idx for idx, word in enumerate(sorted(set(words)))}
token_ids = torch.tensor([[vocab[w] for w in words]])  # add batch dim

print(f"Step 1 — Tokenize: words → integer IDs")
print(f"  Vocabulary: {vocab}")
print(f"  Token IDs: {token_ids}")
print(f"  Shape: {token_ids.shape} (1 sentence, 6 words)")
print(f"  → Now we have numbers, but they're just arbitrary IDs.\n")

# ── Step 2: Token Embedding (IDs → vectors) ──────────────────────────────
# From Module 1.2: Look up each ID in the embedding table
vocab_size = len(vocab)
d_model = 16  # small for demo; GPT-2 uses 768

token_embedding = nn.Embedding(vocab_size, d_model)
tok_vectors = token_embedding(token_ids)  # (1, 6, 16)

print(f"Step 2 — Embed: token IDs → vectors (from Module 1.2)")
print(f"  Embedding table: {vocab_size} tokens × {d_model} dims")
print(f"  Output shape: {tok_vectors.shape}")
print(f"  'cat' vector: {tok_vectors[0, 1, :6].detach().numpy().round(3)}...")
print(f"  → Each word is now a {d_model}-dim vector. But 'the' at pos 0")
print(f"    and 'the' at pos 4 are still identical.\n")

# ── Step 3: Position Embedding (add position info) ───────────────────────
# From Module 1.3: Each position gets its own vector, added to the token vector
seq_len = token_ids.shape[1]
max_len = 50

pos_embedding = nn.Embedding(max_len, d_model)
positions = torch.arange(seq_len).unsqueeze(0)  # [[0, 1, 2, 3, 4, 5]]
pos_vectors = pos_embedding(positions)  # (1, 6, 16)

# THE COMBINATION: simply add them
x = tok_vectors + pos_vectors  # (1, 6, 16)

print(f"Step 3 — Add positions: make identical tokens distinguishable (from Module 1.3)")
print(f"  Position IDs: {positions}")
print(f"  Position vectors shape: {pos_vectors.shape}")
print(f"  Combined shape: {x.shape}")

# Now "the" at position 0 and "the" at position 4 are DIFFERENT
the_pos0 = x[0, 0]
the_pos4 = x[0, 4]
print(f"  'the' at pos 0 vs pos 4 distance: {(the_pos0 - the_pos4).norm():.3f}")
print(f"  → Now they're different! The model knows which 'the' is which.\n")

# ── Step 4: Layer Norm (stabilize before attention) ──────────────────────
# From Module 1.1: Normalize to prevent exploding/vanishing values
layer_norm = nn.LayerNorm(d_model)
x = layer_norm(x)

print(f"Step 4 — LayerNorm: stabilize values (from Module 1.1)")
print(f"  Output shape: {x.shape}")
print(f"  Mean: {x[0, 0].mean():.4f}, Std: {x[0, 0].std():.4f}")
print(f"  → Values are now in a healthy range for the attention layers.\n")

# ── DONE: This is what goes into the first Transformer block ─────────────
print(f"RESULT: Ready for attention!")
print(f"  Input to the model: '{raw_text}'")
print(f"  Output of input pipeline: tensor of shape {x.shape}")
print(f"  = 1 sentence × 6 words × {d_model} dimensions per word")
print(f"  Each word vector encodes: WHAT the word is + WHERE it is")


# =============================================================================
# INTERACTIVE TRACE: Follow one word through the pipeline
# =============================================================================

print(f"\n{'=' * 60}")
print("TRACE: Follow the word 'cat' through every step")
print(f"{'=' * 60}")

print(f"""
  'cat' (string)
    ↓ tokenize
  {vocab['cat']} (integer — token ID)
    ↓ token embedding lookup
  [{tok_vectors[0, 1, :6].detach().numpy().round(3)}...] (16-dim vector — what 'cat' means)
    ↓ + position embedding for position 1
  [{(tok_vectors[0,1] + pos_vectors[0,1])[:6].detach().numpy().round(3)}...] (16-dim — 'cat' at position 1)
    ↓ layer norm
  [{x[0, 1, :6].detach().numpy().round(3)}...] (16-dim — normalized, ready for attention)

  This vector now enters the attention mechanism, where it will
  "look at" every other word's vector and learn relationships.
""")


# =============================================================================
# QUIZ: Test your understanding
# =============================================================================

print("=" * 60)
print("QUIZ: Test your understanding of the foundations")
print("=" * 60)
print("""
Answer these before looking at the answers below:

Q1. If the vocabulary has 50,000 tokens and d_model=4096,
    how many parameters are in the token embedding table alone?

Q2. Why do we ADD position embeddings instead of treating position
    as a separate input?

Q3. What would happen if we skipped LayerNorm before the first
    attention layer?

Q4. In the sentence "bank of the river" vs "bank of America",
    the word "bank" has the same token ID and same position (0).
    At this point (after embeddings), are they the same vector?
    When do they become different?

Q5. Why is this pipeline the same for both Transformer and Mamba?
    (Hint: what's different is what comes AFTER this pipeline)
""")

# =============================================================================
# QUIZ ANSWERS
# =============================================================================

print("\n\n")
print("=" * 60)
print("QUIZ ANSWERS")
print("=" * 60)

print("""
A1. 50,000 × 4,096 = 204,800,000 parameters (204.8M)
    In float32: 204.8M × 4 bytes = ~819 MB just for the embedding table!
    That's why GPT-2 Small is 500MB — a big chunk is just embeddings.
""")

print("""
A2. Addition keeps the dimension the same (d_model stays 4096, not 8192).
    If we concatenated, EVERY weight matrix in the model would need to
    be doubled — hugely expensive. Addition works because the attention
    layers learn to separate and use both pieces of information.
    It's like mixing salt into water — the salt is still there, and
    the model learns to "taste" it.
""")

print("""
A3. Without LayerNorm, the first attention layer receives un-normalized
    vectors. The scale of these vectors depends on random initialization
    of the embedding tables. If values are large, softmax in attention
    becomes extremely peaky (one word gets 99.9% of attention weight).
    If values are tiny, gradients vanish. LayerNorm ensures attention
    gets vectors in a predictable range, regardless of initialization.
""")

print("""
A4. YES — at this point they are EXACTLY the same vector!
    Same token ID → same token embedding. Same position → same position
    embedding. So "bank" in both sentences is identical after step 3.

    They become DIFFERENT after the attention layers process them.
    In "bank of the river", attention connects "bank" to "river" and "of",
    pushing its representation toward the "riverbank" meaning.
    In "bank of America", attention connects "bank" to "America",
    pushing it toward the "financial institution" meaning.

    THIS is why attention matters — it creates CONTEXT-DEPENDENT
    representations from context-independent embeddings.
    (This is also what makes BERT and GPT different from simple
    bag-of-words models.)
""")

print("""
A5. Both Transformer and Mamba need the same input: a sequence of vectors,
    one per token, encoding both content and position.

    The DIFFERENCE is what happens next:
    - Transformer: Attention (every word looks at every other word)
    - Mamba: Selective SSM (state is updated sequentially, token by token)

    But the input pipeline (tokenize → embed → add position → normalize)
    is identical. This is why you can swap a Transformer backbone for
    a Mamba backbone and keep everything else the same.
""")

print("""
╔══════════════════════════════════════════════════════════════════╗
║  MODULE 1 COMPLETE!                                             ║
║                                                                  ║
║  You now understand the ENTIRE input pipeline:                   ║
║    text → tokens → embeddings → + positions → normalize          ║
║                                                                  ║
║  Next: Module 2 (02_attention/) — where the magic happens.      ║
║  Attention is HOW the model builds understanding from these      ║
║  vectors. It's the most important concept in modern AI.          ║
╚══════════════════════════════════════════════════════════════════╝
""")
