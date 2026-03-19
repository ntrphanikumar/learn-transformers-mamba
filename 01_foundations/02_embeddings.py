"""
Module 1.2: Embeddings — How Words Become Vectors
==================================================

WHY THIS MATTERS:
Neural networks only understand numbers. To process text, we need to convert
words (or characters) into numeric vectors. This conversion is called
"embedding" and it's the very first step in both Transformers and Mamba.

KEY CONCEPTS:
- Token = a piece of text (word, subword, or character)
- Vocabulary = the set of all tokens the model knows
- Embedding = a learned vector representation of a token
- Embedding dimension = how many numbers represent each token
"""

import torch
import torch.nn as nn

# =============================================================================
# 1. THE PROBLEM: Text is not numbers
# =============================================================================

text = "the cat sat on the mat"
words = text.split()
print(f"Words: {words}")
print(f"A neural network can't process strings directly.\n")

# =============================================================================
# 2. STEP 1: Build a vocabulary (token -> integer)
# =============================================================================

# In practice, models use sophisticated tokenizers (BPE, SentencePiece).
# For learning, we'll use a simple word-level vocabulary.

vocab = {word: idx for idx, word in enumerate(sorted(set(words)))}
print(f"Vocabulary: {vocab}")

# Convert words to token IDs
token_ids = torch.tensor([vocab[w] for w in words])
print(f"Token IDs: {token_ids}")
print(f"'the' -> {vocab['the']}, 'cat' -> {vocab['cat']}\n")

# =============================================================================
# 3. STEP 2: Embedding layer (integer -> vector)
# =============================================================================

vocab_size = len(vocab)    # how many unique tokens
embed_dim = 8              # how many numbers per token (small for demo)

# nn.Embedding is essentially a lookup table:
# - It stores a matrix of shape (vocab_size, embed_dim)
# - Given a token ID, it returns the corresponding row
embedding = nn.Embedding(vocab_size, embed_dim)

print(f"Embedding table shape: {embedding.weight.shape}")
print(f"  = {vocab_size} tokens × {embed_dim} dimensions\n")

# Look up embeddings for our tokens
embedded = embedding(token_ids)
print(f"Input token IDs shape: {token_ids.shape}")
print(f"Output embeddings shape: {embedded.shape}")
print(f"  = {len(words)} words × {embed_dim} dimensions\n")

# Each word is now a vector of numbers
for word, vec in zip(words, embedded):
    print(f"  '{word:3s}' -> [{vec[:4].detach().numpy().round(2)}...]")

# =============================================================================
# 4. WHY EMBEDDINGS WORK
# =============================================================================

print(f"""
KEY INSIGHT:
These vectors start RANDOM. During training, the model adjusts them so that:
  - Similar words get similar vectors (cat ≈ dog)
  - The vector directions encode meaning (king - man + woman ≈ queen)

The embedding IS the model's understanding of what each word means.
In GPT-3: vocab_size=50,257 and embed_dim=12,288
In Llama-2: vocab_size=32,000 and embed_dim=4,096
""")

# =============================================================================
# 5. REAL-WORLD SCALE
# =============================================================================

# Let's see what real model embeddings look like
print("Typical model sizes:")
for name, v, d in [("GPT-2 Small", 50257, 768),
                    ("GPT-2 Large", 50257, 1280),
                    ("Llama-2 7B", 32000, 4096),
                    ("GPT-3 175B", 50257, 12288)]:
    params = v * d
    print(f"  {name:15s}: {v:,} tokens × {d:,} dims = {params:>14,} parameters "
          f"({params * 4 / 1e6:.0f} MB in float32)")

# =============================================================================
# 6. EXERCISE
# =============================================================================

print(f"""
EXERCISE:
1. Try changing embed_dim to 2 and plotting the word vectors in 2D
2. What happens if two different words get the same token ID?
   (This is the "collision" problem that motivates larger vocabularies)
3. Try character-level tokenization instead of word-level:
   vocab = {{ch: i for i, ch in enumerate(sorted(set(text)))}}
""")
