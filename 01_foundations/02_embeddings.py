"""
Module 1.2: Embeddings — How Words Become Vectors
==================================================

WHY THIS MATTERS:
Neural networks only understand numbers. To process text, we need to convert
words (or characters) into numeric vectors. This conversion is called
"embedding" and it's the very first step in both Transformers and Mamba.

WHEN IS THIS USED:
- The VERY FIRST layer of every language model (GPT, BERT, Llama, Mamba)
- Also used in recommendation systems (user/item embeddings)
- Also used for images (patch embeddings in Vision Transformers)
- Basically: whenever you need to convert discrete things → continuous vectors

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
# WHAT: Neural networks are math machines — they multiply, add, and apply
#       functions. They can't process the string "cat" directly.
# WHEN: This is the very first problem to solve in ANY NLP system.

text = "the cat sat on the mat"
words = text.split()
print(f"Words: {words}")
print(f"A neural network can't process strings directly.")
print(f"We need: string → integer → vector of floats\n")

# =============================================================================
# 2. STEP 1: Build a vocabulary (token -> integer)
# =============================================================================
# WHAT: Assign each unique word a number (its ID).
# WHEN: Before any processing. This is how the model identifies tokens.
# WHY:  The ID is used as an index into the embedding table.

# In practice, models use sophisticated tokenizers (BPE, SentencePiece).
# For learning, we'll use a simple word-level vocabulary.

vocab = {word: idx for idx, word in enumerate(sorted(set(words)))}
print(f"Vocabulary: {vocab}")

# Convert words to token IDs
token_ids = torch.tensor([vocab[w] for w in words])
print(f"Token IDs: {token_ids}")
print(f"'the' -> {vocab['the']}, 'cat' -> {vocab['cat']}\n")

# ── REAL-WORLD: How ChatGPT tokenizes text ───────────────────────────────
print("REAL-WORLD: How ChatGPT actually tokenizes text")
print("-" * 60)
print("""
ChatGPT doesn't use word-level tokens. It uses BPE (Byte-Pair Encoding):
  "unhappiness" → ["un", "happiness"]     (2 tokens, not 1)
  "ChatGPT"     → ["Chat", "G", "PT"]     (3 tokens)
  "the"         → ["the"]                  (1 token — common words stay whole)

WHY subwords instead of words?
  - Word-level: vocabulary would be MILLIONS of words (every name, typo, etc.)
  - Character-level: sequences become very long ("hello" = 5 tokens)
  - Subword (BPE): sweet spot — ~50K tokens covers all of English efficiently

GPT-4 uses ~100K tokens. Each token has a unique ID like our vocab above.
""")


# =============================================================================
# 3. STEP 2: Embedding layer (integer -> vector)
# =============================================================================
# WHAT: A lookup table that maps each token ID to a vector of numbers.
# WHEN: Immediately after tokenization — this IS the first layer of the model.
# WHY:  An integer (like 42) has no meaningful structure. A vector (like
#       [0.3, -0.7, 0.1, ...]) can encode relationships between words.
#       Similar words end up with similar vectors.

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

# ── REAL-WORLD: Embedding is literally just a table lookup ────────────────
print(f"\nREAL-WORLD: What nn.Embedding actually does")
print("-" * 60)

print("The embedding 'layer' is just indexing into a matrix:")
print(f"  Token ID 0 → row 0 of the weight matrix")
print(f"  Token ID 3 → row 3 of the weight matrix\n")

# Prove it — manual lookup vs nn.Embedding gives the same result
manual_lookup = embedding.weight[token_ids[0]]
auto_lookup = embedded[0]
print(f"  Manual: embedding.weight[{token_ids[0]}] = {manual_lookup[:4].detach().numpy().round(3)}...")
print(f"  Auto:   embedding(token_ids[0])    = {auto_lookup[:4].detach().numpy().round(3)}...")
print(f"  Same? {torch.allclose(manual_lookup, auto_lookup)}")
print(f"  → It's literally just matrix[row_index]. No math, just a lookup.\n")


# =============================================================================
# 4. WHY EMBEDDINGS WORK — with real examples
# =============================================================================

print("WHY EMBEDDINGS WORK")
print("=" * 50)

print("""
These vectors start RANDOM. During training, the model adjusts them so that:
  - Similar words get similar vectors (cat ≈ dog, king ≈ queen)
  - Vector arithmetic encodes meaning (king - man + woman ≈ queen)
  - Context-dependent relationships emerge
""")

# ── REAL-WORLD: Word similarity via embeddings ───────────────────────────
print("REAL-WORLD: How Netflix recommends movies using embeddings")
print("-" * 60)
print("""
Netflix represents every movie AND every user as embedding vectors:
  Movie: "Inception" → [0.8, -0.2, 0.5, 0.1, ...]  (maybe 128-dim)
  Movie: "Interstellar" → [0.7, -0.1, 0.6, 0.2, ...]  (similar! both sci-fi)
  Movie: "The Notebook" → [-0.3, 0.9, -0.1, 0.7, ...] (very different)

  User: "Phani" → [0.6, -0.3, 0.4, 0.0, ...]  (close to sci-fi movies)

Recommendation = find movies whose vectors are closest to the user's vector.
If Phani's vector is close to "Inception", the model recommends similar movies.

These embeddings are LEARNED from watch history — not hand-crafted.
Same idea as word embeddings, just applied to movies and users.
""")

# Simulate this
print("Simulating movie recommendations:")
torch.manual_seed(42)
movie_names = ["Inception", "Interstellar", "The Notebook", "Titanic", "The Matrix"]
movie_embs = torch.tensor([
    [0.8, -0.2, 0.5, 0.1],   # Inception (sci-fi)
    [0.7, -0.1, 0.6, 0.2],   # Interstellar (sci-fi)
    [-0.3, 0.9, -0.1, 0.7],  # The Notebook (romance)
    [-0.2, 0.8, 0.0, 0.6],   # Titanic (romance)
    [0.9, -0.3, 0.4, -0.1],  # The Matrix (sci-fi)
])
user_emb = torch.tensor([0.6, -0.3, 0.4, 0.0])  # sci-fi fan

# Score = dot product (how similar are the vectors?)
scores = movie_embs @ user_emb
for name, score in sorted(zip(movie_names, scores), key=lambda x: -x[1]):
    bar = "█" * int(max(0, score.item()) * 20)
    print(f"  {name:15s}: score = {score:.2f}  {bar}")
print("→ Sci-fi movies score highest for this user!\n")


# =============================================================================
# 5. REAL-WORLD SCALE
# =============================================================================

print("Typical model embedding sizes:")
print("-" * 60)
for name, v, d in [("GPT-2 Small", 50257, 768),
                    ("GPT-2 Large", 50257, 1280),
                    ("Llama-2 7B", 32000, 4096),
                    ("GPT-3 175B", 50257, 12288)]:
    params = v * d
    print(f"  {name:15s}: {v:,} tokens × {d:,} dims = {params:>14,} parameters "
          f"({params * 4 / 1e6:.0f} MB in float32)")

print("""
  → GPT-3's embedding table alone is 2.3 GB!
  → This is JUST the first layer. The transformer blocks come after.
""")


# =============================================================================
# 6. EXERCISES
# =============================================================================

print("=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
1. SAME WORD, SAME VECTOR:
   In our vocab, "the" appears at positions 0 and 4 in "the cat sat on the mat".
   Do both get the same embedding vector? Why is this a problem?
   (Hint: "the cat" vs "the mat" — the model needs to know WHICH "the")

2. VOCABULARY SIZE TRADEOFF:
   Why not just use character-level tokenization (vocab_size = 128 for ASCII)?
   Why not word-level (vocab_size = every English word)?
   What's the tradeoff?

3. EMBEDDING DIMENSION:
   GPT-2 Small uses embed_dim=768, GPT-3 uses 12,288.
   What happens if embed_dim is too small (like 2)?
   What happens if it's too large?

4. SPOTIFY PLAYLISTS:
   Spotify embeds songs into vectors (just like words). Songs you play together
   end up with similar vectors. If you listen to a lot of jazz, your user
   vector drifts toward the jazz region of the embedding space.
   → Can you think of what dimensions might represent? (genre? tempo? mood?)
""")


# =============================================================================
# 7. ANSWERS
# =============================================================================

print("\n\n")
print("=" * 60)
print("ANSWERS (try the exercises first!)")
print("=" * 60)

# ── ANSWER 1: Same word, same vector ─────────────────────────────────────
print("\n--- Answer 1: Same Word, Same Vector ---")

# "the" appears at positions 0 and 4
the_pos0 = embedded[0]  # "the" at position 0
the_pos4 = embedded[4]  # "the" at position 4
distance = (the_pos0 - the_pos4).norm()
print(f"'the' at position 0: {the_pos0[:4].detach().numpy().round(3)}...")
print(f"'the' at position 4: {the_pos4[:4].detach().numpy().round(3)}...")
print(f"Distance: {distance:.6f}")
print(f"→ They are IDENTICAL! Same token ID → same row in embedding table.")
print(f"")
print(f"This IS a problem: 'the cat' and 'the mat' have different meanings")
print(f"for 'the', but the embedding is the same.")
print(f"")
print(f"SOLUTION: This is exactly why we need POSITIONAL ENCODING (next file).")
print(f"Position info is ADDED to the embedding, making 'the' at pos 0")
print(f"different from 'the' at pos 4.")

# ── ANSWER 2: Vocabulary size tradeoff ───────────────────────────────────
print("\n--- Answer 2: Vocabulary Size Tradeoff ---")
print("""
Character-level (vocab_size ≈ 128):
  ✓ Tiny vocabulary, handles any word, no unknown tokens
  ✗ "transformer" = 11 tokens — sequences become very long
  ✗ Model must learn spelling before it can learn meaning
  ✗ More tokens = more computation per sentence

Word-level (vocab_size ≈ 500,000+):
  ✓ "transformer" = 1 token — sequences are short
  ✗ Huge embedding table (500K × 4096 = 8 GB just for embeddings!)
  ✗ Can't handle new words, typos, or rare names → "unknown token"
  ✗ Wastes parameters on rare words seen once in training

Subword / BPE (vocab_size ≈ 32K-100K) — what real models use:
  ✓ Common words stay whole: "the" = 1 token
  ✓ Rare words split sensibly: "unhappiness" → "un" + "happiness"
  ✓ Can handle ANY word by splitting into pieces
  ✓ Balanced sequence length and vocabulary size
  This is why GPT uses ~50K tokens and Llama uses ~32K tokens.
""")

# ── ANSWER 3: Embedding dimension ────────────────────────────────────────
print("--- Answer 3: Embedding Dimension ---")

# Too small: can't capture enough meaning
emb_tiny = nn.Embedding(5, 2)
words_demo = torch.tensor([0, 1, 2, 3, 4])
vecs_tiny = emb_tiny(words_demo)
print("embed_dim=2 (too small):")
for i in range(5):
    print(f"  Word {i}: {vecs_tiny[i].detach().numpy().round(3)}")
print("  → Only 2 numbers per word. Can barely encode anything.")
print("  → 'cat' and 'dog' can't both be similar to each other AND")
print("    different from 'table' in just 2 dimensions.\n")

print("""Too large (embed_dim=100,000):
  → Each word has 100K numbers — massive waste of memory
  → Most dimensions would be redundant (unused)
  → More parameters = needs more data to train, slower
  → Overfitting risk: model memorizes training data

Sweet spot depends on vocab size and training data:
  Small model + little data → embed_dim=128-512
  Large model + lots of data → embed_dim=4096-12288
  The model needs enough dimensions to capture all the
  relationships between words, but not so many that it wastes capacity.
""")

# ── ANSWER 4: Spotify dimensions ─────────────────────────────────────────
print("--- Answer 4: Spotify Embedding Dimensions ---")
print("""
The dimensions DON'T have predefined meanings — they're learned.
But after training, researchers find that dimensions often correspond to:

  - Genre (rock vs jazz vs classical)
  - Tempo (fast vs slow)
  - Mood (happy vs melancholy)
  - Energy (calm vs intense)
  - Era (60s vs 2020s)
  - Instrumentation (acoustic vs electronic)
  - Language (English vs Spanish vs Korean)

Key insight: nobody TELLS the model these categories. They EMERGE
from the data. The model discovers that "genre" is a useful concept
for predicting what you'll play next, so it dedicates some dimensions
to encoding it. This is what "representation learning" means —
the model learns its own useful representations of the data.

Same thing happens with word embeddings:
  - Some dimensions end up encoding "is this a noun or verb?"
  - Others encode "is this positive or negative?"
  - Others encode "is this formal or informal?"
  All discovered automatically from text data.
""")
