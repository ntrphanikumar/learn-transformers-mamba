"""
Module 1.1: Tensors & Core Operations
======================================

WHY THIS MATTERS:
Every neural network is just tensor math. Before understanding Transformers
or Mamba, you need to be comfortable with what tensors are and how they
combine. This file covers the operations that appear EVERYWHERE in deep learning.

KEY CONCEPTS:
- Tensor = multi-dimensional array (generalization of vectors and matrices)
- Matrix multiplication = the fundamental building block of neural networks
- Softmax = turns raw numbers into probabilities (used in attention)
- Layer normalization = keeps numbers in a stable range during training

Each section follows: WHAT → WHEN/WHY you'd use it → REAL-WORLD EXAMPLE
"""

import torch
import torch.nn.functional as F

# =============================================================================
# 1. TENSORS: The building blocks
# =============================================================================
# WHAT: Tensors are multi-dimensional arrays of numbers.
# WHEN: Always. Every piece of data in AI is a tensor.

# A scalar (0-dimensional tensor)
scalar = torch.tensor(3.14)
print(f"Scalar: {scalar}, shape: {scalar.shape}")

# A vector (1-dimensional tensor)
vector = torch.tensor([1.0, 2.0, 3.0])
print(f"Vector: {vector}, shape: {vector.shape}")

# A matrix (2-dimensional tensor)
matrix = torch.randn(3, 4)
print(f"Matrix shape: {matrix.shape}")
print(f"Matrix:\n{matrix}\n")

# A 3D tensor
batch = torch.randn(2, 5, 8)
print(f"Batch shape: {batch.shape}")
print(f"This is: {batch.shape[0]} sentences, {batch.shape[1]} words each, "
      f"{batch.shape[2]}-dimensional embeddings\n")

# ── REAL-WORLD EXAMPLE: Sentiment Analysis Input ─────────────────────────
# Imagine you're building a model to classify movie reviews as positive/negative.
# A single review: "this movie was great" → 4 words
# Each word is represented as a vector of numbers (an embedding).
# A batch of reviews is a 3D tensor.

print("REAL-WORLD: Preparing a batch of movie reviews for sentiment analysis")
print("-" * 60)

reviews = [
    "this movie was great",      # review 1 (positive)
    "terrible waste of time",    # review 2 (negative)
]

# In practice, each word gets looked up in an embedding table.
# For now, simulate: 2 reviews × 4 words × 8-dim embedding
review_batch = torch.randn(2, 4, 8)
print(f"Batch shape: {review_batch.shape}")
print(f"  → {review_batch.shape[0]} reviews")
print(f"  → {review_batch.shape[1]} words per review")
print(f"  → {review_batch.shape[2]} numbers describing each word")
print(f"This tensor IS the input to a Transformer/Mamba model.\n")


# =============================================================================
# 2. MATRIX MULTIPLICATION: How layers transform data
# =============================================================================
# WHAT: Multiply two matrices to transform vectors from one space to another.
# WHEN: Every single layer in a neural network does this. It's how the model
#       "thinks" — by transforming representations through learned weight matrices.
# WHY:  The weight matrix encodes WHAT FEATURES to extract. Training adjusts
#       these weights until the right features are found.

x = torch.randn(3, 4)   # 3 words, each a 4-dim vector
W = torch.randn(4, 6)   # Weight matrix: transforms 4-dim -> 6-dim
output = x @ W           # @ is matrix multiply
print(f"Input shape: {x.shape}")
print(f"Weight shape: {W.shape}")
print(f"Output shape: {output.shape}")
print(f"→ Each 4-dim word vector became a 6-dim vector\n")

# ── REAL-WORLD EXAMPLE: Email Spam Classifier ────────────────────────────
# You have an email represented as a 100-dim vector (from an embedding).
# You want to classify it into 2 classes: spam or not-spam.
# A linear layer does: email_vector @ W → 2 scores (one per class).

print("REAL-WORLD: Email spam classification")
print("-" * 60)

num_emails = 5
email_embedding_dim = 100
num_classes = 2  # spam, not-spam

# 5 emails, each a 100-dim vector
emails = torch.randn(num_emails, email_embedding_dim)

# Learned weight matrix: maps 100-dim → 2 scores
classifier_weights = torch.randn(email_embedding_dim, num_classes)

# The classification
scores = emails @ classifier_weights  # (5, 2) — two scores per email
print(f"Email embeddings: {emails.shape}")
print(f"Classifier weights: {classifier_weights.shape}")
print(f"Output scores: {scores.shape}")
print(f"  → Each email now has 2 scores: [spam_score, not_spam_score]")
print(f"  → Email 0 scores: spam={scores[0,0]:.2f}, not_spam={scores[0,1]:.2f}")
print(f"  → Whichever score is higher = the model's prediction")
print(f"  → Training adjusts classifier_weights until predictions are correct\n")


# =============================================================================
# 3. SOFTMAX: Turning scores into probabilities
# =============================================================================
# WHAT: Converts any list of numbers into probabilities (positive, sum to 1).
# WHEN:
#   1. Attention weights — deciding how much each word should look at other words
#   2. Final classification — converting raw scores to class probabilities
#   3. Token generation — choosing the next word in text generation
# WHY: Raw scores (logits) are unbounded. Softmax normalizes them into a
#      probability distribution, which is meaningful and differentiable.

logits = torch.tensor([2.0, 1.0, 0.1])
print(f"Raw logits: {logits}")
probs = F.softmax(logits, dim=0)
print(f"After softmax: {probs}")
print(f"Sum: {probs.sum():.4f}\n")

# ── REAL-WORLD EXAMPLE 1: Next-Word Prediction (Language Model) ──────────
# When ChatGPT generates text, at each step it scores EVERY word in its
# vocabulary and uses softmax to get probabilities.

print("REAL-WORLD: Next-word prediction in a language model")
print("-" * 60)

# Simplified vocabulary
vocab = ["the", "cat", "dog", "sat", "on", "mat", "ran", "jumped"]

# Model's raw scores for next word after "the cat ___"
raw_scores = torch.tensor([0.1, -0.5, -0.3, 3.8, 0.2, -1.0, 1.5, 0.7])

probs = F.softmax(raw_scores, dim=0)
print("After 'the cat', model predicts next word:")
for word, score, prob in zip(vocab, raw_scores, probs):
    bar = "█" * int(prob * 50)
    print(f"  {word:8s}  score={score:5.1f}  prob={prob:.3f}  {bar}")
print(f"→ 'sat' has highest probability — model would likely pick it\n")

# ── REAL-WORLD EXAMPLE 2: Temperature in ChatGPT ────────────────────────
# When you set "temperature" in ChatGPT, you're controlling softmax sharpness.

print("REAL-WORLD: Temperature controls creativity")
print("-" * 60)
print("Same scores, different temperatures:")

for temp, label in [(0.1, "Very focused (code generation)"),
                     (1.0, "Balanced (normal chat)"),
                     (2.0, "Creative (brainstorming)")]:
    probs = F.softmax(raw_scores / temp, dim=0)
    top_word = vocab[probs.argmax()]
    top_prob = probs.max()
    print(f"  temp={temp:.1f} ({label})")
    print(f"    Top word: '{top_word}' at {top_prob:.1%} probability")
    print(f"    Distribution: {probs.numpy().round(3)}")

print("""
→ Low temperature: Model almost always picks the highest-scored word.
  Use when you want deterministic, reliable output (code, math).
→ High temperature: Probabilities spread out, model explores more options.
  Use when you want diverse, creative output (stories, brainstorming).
→ Temperature=1.0: The raw model output, unmodified.
""")

# ── REAL-WORLD EXAMPLE 3: Attention Weights ─────────────────────────────
# In a Transformer, softmax is used to compute attention weights.
# "How much should the word 'sat' pay attention to each other word?"

print("REAL-WORLD: Attention in 'the cat sat on the mat'")
print("-" * 60)

# Imagine the model computed these raw similarity scores for the word "sat"
# against every word in the sentence:
words = ["the", "cat", "sat", "on", "the", "mat"]
similarity_scores = torch.tensor([0.3, 2.8, 1.0, 0.1, 0.2, 0.8])

attention_weights = F.softmax(similarity_scores, dim=0)
print("Word 'sat' attends to:")
for word, weight in zip(words, attention_weights):
    bar = "█" * int(weight * 40)
    print(f"  '{word:3s}' → {weight:.3f}  {bar}")
print("→ 'sat' pays most attention to 'cat' (the subject doing the sitting)")
print("  This is how Transformers understand relationships between words.\n")


# =============================================================================
# 4. LAYER NORMALIZATION: Keeping numbers stable
# =============================================================================
# WHAT: Normalizes a vector to have mean=0 and std=1, then applies learned
#       scale and shift.
# WHEN: After EVERY sub-layer in a Transformer (after attention, after FFN).
#       Also used in Mamba blocks. Without it, training fails.
# WHY:  As data flows through many layers, numbers can grow huge or shrink
#       to near-zero. LayerNorm resets the scale at each layer, keeping
#       gradients healthy and training stable.

x = torch.tensor([[1000.0, 2000.0, 3000.0],
                   [0.001, 0.002, 0.003]])
print(f"Before LayerNorm:\n{x}")

ln = torch.nn.LayerNorm(3)
normalized = ln(x)
print(f"After LayerNorm:\n{normalized}")
print("→ Both rows now have similar scale, despite wildly different inputs\n")

# ── REAL-WORLD EXAMPLE: Why Training Fails Without Normalization ─────────
print("REAL-WORLD: What happens WITHOUT LayerNorm in deep networks")
print("-" * 60)

# Simulate passing data through 50 layers WITHOUT normalization
x = torch.randn(1, 8)  # a single 8-dim vector
print(f"Start:     mean={x.mean():.4f}, std={x.std():.4f}, max={x.abs().max():.4f}")

# Without LayerNorm: numbers explode
x_no_norm = x.clone()
for i in range(50):
    W = torch.randn(8, 8) * 0.5  # even with small weights...
    x_no_norm = x_no_norm @ W
print(f"After 50 layers (no norm):  mean={x_no_norm.mean():.4f}, "
      f"std={x_no_norm.std():.4f}, max={x_no_norm.abs().max():.4f}")

# With LayerNorm: numbers stay stable
x_with_norm = x.clone()
layer_norm = torch.nn.LayerNorm(8)
for i in range(50):
    W = torch.randn(8, 8) * 0.5
    x_with_norm = layer_norm(x_with_norm @ W)
print(f"After 50 layers (with norm): mean={x_with_norm.mean():.4f}, "
      f"std={x_with_norm.std():.4f}, max={x_with_norm.abs().max():.4f}")

print("""
→ Without normalization, values explode or vanish after many layers.
  Gradients become useless → the model can't learn.
→ With normalization, values stay in a healthy range no matter how deep.
  This is why GPT-3 (96 layers) and Llama (80 layers) can train at all.
""")

# ── REAL-WORLD EXAMPLE: Batch of Mixed-Scale Inputs ─────────────────────
print("REAL-WORLD: Processing mixed-scale features")
print("-" * 60)

# Imagine a model processing user data where features have very different scales:
# [age, salary, num_clicks, account_age_days]
user_data = torch.tensor([
    [25.0, 75000.0, 12.0, 365.0],    # young employee
    [55.0, 150000.0, 3.0, 3650.0],   # senior executive
])
print(f"Raw features:\n{user_data}")

ln = torch.nn.LayerNorm(4)
normalized = ln(user_data)
print(f"After LayerNorm:\n{normalized}")
print("→ Now all features are on comparable scales — the model can treat")
print("  them fairly instead of being dominated by 'salary'.\n")


# =============================================================================
# 5. PUTTING IT ALL TOGETHER: A mini neural network layer
# =============================================================================
# Every layer in a Transformer or Mamba model combines these operations.
# Let's trace the data flow through one layer and understand each step's role.

print("=" * 60)
print("PUTTING IT TOGETHER: Trace through a single layer")
print("=" * 60)

# Scenario: We have 4 word embeddings and we want to transform them
# through one layer of processing (like one layer of GPT).

x = torch.randn(4, 8)      # 4 words, 8-dim each
W = torch.randn(8, 8)      # learned weight matrix
ln = torch.nn.LayerNorm(8)

# Step 1: Linear transformation (matmul)
# WHY: Extract features. The weight matrix W has learned what patterns matter.
step1 = x @ W
print(f"\nStep 1 — Matmul (extract features):")
print(f"  {x.shape} @ {W.shape} → {step1.shape}")

# Step 2: Softmax (normalize to probabilities)
# WHY: In attention, this decides "how much to attend to each word."
#      Here we're simulating attention-like weighting.
step2 = F.softmax(step1, dim=-1)
print(f"Step 2 — Softmax (normalize to probabilities):")
print(f"  Row sums to: {step2[0].sum():.4f}")

# Step 3: LayerNorm (stabilize)
# WHY: Keep values in a healthy range before passing to the next layer.
step3 = ln(step2)
print(f"Step 3 — LayerNorm (stabilize for next layer):")
print(f"  Mean: {step3[0].mean():.4f}, Std: {step3[0].std():.4f}")

print(f"""
This is essentially what happens at each layer:
  Input → Transform (matmul) → Normalize (softmax/etc) → Stabilize (layernorm) → Next layer

In a real Transformer block, it's:
  Input → Attention(Q,K,V via matmul) → Softmax → LayerNorm → FFN(matmul) → LayerNorm → Output

You now know every operation used inside that block!
""")

# =============================================================================
# 6. EXERCISES WITH REAL-WORLD CONTEXT
# =============================================================================

print("=" * 60)
print("EXERCISES (try modifying the code above)")
print("=" * 60)
print("""
1. IMAGE CLASSIFICATION:
   An image model takes a 224×224 RGB image. Represent it as a tensor.
   What shape should it be? (Hint: 3 color channels)
   → Try: img = torch.randn(???)   # fill in the shape
   → A batch of 16 images? What shape?

2. RECOMMENDATION SYSTEM:
   You have 1000 users and 500 movies. Each user is a 64-dim vector,
   each movie is a 64-dim vector. Compute the score of every user
   for every movie in ONE matrix multiplication.
   → users = torch.randn(1000, 64)
   → movies = torch.randn(500, 64)
   → scores = ???  # What operation gives a (1000, 500) score matrix?

3. SOFTMAX INTUITION:
   A language model is choosing between 50,000 words. Only the top 5
   have meaningful probability; the rest are near-zero.
   → Create a tensor of 50,000 logits (mostly small, 5 large)
   → Apply softmax. What do you notice about the distribution?
   → This is why "top-k sampling" exists in ChatGPT.

4. LAYERNORM vs NO NORM:
   Modify the "50 layers" example above. Try 100 layers, 200 layers.
   At what point do the un-normalized values become infinity (inf)?
   What does this mean for training a very deep model?
""")

# =============================================================================
# 7. ANSWERS — Scroll down only after you've tried!
# =============================================================================

print("\n\n")
print("=" * 60)
print("ANSWERS (try the exercises first!)")
print("=" * 60)

# ── ANSWER 1: Image Classification ──────────────────────────────────────
print("\n--- Answer 1: Image Classification ---")

# A single 224×224 RGB image: 3 channels × 224 height × 224 width
img = torch.randn(3, 224, 224)
print(f"Single image: {img.shape}")
print(f"  → 3 = color channels (Red, Green, Blue)")
print(f"  → 224×224 = pixel grid")
print(f"  → Total numbers: {img.numel():,} (that's a lot for one image!)")
    
# A batch of 16 images
batch_of_images = torch.randn(16, 3, 224, 224)
print(f"Batch of 16 images: {batch_of_images.shape}")
print(f"  → Convention: (batch_size, channels, height, width)")
print(f"  → GPUs process batches in parallel — that's why we batch.")
print(f"  → Total numbers in batch: {batch_of_images.numel():,}")

# ── ANSWER 2: Recommendation System ─────────────────────────────────────
print("\n--- Answer 2: Recommendation System ---")

users = torch.randn(1000, 64)    # 1000 users, each a 64-dim vector
movies = torch.randn(500, 64)    # 500 movies, each a 64-dim vector

# The key insight: user-movie score = dot product of their vectors.
# To get ALL scores at once, multiply users by movies TRANSPOSED.
scores = users @ movies.T        # (1000, 64) @ (64, 500) → (1000, 500)

print(f"Users:  {users.shape}")
print(f"Movies: {movies.shape}")
print(f"Scores: {scores.shape}")
print(f"  → scores[i][j] = how much user i would like movie j")
print(f"  → User 0's top movie: movie #{scores[0].argmax().item()}")
print(f"  → User 0's score for that movie: {scores[0].max():.2f}")
print(f"")
print(f"  WHY THIS WORKS: If a user vector and movie vector point in the")
print(f"  same direction (high dot product), the user likes that movie.")
print(f"  This is exactly how Netflix/Spotify recommendations work.")
print(f"  The vectors are learned during training to capture taste.")

# ── ANSWER 3: Softmax Intuition ─────────────────────────────────────────
print("\n--- Answer 3: Softmax Intuition ---")

# 50,000 logits: most are small noise, 5 are large
logits_50k = torch.randn(50000) * 0.1   # most words: small random scores
# Make 5 words stand out
logits_50k[42] = 8.0     # "sat"
logits_50k[1337] = 6.5   # "rested"
logits_50k[7777] = 5.0   # "slept"
logits_50k[100] = 4.5    # "lay"
logits_50k[9999] = 4.0   # "waited"

probs_50k = F.softmax(logits_50k, dim=0)

# Look at the distribution
top5_probs = probs_50k[[42, 1337, 7777, 100, 9999]]
rest_prob = probs_50k.sum() - top5_probs.sum()

print(f"Vocabulary size: {len(logits_50k):,}")
print(f"Top 5 words share: {top5_probs.sum():.4f} ({top5_probs.sum()*100:.1f}% of probability)")
print(f"Other 49,995 words share: {rest_prob:.4f} ({rest_prob*100:.1f}% of probability)")
print(f"")
print(f"Individual top-5 probabilities:")
for idx, name in [(42, "sat"), (1337, "rested"), (7777, "slept"),
                   (100, "lay"), (9999, "waited")]:
    print(f"  '{name}': {probs_50k[idx]:.4f} ({probs_50k[idx]*100:.2f}%)")
print(f"  Average of the other 49,995 words: {rest_prob/49995:.8f} (~0%)")
print(f"")
print(f"  KEY INSIGHT: Softmax is EXTREMELY peaky with large vocab sizes.")
print(f"  99%+ of probability lands on just a few words.")
print(f"  This is why 'top-k sampling' (only consider top 50 words) and")
print(f"  'top-p / nucleus sampling' (only consider words until cumulative")
print(f"  probability hits 95%) exist — the rest are essentially zero.")

# ── ANSWER 4: LayerNorm vs No Norm ──────────────────────────────────────
print("\n--- Answer 4: LayerNorm vs No Norm ---")

print("Testing how many layers until values explode to infinity:\n")
for num_layers in [50, 75, 100, 150, 200]:
    x_test = torch.randn(1, 8)
    exploded = False
    for i in range(num_layers):
        W = torch.randn(8, 8) * 0.5
        x_test = x_test @ W
        if torch.isinf(x_test).any() or torch.isnan(x_test).any():
            print(f"  {num_layers:3d} layers: EXPLODED at layer {i+1} 💥 (values hit infinity)")
            exploded = True
            break
    if not exploded:
        max_val = x_test.abs().max().item()
        if max_val > 1e30:
            print(f"  {num_layers:3d} layers: max value = {max_val:.2e} (astronomically large)")
        else:
            print(f"  {num_layers:3d} layers: max value = {max_val:.4f} (still finite)")

print(f"""
  KEY INSIGHT: Without normalization, deep networks die. The exact layer
  where it explodes varies (it's random), but it WILL happen eventually.

  This is called the "exploding gradient problem" and it's why:
  - Normalization (LayerNorm, BatchNorm) is non-negotiable in deep nets
  - Residual connections help (adding input back keeps values anchored)
  - Weight initialization matters (too large → explode, too small → vanish)

  GPT-3 has 96 layers. Without LayerNorm + residual connections,
  it would be completely untrainable. These aren't optional extras —
  they're what make deep learning possible at all.
""")
