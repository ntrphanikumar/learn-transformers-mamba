"""
Module 3.1: Transformer Block — The Complete Building Block
============================================================

WHY THIS MATTERS:
A Transformer model is just a STACK of identical blocks. Understand one
block and you understand the whole model. GPT-2 = 12 blocks stacked.
GPT-3 = 96 blocks stacked. Same block, more of them.

EACH BLOCK HAS TWO SUB-LAYERS:
1. Multi-Head Self-Attention: "look at the whole sequence"
2. Feed-Forward Network (FFN): "think about each position independently"

Plus two critical ingredients:
- Residual connections: add the input back to the output (helps gradients flow)
- Layer normalization: stabilize the numbers

BLOCK STRUCTURE (GPT-style / Pre-norm):
    x → LayerNorm → Attention → + → LayerNorm → FFN → + → output
    └────────────────────────┘   └──────────────────┘
         residual connection       residual connection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# BUILDING BLOCKS (reused from earlier modules)
# =============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)

        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    This is applied INDEPENDENTLY to each position. It's where the model
    does its "thinking" — the attention gathers information, the FFN
    processes it.

    Architecture: Linear → GELU → Linear
    The hidden dimension is typically 4× the model dimension.

    WHY 4×? It gives the model more capacity to transform representations.
    Think of it as: attention says "these words are related", FFN says
    "here's what that relationship means."
    """
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


# =============================================================================
# THE TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    """
    A single Transformer block (GPT-style, pre-norm).

    Pre-norm means we normalize BEFORE the sub-layer, not after.
    Modern models (GPT-2+, Llama, etc.) all use pre-norm because
    it makes training more stable.
    """
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()

        # Sub-layer 1: Multi-Head Self-Attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # Sub-layer 2: Feed-Forward Network
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Sub-layer 1: Attention with residual connection
        # x + Attention(LayerNorm(x))
        normed = self.norm1(x)
        attended = self.attention(normed, mask)
        x = x + self.dropout(attended)   # ← RESIDUAL CONNECTION

        # Sub-layer 2: FFN with residual connection
        # x + FFN(LayerNorm(x))
        normed = self.norm2(x)
        fed_forward = self.ffn(normed)
        x = x + self.dropout(fed_forward)  # ← RESIDUAL CONNECTION

        return x


# =============================================================================
# FULL GPT-STYLE MODEL
# =============================================================================

class MiniGPT(nn.Module):
    """
    A minimal GPT-style Transformer for text generation.

    This is structurally identical to GPT-2 — just smaller.
    Stack of: Embedding → N × TransformerBlock → LayerNorm → Linear
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 max_seq_len, d_ff=None, dropout=0.1):
        super().__init__()

        # Token + position embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm + output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between embedding and output
        # This is a common trick — the model uses the same representation
        # for "understanding input tokens" and "predicting output tokens"
        self.head.weight = self.token_emb.weight

        self.max_seq_len = max_seq_len

    def forward(self, idx):
        """
        Args:
            idx: (batch, seq_len) tensor of token IDs

        Returns:
            logits: (batch, seq_len, vocab_size) — prediction for next token
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence too long: {T} > {self.max_seq_len}"

        # Embeddings
        tok_emb = self.token_emb(idx)                           # (B, T, d_model)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))  # (T, d_model)
        x = self.dropout(tok_emb + pos_emb)

        # Causal mask (lower triangular)
        mask = torch.tril(torch.ones(T, T, device=idx.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Output
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        return logits


# =============================================================================
# DEMO
# =============================================================================

print("MiniGPT — A Tiny GPT-style Transformer")
print("=" * 50)

# Model config (tiny, for demonstration)
config = {
    'vocab_size': 256,       # character-level: 256 ASCII characters
    'd_model': 128,          # embedding dimension
    'num_heads': 4,          # attention heads
    'num_layers': 4,         # transformer blocks
    'max_seq_len': 256,      # max sequence length
    'd_ff': 512,             # feed-forward hidden dim (4 × d_model)
    'dropout': 0.1,
}

model = MiniGPT(**config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel config:")
for k, v in config.items():
    print(f"  {k}: {v}")
print(f"\nTotal parameters: {total_params:,}")
print(f"Model size: ~{total_params * 4 / 1e6:.1f} MB (float32)")

# Forward pass
batch_size = 2
seq_len = 32
dummy_input = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
logits = model(dummy_input)

print(f"\nForward pass:")
print(f"  Input: {dummy_input.shape} (batch={batch_size}, seq_len={seq_len})")
print(f"  Output: {logits.shape} (batch={batch_size}, seq_len={seq_len}, vocab={config['vocab_size']})")
print(f"  → At each position, model predicts probability of next character")

# For comparison
print(f"""
COMPARISON WITH REAL MODELS:
┌──────────────────────────────────────────────────────┐
│ Model       │ Layers │ d_model │ Heads │ Parameters  │
├──────────────────────────────────────────────────────┤
│ MiniGPT     │ {config['num_layers']:5d}  │ {config['d_model']:6d}  │ {config['num_heads']:4d}  │ {total_params:>10,} │
│ GPT-2 Small │    12  │    768  │   12  │ 124,000,000 │
│ GPT-2 XL    │    48  │   1600  │   25  │ 1.5 billion │
│ GPT-3       │    96  │  12288  │   96  │ 175 billion │
│ Llama-2 7B  │    32  │   4096  │   32  │   7 billion │
└──────────────────────────────────────────────────────┘
Same architecture, just more of everything!
""")
