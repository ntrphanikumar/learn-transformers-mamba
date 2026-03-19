"""
Module 4.2: Mamba Block & Full Model
======================================

This builds the complete Mamba architecture:
1. Mamba Block = the repeating unit (like TransformerBlock)
2. MiniMamba = full model for text generation

MAMBA BLOCK STRUCTURE:
    x → Linear (expand) → Conv1D → SiLU → Selective SSM ──→ × → Linear (project) → + → out
    └→ Linear (expand) ──────────────────────────→ SiLU ──→ ×                       │
    └───────────────────────────────────────────────────────────────────────────────→ +
                                                                              (residual)

The "gated" architecture (two parallel paths multiplied together) is
borrowed from gated linear units. One path processes the sequence,
the other provides a gate to control information flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """
    A single Mamba block — the equivalent of a Transformer block.

    Key components:
    1. Input projection (expands dimension for more capacity)
    2. 1D convolution (local context, like a small receptive field)
    3. Selective SSM (the sequence model)
    4. Gating mechanism (controls information flow)
    5. Output projection (back to d_model dimension)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_model * expand  # expanded dimension
        self.d_state = d_state
        self.d_conv = d_conv

        # Input projections (two paths)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Convolution (captures local patterns, like n-grams)
        # This gives the model some local context before the SSM
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # depthwise: each channel independently
        )

        # Selective SSM parameters
        # A: state transition (fixed, initialized for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(self.d_inner, -1).clone()
        )

        # Input-dependent projections
        self.proj_b = nn.Linear(self.d_inner, d_state, bias=False)
        self.proj_c = nn.Linear(self.d_inner, d_state, bias=False)
        self.proj_delta = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # D: skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def selective_scan(self, x):
        """Run the selective SSM over a sequence."""
        B_size, L, D = x.shape
        N = self.d_state

        # Compute input-dependent parameters
        B = self.proj_b(x)                      # (batch, L, N)
        C = self.proj_c(x)                      # (batch, L, N)
        delta = F.softplus(self.proj_delta(x))   # (batch, L, D)

        A = -torch.exp(self.A_log)               # (D, N)

        # Sequential scan
        h = torch.zeros(B_size, D, N, device=x.device)
        outputs = []

        for t in range(L):
            x_t = x[:, t, :]
            delta_t = delta[:, t, :]
            B_t = B[:, t, :]
            C_t = C[:, t, :]

            A_bar = torch.exp(delta_t.unsqueeze(-1) * A)
            B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)

            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = (C_t.unsqueeze(1) * h).sum(-1) + self.D * x_t

            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # Project to 2× inner dimension (split into two paths)
        xz = self.in_proj(x)
        x_path, z_path = xz.chunk(2, dim=-1)  # each (batch, L, d_inner)

        # Path 1: Conv → SiLU → SSM
        x_path = x_path.transpose(1, 2)           # (batch, d_inner, L) for conv
        x_path = self.conv1d(x_path)[:, :, :x.shape[1]]  # trim padding
        x_path = x_path.transpose(1, 2)           # back to (batch, L, d_inner)
        x_path = F.silu(x_path)
        x_path = self.selective_scan(x_path)

        # Path 2: Gate
        z_path = F.silu(z_path)

        # Combine paths (gating)
        output = x_path * z_path

        # Project back to d_model
        output = self.out_proj(output)

        # Residual connection
        return output + residual


# =============================================================================
# FULL MAMBA MODEL
# =============================================================================

class MiniMamba(nn.Module):
    """
    A minimal Mamba model for text generation.

    Stack of: Embedding → N × MambaBlock → LayerNorm → Linear

    Structurally simpler than a Transformer:
    - No attention mechanism (no quadratic cost!)
    - No positional encoding (position is implicit in sequential processing)
    - Just: embedding → mamba blocks → prediction
    """
    def __init__(self, vocab_size, d_model, num_layers, d_state=16,
                 d_conv=4, expand=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (same trick as in Transformers)
        self.head.weight = self.embedding.weight

    def forward(self, idx):
        """
        idx: (batch, seq_len) token IDs
        Returns: (batch, seq_len, vocab_size) logits
        """
        x = self.embedding(idx)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        return self.head(x)


# =============================================================================
# DEMO & COMPARISON
# =============================================================================

print("MiniMamba — A Tiny Mamba Model")
print("=" * 50)

config = {
    'vocab_size': 256,
    'd_model': 128,
    'num_layers': 4,
    'd_state': 16,
    'd_conv': 4,
    'expand': 2,
}

model = MiniMamba(**config)
total_params = sum(p.numel() for p in model.parameters())

print(f"\nMamba config:")
for k, v in config.items():
    print(f"  {k}: {v}")
print(f"\nTotal parameters: {total_params:,}")

# Forward pass
batch_size, seq_len = 2, 32
dummy_input = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
logits = model(dummy_input)

print(f"\nForward pass:")
print(f"  Input:  {dummy_input.shape}")
print(f"  Output: {logits.shape}")

print(f"""
TRANSFORMER vs MAMBA COMPARISON:
┌────────────────────────────────────────────────────────────────┐
│ Feature              │ Transformer          │ Mamba             │
├────────────────────────────────────────────────────────────────┤
│ Sequence modeling    │ Attention (global)   │ SSM (recurrent)   │
│ Complexity           │ O(n²) in seq length  │ O(n) linear       │
│ Position encoding    │ Explicit (learned)   │ Implicit (state)  │
│ Long sequences       │ Expensive            │ Efficient          │
│ Parallelism          │ Fully parallel       │ Scan (sequential)  │
│ Memory (inference)   │ KV cache grows       │ Fixed state size   │
│ Architecture         │ Attention + FFN      │ Conv + SSM + Gate  │
│ Training             │ Well understood      │ Newer, evolving    │
└────────────────────────────────────────────────────────────────┘

WHEN TO USE WHICH:
- Transformer: Shorter sequences, need strong global attention
- Mamba: Long sequences, efficiency matters, sequential data
- Hybrid (Jamba, etc.): Mix both for best of both worlds
""")
