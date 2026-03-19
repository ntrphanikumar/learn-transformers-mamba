"""
Module 4.1: State Space Models & Mamba — The Alternative to Attention
======================================================================

WHY MAMBA EXISTS:
Transformers have a fatal flaw: attention is O(n²) in sequence length.
  - 1,000 tokens → 1M attention computations
  - 10,000 tokens → 100M attention computations
  - 100,000 tokens → 10B attention computations

This quadratic cost makes long sequences expensive. Mamba achieves
LINEAR scaling — O(n) — while matching Transformer quality.

THE KEY IDEAS:
1. State Space Models (SSMs): Model sequences as continuous-time systems
   (borrowed from control theory / signal processing)
2. Discretization: Convert continuous-time to discrete steps
3. Selective State Spaces (Mamba's innovation): Make the state transition
   INPUT-DEPENDENT, so the model can choose what to remember/forget

ANALOGY:
- Transformer: At each word, look back at ALL previous words (expensive)
- RNN: At each word, update a fixed-size hidden state (limited memory)
- Mamba: At each word, selectively update a structured state (best of both)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. STATE SPACE MODELS: The continuous-time foundation
# =============================================================================

print("=" * 60)
print("STATE SPACE MODELS (SSMs)")
print("=" * 60)

print("""
A State Space Model defines how a hidden state h evolves over time:

  Continuous form (from control theory):
    h'(t) = A·h(t) + B·x(t)     ← state update (how input affects state)
    y(t)  = C·h(t) + D·x(t)     ← output (how state maps to output)

  Where:
    x(t) = input at time t
    h(t) = hidden state at time t (the model's "memory")
    y(t) = output at time t
    A = state transition matrix (N×N) — how state evolves
    B = input projection (N×1) — how input enters the state
    C = output projection (1×N) — how state maps to output
    D = skip connection (usually ignored)

  INTUITION: The state h is like a summary of everything seen so far.
  Matrix A controls how quickly old information decays.
  Matrix B controls what new information gets stored.
  Matrix C controls what information gets read out.
""")


# =============================================================================
# 2. DISCRETIZATION: From continuous to step-by-step
# =============================================================================

print("DISCRETIZATION")
print("=" * 60)

print("""
Neural networks process DISCRETE sequences (word by word), not continuous
signals. We need to convert the continuous SSM to a discrete recurrence:

  Discrete form (what we actually compute):
    h[k] = Ā·h[k-1] + B̄·x[k]
    y[k] = C·h[k]

  Where Ā and B̄ are "discretized" versions of A and B.
  The discretization step size Δ controls the "resolution".

  Common method (Zero-Order Hold):
    Ā = exp(Δ·A)
    B̄ = (Δ·A)⁻¹ · (exp(Δ·A) - I) · Δ·B

  Simplified (Euler method, easier to understand):
    Ā = I + Δ·A
    B̄ = Δ·B
""")


class SimpleSSM(nn.Module):
    """
    A simple State Space Model (non-selective, for understanding).

    This is the S4 model's core idea: fixed A, B, C matrices
    that process any sequence recurrently.
    """
    def __init__(self, d_model, state_size=16):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size  # N: dimension of hidden state

        # SSM parameters (learned, but fixed for all inputs)
        # A is initialized with a special structure for stability
        self.A = nn.Parameter(torch.randn(d_model, state_size))
        self.B = nn.Parameter(torch.randn(d_model, state_size))
        self.C = nn.Parameter(torch.randn(d_model, state_size))
        self.delta = nn.Parameter(torch.ones(d_model) * 0.1)  # step size

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        B_size, L, D = x.shape

        # Discretize: convert continuous A, B to discrete Ā, B̄
        delta = F.softplus(self.delta)  # ensure positive step size
        A_discrete = torch.exp(delta.unsqueeze(-1) * self.A)  # (D, N)
        B_discrete = delta.unsqueeze(-1) * self.B              # (D, N)

        # Run the recurrence
        h = torch.zeros(B_size, D, self.state_size, device=x.device)
        outputs = []

        for t in range(L):
            # h[t] = Ā * h[t-1] + B̄ * x[t]
            x_t = x[:, t, :]  # (batch, D)
            h = A_discrete * h + B_discrete * x_t.unsqueeze(-1)

            # y[t] = C * h[t]
            y_t = (self.C * h).sum(dim=-1)  # (batch, D)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, D)


# Demo
print("\nSimple SSM Demo:")
batch, seq_len, d_model = 2, 20, 32
ssm = SimpleSSM(d_model, state_size=16)
x = torch.randn(batch, seq_len, d_model)
y = ssm(x)
print(f"  Input:  {x.shape}")
print(f"  Output: {y.shape}")
print(f"  State size: 16 (the model's memory capacity per feature)")
params = sum(p.numel() for p in ssm.parameters())
print(f"  Parameters: {params:,}\n")


# =============================================================================
# 3. THE PROBLEM WITH FIXED SSMs → WHY MAMBA
# =============================================================================

print("THE PROBLEM: Fixed SSMs can't be selective")
print("=" * 60)
print("""
In a simple SSM, matrices A, B, C are the SAME for all inputs.
The model processes "important" and "unimportant" tokens identically.

Example: "The cat, which was very fluffy and had been sleeping on
         the old couch for hours, finally sat on the mat."

A fixed SSM treats every word the same — it can't "decide" to
focus on "cat" and "sat" while ignoring filler words.

MAMBA'S KEY INSIGHT: Make B, C, and Δ depend on the INPUT.
→ The model can selectively remember or forget information.
→ "Selection mechanism" = content-aware state transitions.
""")


# =============================================================================
# 4. SELECTIVE SSM (Mamba's core innovation)
# =============================================================================

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model — the core of Mamba.

    KEY DIFFERENCE from SimpleSSM:
    B, C, and delta are COMPUTED FROM THE INPUT, not fixed parameters.
    This lets the model dynamically choose what to store in its state.
    """
    def __init__(self, d_model, state_size=16):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size

        # A remains a fixed parameter (initialized for stability)
        # Using negative values so exp(Δ·A) < 1 (state decays, not explodes)
        A = torch.arange(1, state_size + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(d_model, -1))

        # These LINEAR PROJECTIONS compute B, C, Δ from the input
        # THIS is the "selective" part — parameters depend on input
        self.proj_b = nn.Linear(d_model, state_size)   # input → B
        self.proj_c = nn.Linear(d_model, state_size)   # input → C
        self.proj_delta = nn.Linear(d_model, d_model)  # input → Δ

        # D: skip connection (direct input-to-output)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        B_size, L, D = x.shape
        N = self.state_size

        # Compute input-dependent parameters
        B = self.proj_b(x)                              # (batch, L, N)
        C = self.proj_c(x)                              # (batch, L, N)
        delta = F.softplus(self.proj_delta(x))           # (batch, L, D)

        # Fixed A (negative for stability)
        A = -torch.exp(self.A_log)                       # (D, N)

        # Run selective scan (recurrence)
        h = torch.zeros(B_size, D, N, device=x.device)
        outputs = []

        for t in range(L):
            x_t = x[:, t, :]         # (batch, D)
            delta_t = delta[:, t, :]  # (batch, D)
            B_t = B[:, t, :]          # (batch, N)
            C_t = C[:, t, :]          # (batch, N)

            # Discretize with input-dependent delta
            A_discrete = torch.exp(delta_t.unsqueeze(-1) * A)  # (batch, D, N)
            B_discrete = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (batch, D, N)

            # State update: h = Ā * h + B̄ * x
            h = A_discrete * h + B_discrete * x_t.unsqueeze(-1)

            # Output: y = C * h + D * x
            y_t = (C_t.unsqueeze(1) * h).sum(dim=-1)  # (batch, D)
            y_t = y_t + self.D * x_t                    # skip connection

            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


# Demo
print("Selective SSM (Mamba-style) Demo:")
selective_ssm = SelectiveSSM(d_model, state_size=16)
y_selective = selective_ssm(x)
print(f"  Input:  {x.shape}")
print(f"  Output: {y_selective.shape}")
params = sum(p.numel() for p in selective_ssm.parameters())
print(f"  Parameters: {params:,}")

print("""
KEY TAKEAWAY:
┌──────────────────────────────────────────────────────┐
│ Simple SSM: Same A,B,C for all tokens → can't select │
│ Selective SSM: B,C,Δ depend on input → CAN select   │
│                                                       │
│ Selection lets Mamba:                                 │
│   - Focus on important tokens                         │
│   - Ignore irrelevant filler                          │
│   - Adaptively control memory retention               │
└──────────────────────────────────────────────────────┘
""")
