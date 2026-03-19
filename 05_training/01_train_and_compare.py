"""
Module 5.1: Train & Compare — Transformer vs Mamba on Text Generation
=======================================================================

This is where it all comes together. We train BOTH models on the same
character-level text generation task and compare them.

Task: Given a sequence of characters, predict the next character.
Data: A small text corpus (Shakespeare or similar).

This is exactly how GPT works, just at character level instead of
token level (for simplicity and speed).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# =============================================================================
# MODELS (copied for self-contained script)
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
        out = (F.softmax(scores, dim=-1) @ V)
        return self.W_o(out.transpose(1, 2).contiguous().view(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.max_seq_len = max_seq_len

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv - 1, groups=self.d_inner)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1).clone())
        self.proj_b = nn.Linear(self.d_inner, d_state, bias=False)
        self.proj_c = nn.Linear(self.d_inner, d_state, bias=False)
        self.proj_delta = nn.Linear(self.d_inner, self.d_inner)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)
        x_path = F.silu(self.conv1d(x_path.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2))

        # Selective scan
        B_sz, L, D = x_path.shape
        N = self.d_state
        B = self.proj_b(x_path)
        C = self.proj_c(x_path)
        delta = F.softplus(self.proj_delta(x_path))
        A = -torch.exp(self.A_log)
        h = torch.zeros(B_sz, D, N, device=x_path.device)
        outs = []
        for t in range(L):
            xt = x_path[:, t]
            dt = delta[:, t]
            A_bar = torch.exp(dt.unsqueeze(-1) * A)
            B_bar = dt.unsqueeze(-1) * B[:, t].unsqueeze(1)
            h = A_bar * h + B_bar * xt.unsqueeze(-1)
            outs.append((C[:, t].unsqueeze(1) * h).sum(-1) + self.D * xt)

        x_path = torch.stack(outs, dim=1)
        return self.out_proj(x_path * F.silu(z)) + residual


class MiniMamba(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, d_state=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, idx):
        x = self.emb(idx)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


# =============================================================================
# TRAINING SETUP
# =============================================================================

def get_training_data(seq_len=64, num_batches=200, batch_size=16):
    """
    Generate a simple training dataset.
    We'll generate text with clear patterns that the model should learn.
    """
    # Create a simple pattern-based dataset
    # Pattern: repeated phrases with slight variation
    corpus = ""
    phrases = [
        "the quick brown fox jumps over the lazy dog ",
        "a fast red cat leaps across the slow mouse ",
        "the big green frog hops over the small pond ",
        "one bright blue bird flies above the dark cloud ",
    ]
    # Repeat to get enough data
    while len(corpus) < seq_len * num_batches * batch_size:
        for phrase in phrases:
            corpus += phrase

    # Character-level encoding
    chars = sorted(set(corpus))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    data = torch.tensor([char_to_idx[c] for c in corpus])

    # Create batches
    batches = []
    for i in range(num_batches):
        start = i * batch_size * seq_len
        batch_x = []
        batch_y = []
        for b in range(batch_size):
            offset = start + b * seq_len
            if offset + seq_len + 1 > len(data):
                break
            batch_x.append(data[offset:offset + seq_len])
            batch_y.append(data[offset + 1:offset + seq_len + 1])
        if len(batch_x) == batch_size:
            batches.append((torch.stack(batch_x), torch.stack(batch_y)))

    return batches, len(chars), char_to_idx, idx_to_char


def train_model(model, batches, epochs=3, lr=3e-4, model_name="Model"):
    """Train a model and return loss history."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []

    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in batches:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(batches)
        losses.append(avg_loss)
        elapsed = time.time() - start_time
        print(f"  [{model_name}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {elapsed:.1f}s")

    return losses


@torch.no_grad()
def generate(model, idx_to_char, char_to_idx, prompt="the ", max_len=100, temperature=0.8):
    """Generate text from a trained model."""
    model.eval()
    tokens = [char_to_idx.get(c, 0) for c in prompt]
    idx = torch.tensor([tokens])

    for _ in range(max_len):
        # Only use last 64 tokens (context window)
        idx_cond = idx[:, -64:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_token], dim=1)

    return ''.join(idx_to_char.get(t.item(), '?') for t in idx[0])


# =============================================================================
# MAIN: Train and compare
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING: Transformer vs Mamba on Character-Level Text")
    print("=" * 60)

    # Prepare data
    print("\nPreparing data...")
    batches, vocab_size, c2i, i2c = get_training_data()
    print(f"Vocab size: {vocab_size} characters")
    print(f"Batches: {len(batches)}")

    d_model = 64
    num_layers = 3

    # Create models
    gpt = MiniGPT(vocab_size, d_model, num_heads=4, num_layers=num_layers, max_seq_len=64)
    mamba = MiniMamba(vocab_size, d_model, num_layers=num_layers, d_state=16)

    gpt_params = sum(p.numel() for p in gpt.parameters())
    mamba_params = sum(p.numel() for p in mamba.parameters())
    print(f"\nMiniGPT  parameters: {gpt_params:,}")
    print(f"MiniMamba parameters: {mamba_params:,}")

    # Train Transformer
    print(f"\n--- Training MiniGPT (Transformer) ---")
    gpt_losses = train_model(gpt, batches, epochs=5, model_name="GPT")

    # Train Mamba
    print(f"\n--- Training MiniMamba ---")
    mamba_losses = train_model(mamba, batches, epochs=5, model_name="Mamba")

    # Generate samples
    print(f"\n{'='*60}")
    print("GENERATION SAMPLES")
    print(f"{'='*60}")

    print(f"\nMiniGPT generates:")
    print(f"  '{generate(gpt, i2c, c2i)}'")

    print(f"\nMiniMamba generates:")
    print(f"  '{generate(mamba, i2c, c2i)}'")

    # Compare
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  MiniGPT  — Final loss: {gpt_losses[-1]:.4f}, Params: {gpt_params:,}")
    print(f"  MiniMamba — Final loss: {mamba_losses[-1]:.4f}, Params: {mamba_params:,}")

    print(f"""
WHAT YOU'VE LEARNED:
1. Transformers use attention to relate all tokens (powerful but O(n²))
2. Mamba uses selective state spaces (efficient, O(n))
3. Both can learn the same patterns on small tasks
4. At scale, Mamba's linear cost becomes a big advantage

NEXT STEPS:
- Try larger d_model and more layers
- Use a real text corpus (download Shakespeare from the internet)
- Add learning rate scheduling and proper evaluation
- Try the Hugging Face 'transformers' library for pre-trained models
- Read the papers:
  * "Attention Is All You Need" (Vaswani et al., 2017)
  * "Mamba: Linear-Time Sequence Modeling" (Gu & Dao, 2023)
""")
