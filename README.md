# Learn Transformers & Mamba — From Scratch

Build Transformer and Mamba architectures from first principles using PyTorch. No API calls, no pre-trained models — you write the neural networks yourself and understand every layer.

## Why This Project

Most AI tutorials stop at `openai.chat.completions.create()`. This project goes deeper — you implement the actual architectures that power GPT, Llama, and Mamba, understanding *why* each component exists, *when* you'd use it, and *how* it works mathematically.

## What You'll Build

By the end, you'll have implemented from scratch:
- A **GPT-style Transformer** (same architecture as GPT-2/3, just smaller)
- A **Mamba model** (the linear-time alternative to Transformers)
- Trained both on the **same task** and compared their behavior

## Learning Path

### Module 1: Foundations (`01_foundations/`)

The atoms of deep learning — every concept here is used in every layer of every model.

| File | What You Learn | Real-World Connection |
|------|---------------|----------------------|
| `01_tensors.py` | Tensors, matrix multiplication, softmax, layer normalization | Spam classifiers, ChatGPT temperature, recommendation systems |
| `02_embeddings.py` | Token vocabularies, embedding lookup tables | How "cat" becomes a vector of 4,096 numbers in Llama |
| `03_positional_encoding.py` | Sinusoidal encoding, learned positions | Why "dog bites man" ≠ "man bites dog" to a Transformer |

### Module 2: Attention (`02_attention/`)

The breakthrough idea that makes Transformers work — letting every word look at every other word.

| File | What You Learn | Real-World Connection |
|------|---------------|----------------------|
| `01_attention.py` | Scaled dot-product attention, causal masking | How "sat" knows to attend to "cat" in "the cat sat on the mat" |
| `02_multi_head_attention.py` | Multiple parallel attention heads, self vs cross attention | Why models can track syntax, semantics, and position simultaneously |

### Module 3: Transformer (`03_transformer/`)

Assemble the building blocks into a complete GPT-style model.

| File | What You Learn | Real-World Connection |
|------|---------------|----------------------|
| `01_transformer_block.py` | Transformer block, residual connections, feed-forward network, full MiniGPT model | GPT-2 = 12 of these blocks stacked. GPT-3 = 96. Same block, more of them. |

### Module 4: Mamba (`04_mamba/`)

The linear-time alternative — why attention's O(n²) cost is a problem and how state space models solve it.

| File | What You Learn | Real-World Connection |
|------|---------------|----------------------|
| `01_ssm_basics.py` | State space models, discretization, fixed vs selective SSMs | Why 100K-token contexts are expensive for Transformers but cheap for Mamba |
| `02_mamba_block.py` | Mamba block (conv + selective SSM + gating), full MiniMamba model | The architecture behind Mamba-1/2, Jamba, and hybrid models |

### Module 5: Training (`05_training/`)

Train both models on the same task and compare.

| File | What You Learn | Real-World Connection |
|------|---------------|----------------------|
| `01_train_and_compare.py` | Training loop, loss computation, text generation, Transformer vs Mamba comparison | Exactly how GPT learns to predict the next character/token |

## Quick Comparison

```
┌────────────────────────────────────────────────────────────────┐
│ Feature              │ Transformer          │ Mamba             │
├────────────────────────────────────────────────────────────────┤
│ Core mechanism       │ Attention (global)   │ SSM (recurrent)   │
│ Sequence cost        │ O(n²)               │ O(n) linear       │
│ Position encoding    │ Explicit (learned)   │ Implicit (state)  │
│ Long sequences       │ Expensive            │ Efficient          │
│ Parallelism          │ Fully parallel       │ Sequential scan    │
│ Memory at inference  │ KV cache grows       │ Fixed state size   │
└────────────────────────────────────────────────────────────────┘
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, numpy, matplotlib

## How to Use

Each module has numbered files. Run them in order — each builds on the previous:

```bash
python 01_foundations/01_tensors.py
python 01_foundations/02_embeddings.py
# ... continue through all modules
python 05_training/01_train_and_compare.py
```

Every file is:
- **Self-contained** — runs independently, no imports between modules
- **Heavily commented** — explains the *why*, not just the *what*
- **Real-world grounded** — each concept tied to where it's used in production models
- **Has exercises with answers** — validate your understanding

## Prerequisites

- **Python** — comfortable writing classes and functions
- **Basic linear algebra** — matrix multiply, transpose, dot product
- **Basic calculus** — what a gradient is (you don't need to compute them; PyTorch does that)

No prior ML/AI experience needed. The modules build up from zero.

## Key Papers

If you want to go deeper after completing the modules:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) — the original Transformer
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023) — the Mamba architecture
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019) — GPT-2
