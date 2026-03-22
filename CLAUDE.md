# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Parameter Golf** is an OpenAI Model Craft Challenge: train the best language model that fits in a **16MB artifact** (code + compressed model) and trains in under **10 minutes on 8×H100s**, evaluated by compression on the FineWeb validation set (bits-per-byte / BPB — lower is better).

The challenge targets L(N) scaling: minimize loss for a fixed parameter count, unconstrained by architecture or training recipe.

## Commands

```bash
# Setup (Apple Silicon)
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm torch

# Download data (sp1024 tokenizer variant)
python3 data/cached_challenge_fineweb.py --variant sp1024              # full (80 shards, 8B tokens)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1  # small local subset

# MLX training (Apple Silicon — fast local iteration)
RUN_ID=smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 python3 train_gpt_mlx.py

# PyTorch training (single GPU)
torchrun --standalone --nproc_per_node=1 train_gpt.py

# PyTorch training with custom config; VAL_LOSS_EVERY=200 for periodic logs; MAX_WALLCLOCK_SECONDS=0 to disable cap
RUN_ID=run1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 NUM_LAYERS=9 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
```

All hyperparameters are controlled via environment variables (no CLI flags). There are no lint or test commands — the codebase is two training scripts and a data pipeline.

## Architecture

### Training Scripts

- **`train_gpt.py`**: PyTorch / CUDA, primary submission script. Hard limit: 1500 lines. Supports DDP via `torchrun`.
- **`train_gpt_mlx.py`**: MLX port for Apple Silicon local development. Mirrors `train_gpt.py` semantics; uses eager evaluation + microbatching for 16GB machines.

Both scripts are self-contained: model definition, training loop, validation, quantization, and serialization are all inline.

### Model Architecture (baseline)

- 9 transformer blocks, 512 dim, 8 heads / 4 KV heads (Grouped Query Attention)
- RMSNorm, RoPE positional embeddings
- relu² MLP activation, 2× expansion ratio
- Tied embeddings (token embedding = output projection) — reduces parameters
- Per-block learnable scalars: `resid_mix`, `attn_scale`, `mlp_scale`

### Optimizer

- **Muon** for all 2D weight matrices (orthogonal updates via Newton-Schulz iterations)
- **AdamW** for embeddings, output head, and scalars — separate param groups with independent LRs
- Env vars: `EMBED_LR`, `HEAD_LR`, `MATRIX_LR`, `SCALAR_LR`, `MUON_MOMENTUM`, `BETA1`, `BETA2`

### Quantization & Serialization (for 16MB budget)

Post-training, the model is quantized and compressed:
- Int8 per-row scaling for 2D tensors; per-tensor for vectors/scalars
- Small float tensors (<65KB) kept in FP16
- Final artifact = zlib-compressed serialized model + training script

The artifact must be **<16MB decimal** (not MiB) and fully self-contained (no network calls at eval time).

### Data

- Format: binary shards, 256 int32 header + uint16 tokens
- Location: `data/datasets/fineweb10B_sp1024/` (train + val splits)
- Tokenizer: SentencePiece BPE, 1024 vocab — `data/tokenizers/fineweb_1024_bpe.model`
- Evaluation: fixed first-50k FineWeb documents, bits-per-byte metric

### Submissions

`records/track_10min_16mb/` contains timestamped SOTA submissions, each with:
- `train_gpt.py` — the modified training script
- `submission.json` — metadata (author, val_bpb, artifact_size)
- `README.md` — approach description and ablations
- `train_seedN.log` — logs for 3 seeds (required for p<0.01 significance test)

New SOTA must beat prior best by **≥0.005 nats** at p<0.01 significance over seeds {42, 1337, 2024}. Evaluation on 8×H100s also has a 10-minute limit (separate from training). You cannot access validation data during training.

## Key Env Vars Reference

| Variable | Default | Description |
|---|---|---|
| `NUM_LAYERS` | 9 | Transformer depth |
| `MODEL_DIM` | 512 | Hidden dimension |
| `NUM_HEADS` / `NUM_KV_HEADS` | 8 / 4 | Attention heads (GQA) |
| `MLP_MULT` | 2 | MLP expansion ratio |
| `VOCAB_SIZE` | 1024 | Must match tokenizer |
| `TRAIN_SEQ_LEN` | 1024 | Sequence length |
| `ITERATIONS` | (computed) | Training steps |
| `MAX_WALLCLOCK_SECONDS` | 580 | Wall-clock cap |

## Techniques from Top Submissions

- Mixed Int5/Int6 quantization (Int5 for MLPs, Int6 for attention) to stay under 16MB with larger models
- BigramHash embeddings — hash consecutive token pairs into a larger embedding table
- Stochastic Weight Averaging (SWA) starting at ~40% through training
- U-Net skip connections between early and late blocks
- Sliding-window validation (stride=64) instead of fixed non-overlapping sequences
- Spectral / orthogonal weight initialization
