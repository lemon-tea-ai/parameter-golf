# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Parameter Golf** is an open competition to train the best language model that fits within a **16MB artifact** and trains in under **10 minutes on 8xH100s**, evaluated by compression on the FineWeb validation set (bits per byte / BPB). Lower BPB = better. Challenge runs March 18 – April 30, 2026.

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Download and prepare data (run once)
python data/cached_challenge_fineweb.py       # downloads FineWeb dataset chunks
python data/download_hf_docs_and_tokenize.py  # prepares tokenizer

# Run training locally (Mac/Apple Silicon)
python train_gpt_mlx.py

# Run training on GPU cluster (8xH100)
torchrun --nproc_per_node=8 train_gpt.py
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `train_gpt.py` | Main PyTorch training script for distributed GPU (torchrun + DDP) |
| `train_gpt_mlx.py` | Apple Silicon MLX version for local iteration |

Both scripts use environment variables to configure all hyperparameters (defined in the `HyperParameters` class at the top of each file). The scripts self-document their parameters — read them directly.

## Architecture

**Model**: GPT with RoPE positional embeddings, GQA (grouped query attention), tied input/output embeddings, configurable depth/width.

**Optimizer**: Muon (Newton-Schulz orthogonalization) for weight matrices + AdamW for embeddings/biases/scalars.

**Evaluation**: After training, validation BPB is computed on the FineWeb validation set. The model weights are serialized and checked against the 16MB limit.

**Key hyperparameters** (set via env vars or modifying defaults in the script):
- Architecture: `model_dim`, `num_layers`, `num_heads`, `num_kv_heads`, `mlp_mult`
- Training: `batch_size`, `grad_accum`, `learning_rate`, `warmup_steps`
- Techniques: quantization (`qat_int6`, `qat_int8`), weight averaging (`ema_alpha`, `swa_start_step`)

## Submission Structure

Submissions live under `records/`:
- `records/track_10min_16mb/` — official competition submissions
- `records/track_non_record_16mb/` — experimental / unlimited compute

Each submission folder (`YYYY-MM-DD_description/`) must contain:
1. `train_gpt.py` — the modified training script
2. `submission.json` — metadata: `name`, `val_bpb`, `bytes_total`, `author`, `date`, `description`
3. `README.md` — ablations, methodology, results across ≥3 random seeds

## Submission Requirements

- Beat current SOTA by ≥0.005 nats (BPB), with p < 0.01 statistical significance
- Must reproduce in <10 min on 8xH100s with ≥3 random seeds
- Model artifact (weights) must be ≤16MB on disk
- Evaluation restricted to FineWeb validation set (no test set leakage)

## Data Pipeline

The `data/` directory handles dataset acquisition:
- `cached_challenge_fineweb.py` — downloads pre-tokenized FineWeb chunks from HuggingFace
- `tokenizer_specs.json` — defines supported tokenizer variants (GPT-2 BPE, etc.)
- Downloaded data goes to `data/datasets/` (gitignored)

## What Participants Typically Modify

Looking at the `records/` submissions, the main levers are:
1. **Architecture changes**: depth vs. width tradeoffs, attention variants (MLA, GQA ratios), MLP designs (MoE, GLU variants)
2. **Optimizer changes**: Muon variants, learning rate schedules, gradient clipping
3. **Quantization**: QAT int6/int8 for packing more parameters into 16MB
4. **Training dynamics**: weight averaging strategies, warmup schedules, batch size scaling
5. **Novel techniques**: depth recurrence, test-time compute, 1-bit/ternary weights
