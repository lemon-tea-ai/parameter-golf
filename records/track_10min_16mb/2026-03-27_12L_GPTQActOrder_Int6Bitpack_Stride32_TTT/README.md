# Star 8: SGD TTT + GPTQ ActOrder + IZL + Int6 Bitpacking + Stride-32

**Target BPB:** ~1.110 (conservative) — beat SOTA 1.1194

**Status:** Pending 8×H100 verification (3-seed runs: 42, 1337, 2023)

## Innovations over SOTA (1.1194 BPB)

### 1. GPTQ with ActOrder (quantization)
Replaces naive per-row int6 with Hessian-sorted column-wise optimal rounding:

- Hessian H = X^T X collected via forward hooks on 256 calibration samples
- Columns sorted by Hessian diagonal magnitude (most important first = quantized last)
- Cholesky decomposition for efficient block-wise error propagation (block_size=128)
- Falls back to naive int6 on Cholesky failure or dimension mismatch
- **Expected improvement:** −0.002 to −0.004 BPB over naive int6

### 2. Int6 Bitpacking (compression)
Packs 4 int6 values into 3 bytes (24 bits), saving 25% vs int8 storage:
```
byte0 = (v0 << 2) | (v1 >> 4)
byte1 = ((v1 & 0xF) << 4) | (v2 >> 2)
byte2 = ((v2 & 0x3) << 6) | (v3 & 0x3F)
```
Frees ~1.5 MB of artifact budget for an extra transformer layer.

### 3. 12 Transformer Layers (up from 11)
Budget freed by bitpacking allows a 12th layer (6 encoder + 6 decoder with skip connections).
- VE embeddings at layers 10, 11
- XSA on last 4 layers (8, 9, 10, 11)

### 4. Stride-32 Sliding Window Eval (down from 64)
Every token appears in more windows → better BPB estimate with maximum context.
- **Expected improvement:** −0.001 to −0.002 BPB

### 5. Zstd-22 Compression (replacing lzma)
Better compression ratio with faster decompression. Falls back to zlib if zstandard not installed.

### 6. SGD TTT — All Blocks Unfrozen
Legal score-first TTT (PR #461) with SGD+momentum. All blocks adapt (ttt_freeze_blocks=0), matching IZL architecture's full adaptation. TTT LR=0.001, grad_clip=1.8.

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 12 (6 encoder + 6 decoder) |
| Model dim | 512 |
| Heads / KV heads | 8 / 4 |
| MLP mult | 3.0 |
| BigramHash | 2048 |
| XSA | Last 4 layers |
| VE128 | Layers 10, 11 |
| Activation | LeakyReLU(0.5)² |
| Optimizer | Parallel Muon + AdamW |
| TTT | SGD, all blocks, LR=0.001, 3 epochs |

## Run Command (8×H100)

```bash
NUM_LAYERS=12 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
TTT_ENABLED=1 TTT_LR=0.001 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.8 \
MATRIX_LR=0.025 SCALAR_LR=0.015 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMUP_STEPS=1500 \
ITERATIONS=3337 EVAL_STRIDE=32 \
GPTQ_ACTORDER=1 GPTQ_CALIB_SAMPLES=256 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Timing Budget

| Phase | Time |
|-------|------|
| Training (3337 steps) | ~600s |
| GPTQ calibration (256 samples) | ~7s |
| Standard int6 roundtrip eval | ~1s |
| Sliding window eval (stride-32) | ~130s |
| Legal TTT (score-first + adapt) | ~480s |
| **Total eval** | **~620s (~10 min)** |

## Expected Impact Summary

| Innovation | BPB delta |
|------------|-----------|
| GPTQ ActOrder | −0.002 to −0.004 |
| 12th layer | −0.003 to −0.004 |
| Stride-32 eval | −0.001 to −0.002 |
| TTT all-unfrozen | −0.001 to −0.002 |
| **Conservative total** | **~−0.009 → ~1.110** |

## Credits

- IZL model + architecture: PR #414 by @alpnalyush
- TTT (Score-First SGD): PR #461 recipe by @babybektursun
- GPTQ recipe inspiration: PR #601 by @Christopher-Lee-McLendon
- Zstd compression: PR #634 by @Quatlad3
- Parallel Muon: PR #393 by @Christopher-Lee-McLendon
- Previous SOTA: @abaybektursun (1.1194 BPB)
