# Star 8: 11L GPTQ-ActOrder + Int6 Bitpacking + Stride-32 TTT

**Target BPB:** ~1.114 (conservative) — beat SOTA 1.1194

**Status:** Pending 8×H100 verification (3-seed runs: 42, 1337, 2023)

## Innovations over SOTA (1.1194 BPB)

### 1. GPTQ with ActOrder (quantization)
Replaces naive per-row int6 with Hessian-sorted column-wise optimal rounding:

- Hessian H = X^T X collected via forward hooks on 256 calibration samples (~7s)
- Columns sorted by Hessian diagonal magnitude (most important quantized last)
- Cholesky decomposition for efficient block-wise error propagation (block_size=256)
- Falls back to naive int6 on Cholesky failure or dimension mismatch
- **Expected improvement:** −0.003 to −0.004 BPB over naive int6

### 2. Int6 Bitpacking (compression)
Packs 4 int6 values into 3 bytes (24 bits), saving 25% vs int8 storage:
```
byte0 = (v0 << 2) | (v1 >> 4)
byte1 = ((v1 & 0xF) << 4) | (v2 >> 2)
byte2 = ((v2 & 0x3) << 6) | (v3 & 0x3F)
```

### 3. Stride-32 TTT Eval
TTT sliding window uses stride=32 (vs SOTA's stride=64) for ~2× more adaptation steps.
Static SW skipped when TTT enabled — eval time dominated by TTT only.
- TTT LR=0.002 (matches proven SOTA recipe), grad_clip=1.8, 3 epochs
- All blocks unfrozen (`ttt_freeze_blocks=0`)
- **Expected improvement:** −0.001 to −0.002 BPB

### 4. Zstd-22 Compression (replacing lzma)
Better compression ratio. Falls back to zlib if zstandard not installed.

## Timing Budget Analysis

Two separate 10-minute budgets per the challenge rules.

### Training (≤600s)
| Phase | Time |
|-------|------|
| Warmup (20 steps, reset) | ~2s |
| Main training (~7200 steps @ ~83ms) | ~598s |
| **Total** | **~600s** ✓ |

`max_wallclock_seconds=600` caps training automatically.

### Eval (≤600s)
| Phase | Time |
|-------|------|
| Post-EMA diagnostic eval | ~2s |
| GPTQ calibration (256 samples) | ~7s |
| Int6 quantize + compress + save | ~3s |
| Int6 roundtrip eval | ~5s |
| Legal TTT sliding window (stride=32, 11L) | ~521s |
| **Total** | **~538s** ✓ |

Key design choices to fit eval within 600s:
- **Static sliding window skipped when TTT enabled** — TTT subsumes it and is the official BPB
- **stride=32** for TTT only (no static SW overhead)
- **`max_eval_wallclock_seconds=580`** — soft guard: skips TTT if budget is dangerously tight

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (IZL U-Net style) |
| Model dim | 512 |
| Heads / KV heads | 8 / 4 |
| MLP mult | 3.0 |
| BigramHash | 2048 |
| XSA | Last 4 layers |
| VE128 | Layers 9, 10 |
| Activation | LeakyReLU(0.5)² |
| Optimizer | Parallel Muon + AdamW |
| TTT | SGD, all blocks, LR=0.002, 3 epochs, stride=32 |
| GPTQ | ActOrder, 256 calib samples, block_size=256 |

## Run Command (8×H100)

```bash
BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.8 \
MATRIX_LR=0.025 SCALAR_LR=0.015 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
GPTQ_ACTORDER=1 GPTQ_CALIB_SAMPLES=256 GPTQ_BLOCK_SIZE=256 \
EVAL_STRIDE=32 MAX_EVAL_WALLCLOCK_SECONDS=580 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Impact Summary

| Innovation | BPB delta |
|------------|-----------|
| GPTQ ActOrder (256 samples, bs=256) | −0.003 to −0.004 |
| Stride-32 TTT (vs stride-64) | −0.001 to −0.002 |
| TTT LR=0.002 (proven SOTA recipe) | −0.000 to −0.001 |
| **Conservative total** | **~−0.005 → ~1.114** |

## Credits

- IZL model + architecture: PR #414 by @alpnalyush
- TTT (Score-First SGD): PR #461 recipe by @babybektursun
- GPTQ recipe inspiration: PR #601 by @Christopher-Lee-McLendon
- Zstd compression: PR #634 by @Quatlad3
- Parallel Muon: PR #393 by @Christopher-Lee-McLendon
- Previous SOTA: @abaybektursun (1.1194 BPB)
