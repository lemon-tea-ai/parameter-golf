# QAT + Curriculum Learning + Multi-Token Prediction

## Score: TBD (pending 3-seed validation on 8×H100)

Built on top of the current SOTA (`2026-03-20_10L_Int5MLP_MuonWD04_SWA50`, val_bpb=1.14276).

## Approach

Three orthogonal improvements stacked on the 10-layer Int5/Int6 SOTA:

### 1. Quantization-Aware Training (QAT)

The prior SOTA has a ~0.016 BPB quantization penalty (pre-quant vs post-quant) but no QAT. We add STE (Straight-Through Estimator) fake-quantization during the forward pass that exactly matches the export quantization:
- **Int5 [-16, 15]** for MLP weights (matching `quantize_intN_per_row(t, clip_range=15)`)
- **Int6 [-32, 31]** for attention weights (matching `quantize_intN_per_row(t, clip_range=31)`)
- Embeddings and small tensors are not QAT'd (kept in FP16)

Expected impact: -0.005 to -0.010 BPB (halving the quantization penalty).

### 2. Curriculum Learning on Sequence Length

- **Phase 1** (first 50% of wall-clock time): train with `seq_len=1024` (~40ms/step)
- **Phase 2** (remaining 50%): train with `seq_len=2048` (~60ms/step)

At 786K batch tokens per step:
- Phase 1: ~7,500 steps in 300s at 40ms/step
- Phase 2: ~5,000 steps in 300s at 60ms/step
- Total: ~12,500 steps vs ~7,500 without curriculum (+67% more gradient updates)

Short-range patterns (which dominate BPE-1024 text) are learned faster in Phase 1. Long-range dependencies are refined in Phase 2. Both torch.compile paths are primed during warmup.

Expected impact: -0.002 to -0.005 BPB.

### 3. Multi-Token Prediction (Auxiliary t+2 Head)

An auxiliary linear head predicts the token at position t+2 from the hidden state at position t. This forces richer internal representations (the model must encode information about multiple future tokens). The auxiliary loss is weighted at 0.15× the primary loss.

The auxiliary head is **discarded before serialization** — zero bytes in the artifact. Pure training-time regularizer.

Expected impact: -0.001 to -0.003 BPB.

## Architecture (unchanged from SOTA base)

- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3× expansion (hidden=1536), relu² activation
- SmearGate + BigramHash(10240, dim=128)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| qat_enabled | 1 (MLP: int5, attn: int6) |
| curriculum_short_seq | 1024 |
| curriculum_switch_frac | 0.5 |
| aux_loss_weight | 0.15 |
| All other params | Same as SOTA base |

## How to Run

```bash
# On 8×H100 (all defaults baked into the script)
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_QAT_Curriculum_MultiToken/train_gpt.py

# With specific seed
SEED=42 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_QAT_Curriculum_MultiToken/train_gpt.py
```

## Risk Assessment

| Technique | Confidence | Validated? |
|-----------|-----------|------------|
| QAT | High | Yes (submission #3 proved it works) |
| Curriculum | Medium | Not tried before in this challenge |
| Multi-token prediction | Medium | Proven in Meta research, untested here |
