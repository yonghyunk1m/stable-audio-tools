# SAO-Small Architecture Reference

> See [pipeline.md](pipeline.md) for the full training/inference data flow.

## Model Components (497M total)

| Component | Architecture | Params | Role |
|-----------|-------------|--------|------|
| VAE Encoder | Oobleck CNN, strides=[2,4,4,8,8] | 78M | Audio → latent |
| VAE Decoder | Oobleck CNN | 78M | Latent → audio |
| T5-base | Frozen text encoder | 109M | Text → 768d embeddings |
| **DiT** | 16-block transformer | **339M** | Denoising in latent space |

## DiT Architecture

```
Input: (B, 64, 256)  →  preprocess_conv  →  rearrange "bct→btc"  →  project_in
                                                                        ↓
                                                                   (B, 256, 1024)
                                                                        ↓
global_cond (prepend) → [prepend_tok, lat_1, ..., lat_256] = (B, 257, 1024)
                                                                        ↓
                                                              TransformerBlock x16
                                                              ├─ Self-Attention (RoPE, 8 heads)
                                                              ├─ Cross-Attention (cond: 66 tokens)
                                                              └─ FFN (1024→4096→1024)
                                                                        ↓
project_out → rearrange "btc→bct" → postprocess_conv → (B, 64, 256)
```

### Key dimensions
- embed_dim: 1024, num_heads: 8, head_dim: 128
- latent: 64 channels, 256 timesteps (from 524,288 audio samples @ 44.1kHz)
- cond_token_dim: 768 (T5-base output)
- downsampling_ratio: 2048x (VAE)

## Conditioning Pathways

| Pathway | Mechanism | Used by |
|---------|-----------|---------|
| Cross-attention | K/V from cond tokens, Q from audio | text, seconds_total, score |
| Global (prepend) | Prepend token to self-attention sequence | seconds_total |
| adaLN | Scale/shift/gate per layer | *Not used (failed in experiments)* |
| Input-add | Additive residual to latent | *Not used* |

## Rectified Flow Training

```
Forward:  x_t = (1-t) * x_0 + t * noise
Loss:     ||v_pred(x_t, t, cond) - (noise - x_0)||^2
```

CFG at inference: `output = uncond + cfg_scale * (cond - uncond)`

## ICME Parameter Budget

| | Params | Classification |
|--|--------|---------------|
| DiT + conditioners | ~340M | Core (500M limit) |
| VAE + T5 | ~265M | Auxiliary (excluded) |
| **Headroom** | **~160M** | Available for expansion |
