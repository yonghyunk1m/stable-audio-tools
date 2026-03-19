# SAO-Small Score Conditioning: Experiment Log

> Last updated: 2026-03-19
> Project: SAO-Small + Music-RankNet Score Conditioning

---

## 1. Overview

Add reward score conditioning to SAO-Small (497M) so generation quality can be steered at inference time.

- **ISMIR track**: FMA-Large (106K), score conditioning experiments
- **ICME track**: MTG-Jamendo (55K), scratch training with vocal separation

## 2. Architecture (497M total)

| Component | Params | ICME classification |
|-----------|--------|-------------------|
| DiT (16-depth, 1024-embed) | 339M | Core (500M limit) |
| VAE (Oobleck enc+dec) | 156M | Auxiliary |
| T5-base text encoder | 109M | Auxiliary |
| Score conditioners | ~1-2M | Core addon |

Core total ~340M → 160M headroom under ICME 500M limit.

## 3. Score Conditioning Experiments (ISMIR)

### Approaches tried

| Approach | Params | Best Corr | Audio Quality | Verdict |
|----------|--------|-----------|---------------|---------|
| adaLN (v5-v10) | 7-9M | ~0 | OK | **Failed** — score signal too weak for indirect pathway |
| Cross-attn, full unfreeze | 1.8M | 0.357 | Some genres degrade | Corr OK but Classical/Blues → silent |
| LoRA r=8 on to_cond_embed | 802K | 0.425 | Noise across all | Higher corr but global noise degradation |
| Separate score projection | ~2M | TBD | TBD | Cleanest architecture, not yet validated |

### Key findings

1. **adaLN doesn't work** for weak score signals (6 attempts, v5-v10)
2. **Cross-attention works** but shared `to_cond_embed` causes text degradation
3. **LoRA causes global noise** — rank=8 too constrained to preserve text quality
4. **CFG fix critical**: uncond pass must use zeros for score (not duplicate)
5. **Fourier score embedding** >> Linear(1,768) for discriminative power

### Reward model analysis

- **Text bias discovered**: text-aware model ranks Noise/Glitch genres as "best quality"
- `ablation_dropout_100.pt` (audio-only) gives most sensible rankings
- New model trained (2026-03-18): `reward_model_20260318_0347.pt`, Test Acc 67-96%

## 4. ICME Challenge Plan

### Data pipeline
1. MTG-Jamendo 55K tracks (full length, ~508GB)
2. Vocal separation via Mel-Band Roformer (**in progress**, GPU 0-3)
3. Qwen2-Audio captions: 54,753 generated (tag→caption style, matching test format)
4. ICME reference captions: 55,701 (Qwen + MusicFlamingo, already provided)

### Training plan
| Exp | Data | Score | Purpose |
|-----|------|-------|---------|
| (i) | Jamendo | w/ score | ICME submission |
| (ii) | FMA | none | Baseline |
| (iii) | FMA | w/ score | Score conditioning effect |

### Evaluation metrics (ICME)
- **FAD**: Audio quality (distributional)
- **CLAP Score**: Text-audio alignment
- **CCS (K/M)**: Concept coverage per prompt

### Test prompt format
Tags → Qwen2-Audio → caption-like descriptions (<100 words).

## 5. Current Status (2026-03-19)

- **Vocal separation**: Running on GPU 0-3, ~2-3 days ETA
- **LoRA v2 training**: GPU 8-9, Epoch 8, Best Corr=0.425 (but noise issues)
- **Qwen captions**: Complete (54,753 Jamendo tracks)
- **Disk**: 589GB free after cleanup

## 6. Repositories

- `stable-audio-tools`: SAO-Small training/inference code
- `music-ranknet`: Reward model, feature extraction, scoring pipelines
