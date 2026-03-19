# SAO-Small Score Conditioning: Experiment Log

> Last updated: 2026-03-19
> Project: SAO-Small + Music-RankNet Score Conditioning

---

## 1. Overview

Add reward score conditioning to SAO-Small (497M) so generation quality can be steered at inference time.

- **ISMIR track**: FMA-Large (106K), score conditioning finetuning experiments
- **ICME track**: MTG-Jamendo (55K), scratch training with vocal separation

## 2. Architecture (497M total)

| Component | Params | ICME classification |
|-----------|--------|-------------------|
| DiT (16-depth, 1024-embed) | 339M | Core (500M limit) |
| VAE (Oobleck enc+dec) | 156M | Auxiliary |
| T5-base text encoder | 109M | Auxiliary |
| Score conditioners | ~1-2M | Core addon |

Core total ~340M → 160M headroom under ICME 500M limit.

## 3. Score Conditioning Experiments (Finetuning on pretrained SAO-Small)

### 3.1 Approaches tried

| Approach | Trainable Params | Best Corr | Audio Quality | Verdict |
|----------|-----------------|-----------|---------------|---------|
| adaLN v5-v10 (broken gate init) | 7-9M | ~0 | OK | Failed — `to_scale_shift_gate` randomly initialized, gates reduce signal by 27%/block → 0.73^16 ≈ 1% passthrough |
| adaLN gate-init v1 (fixed) | 10.1M | In progress | TBD | Proper init: scale=0, shift=0, gate=-10 → sigmoid(11)≈1.0 passthrough |
| Cross-attn, full unfreeze of `to_cond_embed` | 1.8M | 0.357 | Some genres silent | `to_cond_embed` shared by 66 tokens — modifying for 1 score token corrupts 65 text tokens |
| LoRA r=8 on `to_cond_embed` | 802K | 0.425 | Global noise | Rank-8 too constrained: can disrupt but not recover. Spectral flatness 0.206 vs 0.143 (xattn) |
| Separate score projection (`to_score_embed`) | ~2M | Not validated | Expected clean | Score token gets own projection, text tokens use frozen `to_cond_embed` |

### 3.2 Key findings

1. **adaLN failure was partly a bug**: `to_scale_shift_gate` was zero-initialized, causing gate=sigmoid(1)=0.73 per block. Over 16 blocks: 0.73^16 ≈ 0.01 → signal nearly vanishes. Fixed with gate=-10 → sigmoid(11)≈1.0.
2. **Cross-attention works** but shared `to_cond_embed` causes text degradation (Classical, Blues, Lo-Fi → near silent).
3. **LoRA causes global noise** — spectral flatness increases from 0.100 (epoch 0) to 0.206 (epoch 8). User listening confirmed all-noise output.
4. **CFG fix critical**: uncond pass must use `torch.zeros_like(global_embed)` for score, not duplicate. Otherwise CFG cannot amplify score signal.
5. **Fourier score embedding** >> `Linear(1,768)` — sinusoidal features make score=0.1 and score=0.9 completely different patterns in 768d, vs single direction vector with Linear.
6. **Score dropout 30%** creates stronger CFG contrast than default 15%.

### 3.3 Conditioning pathway analysis

| Pathway | Mechanism | Pros | Cons |
|---------|-----------|------|------|
| Cross-attention | Score becomes K/V token, audio Q attends | Pretrained pathway, proven | Competes with 64 text tokens for attention weight |
| adaLN | Score modulates LayerNorm scale/shift/gate | Direct, affects all layers uniformly | New params needed, initialization critical |
| Input-concat | Score channels concatenated to latent input | No attention competition, direct | Modifies `project_in`, spatially uniform |
| Prepend (global_cond) | Score prepended to self-attention sequence | Bidirectional interaction | Needs transformer unfreezing to learn new token |

## 4. Reward Model Analysis

### 4.1 Text embedding bias

Discovered that text-aware reward model rankings are influenced by CLAP text-audio similarity:

| Condition | FMA Top genres | Reasonable? |
|-----------|---------------|------------|
| With text embedding | Noise, Dubstep, Glitch, Breakcore | Questionable — high CLAP similarity inflates score for noisy genres |
| Without text (zeros) | Hip-Hop, Rock, Jazz, Pop, Soul | More aligned with perceived audio quality |
| `ablation_dropout_100` model | Hip-Hop, Pop, Techno, Soul, Indie-Rock | Best — trained without text, pure audio quality |

**Key insight**: This is not necessarily "bias" — the text-aware model correctly gives high prompt-adherence scores to Noise tracks with "A Noise song" prompts. The issue is whether we want the SAO score conditioning to reflect **audio quality only** (→ audio-only model) or **quality + prompt adherence** (→ text-aware model). Since SAO's T5 encoder already handles prompt adherence, using audio-only reward scores provides cleaner role separation.

### 4.2 Reward model comparison (FMA, 106K tracks)

| Model | Spearman (vs dropout_100) | Score range | Stability |
|-------|--------------------------|-------------|-----------|
| `ultimate_train_all(brainmusic).pt` (old) | 0.750 | [-7.4, +2.7] | Moderate |
| `reward_model_20260318_0347.pt` (new, text-aware) | 0.744 | [-73, +3.1] | Unstable (text=0 → extreme negatives) |
| `ablation_dropout_100.pt` (audio-only) | 1.000 | [-6.6, +3.8] | Most stable |

### 4.3 Architecture update (2026-03-19)

Collaborator unified model architecture:
- **Old**: `[1024, 512, 256, 128, 1]`, dropout=0.3, key prefix `net.`
- **New**: `[1024, 512, 128, 1]`, dropout=0.5, key prefix `score_predictor.`
- Quick comparison: 4-layer (72.02% test acc) slightly better than 5-layer (71.33%)
- **Breaking change**: old checkpoints incompatible with new code
- Both text-aware and audio-only models being retrained with new architecture

### 4.4 Sanity checks performed

- **Score consistency**: Pre-computed metadata scores exactly match live model inference (20/20, delta=0.000)
- **Feature order verified**: `[flag(1), clap(512), mert(1024), text(512)] = 2049 dim`
- **Text ablation**: Mismatched text prompts cause moderate score drops (~0.5-1.5 points), not catastrophic. Model relies primarily on audio features.

## 5. ICME Challenge Pipeline

### 5.1 Data preparation
1. [x] MTG-Jamendo 55K tracks (full length, ~508GB)
2. [IN PROGRESS] Vocal separation via Mel-Band Roformer (GPU 0-3, ~2 days remaining)
3. [x] Qwen2-Audio tag→caption: 54,753 generated (ICME test prompt style)
4. [x] ICME reference captions downloaded (55,701 from organizers, Qwen + MusicFlamingo)
5. [x] Reward models retrained with new 4-layer architecture (text-aware + audio-only)
6. [x] Jamendo re-scored with new text-aware model → `metadata_jamendo_scored_new.jsonl`

### 5.2 Caption strategy
- **ICME test prompts**: Tags → Qwen2-Audio → short caption (<100 words)
- **ICME reference captions**: Audio → Qwen/MusicFlamingo → longer descriptions
- **Our Qwen captions**: Tags → Qwen2-Audio → ICME test style (generated)
- **Plan**: Train with mixed captions (reference + ours) for robustness

### 5.3 Training plan

| Exp | Data | Score | Purpose |
|-----|------|-------|---------|
| (i) | Jamendo instrumental | w/ score | ICME submission |
| (ii) | FMA | none | Baseline |
| (iii) | FMA | w/ score | Score conditioning ablation |

All three: scratch training (random init).

### 5.4 Evaluation metrics
- **FAD** (Frechet Audio Distance): audio quality
- **CLAP Score**: text-audio alignment
- **CCS** (K/M): concept coverage per prompt (Audio LM judges)

## 6. Current Status (2026-03-19)

| Task | GPU | Status |
|------|-----|--------|
| Vocal separation | 0-3 | Running (~2 days remaining) |
| adaLN gate-init v3 | 8-9 | Running (Jamendo, fixed reward monitor) |
| Reward models (4-layer) | — | Complete (text-aware + audio-only) |
| Jamendo re-scoring | — | Complete (54,330 tracks, text-aware) |
| Qwen captions | — | Complete (54,753) |
| Disk space | — | ~580GB free |

## 7. Repositories

- `stable-audio-tools` (branch: `2026-03-12-ema5`): SAO-Small training/inference + score conditioning
- `music-ranknet` (branch: `main`): Reward model, feature extraction, scoring pipelines
