# Score-Conditioned Audio Generation with SAO-Small: A Complete Tutorial

**Last updated**: 2026-03-14

---

## Table of Contents

1. [What Are We Building?](#1-what-are-we-building)
2. [Background: How Text-to-Audio Generation Works](#2-background-how-text-to-audio-generation-works)
3. [SAO-Small Architecture Deep Dive](#3-sao-small-architecture-deep-dive)
4. [The Score Conditioning Problem](#4-the-score-conditioning-problem)
5. [Conditioning Pathways: How Signals Enter the Model](#5-conditioning-pathways-how-signals-enter-the-model)
6. [Implementation: Code Walkthrough](#6-implementation-code-walkthrough)
7. [Training Pipeline](#7-training-pipeline)
8. [Validation & Reward Monitoring](#8-validation--reward-monitoring)
9. [Experiment Log: What We Tried and What We Learned](#9-experiment-log-what-we-tried-and-what-we-learned)
10. [Current Best Approach: Cross-Attention Score Conditioning](#10-current-best-approach-cross-attention-score-conditioning)
11. [Key Debugging Lessons](#11-key-debugging-lessons)
12. [Appendix: Tensor Shape Reference](#12-appendix-tensor-shape-reference)

---

## 1. What Are We Building?

### The Goal

We want to build a text-to-audio model that not only follows text prompts but also
controls the **quality** of its output. Given:

- A text prompt: *"A jazz piano solo"*
- A quality score: **9.0** (high quality) or **2.0** (low quality)

The model should generate audio that matches the text AND the desired quality level.

### Why This Matters

Text-to-audio models generate audio from text descriptions, but they have no notion
of "quality." A model might produce mediocre audio even when a user wants the best
possible output. By conditioning on a quality score, we enable:

1. **Quality steering**: Generate high-quality audio on demand
2. **Curriculum learning**: Train on all data (including low-quality) while generating only high-quality at inference
3. **Controllable generation**: Users can trade off quality vs. diversity

### The System at a Glance

```
                                    Score Conditioning
                                    (our addition)
                                         |
                                         v
  Text Prompt -----> [ T5 Encoder ] ---+---> [ DiT Transformer ] ---> [ VAE Decoder ] ---> Audio
                                       |            ^
  Duration --------> [ Number Emb ] ---+            |
                                                    |
  Gaussian Noise ---> (iterative denoising) --------+
```

We take an existing text-to-audio model (Stable Audio Open Small, "SAO-Small")
and add a **quality score** as an additional conditioning signal.

---

## 2. Background: How Text-to-Audio Generation Works

### 2.1 Diffusion Models in 60 Seconds

Diffusion models learn to generate data by learning to **remove noise**.

**Training**: Take real audio, add noise, train the model to predict the noise.
```
Clean audio x0 ---(add noise)--> Noisy audio xt ---(model predicts)--> Predicted noise
                                                                         |
Loss = || true_noise - predicted_noise ||^2           <---(minimize)------+
```

**Generation**: Start from pure noise, iteratively remove noise to get clean audio.
```
Pure noise z_T --> (denoise) --> z_{T-1} --> (denoise) --> ... --> z_0 --> Clean audio
```

### 2.2 Latent Diffusion: Working in Compressed Space

Raw audio is huge (44,100 samples/sec * 2 channels * 12 sec = ~1M numbers).
Running diffusion directly on audio is too expensive. Instead:

```
Audio (2 x 524,288) ---> [ VAE Encoder ] ---> Latent (64 x 256) ---> [ Diffusion ] ---> ...
       ~1M numbers           compress            ~16K numbers
                             2048x smaller
```

The **Variational Autoencoder (VAE)** compresses audio by 2048x. Diffusion operates
in this small latent space, then the VAE decoder converts back to audio.

### 2.3 Rectified Flow (SAO-Small's Training Objective)

SAO-Small uses **rectified flow** instead of classic diffusion. The idea is simple:
learn a straight-line path from noise to data.

```
Forward:  x_t = (1 - t) * x_0 + t * noise     (mix data and noise)
Loss:     || model(x_t, t) - (noise - x_0) ||^2   (predict the "velocity")
```

The model predicts the velocity `v = noise - x_0`, which tells it the direction
from data to noise (or vice versa). At inference, it follows this velocity backward
from noise to data using an ODE solver.

### 2.4 Classifier-Free Guidance (CFG)

CFG is a trick to make the model follow its conditioning more strongly:

```python
output = uncond_output + cfg_scale * (cond_output - uncond_output)
```

During **training**, 15% of the time we drop the condition (replace with null).
This teaches the model both conditional and unconditional generation.

During **inference**, we run the model twice (with and without condition) and
amplify the difference. `cfg_scale=3.5` means "follow the condition 3.5x more
strongly than you naturally would."

---

## 3. SAO-Small Architecture Deep Dive

### 3.1 Model Overview

| Component | Architecture | Parameters | Purpose |
|-----------|-------------|------------|---------|
| VAE Encoder | Oobleck (CNN) | ~78M | Audio -> Latent |
| VAE Decoder | Oobleck (CNN) | ~78M | Latent -> Audio |
| Text Encoder | T5-base | ~109M | Text -> Embeddings |
| DiT (Denoiser) | Transformer | ~340M | Noise prediction |
| **Total** | | **~497M** | |

### 3.2 The DiT (Diffusion Transformer)

The DiT is the core of the model. It's a standard Transformer with modifications
for diffusion:

```
Input latent: (batch, 64, 256)      # 64 channels, 256 timesteps
                  |
          [Patch Embedding]          # Conv1d: 64 -> 1024
                  |
          (batch, 1024, 256)         # Now in transformer dim
                  |
    +--->[TransformerBlock x16]<---+
    |             |                |
    |     Self-Attention           |  Text tokens via
    |     Cross-Attention   <------+  Cross-Attention
    |     Feed-Forward             |
    |             |                |
    +-------------+                |
                  |                |
          [Output Projection]      |
                  |                |
          (batch, 64, 256)         # Back to latent dim
```

**Key numbers for SAO-Small**:
- Embedding dimension: 1024
- Number of heads: 8 (head dim = 128)
- Depth: 16 transformer blocks
- Conditioning token dim: 768 (from T5-base)
- Latent shape: (batch, 64, 256) = ~16K values per sample

### 3.3 Conditioning in the Original SAO

The original SAO-Small has two conditions:

1. **Text prompt** (via T5-base): Produces `(batch, 64, 768)` embeddings
   - Fed through **cross-attention** in each transformer block
   - Each position in the latent can "look at" relevant text tokens

2. **Duration** (`seconds_total`): A single number (e.g., 10.0)
   - Projected: `Linear(1, 768)` -> 768-dim embedding
   - Fed through **both** cross-attention and prepend/global conditioning

---

## 4. The Score Conditioning Problem

### 4.1 Where Do Scores Come From?

We use **Music-RankNet**, a trained quality assessment model:

```
Audio --> [MERT (1024d)] ----+
      --> [CLAP audio (512d)] +---> [RankNet MLP] ---> Score (float)
      --> [CLAP text (512d)] -+         |
      --> [Handcrafted (1d)] -+    [1024,512,256,128]
                                   Siamese network
```

Music-RankNet produces a continuous score for each audio file. We pre-compute
scores for all training data (FMA-Large, ~106K tracks) and store them in metadata:

```jsonl
{"file": "000002.mp3", "prompt": "A Hip-Hop song", "seconds_total": 30, "reward_score": 0.42}
{"file": "000005.mp3", "prompt": "A Rock song",    "seconds_total": 30, "reward_score": -0.87}
```

Score distribution: roughly [-1.5, +0.9], where higher = better quality.

### 4.2 The Challenge

Adding a new conditioning signal to a pretrained model is hard because:

1. **Pretrained weights are fragile**: Random initialization of new parameters can
   break the model's existing capabilities
2. **The model doesn't need the score**: It can predict noise perfectly well using
   just text + the audio latent itself. The score is "redundant" for denoising.
3. **Few trainable parameters**: We freeze most of the model (~497M params) and
   only train the score-related parameters (~1-9M params)

---

## 5. Conditioning Pathways: How Signals Enter the Model

The DiT has **four** pathways for conditioning signals:

### 5.1 Cross-Attention (proven, used by text + duration)

```
                    Latent tokens         Conditioning tokens
                   (batch, 256, 1024)     (batch, seq, 1024)
                         |                       |
                     [Q = W_q * x]          [K = W_k * c]
                         |                  [V = W_v * c]
                         |                       |
                    Attention(Q, K, V) ----------+
                         |
                   (batch, 256, 1024)   # Each position attends to conditioning
```

**How it works**: The latent sequence has 256 positions. Each position computes
attention over the conditioning tokens (text + duration + score). The model
learns which conditioning tokens are relevant at each position.

**Strengths**: Proven pathway, flexible, position-specific conditioning.

### 5.2 Global Conditioning via Prepend

```
    Score embedding (batch, 1, 1024)
              |
    Prepend to sequence:  [score_token, latent_1, latent_2, ..., latent_256]
              |
    Self-attention sees the score token as part of the sequence
```

**How it works**: The score embedding becomes an extra token at the start of the
sequence. Self-attention can attend to it from any position.

### 5.3 Global Conditioning via adaLN (Adaptive Layer Normalization)

```
    Score embedding (batch, 768)
              |
    [global_cond_embedder]   Linear(768->1024) -> SiLU -> Linear(1024->6144)
              |
    (batch, 6144) -> chunk into 6 x (batch, 1024)
              |
    scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff
              |
    For each TransformerBlock:
        x = LayerNorm(x)
        x = x * (1 + scale) + shift        # Modulate statistics
        x = SelfAttention(x)
        x = x * sigmoid(1 - gate)          # Gate the output
```

**How it works**: The score modulates the **statistics** (mean, variance) of every
layer's activations. This is how the original DiT paper conditions on ImageNet
class labels.

**Problem we found**: When `global_cond_embedder` is zero-initialized (because
it's a new parameter not in the pretrained checkpoint), the output is always
zero. The score has NO effect, and gradients can't flow to update the weights.
See [Section 9](#9-experiment-log-what-we-tried-and-what-we-learned) for details.

### 5.4 Input-Add Adapter (Channel-wise Residual)

```
    Score embedding (batch, 768, 1)
              |
    [Interpolate to sequence length]  -> (batch, 768, 256)
              |
    [Conv1d(768, 64, kernel_size=1)]  -> (batch, 64, 256)   # zero-initialized
              |
    Add to latent:  latent = latent + adapter_output
```

**How it works**: The score is broadcast across the sequence and added directly
to the latent. The Conv1d is zero-initialized (ControlNet-style) so it starts
with zero effect.

**Problem we found**: A single Linear(1->768) followed by Conv1d(768->64) has
very limited expressiveness. It can only add a single "direction vector" scaled
by the score value.

### 5.5 Comparison Table

| Pathway | Mechanism | Pretrained? | New Params | Works for Score? |
|---------|-----------|------------|------------|-----------------|
| Cross-attention | Attention over tokens | Yes | ~769 | Current best approach |
| Prepend | Extra self-attn token | Yes | ~769 | Untested separately |
| adaLN | Scale/shift/gate per layer | No | ~9M | Failed (v5-v10) |
| Input-add | Channel residual | No | ~49K | Failed (too limited) |

---

## 6. Implementation: Code Walkthrough

### 6.1 The Score Conditioner

The simplest component: maps a scalar score to a 768-dim embedding.

```python
# stable_audio_tools/models/conditioners.py

class ContinuousScoreConditioner(nn.Module):
    def __init__(self, output_dim=768, cond_dim=1):
        super().__init__()
        self.mapper = nn.Linear(cond_dim, output_dim)   # 1 -> 768

    def forward(self, x, device):
        x = x.view(-1, 1)                              # (batch, 1)
        embeds = self.mapper(x)                         # (batch, 768)

        # Null condition: score == -999.0 -> zero embedding
        null_idx = (x.squeeze(-1) == -999.0)
        if null_idx.any():
            embeds[null_idx] = 0.0

        mask = torch.ones(embeds.shape[0], 1, device=device)
        return embeds, mask                             # (batch, 768), (batch, 1)
```

**Why -999.0?** During CFG dropout, we replace the real score with -999.0 (a
sentinel value the model will never see in real data). The conditioner zeroes
out the embedding for this value, teaching the model what "no score" looks like.
At inference, CFG compares "with score" vs "without score" to amplify the effect.

### 6.2 Model Configuration (JSON)

The config file tells the model which conditioners exist and which pathway they use:

```json
{
    "conditioning": {
        "configs": [
            {"id": "prompt",           "type": "t5",              "config": {"t5_model_name": "t5-base"}},
            {"id": "seconds_total",    "type": "number",          "config": {"min_val": 0, "max_val": 256}},
            {"id": "continuous_score", "type": "continuous_score", "config": {"output_dim": 768}}
        ]
    },
    "diffusion": {
        "cross_attention_cond_ids": ["prompt", "seconds_total", "continuous_score"],
        "global_cond_ids": []
    }
}
```

`cross_attention_cond_ids` determines which conditioners feed into cross-attention.
`global_cond_ids` determines which feed into adaLN/prepend global conditioning.

### 6.3 How Conditioning Flows Through the Model

In `diffusion.py`, the `get_conditioning_inputs()` method assembles all conditions:

```python
def get_conditioning_inputs(self, conditioning_tensors):
    # Cross-attention: concatenate all condition tokens along sequence dim
    cross_attention_input = []
    for key in self.cross_attn_cond_ids:       # ["prompt", "seconds_total", "continuous_score"]
        cond_in, mask = conditioning_tensors[key]

        if len(cond_in.shape) == 2:            # (B, 768) -> (B, 1, 768)
            cond_in = cond_in.unsqueeze(1)

        cross_attention_input.append(cond_in)

    cross_attention_input = torch.cat(cross_attention_input, dim=1)
    # Result: (batch, 64 + 1 + 1, 768) = (batch, 66, 768)
    #          text tokens  seconds  score
```

The combined conditioning sequence:
```
[text_tok_1, text_tok_2, ..., text_tok_64, seconds_tok, score_tok]
                                                           ^
                                                    Our addition!
```

### 6.4 CFG: Amplifying the Score Signal

In `dit.py`, Classifier-Free Guidance runs the model twice:

```python
# Forward pass: conditioned (with score) and unconditioned (without score)
batch_inputs = torch.cat([x, x], dim=0)                          # Double the batch

if global_embed is not None:
    batch_global_cond = torch.cat([
        global_embed,                     # Conditioned: real score embedding
        torch.zeros_like(global_embed)    # Unconditioned: zero (= no score)
    ], dim=0)

# Cross-attention conditions also doubled:
# Top half: real text + real seconds + real score
# Bottom half: null text + null seconds + null score (zeros)

output = model(batch_inputs, ...)

cond_output, uncond_output = output.chunk(2, dim=0)
final = uncond_output + cfg_scale * (cond_output - uncond_output)
```

**Critical bug we fixed**: Previously, the unconditioned pass used the **same**
score embedding as the conditioned pass. This means CFG couldn't distinguish
"with score" from "without score" -- the score was invisible to CFG!

Fix: Use `torch.zeros_like(global_embed)` for the unconditioned pass.

### 6.5 Unfreeze Profiles: What Parameters to Train

We freeze the entire pretrained model and only train score-related parameters:

```python
UNFREEZE_PROFILES = {
    "xattn":   ["continuous_score", "to_cond_embed"],        # 1.8M params
    "adaln":   ["continuous_score", "to_global_embed",       # 9.3M params
                "to_scale_shift_gate", "global_cond_embedder"],
    "minimal": ["continuous_score"],                         # ~769 params
}

def unfreeze_finetune_params(model):
    model.requires_grad_(False)                # Freeze everything
    for name, param in model.named_parameters():
        if any(key in name for key in trainable_name_keys):
            param.requires_grad_(True)         # Unfreeze matching params
```

The `xattn` profile unfreezes:
- `continuous_score`: The `Linear(1, 768)` mapper (769 params)
- `to_cond_embed`: The `Sequential(Linear(768,1024), SiLU, Linear(1024,1024))`
  that projects conditioning tokens for cross-attention (~1.8M params total)

### 6.6 Weight Initialization: The ControlNet Zero-Conv Pattern

New parameters not in the pretrained checkpoint need careful initialization:

```python
def zero_init_new_params(model, pretrained_keys):
    """
    ControlNet zero-conv principle:
    - OUTPUT layers: zero-init (new pathways start with zero effect)
    - INTERMEDIATE layers: small random init (gradients can flow)

    If BOTH layers are zero-init'd:
      Forward:  h = W1 @ x = 0,  out = W2 @ h = 0     (always zero)
      Backward: dL/dW2 = dL/d(out) * h^T = ... * 0 = 0 (dead gradient!)
      The network can NEVER learn!

    If only the output layer is zero:
      Forward:  h = W1 @ x != 0, out = W2 @ h = 0     (starts at zero)
      Backward: dL/dW2 = dL/d(out) * h^T != 0          (gradient flows!)
      W2 updates, then W1 can update too.
    """
    KEEP_RANDOM = ["continuous_score", "global_cond_embedder.0"]

    for name, param in model.named_parameters():
        if name not in pretrained_keys:
            if any(pat in name for pat in KEEP_RANDOM):
                param.data.normal_(0, 0.02)     # Small random
            else:
                param.zero_()                   # Zero
```

---

## 7. Training Pipeline

### 7.1 Data Flow

```
FMA-Large Audio Files (106K tracks)
         |
         v
[audio_dir Dataset] --> (audio_tensor, metadata_dict)
         |
         v
[ContinuousScoreDatasetWrapper]
    - Reads metadata['reward_score']
    - 15% CFG dropout: score -> -999.0 (null)
    - Adds metadata['continuous_score'] = score_val
         |
         v
[DataLoader] --> batches of (audio, metadata)
         |
         v
[Training Step]
    1. VAE encode: audio (2, 524288) -> latent (64, 256)
    2. Add noise:  latent_noisy = (1-t) * latent + t * noise
    3. Conditioner: metadata -> conditioning tensors
    4. Model forward: predict velocity from (latent_noisy, t, conditions)
    5. Loss = MSE(predicted_velocity, true_velocity)
    6. Backprop & update only unfrozen params
```

### 7.2 Launch Script

```bash
# Key environment variables
export SA_UNFREEZE_PROFILE=xattn          # Which params to train
export MODEL_CONFIG=./checkpoints/sao_small/model_config_with_score_xattn.json
export CUDA_VISIBLE_DEVICES=8,9           # Two GPUs
export SA_CFG_DROP_RATE=0.15              # 15% CFG dropout
export SA_LR=5e-5                         # Learning rate
export SA_WARMUP_STEPS=1000               # LR warmup
export SA_TOTAL_STEPS=300000              # Total training steps

# Launch
python finetune.py \
    --dataset-config ./configs/dataset_fma_scored.json \
    --model-config $MODEL_CONFIG \
    --pretrained-ckpt-path ./checkpoints/sao_small/model.safetensors \
    --batch-size 2 --accum-batches 4 \   # Effective batch = 2*4*2GPUs = 16
    --precision 16-mixed \                # FP16 mixed precision
    --checkpoint-every 5000 \
    --val-every 5000 \
    --logger wandb
```

### 7.3 LR Schedule

Cosine annealing with warm restarts:
```
LR
 ^
 |  /\    /\    /\
 | /  \  /  \  /  \
 |/    \/    \/    \
 +-----|-----|-------> steps
  warmup  cycle1 cycle2
```

---

## 8. Validation & Reward Monitoring

### 8.1 How Validation Works

Every 5,000 steps, the `RewardMonitorCallback` runs:

1. **Generate audio**: Sample 100 audio clips at different target scores
   - 10 clips each at score targets: top 10%, 20%, ..., 100% percentiles
   - Uses real text prompts from validation set

2. **Evaluate with Music-RankNet**: Score each generated clip
   - Extract MERT (1024d) + CLAP audio (512d) + CLAP text (512d) features
   - Run through the same RankNet that produced training scores

3. **Compute metrics**:
   - **Correlation**: Pearson correlation between target score and measured score
     - Perfect conditioning: correlation -> 1.0
     - No conditioning effect: correlation -> 0.0
   - **Monotonicity**: Fraction of pairs where higher target -> higher measured
     - Perfect: 1.0, Random: 0.5

### 8.2 Interpreting Results

```
Step 0 (before training):
  Correlation: 0.047   (essentially random)
  Monotonicity: 0.510  (essentially random)
  -> Expected! Model hasn't learned to use the score yet.

Step 5000 (after training):
  Correlation: 0.35    (positive correlation!)
  Monotonicity: 0.65   (better than random!)
  -> The model is learning to use the score!

  Per-bin measured scores:
    Top  10% (target=0.91): measured= +0.42   # High target -> high measured
    Top  50% (target=0.25): measured= -0.15   # Mid target -> mid measured
    Top 100% (target=-1.52): measured= -0.88  # Low target -> low measured
    -> Monotonically decreasing = score conditioning works!
```

---

## 9. Experiment Log: What We Tried and What We Learned

### Timeline of Approaches

| Version | Config | Pathway | Trainable | Result | Root Cause |
|---------|--------|---------|-----------|--------|------------|
| v1-v4 | prepend | global_cond (prepend) | 1.5K | No effect | adaLN params don't exist in prepend mode |
| v5 | adaln | global_cond (adaLN) | 7.4M | Corr=-0.03 | sum-merge + dual-path redundancy |
| v6 | adaln_pure | global_cond (adaLN only) | 7.4M | Corr=-0.12 | Dead gradient in global_cond_embedder |
| v7 | adaln_pure | adaLN + small-random init | 9.3M | Corr~0 | Score redundant for denoising loss |
| v8 | adaln_pure | adaLN + all-random init | 9.3M | NaN | fp16 overflow from large adaLN values |
| v9 | adaln_pure | adaLN + to_global_embed | 9.3M | Corr~0 | Same fundamental issue |
| v10 | adaln_pure | adaLN + CFG fix | 9.3M | Corr=-0.04 | adaLN not suitable for quality scores |
| **v11** | **xattn** | **cross-attention** | **1.8M** | **TBD** | **Currently running** |

### Detailed Failure Analysis

#### Failure 1: The adaLN No-Op (v1-v4)

**Setup**: Used `model_config_with_score.json` (prepend mode) with `SA_UNFREEZE_PROFILE=adaln`.

**Bug**: The config had `global_cond_type: "prepend"`, which means the model uses
prepend tokens, NOT adaLN. The adaLN parameters (`to_scale_shift_gate`,
`global_cond_embedder`) simply don't exist in prepend mode. Unfreezing them
is a no-op -- only 1.5K params (the Linear(1,768) mapper) were actually trained.

**Lesson**: Always verify that the unfreeze profile matches the model config.

#### Failure 2: Dead Gradient in Zero-Init (v5-v7)

**Setup**: Correct adaLN config, global_cond_embedder zero-initialized.

**Bug**: `global_cond_embedder` is a 2-layer MLP:
```
Linear(1024, 1024) -> SiLU() -> Linear(1024, 6144)
```
Both layers were zero-initialized. This creates a dead network:

```
Forward:  h = W1 @ x + b1 = 0 + 0 = 0
          out = W2 @ SiLU(h) + b2 = W2 @ SiLU(0) + 0 = 0

Backward: dL/dW2 = dL/d(out) * SiLU(h)^T = dL/d(out) * 0^T = 0
          dL/dW1 = (dL/d(out) * W2^T * SiLU'(h)) * x^T
                 = (... * 0^T * ...) * x^T = 0

All gradients are ZERO. The network can never learn!
```

**Fix**: Only zero-init the output layer, keep the intermediate layer with small
random values. This is the "ControlNet zero-conv" pattern.

#### Failure 3: NaN from Large Initialization (v8)

**Setup**: Random-init on ALL layers of global_cond_embedder.

**Bug**: With `N(0, 0.02)` init on the output layer (shape 1024x6144), the adaLN
modulation values were too large:
```
output_norm ~ sqrt(1024) * 0.02 ~ 0.64
```
Combined with `to_scale_shift_gate` (also non-zero), the scale/shift modulation
immediately pushed activations out of fp16 range, causing NaN.

**Lesson**: Output layers of new conditioning pathways MUST start at zero.

#### Failure 4: Score Redundancy for Denoising (v9-v10)

**Setup**: Correct initialization, CFG fix applied, to_global_embed unfrozen.

**Why it still failed**: Even with everything technically correct, the adaLN
pathway doesn't learn because **the denoising loss has no incentive to use
the score signal**.

The MSE loss is: `||v_pred(x_t, t, text, score) - v_target||^2`

The model can minimize this loss perfectly well using just `x_t` and `text`.
The score is **redundant information** for noise prediction. Adding score through
adaLN (which modulates layer statistics) is a very indirect signal that the
model can simply ignore by keeping `global_cond_embedder` output near zero.

**Why cross-attention is different**: Cross-attention adds the score as an
explicit token that the model can attend to. The pretrained cross-attention
mechanism is already trained to extract useful information from conditioning
tokens (text, duration). Adding score as another token leverages this existing,
proven mechanism.

---

## 10. Current Best Approach: Cross-Attention Score Conditioning

### 10.1 Architecture

```
    Text (64 tokens, 768d each)     Seconds (1 token, 768d)     Score (1 token, 768d)
              |                              |                          |
              +------------- concat ---------+--------------------------+
              |
        (batch, 66, 768)                     # 66 conditioning tokens total
              |
        [to_cond_embed]                      # Pretrained: Linear(768,1024) -> SiLU -> Linear(1024,1024)
              |
        (batch, 66, 1024)
              |
        Fed into Cross-Attention of each TransformerBlock
```

### 10.2 Why This Should Work

1. **Proven pathway**: Text and duration already work through cross-attention
2. **Pretrained projection**: `to_cond_embed` already maps 768d -> 1024d;
   score embeddings are also 768d, so they're compatible
3. **Attention is flexible**: Each latent position can choose how much to
   attend to the score token based on learned attention patterns
4. **CFG amplification**: With zeros for the unconditioned pass, CFG naturally
   amplifies the score signal just like it amplifies text

### 10.3 Config

```json
{
    "diffusion": {
        "cross_attention_cond_ids": ["prompt", "seconds_total", "continuous_score"],
        "global_cond_ids": [],
        "config": {
            "global_cond_type": "prepend",
            "cond_token_dim": 768
        }
    }
}
```

### 10.4 Training Setup

```bash
SA_UNFREEZE_PROFILE=xattn                  # Unfreeze: continuous_score + to_cond_embed
MODEL_CONFIG=model_config_with_score_xattn.json
# Trainable params: ~1.8M out of 497M total (0.36%)
```

---

## 11. Key Debugging Lessons

### Lesson 1: Verify What You're Actually Training

```python
# Always print unfrozen parameters and their shapes
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  [UNFROZEN] {name}: {list(param.shape)}")
```

If you expect 7.4M trainable params but see 1.5K, something is wrong.

### Lesson 2: Config Must Match Unfreeze Profile

| Config `global_cond_type` | Available Params | `adaln` Profile Effect |
|---------------------------|-----------------|----------------------|
| `"prepend"` | `to_global_embed` only | **No-op** (adaln params don't exist) |
| `"adaLN"` | `to_global_embed` + `to_scale_shift_gate` + `global_cond_embedder` | Works as intended |

### Lesson 3: Zero-Init Must Be Layer-Selective

```
BAD:  Zero W1, Zero W2  ->  Dead gradient, network can never learn
GOOD: Random W1, Zero W2 ->  Starts at zero effect, but gradients flow
```

This is the ControlNet zero-convolution principle.

### Lesson 4: CFG Must Differentiate Conditioned vs Unconditioned

```python
# WRONG: Both passes use the same score
batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
# CFG: cond - uncond = 0 for the score component!

# RIGHT: Unconditioned pass uses zero (null score)
batch_global_cond = torch.cat([global_embed, torch.zeros_like(global_embed)], dim=0)
# CFG: cond - uncond = score_effect (amplified by cfg_scale)
```

### Lesson 5: Use Proven Pathways When Possible

When adding new conditioning to a pretrained model:
- **Prefer cross-attention** if you have an embedding that fits the existing space
- adaLN requires training new per-layer modulation parameters from scratch
- The pretrained cross-attention already knows how to extract information from tokens

### Lesson 6: Check for NaN Early

```python
# In the training log, look for:
train/loss=nan.0    # Immediate NaN = initialization problem
                    # Usually fp16 overflow from large values
```

---

## 12. Appendix: Tensor Shape Reference

### A. Audio Processing Pipeline

```
Raw audio:           (batch, 2, 524288)    # 2 channels, ~11.9s at 44.1kHz
After VAE encode:    (batch, 64, 256)      # 64 latent channels, 256 timesteps
After patch embed:   (batch, 256, 1024)    # 256 tokens, 1024 embedding dim
After transformer:   (batch, 256, 1024)
After project out:   (batch, 256, 64)      # Back to latent space
Reshape to latent:   (batch, 64, 256)
After VAE decode:    (batch, 2, 524288)    # Back to audio
```

### B. Conditioning Shapes

```
T5 text embeddings:  (batch, 64, 768)      # 64 text tokens
Duration embedding:  (batch, 768)          # Single scalar -> 768d
Score embedding:     (batch, 768)          # Single scalar -> 768d

Cross-attn combined: (batch, 66, 768)      # Concatenated along seq dim
After to_cond_embed: (batch, 66, 1024)     # Projected to model dim
```

### C. adaLN Shapes (for reference)

```
to_global_embed:     (batch, 768) -> (batch, 1024)
global_cond_embedder:(batch, 1024) -> (batch, 6144)
to_scale_shift_gate: (6144,) per block     # 16 blocks total
Chunk into 6:        6 x (batch, 1024)     # scale, shift, gate x2
```

### D. SAO-Small vs SAO-Original

| | SAO-Small | SAO-Original |
|---|---|---|
| DiT depth | 16 | 24 |
| DiT embed_dim | 1024 | 1536 |
| DiT params | ~340M | ~1.06B |
| Total params | ~497M | ~1.2B |
| Sample size | 524,288 (~11.9s) | 2,097,152 (~47.5s) |
| Text encoder | T5-base | T5-base |
| Latent dim | 64 | 64 |
| Downsampling | 2048x | 2048x |

---

## Glossary

- **adaLN**: Adaptive Layer Normalization. Modulates LayerNorm statistics per-sample.
- **CFG**: Classifier-Free Guidance. Amplifies conditioning signal at inference.
- **DiT**: Diffusion Transformer. Transformer architecture for diffusion models.
- **FMA**: Free Music Archive. Open dataset of ~106K music tracks.
- **Latent space**: Compressed representation of audio (2048x smaller).
- **MERT**: Music Encoding and Representation with Transformers. Audio feature extractor.
- **Music-RankNet**: Our trained quality assessment model using Siamese RankNet.
- **Rectified flow**: Training objective that learns straight-line noise->data paths.
- **SAO**: Stable Audio Open. Open-source text-to-audio model by Stability AI.
- **VAE**: Variational Autoencoder. Compresses audio to/from latent space.

---

*This document is part of the SAO Score Conditioning project for ISMIR 2026.*
