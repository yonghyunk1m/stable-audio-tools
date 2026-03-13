# SAO-Small Score Conditioning: 전체 아키텍처 및 학습 플로우 상세 문서

> 작성일: 2026-03-13
> 목적: 데이터 → 모델 → 학습 → 추론까지의 전체 흐름을 코드 수준에서 추적

---

## 1. 전체 시스템 개요도

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE 전체 흐름                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │ FMA-Large│    │  JSON Sidecar    │    │   ContinuousScoreDataset     │   │
│  │  .mp3    │───▶│ {reward_score,   │───▶│   Wrapper                    │   │
│  │ 103K files│   │  text}           │    │ • CFG dropout (15%)          │   │
│  └──────────┘    └──────────────────┘    │ • score → metadata           │   │
│                                          └───────────┬──────────────────┘   │
│                                                      │                       
│                              ┌───────────────────────┘                      │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    DataLoader (batch_size=2)                         │   │
│  │   Output: (audio_tensor, metadata_dict) × batch                      │   │
│  │   • audio: (B, 2, 524288) ─ stereo, 44.1kHz, ~11.9초                 │   
│  │   • metadata: {prompt, seconds_total, continuous_score, path, ...}   │   │
│  └──────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │              DiffusionCondTrainingWrapper.training_step()            │   │
│  │                                                                      │   │
│  │  1. audio → Oobleck VAE Encoder → latent (B, 64, 256)              │   │
│  │  2. metadata → MultiConditioner → conditioning_tensors              │   │
│  │  3. t ~ Uniform(0,1), noise ~ N(0,I)                               │   │
│  │  4. noised = latent × (1-t) + noise × t                            │   │
│  │  5. target = noise - latent                                         │   │
│  │  6. output = DiT(noised, t, conditions)                             │   │
│  │  7. loss = MSE(output, target)                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 데이터 파이프라인 상세

### 2.1 원본 데이터 구조

```
/home/yonghyun/fma/data/fma_large/
├── 000/
│   ├── 000002.mp3          ← 오디오 파일
│   ├── 000002.json         ← JSON sidecar (메타데이터)
│   ├── 000005.mp3
│   ├── 000005.json
│   └── ...
├── 001/
│   └── ...
└── 155/
    └── ...

총 103,103개 오디오 + JSON sidecar 쌍
```

### 2.2 JSON Sidecar 내용

```json
{
    "reward_score": 0.798,
    "text": "A Hip-Hop song."
}
```

- `reward_score`: Music-RankNet이 부여한 품질 점수 (float, 0~1 범위)
- `text`: 장르 기반 텍스트 프롬프트

### 2.3 데이터 로딩 흐름

```
┌────────────────────────────────────────────────────────────────────┐
│                          데이터 로딩 흐름                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   1. create_dataloader_from_config()                               │
│      ├── SampleDataset: 오디오를 44.1kHz, stereo로 로드             │
│      │   • sample_size=524288 (≈11.9초)                            │
│      │   • 긴 파일은 랜덤 crop, 짧으면 zero-pad                    │
│      └── 반환: (audio_tensor, metadata_dict)                       │
│                                                                    │
│   2. ContinuousScoreDatasetWrapper (finetune.py)                   │
│      ├── 내부 dataset을 감싸서 score를 metadata에 주입              │
│      │                                                              │
│      │   ┌─ 정상 (85%) ─────────────────────────────────┐          │
│      │   │ metadata['continuous_score'] = 0.798 (실제값) │          │
│      │   └──────────────────────────────────────────────┘          │
│      │   ┌─ CFG Dropout (15%) ──────────────────────────┐          │
│      │   │ metadata['continuous_score'] = -999.0 (null)  │          │
│      │   └──────────────────────────────────────────────┘          │
│      │                                                              │
│      └── 손상된 mp3 → try/except로 다음 인덱스 시도 (DDP 안전)     │
│                                                                    │
│   3. DataLoader(batch_size=2, num_workers=6, pin_memory=True)      │
│      └── 출력 형태:                                                 │
│          audio:  (2, 2, 524288)  — [batch, channels, samples]      │
│          meta:   [{prompt: "...", seconds_total: 10,                │
│                    continuous_score: 0.798}, ...]                   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Conditioning 시스템 상세

### 3.1 MultiConditioner 처리 과정

MultiConditioner는 metadata dict를 받아 각 conditioner별로 텐서를 생성합니다.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MultiConditioner.forward()                            │
│                    (conditioners.py:675)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   입력: batch_metadata = [                                              │
│     {"prompt":"Hip-Hop song", "seconds_total":10, "continuous_score":0.8,│
│      "path":"/fma/000/000002.mp3"},                                     │
│     {"prompt":"Piano riff",   "seconds_total":10, "continuous_score":-999}│
│   ]                                                                     │
│                                                                         │
│   Step 1: JSON sidecar 주입                                             │
│   ├── path에서 .json 찾아서 reward_score, text 주입                     │
│   └── 기본값: prompt="A music song.", score_bin=0, reward_score=0.0     │
│                                                                         │
│   Step 2: 각 conditioner 실행                                           │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │ "prompt" → T5Conditioner                                      │     │
│   │   • T5-base 인코더 (110M params, frozen)                      │     │
│   │   • 텍스트 → 토큰화(max_length=64) → T5 encoder               │     │
│   │   • 출력: (B, 64, 768) + attention_mask (B, 64)               │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │ "seconds_total" → NumberConditioner                            │     │
│   │   • NumberEmbedder: Fourier features → Linear → 768-dim       │     │
│   │   • 입력: [10.0, 10.0] (초 단위)                               │     │
│   │   • 정규화: (x - min) / (max - min), min=0, max=256            │     │
│   │   • 출력: (B, 1, 768) + mask (B, 1)                           │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │ "continuous_score" → ContinuousScoreConditioner                │     │
│   │   • Linear(1 → 768)                                           │     │
│   │   • 입력: [0.798, -999.0]                                      │     │
│   │   • -999.0(null) → zero vector 강제                            │     │
│   │   • 출력: (B, 768, 1) + mask (B, 1)                           │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   출력: conditioning_tensors = {                                        │
│     "prompt":           [(B,64,768), (B,64)],                           │
│     "seconds_total":    [(B,1,768),  (B,1)],                            │
│     "continuous_score": [(B,768,1),  (B,1)],                            │
│   }                                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Conditioning 라우팅 (get_conditioning_inputs)

JSON config에서 지정된 라우팅 규칙에 따라 텐서가 각 경로로 분배됩니다.

```
┌─────────────────────────────────────────────────────────────────────────┐
│               Conditioning Routing (adaLN config 기준)                   │
│               (diffusion.py:165 get_conditioning_inputs)                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Config에서 정의된 라우팅 규칙:                                        │
│   ┌────────────────────────────────────────────────────────┐            │
│   │ cross_attention_cond_ids: ["prompt", "seconds_total"]  │            │
│   │ global_cond_ids:          ["seconds_total",            │            │
│   │                            "continuous_score"]          │            │
│   │ input_add_ids:            ["continuous_score"]          │            │
│   └────────────────────────────────────────────────────────┘            │
│                                                                         │
│   ※ 하나의 conditioner가 여러 경로에 동시 사용 가능                    │
│      (seconds_total → cross_attn + global)                              │
│      (continuous_score → global + input_add)                            │
│                                                                         │
│                                                                         │
│   ┌─ Cross-Attention 경로 ──────────────────────────────────────┐      │
│   │                                                              │      │
│   │  prompt:        (B, 64, 768)  ─┐                             │      │
│   │                                ├─ cat(dim=1) ─▶ (B, 65, 768) │      │
│   │  seconds_total: (B,  1, 768)  ─┘                             │      │
│   │                                                              │      │
│   │  → DiT의 cross-attention context로 전달                     │      │
│   │  → 모든 latent 토큰이 이 65개 토큰에 attend                 │      │
│   │                                                              │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│   ┌─ Global Conditioning 경로 ──────────────────────────────────┐      │
│   │                                                              │      │
│   │  seconds_total:    (B, 1, 768) ─squeeze─▶ (B, 768) ─┐       │      │
│   │                                                      │       │      │
│   │                                            같은 dim? │ YES   │      │
│   │                                                      ▼       │      │
│   │  continuous_score: (B, 768, 1) ─squeeze─▶ (B, 768) ─ SUM    │      │
│   │                                                      │       │      │
│   │                                              ┌───────┘       │      │
│   │                                              ▼               │      │
│   │                                         (B, 768)             │      │
│   │                                                              │      │
│   │  → DiT의 global_embed로 전달                                │      │
│   │  → adaLN: 모든 블록의 scale/shift/gate 변조                 │      │
│   │  → prepend: 시퀀스 앞에 토큰으로 추가 (비-adaLN 모드)       │      │
│   │                                                              │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│   ┌─ Input-Add 경로 ────────────────────────────────────────────┐      │
│   │                                                              │      │
│   │  continuous_score: (B, 768, 1) ───▶ Conv1d(768→64, k=1)     │      │
│   │                                     (zero-init weights)      │      │
│   │                                          │                   │      │
│   │                                          ▼                   │      │
│   │                                     (B, 64, 1)               │      │
│   │                                          │                   │      │
│   │                              interpolate to seq_len          │      │
│   │                                          │                   │      │
│   │                                          ▼                   │      │
│   │                                     (B, 64, 256)             │      │
│   │                                          │                   │      │
│   │                         latent_input  +  residual             │      │
│   │                         (B,64,256)    =  (B,64,256)          │      │
│   │                                                              │      │
│   │  → latent에 직접 더해지는 잔차 신호                         │      │
│   │  → zero-init이라 학습 초기에는 영향 없음                    │      │
│   │                                                              │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 모델 아키텍처 상세

### 4.1 전체 모델 구조

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ConditionedDiffusionModelWrapper                      │
│                    (전체 모델, ~504M params)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─ Pretransform (Oobleck VAE, ~115M params, FROZEN) ──────────────┐  │
│   │                                                                  │  │
│   │   Encoder:                                                       │  │
│   │   (B, 2, 524288) ─▶ Conv1d layers ─▶ Snake activations          │  │
│   │                     channels: [128, 256, 512, 1024, 2048]        │  │
│   │                     strides:  [2,   4,   4,   8,    8]           │  │
│   │                     총 downsampling: 2×4×4×8×8 = 2048            │  │
│   │                  ─▶ Linear(2048→128) ─▶ VAE bottleneck           │  │
│   │                  ─▶ μ, σ → sample → (B, 64, 256)                │  │
│   │                                                                  │  │
│   │   ※ latent_dim = 64 (encoder 128 → VAE 분할 → decoder 64)      │  │
│   │   ※ 524288 / 2048 = 256 (시간축 길이)                           │  │
│   │                                                                  │  │
│   │   Decoder:                                                       │  │
│   │   (B, 64, 256) ─▶ 역순 upsampling ─▶ (B, 2, 524288)            │  │
│   │                                                                  │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   ┌─ MultiConditioner (~110M params, T5 부분 FROZEN) ────────────────┐ │
│   │                                                                  │  │
│   │   T5Conditioner("prompt"):     T5-base encoder (110M, frozen)    │  │
│   │   NumberConditioner("seconds"): NumberEmbedder → (B, 1, 768)     │  │
│   │   ContinuousScoreConditioner:   Linear(1→768) → (B, 768, 1)     │  │
│   │                                                                  │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   ┌─ DiTWrapper → DiffusionTransformer (~280M params) ───────────────┐ │
│   │                                                                  │  │
│   │   (아래 4.2절에서 상세 설명)                                     │  │
│   │                                                                  │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 DiffusionTransformer (DiT) 내부 구조

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DiffusionTransformer._forward()                       │
│                    (dit.py:150)                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   입력:                                                                 │
│   x             = (B, 64, 256)   noised latent                          │
│   t             = (B,)           timestep [0, 1]                        │
│   cross_attn    = (B, 65, 768)   text + seconds tokens                  │
│   global_embed  = (B, 768)       seconds + score (sum-merged)           │
│   input_add     = (B, 768, 1)    score for channel residual             │
│                                                                         │
│                                                                         │
│   Step 1: Input-Add Residual                                            │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │  input_add_cond: (B, 768, 1)                                 │      │
│   │       │                                                      │      │
│   │       ▼                                                      │      │
│   │  input_add_adapter: Conv1d(768→64, k=1, zero-init)           │      │
│   │       │                                                      │      │
│   │       ▼                                                      │      │
│   │  (B, 64, 1) ─── interpolate ──▶ (B, 64, 256)                │      │
│   │       │                                                      │      │
│   │  x = x + adapter_output   ← 잔차 더하기                     │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│   Step 2: Timestep Embedding                                            │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │  t: (B,) ─▶ FourierFeatures(1→256) ─▶ MLP(256→1024→1024)    │      │
│   │            ─▶ timestep_embed: (B, 1024)                      │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│   Step 3: Global Embed Processing                                       │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │  global_embed: (B, 768)                                      │      │
│   │       │                                                      │      │
│   │       ▼                                                      │      │
│   │  to_global_embed: Linear(768→1024) ─▶ SiLU ─▶ Linear(1024)  │      │
│   │       │                      (pretrained layers)             │      │
│   │       ▼                                                      │      │
│   │  global_embed: (B, 1024)                                     │      │
│   │       │                                                      │      │
│   │       + timestep_embed: (B, 1024)                            │      │
│   │       │                                                      │      │
│   │       ▼                                                      │      │
│   │  global_embed: (B, 1024)  ← timestep + conditioning 결합    │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│   Step 4: Preprocess                                                    │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │  x: (B, 64, 256) ─▶ preprocess_conv(zero-init) + x          │      │
│   │  x: (B, 64, 256) ─▶ rearrange("b c t -> b t c")             │      │
│   │  x: (B, 256, 1024) ← project_in: Linear(64→1024)            │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│   Step 5: adaLN Projection (NEW, zero-init)                             │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │  global_embed: (B, 1024)                                     │      │
│   │       │                                                      │      │
│   │       ▼                                                      │      │
│   │  global_cond_embedder:                                       │      │
│   │    Linear(1024→1024) ─▶ SiLU ─▶ Linear(1024→6144)           │      │
│   │       │                                                      │      │
│   │       ▼                                                      │      │
│   │  global_cond: (B, 6144)  ← 6×1024 for scale/shift/gate      │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│   Step 6: Transformer Blocks × 16                                       │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │                                                              │      │
│   │  (아래 4.3절에서 상세 설명)                                  │      │
│   │                                                              │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│   Step 7: Postprocess                                                   │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │  x: (B, 256, 1024) ─▶ project_out: Linear(1024→64)          │      │
│   │  x: (B, 256, 64)   ─▶ rearrange("b t c -> b c t")           │      │
│   │  x: (B, 64, 256)   ─▶ postprocess_conv(zero-init) + x       │      │
│   │  output: (B, 64, 256)  ← 예측된 velocity field              │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 TransformerBlock (adaLN 모드) 상세

```
┌─────────────────────────────────────────────────────────────────────────┐
│               TransformerBlock with adaLN (× 16 blocks)                 │
│               (transformer.py:658)                                      │
│                                                                         │
│   각 블록의 학습 가능 파라미터:                                         │
│   • to_scale_shift_gate: nn.Parameter(6144)  ← 블록별 bias term        │
│   • self_attn: MultiHeadAttention (1024-dim, 8 heads, RoPE)            │
│   • cross_attn: MultiHeadAttention (context: 768→1024)                 │
│   • ff: FeedForward (1024→4096→1024)                                   │
│   • pre_norm, ff_norm: LayerNorm(1024)                                 │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   입력: x = (B, 256, 1024),  global_cond = (B, 6144)                   │
│                                                                         │
│   ┌─ Scale/Shift/Gate 분리 ──────────────────────────────────────┐     │
│   │                                                              │     │
│   │  (to_scale_shift_gate + global_cond)                         │     │
│   │          (B, 6144)                                           │     │
│   │              │                                               │     │
│   │         .chunk(6, dim=-1)                                    │     │
│   │              │                                               │     │
│   │    ┌─────────┼─────────┬─────────┬─────────┬─────────┐      │     │
│   │    ▼         ▼         ▼         ▼         ▼         ▼      │     │
│   │ scale_s  shift_s  gate_s   scale_f  shift_f  gate_f         │     │
│   │ (B,1024) (B,1024) (B,1024) (B,1024) (B,1024) (B,1024)      │     │
│   │                                                              │     │
│   │ _s = self-attention용,  _f = feed-forward용                  │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   ┌─ Self-Attention with adaLN ──────────────────────────────────┐     │
│   │                                                              │     │
│   │  residual = x                                                │     │
│   │                                                              │     │
│   │  x = LayerNorm(x)                        ← pre_norm         │     │
│   │  x = x × (1 + scale_s) + shift_s         ← adaLN 변조      │     │
│   │  x = MultiHeadAttention(x, rope=RoPE)    ← self-attention   │     │
│   │  x = x × sigmoid(1 - gate_s)             ← gating           │     │
│   │  x = LayerScale(x)                       ← learnable scale  │     │
│   │  x = x + residual                        ← skip connection  │     │
│   │                                                              │     │
│   │  ※ zero-init 시: scale_s=0, shift_s=0, gate_s=0             │     │
│   │     → x×(1+0)+0 = x (identity)                              │     │
│   │     → x×sigmoid(1-0) = x×0.731 (slight scaling)             │     │
│   │                                                              │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   ┌─ Cross-Attention (텍스트 조건) ──────────────────────────────┐     │
│   │                                                              │     │
│   │  x = x + CrossAttention(                                     │     │
│   │            query = LayerNorm(x),      (B, 256, 1024)         │     │
│   │            context = cross_attn_cond  (B, 65, 768→1024)      │     │
│   │        )                                                     │     │
│   │                                                              │     │
│   │  ※ cross-attention은 adaLN 변조를 받지 않음                 │     │
│   │                                                              │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   ┌─ Feed-Forward with adaLN ────────────────────────────────────┐     │
│   │                                                              │     │
│   │  residual = x                                                │     │
│   │                                                              │     │
│   │  x = LayerNorm(x)                        ← ff_norm          │     │
│   │  x = x × (1 + scale_f) + shift_f         ← adaLN 변조      │     │
│   │  x = FeedForward(x)                      ← MLP (1024→4096)  │     │
│   │  x = x × sigmoid(1 - gate_f)             ← gating           │     │
│   │  x = LayerScale(x)                       ← learnable scale  │     │
│   │  x = x + residual                        ← skip connection  │     │
│   │                                                              │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   출력: x = (B, 256, 1024)                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 학습 과정 상세

### 5.1 Training Step (한 스텝의 전체 흐름)

```
┌─────────────────────────────────────────────────────────────────────────┐
│            DiffusionCondTrainingWrapper.training_step()                  │
│            (training/diffusion.py:350~500)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   입력: batch = (audio, metadata)                                       │
│                                                                         │
│   ┌─ Step 1: Pretransform Encode ────────────────────────────────┐     │
│   │                                                              │     │
│   │   audio: (B, 2, 524288)                                      │     │
│   │     │                                                        │     │
│   │     ▼  Oobleck VAE Encoder (frozen, fp16)                    │     │
│   │                                                              │     │
│   │   diffusion_input: (B, 64, 256)                              │     │
│   │                                                              │     │
│   │   ※ downsampling: 524288 / 2048 = 256 timesteps              │     │
│   │   ※ 64 latent channels                                      │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   ┌─ Step 2: Conditioning ───────────────────────────────────────┐     │
│   │                                                              │     │
│   │   metadata → MultiConditioner.forward()                      │     │
│   │                                                              │     │
│   │   conditioning = {                                           │     │
│   │     "prompt":           [(B,64,768), (B,64)],                │     │
│   │     "seconds_total":    [(B,1,768),  (B,1)],                 │     │
│   │     "continuous_score": [(B,768,1),  (B,1)],                 │     │
│   │   }                                                          │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   ┌─ Step 3: Noise Schedule (Rectified Flow) ────────────────────┐     │
│   │                                                              │     │
│   │   t ~ Uniform(0, 1)           배치별 랜덤 timestep           │     │
│   │                                                              │     │
│   │   alpha = 1 - t               signal 비율 (t=0이면 clean)    │     │
│   │   sigma = t                   noise 비율  (t=1이면 pure noise)│     │
│   │                                                              │     │
│   │   noise = randn_like(diffusion_input)                        │     │
│   │                                                              │     │
│   │   noised = diffusion_input × alpha + noise × sigma           │     │
│   │         = diffusion_input × (1-t) + noise × t                │     │
│   │                                                              │     │
│   │   ※ t=0: noised = diffusion_input (원본)                    │     │
│   │   ※ t=1: noised = noise (순수 노이즈)                       │     │
│   │   ※ t=0.5: 반반 섞임                                        │     │
│   │                                                              │     │
│   │   target = noise - diffusion_input                           │     │
│   │                                                              │     │
│   │   ※ RF objective: velocity = d/dt(noised)                   │     │
│   │     = d/dt[(1-t)×data + t×noise]                             │     │
│   │     = noise - data                                           │     │
│   │                                                              │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   ┌─ Step 4: Model Forward ──────────────────────────────────────┐     │
│   │                                                              │     │
│   │   output = DiT(                                              │     │
│   │     x = noised,              (B, 64, 256)                    │     │
│   │     t = timestep,            (B,)                            │     │
│   │     cond = conditioning,     위의 dict                       │     │
│   │   )                                                          │     │
│   │                                                              │     │
│   │   output: (B, 64, 256)  ← 예측된 velocity                   │     │
│   │                                                              │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   ┌─ Step 5: Loss Computation ───────────────────────────────────┐     │
│   │                                                              │     │
│   │   loss = MSE(output, target)                                 │     │
│   │                                                              │     │
│   │   = MSE(predicted_velocity, noise - data)                    │     │
│   │                                                              │     │
│   │   ※ padding_mask가 있으면 해당 영역 제외                    │     │
│   │                                                              │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   ┌─ Step 6: Backward & Optimizer ───────────────────────────────┐     │
│   │                                                              │     │
│   │   loss.backward()                                            │     │
│   │                                                              │     │
│   │   ※ gradient_clip_val=1.0                                   │     │
│   │   ※ accumulate_grad_batches=4                               │     │
│   │     → 4 forward passes 후 1번 optimizer.step()               │     │
│   │   ※ effective_batch = 2 × 2(GPU) × 4(accum) = 16            │     │
│   │                                                              │     │
│   │   AdamW(lr=5e-5, betas=(0.9,0.95), weight_decay=1e-3)       │     │
│   │   Scheduler: CosineWithHardRestarts(warmup=1K, cycles=18)    │     │
│   │                                                              │     │
│   │   ※ adaln profile: 22 tensors (7.4M params)만 업데이트      │     │
│   │     나머지 ~497M params는 frozen                             │     │
│   │                                                              │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Freeze/Unfreeze 전략

```
┌─────────────────────────────────────────────────────────────────────────┐
│               Parameter Freeze Map (adaln profile)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ██ = FROZEN (gradient 비활성)                                        │
│   ░░ = TRAINABLE (gradient 활성)                                       │
│                                                                         │
│   Oobleck VAE Encoder          ██████████████████ (frozen, ~57M)        │
│   Oobleck VAE Decoder          ██████████████████ (frozen, ~58M)        │
│   T5-base Encoder              ██████████████████ (frozen, ~110M)       │
│   NumberConditioner             ██████████████████ (frozen)              │
│   ContinuousScoreConditioner    ░░░░░░░░░░░░░░░░ (trainable, 769)      │
│                                                                         │
│   DiffusionTransformer:                                                 │
│     to_cond_embed (cross-attn)  ██████████████████ (frozen)             │
│     to_global_embed (global)    ██████████████████ (frozen)             │
│     preprocess_conv             ██████████████████ (frozen)             │
│     postprocess_conv            ██████████████████ (frozen)             │
│     to_timestep_embed           ██████████████████ (frozen)             │
│     input_add_adapter           ██████████████████ (frozen*)            │
│                                                                         │
│   ContinuousTransformer:                                                │
│     global_cond_embedder        ░░░░░░░░░░░░░░░░ (trainable, 7.3M)    │
│     project_in / project_out    ██████████████████ (frozen)             │
│     rotary_pos_emb              ██████████████████ (frozen)             │
│                                                                         │
│   TransformerBlock × 16:                                                │
│     to_scale_shift_gate         ░░░░░░░░░░░░░░░░ (trainable, 98K)     │
│     pre_norm, ff_norm           ██████████████████ (frozen)             │
│     self_attn (QKV, O)          ██████████████████ (frozen)             │
│     cross_attn (QKV, O)         ██████████████████ (frozen)             │
│     ff (MLP)                    ██████████████████ (frozen)             │
│                                                                         │
│   ※ hybrid profile에서는 input_add_adapter,                           │
│     to_global_embed도 trainable                                        │
│                                                                         │
│   Total trainable (adaln): 22 tensors, 7.4M params (1.48%)             │
│   Total frozen:            ~497M params (98.52%)                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Learning Rate Schedule

```
    LR
   5e-5 ┤         ╱╲    ╱╲    ╱╲    ╱╲    ╱╲    ╱╲
        │        ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲     ...×18 cycles
        │       ╱    ╲╱    ╲╱    ╲╱    ╲╱    ╲╱    ╲╱
        │      ╱
        │     ╱  warmup (1000 steps)
   0    ┤────╱
        └─────┬──────────────────────────────────────── steps
              0    1K        ...                 300K

   • Cosine with Hard Restarts
   • 18 cycles over 300K steps = ~16.7K steps/cycle
   • 각 cycle 시작마다 LR이 peak(5e-5)로 리셋
```

---

## 6. 추론 (Generation) 과정

### 6.1 Classifier-Free Guidance (CFG) with Selective CFG

```
┌─────────────────────────────────────────────────────────────────────────┐
│              CFG Inference (DiTWrapper.forward, dit.py)                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   입력: x_t (noised latent), t (timestep), conditions, cfg_scale=3.5   │
│                                                                         │
│   ┌─ Selective CFG Check ────────────────────────────────────────┐     │
│   │                                                              │     │
│   │  if t.max() < SA_SELECTIVE_CFG_THRESHOLD (default: 0.8):     │     │
│   │      cfg_scale = 1.0  ← CFG 비활성화                        │     │
│   │                                                              │     │
│   │  ※ sigma(=t) < 0.8 → denoising 후반부 (낮은 노이즈)        │     │
│   │  ※ 후반부에서 CFG를 끄면 artifact 감소                      │     │
│   │                                                              │     │
│   │  Timeline:                                                   │     │
│   │  t=1.0  ────── 0.8 ────────────────── 0.0                   │     │
│   │  [pure noise]  │     [low noise]      [clean]               │     │
│   │  ◄── CFG ON ──►│◄──── CFG OFF ──────►│                      │     │
│   │                                                              │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   ┌─ Batch-Doubled Forward (cfg_scale != 1.0일 때) ──────────────┐    │
│   │                                                              │     │
│   │  batch_x = cat([x_t, x_t], dim=0)     ← 2배 배치           │     │
│   │  batch_t = cat([t, t], dim=0)                                │     │
│   │                                                              │     │
│   │  ┌─ Conditional half ────────────────────────────┐           │     │
│   │  │  cross_attn: 실제 텍스트 임베딩               │           │     │
│   │  │  global:     실제 global conditioning          │           │     │
│   │  │  input_add:  실제 score conditioning           │           │     │
│   │  └───────────────────────────────────────────────┘           │     │
│   │                                                              │     │
│   │  ┌─ Unconditional half ──────────────────────────┐           │     │
│   │  │  cross_attn: zero embeddings (null prompt)    │           │     │
│   │  │  global:     negative_global_embed (있으면)    │           │     │
│   │  │  input_add:  negative_input_add (있으면)       │           │     │
│   │  └───────────────────────────────────────────────┘           │     │
│   │                                                              │     │
│   │  batch_output = DiT._forward(batch_x, batch_t, ...)         │     │
│   │                                                              │     │
│   │  cond_out, uncond_out = batch_output.chunk(2, dim=0)         │     │
│   │                                                              │     │
│   │  output = uncond_out + cfg_scale × (cond_out - uncond_out)   │     │
│   │                                                              │     │
│   │  ※ cfg_scale=1.0: output = cond_out (no guidance)           │     │
│   │  ※ cfg_scale=3.5: conditional 방향으로 3.5배 증폭           │     │
│   │                                                              │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 ODE Sampling (Rectified Flow)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 Sampling Process (N steps)                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   x_1 ~ N(0, I)                    ← 순수 노이즈에서 시작              │
│                                                                         │
│   for i = N-1 ... 0:                                                    │
│       t = i / N                                                         │
│       dt = -1 / N                                                       │
│                                                                         │
│       velocity = DiT(x_t, t, conditions, cfg_scale)                     │
│                                                                         │
│       x_{t+dt} = x_t + velocity × dt                                   │
│                                                                         │
│       ※ RF에서 velocity = noise - data                                 │
│       ※ noised = (1-t)×data + t×noise                                  │
│       ※ d(noised)/dt = noise - data = velocity                         │
│       ※ 따라서 ODE를 t=1→0으로 풀면 data 복원                         │
│                                                                         │
│   x_0 ≈ clean latent                                                    │
│                                                                         │
│   audio = VAE_Decoder(x_0)          ← (B, 64, 256) → (B, 2, 524288)   │
│                                                                         │
│   Demo 설정: demo_steps=8 (빠른 미리보기)                               │
│   Validation: gen_steps=50 (고품질)                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Score Conditioning의 작동 원리

### 7.1 학습 시 Score의 흐름

```
                          reward_score = 0.798
                                │
                    ┌───────────┼───────────────────┐
                    │           │                    │
                    │     15% 확률로                 │
                    │     -999.0 대체                │
                    │     (CFG dropout)              │
                    │           │                    │
                    │           ▼                    │
                    │   ContinuousScoreConditioner   │
                    │   Linear(1 → 768)              │
                    │           │                    │
                    │           ▼                    │
                    │      (B, 768, 1)               │
                    │     ╱           ╲              │
                    │    ╱             ╲             │
                    ▼   ▼               ▼            │
               Global Path         Input-Add Path   │
              (sum with            (Conv1d adapter)  │
               seconds_total)                        │
                    │                    │            │
                    ▼                    ▼            │
              adaLN 변조          Latent 잔차 더하기  │
           (매 블록 scale/         (채널 단위)        │
            shift/gate)                              │
                    │                    │            │
                    └────────┬───────────┘            │
                             │                        │
                             ▼                        │
                    DiT가 score를 인식하고             │
                    score에 맞는 denoising 학습        │
                                                      │
                                                      │
     -999.0 (null)인 경우:                             │
       → ContinuousScoreConditioner가 zero vector 반환│
       → Global path: seconds_total만 남음            │
       → Input-Add path: zero contribution            │
       → 모델은 "score 없이" denoising 학습           │
       → 추론 시 unconditional branch로 활용           │
                                                      │
└─────────────────────────────────────────────────────┘
```

### 7.2 추론 시 Score 조절

```
추론 시나리오: "A beautiful piano melody" + score=10.0 (고품질 유도)

   score=10.0 ──▶ ContinuousScoreConditioner ──▶ 강한 conditioning signal
                                                     │
                    ┌────────────────────────────────┘
                    │
                    ▼
          CFG가 conditional(score=10) vs unconditional(score=null) 차이를 증폭
          → 고품질 방향으로 생성 유도

   score=0.0 ──▶ 약한 conditioning signal
          → 낮은 품질 (또는 "평균적" 품질) 방향

   score의 크기가 클수록 → Linear(1→768)의 출력 크기 증가
   → adaLN scale/shift가 더 강하게 변조
   → input_add residual이 더 강하게 작용
   → 모델이 "이건 고품질이어야 해"라고 인식
```

---

## 8. Validation & Monitoring

### 8.1 RewardMonitorCallback 흐름

```
┌─────────────────────────────────────────────────────────────────────────┐
│                RewardMonitorCallback (매 5000 steps)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   1. Validation 데이터에서 100개 샘플 추출                              │
│      └── prompt, seconds_total, continuous_score 포함                   │
│                                                                         │
│   2. 각 샘플에 대해 오디오 생성 (50 steps, cfg=3.5)                    │
│      └── 생성된 오디오: (100, 2, 524288) — stereo, 44.1kHz             │
│                                                                         │
│   3. Music-RankNet으로 생성 오디오 점수 매기기                          │
│      ├── CLAP embedding 추출 (512-dim)                                 │
│      ├── MERT embedding 추출 (1024-dim)                                │
│      ├── Text embedding 추출 (512-dim)                                 │
│      ├── Feature concat: [FLAG(1), CLAP(512), MERT(1024), TEXT(512)]   │
│      │                    = 2049-dim                                    │
│      └── RankNet forward → predicted quality score                     │
│                                                                         │
│   4. Metrics 계산 및 wandb 로깅                                        │
│      ├── Spearman correlation (입력 score vs 생성 품질)                │
│      ├── Score monotonicity (높은 score → 높은 품질?)                  │
│      ├── Mean predicted score                                          │
│      └── (선택) 오디오 파일 저장 및 wandb에 업로드                     │
│                                                                         │
│   목표: 입력 score와 생성 품질의 상관관계가 높아지는 것                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 텐서 Shape 요약 (Quick Reference)

```
위치                          텐서 이름              Shape            비고
─────────────────────────────────────────────────────────────────────────
DataLoader 출력              audio                  (B, 2, 524288)   stereo raw audio
                             metadata               dict             per-sample

VAE Encoder 출력             diffusion_input        (B, 64, 256)     latent space

T5 Encoder 출력              prompt embeddings      (B, 64, 768)     max_length=64
NumberConditioner 출력        seconds embedding      (B, 1, 768)      single token
ContinuousScoreConditioner   score embedding        (B, 768, 1)      single value

Cross-Attention              context                (B, 65, 768)     prompt+seconds
Global Cond (sum-merged)     global_embed           (B, 768)         seconds+score
Input-Add Cond               input_add_cond         (B, 768, 1)      score only

DiffusionTransformer 내부:
  to_global_embed 출력       global_embed           (B, 1024)        projected
  + timestep_embed           global_embed           (B, 1024)        combined
  global_cond_embedder 출력  global_cond            (B, 6144)        for adaLN
  to_scale_shift_gate        (per block)             (6144,)         learnable bias
  transformer input          x                      (B, 256, 1024)   after project_in
  transformer output         x                      (B, 256, 1024)   16 blocks later
  model output               velocity               (B, 64, 256)     after project_out

Training:
  noise                      noise                  (B, 64, 256)     ~ N(0,I)
  noised_inputs              noised                 (B, 64, 256)     (1-t)×data + t×noise
  targets                    velocity target        (B, 64, 256)     noise - data
  loss                       MSE                    scalar           mean over all dims
```

---

## 10. 파일별 역할 요약

```
finetune.py
 ├── main()                          학습 진입점, 모델/데이터/옵티마이저 생성
 ├── ContinuousScoreDatasetWrapper   score 주입 + CFG dropout
 ├── zero_init_new_params()          pretrained에 없는 파라미터 zero-init
 ├── unfreeze_finetune_params()      profile에 따라 파라미터 선택적 unfreeze
 └── attach_custom_optimizer()       AdamW + CosineHardRestart scheduler

models/conditioners.py
 ├── T5Conditioner                   텍스트 → T5 → (B, seq, 768)
 ├── NumberConditioner               숫자 → Fourier → (B, 1, 768)
 ├── ContinuousScoreConditioner      score → Linear → (B, 768, 1)
 └── MultiConditioner.forward()      JSON sidecar 읽기 + 모든 conditioner 실행

models/diffusion.py
 ├── ConditionedDiffusionModelWrapper
 │   ├── get_conditioning_inputs()   conditioner 출력을 4개 경로로 라우팅
 │   └── forward()                   conditioning + 모델 forward
 └── DiTWrapper
     └── forward()                   Selective CFG + batch-doubled CFG

models/dit.py
 └── DiffusionTransformer
     ├── _forward()                  입력 처리 → transformer → 출력 처리
     └── forward()                   CFG 로직 (conditional/unconditional 분기)

models/transformer.py
 ├── TransformerBlock                adaLN scale/shift/gate + attention + FFN
 └── ContinuousTransformer          16개 block 스택 + global_cond_embedder

training/diffusion.py
 └── DiffusionCondTrainingWrapper
     └── training_step()             noise schedule → model forward → MSE loss

training/reward_monitor.py
 └── RewardMonitorCallback           validation 시 생성 → 리워드 평가 → 로깅
```
