# SAO-Small Score Conditioning Fine-tuning

> 작성일: 2026-03-13 (최종 업데이트: 2026-03-14)
> 프로젝트: Stable Audio Open Small + Music-RankNet Score Conditioning
> 목적: 리워드 모델 점수를 조건으로 사용하여 음악 생성 품질을 제어하는 파인튜닝 실험

---

## 1. 프로젝트 개요

Stable Audio Open Small (SAO-Small, 497M params) 모델에 Music-RankNet 리워드 점수를 conditioning 신호로 주입하여, 생성 음악의 품질을 제어 가능하게 만드는 연구.

### 핵심 아이디어
- 리워드 모델(Music-RankNet)이 FMA-Large 전체 데이터에 품질 점수를 부여
- 이 점수를 DiT(Diffusion Transformer)에 조건으로 주입
- 추론 시 높은 점수를 조건으로 주면 → 고품질 음악 생성 유도

---

## 2. 실험 설계 (ISMIR Track — FMA-Large)

| Case | 설명 | Model Config | Dataset | Unfreeze Profile | 목적 |
|------|------|-------------|---------|-----------------|------|
| **1** | SAO 원본 (학습 없음) | `model_config.json` | - | - | Baseline 성능 측정 |
| **2** | 전체 FMA SFT | `model_config.json` | `dataset_fma_all.json` | `global` | 파인튜닝 자체의 효과 측정 |
| **3a** | Score: Adapter only | `model_config_with_score.json` | `dataset_fma_scored.json` | `adapter` | 최소한의 파라미터로 점수 주입 |
| **3b** | Score: adaLN only | `model_config_with_score_adaln.json` | `dataset_fma_scored.json` | `adaln` | 모든 트랜스포머 블록에서 점수 모듈레이션 |
| **3c** | Score: Hybrid | `model_config_with_score_adaln.json` | `dataset_fma_scored.json` | `hybrid` | adaLN + Adapter 동시 사용 |
| **4** | 필터링된 FMA | `model_config.json` | `dataset_fma_filtered.json` | `global` | 리워드 기반 데이터 필터링 효과 |

### 실행 명령어
```bash
# Case 3b (adaln) — ★ MODEL_CONFIG 반드시 지정 필요
CUDA_VISIBLE_DEVICES=8,9 SA_UNFREEZE_PROFILE=adaln \
  MODEL_CONFIG=./checkpoints/sao_small/model_config_with_score_adaln.json \
  RUN_NAME=case3b_adaln VAL_EVERY=5000 \
  ./scripts/run_finetune_case3.sh

# Case 3a (adapter) — prepend config 사용 가능
SA_UNFREEZE_PROFILE=adapter RUN_NAME=case3a_adapter ./scripts/run_finetune_case3.sh

# 사용 가능: case2, case3a, case3b, case3c, case4
```

> **주의**: adaln/hybrid profile은 반드시 `model_config_with_score_adaln.json`을 사용해야 함.
> 기본 config(`model_config_with_score.json`)는 `global_cond_type: "prepend"`이므로 adaLN 파라미터가 모델에 존재하지 않아 adaln profile이 no-op이 됨 (03-14 발견).

### ICME Challenge Track (MTG-Jamendo)
- SAO-Small을 MTG-Jamendo로 학습
- **500M 파라미터 제한** → adapter(497.2M) 또는 global(499.0M) profile만 사용 가능
- 전체 데이터 vs 리워드 필터링 vs 점수 조건부 비교
- MTG-Jamendo 오디오 다운로드 완료 (55,701 tracks, 508GB)
- Feature 추출 진행 중 (2026-03-14)

---

## 3. 아키텍처 상세

### 3.1 모델 구조
```
Audio → Oobleck VAE (↓2048x) → 64ch Latent → DiT (16 blocks, 1024 dim) → Latent → VAE Decoder → Audio
                                                  ↑
                                         Conditioning 주입 지점들
```

- **전체 파라미터**: 497M (score 없음) / 504M (adaLN 포함)
- **Diffusion Objective**: Rectified Flow (`rf_denoiser`) — alpha=1-t, sigma=t
- **VAE**: Oobleck, downsampling ratio 2048, latent dim 64

### 3.2 Conditioning 경로 (4가지)

| 경로 | 입력 형태 | 동작 | 사용처 |
|------|----------|------|--------|
| **cross_attention** | (B, Seq, 768) | 트랜스포머 cross-attention | prompt (T5), seconds_total |
| **global_cond (prepend)** | (B, 768) | 시퀀스 앞에 토큰 추가 | seconds_total (기본 모드) |
| **global_cond (adaLN)** | (B, 768) | 매 블록 scale/shift/gate 변조 | seconds_total + score (adaLN 모드) |
| **input_add** | (B, 768, 1) → Conv1d → (B, 64, 1) | latent에 잔차 더하기 | continuous_score |

### 3.3 adaLN (Adaptive Layer Normalization) 상세

adaLN 모드에서는 global conditioning이 트랜스포머의 **모든 블록**에서 scale/shift/gate로 작용:

```python
# TransformerBlock.forward() — adaLN 경로
scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff =
    (to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1)

# Self-Attention에 적용
x = LayerNorm(x) * (1 + scale_self) + shift_self
x = SelfAttention(x) * sigmoid(1 - gate_self)

# FeedForward에 적용
x = LayerNorm(x) * (1 + scale_ff) + shift_ff
x = FeedForward(x) * sigmoid(1 - gate_ff)
```

**파라미터 이름** (unfreeze profile 매칭에 중요):
- `to_scale_shift_gate`: 블록별 학습 가능 파라미터 (nn.Parameter, shape: 6*dim)
- `global_cond_embedder`: 공유 프로젝션 네트워크 (Linear→SiLU→Linear, dim→6*dim)

### 3.4 Score Conditioner

```python
# ContinuousScoreConditioner
score (float) → Linear(1, 768) → unsqueeze(-1) → (B, 768, 1)
# CFG dropout: score = -999.0 → zero vector
```

### 3.5 Dual-Path 설계 (adaLN config)

adaLN 설정에서 `continuous_score`는 **두 경로**로 동시에 들어감:
1. `global_cond_ids` → adaLN scale/shift/gate (매 블록 변조)
2. `input_add_ids` → Conv1d adapter (latent 잔차)

`seconds_total`도 `global_cond_ids`에 포함 → 같은 차원(768)이므로 **sum-merge** 처리.

---

## 4. Unfreeze Profile별 학습 파라미터

| Profile | 텐서 수 | 파라미터 수 | 총 모델 | 학습 대상 | 필요 config |
|---------|---------|-----------|--------|----------|------------|
| **hybrid** | 25 | 9.3M | 504.7M | adaLN 전체 + adapter + global_embed + conditioner | `adaln` config 필수 |
| **adaln** | 22 | 7.4M | 504.7M | to_scale_shift_gate (16블록) + global_cond_embedder + conditioner | `adaln` config 필수 |
| **adapter** | 3 | 50K | 497.2M | input_add_adapter (Conv1d 768→64) + conditioner | prepend OK |
| **global** | 4 | 1.8M | 499.0M | to_global_embed (2 linear layers) + conditioner | prepend OK |
| **minimal** | 2 | 1.5K | 497M | continuous_score.mapper (Linear 1→768) only | prepend OK |

> **ICME 500M 제한**: adapter(497.2M), global(499.0M)만 적합. adaln/hybrid는 504.7M으로 초과.

---

## 5. 학습 설정

### Optimizer & Scheduler
| 항목 | 기본값 | 환경변수 |
|------|--------|---------|
| Optimizer | AdamW | - |
| Learning Rate | 5e-5 | `SA_LR` |
| Weight Decay | 1e-3 | `SA_WEIGHT_DECAY` |
| Warmup Steps | 1,000 | `SA_WARMUP_STEPS` |
| Total Steps | 300,000 | `SA_TOTAL_STEPS` |
| Scheduler | Cosine Hard Restart | - |
| Restart Cycles | 18 | `SA_NUM_CYCLES` |
| CFG Dropout Rate | 0.15 | `SA_CFG_DROP_RATE` |

### Selective CFG (LatCHs 기반)
- `SA_SELECTIVE_CFG_THRESHOLD=0.8` (기본값)
- sigma(=t) < 0.8일 때 CFG 비활성화 → denoising 후반부에서는 무조건 생성
- 1.0으로 설정하면 항상 CFG 적용

### Validation (RewardMonitorCallback)
- 생성 샘플 수: 100 (`SA_VAL_NUM_SAMPLES`) — 10 bins × 10 repeats
- 생성 스텝: 50 (`SA_VAL_GEN_STEPS`)
- CFG Scale: 3.5 (`SA_VAL_CFG_SCALE`)
- 리워드 모델로 점수 매기고, correlation/monotonicity 측정
- Target score: 10-bin 중앙값 사용 (경계값 아님, 03-14 수정)
- Feature 추출: scoring pipeline(04_extract)과 동일한 방식으로 정합 (03-14 수정)

---

## 6. 데이터

### FMA-Large
| 항목 | 값 |
|------|-----|
| 오디오 경로 | `/home/yonghyun/fma/data/fma_large/` |
| JSON sidecar 수 | 103,103개 |
| Sidecar 형식 | `{"reward_score": 0.798, "text": "A Hip-Hop song."}` |
| 전체 메타데이터 | `metadata_fma_all.jsonl` (106,401줄) |
| 스코어 메타데이터 | `metadata_fma_scored.jsonl` (106,401줄) |
| 필터링 메타데이터 | `metadata_fma_filtered.jsonl` (10,640줄, 상위 ~10%) |

> 참고: `custom_metadata_path`는 dataset config에 있지만 실제로 데이터 로딩 코드에서 사용하지 않음. 메타데이터는 오디오 옆의 JSON sidecar에서 읽음.

### Music-RankNet (리워드 모델)
- 아키텍처: Siamese RankNet
- 입력: FLAG(1) + CLAP(512) + MERT(1024) + TEXT(512) = 2049차원
- Hidden: [1024, 512, 256, 128], dropout=0.5
- 체크포인트: `music-ranknet/checkpoints/ultimate_train_all(brainmusic).pt`

---

## 7. 코드 수정 이력

### 7.1 버그 수정 (2026-03-13)

| # | 파일 | 수정 내용 | 이유 |
|---|------|----------|------|
| 1 | `configs/dataset_fma_*.json` (3개) | `"path": ""` → 실제 FMA 경로 | 빈 경로로 데이터 로드 실패 |
| 2 | `finetune.py` UNFREEZE_PROFILES | `"adaLN"` → `"to_scale_shift_gate"`, `"global_cond_embedder"` | 이전 키가 실제 파라미터명과 불일치하여 adaLN 파라미터가 동결된 채 학습됨 |
| 3 | `diffusion.py` get_conditioning_inputs() | 3D→2D shape 정규화 + 같은 dim이면 sum-merge | seconds_total (B,1,768)과 score (B,768,1) concat 시 shape 에러 |
| 4 | `dit.py` forward() | 1536→768 하드코딩 분리 로직 삭제 | 상위에서 이미 처리하므로 dead code |

### 7.2 버그 수정 (2026-03-14) — RewardMonitorCallback

| # | 파일 | 수정 내용 | 이유 |
|---|------|----------|------|
| 5 | `reward_monitor.py` | threshold 키 `bin_{i}_median` → `top_{i}_percent` | 잘못된 키로 모든 target=0.0 → corr/mono 항상 0 |
| 6 | `reward_monitor.py` | 첫 10개 null sample 제거 | 100개 중 10개 낭비 + top_100% bin 미측정 |
| 7 | `reward_monitor.py` | wandb audio key에 bin index 추가 | 동일 키 덮어쓰기로 2개만 표시됨 |
| 8 | `reward_monitor.py` | `sample_rate` → `model_config.get()` | pl_module에 sample_rate 속성 없음 |
| 9 | `reward_monitor.py` | CLAP 로딩: `load_ckpt()` → 수동 `load_state_dict` | laion_clap 1.1.4 `position_ids` 키 호환성 |
| 10 | `reward_monitor.py` | CLAP audio: `get_audio_embedding_from_filelist()` via temp file | `get_audio_embedding_from_data()`와 전처리 차이로 cos sim ~0.05 |
| 11 | `reward_monitor.py` | CLAP text: `get_text_embedding()` + custom tokenizer | 수동 RobertaTokenizer 결과 불일치 + 1.1.4 squeeze 버그 |
| 12 | `reward_thresholds.json` | 경계값(boundary) → 중앙값(median) | 경계값은 bin 대표값으로 부적합 (특히 Top 100%: -5.80 → -1.52) |
| 13 | 실행 시 | `MODEL_CONFIG=model_config_with_score_adaln.json` 필수 | 기본 config(prepend)에는 adaLN 파라미터 미존재 → adaln profile이 no-op |

### 7.3 기능 추가 (2026-03-13)

| # | 파일 | 내용 |
|---|------|------|
| 14 | `model_config_with_score_adaln.json` | adaLN 활성화 config 생성 (`global_cond_type: "adaLN"`, dual-path) |
| 15 | `diffusion.py` DiTWrapper | Selective CFG threshold 환경변수화 (`SA_SELECTIVE_CFG_THRESHOLD`) |
| 16 | `scripts/run_experiments.sh` | 전체 실험 케이스 마스터 런처 생성 |
| 17 | `scripts/run_finetune_case3.sh` | Python 경로, 새 환경변수, help 텍스트 업데이트 |

### 7.4 Pretrained Weight 호환성
- `copy_state_dict()`가 `strict=False`로 동작하여 매칭되는 키만 로드
- adaLN config 사용 시 23개 새 파라미터는 랜덤 초기화 상태로 학습 시작
- `input_add_adapter`는 zero-init (Conv1d weight=0)이므로 초기에 pretrained 동작 보존

---

## 8. 주요 파일 위치

```
stable-audio-tools/
├── finetune.py                          # 파인튜닝 진입점
├── checkpoints/sao_small/
│   ├── model.safetensors                # pretrained weights
│   ├── model_config.json                # 원본 (score 없음)
│   ├── model_config_with_score.json     # score + prepend mode
│   └── model_config_with_score_adaln.json  # score + adaLN mode ★
├── configs/
│   ├── dataset_fma_all.json             # 전체 FMA
│   ├── dataset_fma_scored.json          # 스코어 포함 FMA
│   ├── dataset_fma_filtered.json        # 상위 10% 필터링
│   ├── metadata_fma_all.jsonl
│   ├── metadata_fma_scored.jsonl
│   └── metadata_fma_filtered.jsonl
├── scripts/
│   ├── run_experiments.sh               # 마스터 런처 ★
│   └── run_finetune_case3.sh            # 개별 실행 스크립트
└── stable_audio_tools/
    ├── models/
    │   ├── conditioners.py              # ContinuousScoreConditioner
    │   ├── diffusion.py                 # DiTWrapper, ConditionedDiffusionModelWrapper
    │   ├── dit.py                       # DiffusionTransformer (adaLN 포함)
    │   └── transformer.py              # TransformerBlock (to_scale_shift_gate)
    └── training/
        ├── diffusion.py                 # DiffusionCondTrainingWrapper
        └── reward_monitor.py            # RewardMonitorCallback

music-ranknet/
├── checkpoints/
│   ├── ultimate_train_all(brainmusic).pt  # 리워드 모델
│   └── music_audioset_epoch_15_esc_90.14.pt  # CLAP 체크포인트
└── data/processed/FMA_Scoring/
    └── reward_thresholds.json
```

---

## 9. 현재 진행 상태 & 다음 단계

### 진행 중 (2026-03-14)
- [x] Case 3b v5 (adaln, `model_config_with_score_adaln.json`, 7.4M trainable) 학습 중 — GPU 8,9
- [x] MTG-Jamendo feature 추출 중 — GPU 0,1 (54,753 tracks, ~27시간 예상)

### TODO
- [ ] v5 step 5000 validation에서 correlation/monotonicity 개선 확인
- [ ] Case 2 (SFT baseline) 실행하여 비교군 확보
- [ ] Case 3a (adapter), Case 3c (hybrid) 실행
- [ ] ICME Track: Jamendo scoring → threshold 계산 → 학습 파이프라인 구축
- [ ] Score normalization 전략 검토 (현재 raw score, 범위 ~[-5.8, +1.5])
