# Score-Conditioned SAO-Small: 전체 파이프라인 완전 해설

**최종 수정일**: 2026-03-17

이 문서는 스코어 조건부 SAO-Small 파이프라인의 **모든 구성 요소**를
코드 레벨에서 추적하여 설명합니다. 텐서 형상, 파일 경로, 라인 번호를 포함합니다.

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [데이터 파이프라인](#2-데이터-파이프라인)
3. [VAE (Oobleck 사전변환)](#3-vae-oobleck-사전변환)
4. [컨디셔닝 시스템](#4-컨디셔닝-시스템)
5. [DiT 아키텍처](#5-dit-아키텍처)
6. [LoRA 어댑테이션](#6-lora-어댑테이션)
7. [노이즈 스케줄 및 손실 함수](#7-노이즈-스케줄-및-손실-함수)
8. [분류기 없는 가이던스 (CFG)](#8-분류기-없는-가이던스-cfg)
9. [옵티마이저 및 스케줄러](#9-옵티마이저-및-스케줄러)
10. [프리즈/언프리즈 전략](#10-프리즈언프리즈-전략)
11. [검증 및 보상 모니터링](#11-검증-및-보상-모니터링)
12. [실험 히스토리](#12-실험-히스토리)
13. [텐서 형상 완전 참조표](#13-텐서-형상-완전-참조표)

---

## 1. 시스템 개요

### 목표

사전학습된 텍스트-투-오디오 모델(SAO-Small)에 **품질 스코어 컨디셔닝**을 추가하여,
추론 시 생성 품질을 조향할 수 있게 한다. 높은 스코어 → 높은 품질의 오디오 생성.

### 전체 아키텍처

```
원시 오디오 (스테레오, 524,288 샘플 @ 44.1kHz ≈ 11.9초)
    │
    ▼
┌──────────────────┐
│  Oobleck VAE     │  프리즈됨. 인코더 78M + 디코더 78M.
│  인코더          │  스트라이드 [2,4,4,8,8] = 2048배 압축
│  (2, 524288)     │
│  → (64, 256)     │
└────────┬─────────┘
         │ 잠재 벡터 z₀: (B, 64, 256)
         │
         │  ┌── 노이즈: ε ~ N(0,I), 같은 형상
         │  │   타임스텝: t ~ U(0,1)
         ▼  ▼
    z_t = (1-t)·z₀ + t·ε        ← 정류 흐름 순방향 프로세스
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│                     DiT (340M 파라미터)                      │
│                                                            │
│  컨디셔닝 입력:                                             │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │ T5-base      │  │ NumberCond    │  │ FourierScore   │  │
│  │ "록 노래"    │  │ seconds=10.0  │  │ score=0.73     │  │
│  │→(B,64,768)   │  │→(B,1,768)     │  │→(B,768)        │  │
│  └──────┬───────┘  └──────┬────────┘  └──────┬─────────┘  │
│         │                 │                   │            │
│         └── cross_attn_cond (B, 66, 768) ─────┘            │
│                      │                                     │
│              ┌───────┴───────┐                              │
│              ▼               ▼                              │
│  to_cond_embed (프리즈)  to_score_embed (학습, 1.8M)        │
│  65 토큰 (텍스트+시간)   1 토큰 (스코어 전용)                │
│  → (B, 65, 1024)        → (B, 1, 1024)                    │
│              │               │                              │
│              └───── cat ─────┘                              │
│              → (B, 66, 1024)                               │
│                      │                                     │
│  z_t → preprocess_conv → rearrange → project_in            │
│  (B,64,256)  (B,64,256)  (B,256,64)  (B,256,1024)         │
│                                           │                │
│          ┌── Prepend: global_embed (B,1,1024)              │
│          │   (타임스텝 + seconds_total)                     │
│          ▼                                                 │
│     (B, 257, 1024)                                         │
│          │                                                 │
│    ┌─────────────────┐  ×16 블록                            │
│    │ TransformerBlock │                                     │
│    │  Self-Attention  │◄── RoPE 위치 인코딩                  │
│    │  Cross-Attention │◄── (B, 66, 1024) 컨디셔닝의 K,V     │
│    │  Feed-Forward    │                                     │
│    └─────────────────┘                                     │
│          │                                                 │
│     project_out → prepend 제거 → rearrange                  │
│     (B,257,64)    (B,256,64)     (B,64,256)                │
│          │                                                 │
│     postprocess_conv (제로 초기화 잔차)                      │
│     출력: v̂ = (B, 64, 256)                                │
└────────────────────────────────────────────────────────────┘
         │
         ▼
    손실 = MSE(v̂, v_target),  v_target = ε - z₀
```

### 파라미터 예산

| 구성 요소 | 파라미터 수 | ICME 분류 |
|-----------|-----------|----------|
| VAE 인코더 (Oobleck) | 77,989,888 | 보조 (제외) |
| VAE 디코더 (Oobleck) | 78,122,626 | 보조 (제외) |
| T5-base 텍스트 인코더 | ~109,000,000 | 보조 (제외) |
| **DiT 핵심** | **339,063,168** | **핵심 모델** |
| NumberConditioner | 198,272 | 핵심 (컨디셔너) |
| FourierScoreConditioner | 788,096 | 핵심 (컨디셔너) |
| **핵심 합계** | **~340M** | **500M 한도 대비 160M 여유** |

---

## 2. 데이터 파이프라인

### 출처: `finetune.py` 46-124행

### 데이터셋 구조

**FMA-Large**: `/home/yonghyun/fma/data/fma_large/`에 106,574개 트랙

**메타데이터 JSONL** (`configs/metadata_fma_scored.jsonl`):
```jsonl
{"relpath": "/home/yonghyun/fma/data/fma_large/000/000002.mp3",
 "prompt": "A Hip-Hop song.",
 "reward_score": 0.627}
```

각 트랙에 대해 Music-RankNet이 사전 계산한 `reward_score`가 포함됨.
**FMA 스코어 분포**: 범위 [-7.37, +2.73], 중앙값=-0.10, 평균=-0.15, 표준편차=0.77.

**MTG-Jamendo**: `/home/yonghyun/music-ranknet/data/raw/mtg-jamendo-audio/`에 54,330개 트랙

**메타데이터 JSONL** (`configs/metadata_jamendo_scored.jsonl`):
```jsonl
{"relpath": "/home/yonghyun/music-ranknet/data/raw/mtg-jamendo-audio/46/40146.mp3",
 "prompt": "A slow-paced ambient track...",
 "reward_score": 3.215}
```

**Jamendo 스코어 분포**: 범위 [-3.50, +3.21], 중앙값=+0.94, 평균=+0.91, 표준편차=0.51.

**Sanity check (2026-03-17)**: FMA 랜덤 20개 트랙에 대해 메타데이터 스코어와
MusicRankNet(피처) 라이브 스코어를 비교 — **전부 delta=0.0000**. 사전 계산 스코어 정확성 검증 완료.

> **데이터셋 결합 시 주의사항**: Jamendo 스코어가 FMA보다 평균 ~1.0 높음
> (상업 음악 vs 사용자 업로드). 단순 결합 시 모델이 품질 대신 데이터셋
> 아이덴티티를 학습할 위험. 스코어 정규화 또는 균형 샘플링 전략 필요.

### ContinuousScoreDatasetWrapper

**파일**: `finetune.py` 46-82행

데이터셋을 감싸서 `reward_score`를 `continuous_score`로 변환:

```python
class ContinuousScoreDatasetWrapper(Dataset):
    def __getitem__(self, idx):
        audio, metadata = self.dataset[idx]
        metadata = metadata.copy()

        raw_score = metadata.get('reward_score', 0.0)

        # CFG 드롭아웃: cfg_drop_rate 확률로 스코어를 null로 교체
        if self.is_training and torch.rand(1).item() < self.cfg_drop_rate:
            score_val = -999.0          # NULL 센티넬 값
        else:
            score_val = float(raw_score)

        metadata['continuous_score'] = score_val
        return audio, metadata
```

**CFG 드롭아웃 비율**: 기본 15% (`SA_CFG_DROP_RATE`), 현재 실험에서는 30%.

### 배치 형상

`batch_size=2`로 DataLoader collation 후:
```
audio:    (2, 2, 524288)    # 배치=2, 스테레오, ~11.9초 @ 44.1kHz
metadata: dict {
    'prompt':           길이 2의 list[str]
    'seconds_total':    tensor(2,)          # 재생 시간
    'continuous_score': tensor(2,)          # 스코어 (-999=null)
    'reward_score':     tensor(2,)          # 원본 스코어
    'padding_mask':     (2, 524288)         # 유효 오디오 마스크
}
```

### 필터링된 데이터셋

| 설정 파일 | 트랙 수 | 필터 조건 |
|-----------|--------|---------|
| `metadata_fma_scored.jsonl` | 106,401 | FMA 전체 |
| `metadata_fma_scored_top50.jsonl` | 53,201 | FMA score > -0.102 (중앙값) |
| `metadata_fma_scored_top30.jsonl` | 31,921 | FMA score > 0.270 |
| `metadata_jamendo_scored.jsonl` | 54,330 | Jamendo 전체 (Qwen2-Audio 캡션) |

---

## 3. VAE (Oobleck 사전변환)

### 출처: 모델 설정 파일 `pretransform` 섹션

**모든 파인튜닝 동안 프리즈됨.** 사전학습된 SAO-Small의 가중치 사용.

### 인코더

```
입력:  (B, 2, 524288)     # 스테레오 파형

CNN 레이어, 스트라이드 [2, 4, 4, 8, 8]:
  레이어 1: 스트라이드=2  → (B, 128, 262144)
  레이어 2: 스트라이드=4  → (B, 256, 65536)
  레이어 3: 스트라이드=4  → (B, 512, 16384)
  레이어 4: 스트라이드=8  → (B, 1024, 2048)
  레이어 5: 스트라이드=8  → (B, 2048, 256)

보틀넥 (VAE): → latent_dim=128 → 분할 → μ, logσ → 샘플링 → 64

출력: (B, 64, 256)       # 64개 잠재 채널, 256 타임스텝
                          # 압축률: 524288/256 = 2048배
```

### 디코더

인코더의 거울 구조. `(B, 64, 256) → (B, 2, 524288)`

---

## 4. 컨디셔닝 시스템

### 출처: `stable_audio_tools/models/conditioners.py`

### MultiConditioner 흐름

```
메타데이터 (샘플별 dict)
    │
    ├── 'prompt' ───────► T5Conditioner ──────► (B, 64, 768)
    │                     T5-base 인코더 (프리즈)
    │
    ├── 'seconds_total' ► NumberConditioner ──► (B, 1, 768)
    │                     정규화 + 시간 위치 임베딩
    │
    └── 'continuous_score' ► FourierScoreCond ► (B, 768)
                              푸리에 특징 + MLP
```

### T5Conditioner (프롬프트)

```
입력:  ["A Hip-Hop song.", "A Rock song."]

T5-base 토크나이저 (max_length=64)
    → input_ids: (B, 64)
T5-base 인코더 (프리즈됨)
    → last_hidden_state: (B, 64, 768)

출력: (B, 64, 768), attention_mask (B, 64)
```

각 프롬프트가 최대 64개 토큰으로 인코딩됨. 각 토큰은 768차원 벡터.

### NumberConditioner (seconds_total)

```
입력:  [10.0, 10.0]

정규화: (값 - min) / (max - min)      # [0, 256] → [0, 1]
TimePositionalEmbedding → (B, 768)
    사인/코사인 주파수 기반 인코딩 (타임스텝 인코딩과 동일)
Unsqueeze → (B, 1, 768)

출력: (B, 1, 768), mask (B, 1)
```

### FourierScoreConditioner (continuous_score)

**핵심 스코어 임베딩**. 스칼라 품질 스코어를 풍부한 768차원 표현으로 변환.

```
입력:  [0.73, -999.0]                        # -999 = null (CFG 드롭아웃)

단계 1: 텐서 변환
    x: (B, 1)

단계 2: 푸리에 특징 인코딩
    가중치: (128, 1), 랜덤 초기화, std=1.0
    f = 2π × x @ weight.T                    # (B, 128)
    fourier_embed = [cos(f), sin(f)]          # (B, 256)

단계 3: MLP 매퍼
    Linear(256, 768) → SiLU → Linear(768, 768)
    embeds: (B, 768)

단계 4: Null 처리
    x == -999.0인 곳: embeds[null_idx] = 0.0  # 제로 벡터

출력: (B, 768), mask (B, 1)
```

**왜 Linear(1,768)이 아닌 푸리에인가?**

Linear는 스코어를 768d 공간의 **하나의 방향**으로만 매핑한다.
score=0.1과 score=0.9는 크기만 다르고 패턴은 같다.
푸리에 특징은 각 스코어 값에 대해 **완전히 다른 주기적 패턴**을 생성하여,
cross-attention이 구별하기 훨씬 쉽다.

### 컨디셔닝 라우팅

**파일**: `diffusion.py` 166-290행, `get_conditioning_inputs()`

설정 파일이 어떤 컨디셔너가 어떤 경로로 들어가는지 결정:

```python
"cross_attention_cond_ids": ["prompt", "seconds_total", "continuous_score"]
"global_cond_ids": ["seconds_total"]          # 원래 SAO 경로 복원

# Cross-attention 조립:
cross_attn_cond = cat([
    prompt_embed,          # (B, 64, 768)     텍스트 64 토큰
    seconds_total_embed,   # (B, 1, 768)      시간 1 토큰
    score_embed,           # (B, 1, 768)      스코어 1 토큰
], dim=1)
# 결과: (B, 66, 768)    — 총 66개 컨디셔닝 토큰

# 글로벌 컨디셔닝:
global_cond = seconds_total_embed  # (B, 768)
# → DiT에서 prepend 토큰으로 self-attention 시퀀스 앞에 추가됨
```

---

## 5. DiT 아키텍처

### 출처: `stable_audio_tools/models/dit.py`

### 순방향 패스 전체 흐름

#### 단계 1: Cross-Attention 프로젝션 (스코어 분리 경로)

```python
# 텍스트/시간 토큰과 스코어 토큰을 분리
n_score = 1
main_tokens  = cond_input[:, :-n_score, :]   # (B, 65, 768) — 텍스트+시간
score_tokens = cond_input[:, -n_score:, :]   # (B, 1, 768)  — 스코어 전용

# 텍스트/시간: 프리즈된 사전학습 프로젝션 (열화 없음)
main_proj  = to_cond_embed(main_tokens)      # (B, 65, 1024)

# 스코어: 전용 학습 가능 프로젝션
score_proj = to_score_embed(score_tokens)     # (B, 1, 1024)

# 재결합
cross_attn_cond = cat([main_proj, score_proj], dim=1)  # (B, 66, 1024)
```

핵심 아키텍처 변경: 텍스트/시간 토큰과 스코어 토큰이 **독립적으로** 프로젝션됨.
교차 오염 없음.

#### 단계 2: 글로벌 컨디셔닝 프로젝션

```python
global_embed = to_global_embed(global_embed)
# (B, 768) → (B, 1024)     seconds_total 전용, 프리즈됨
```

#### 단계 3: 타임스텝 임베딩

```python
timestep_embed = to_timestep_embed(FourierFeatures(t))
# t: (B,) → 푸리에: (B, 256) → MLP: (B, 1024)

# 글로벌에 합산:
global_embed = global_embed + timestep_embed
# 형상: (B, 1024) = seconds_total 프로젝션 + 타임스텝 프로젝션
```

타임스텝과 재생시간이 **하나의 글로벌 임베딩으로 합쳐진다.**
스코어는 여기에 포함되지 않음 — cross-attention으로만 들어감.

#### 단계 4: 입력 처리

```python
x = preprocess_conv(x) + x     # 잔차 컨볼루션 (제로 초기화)
# (B, 64, 256) → (B, 64, 256)

x = rearrange(x, "b c t -> b t c")
# (B, 64, 256) → (B, 256, 64)   채널 → 시퀀스 차원 교환
```

#### 단계 5: 글로벌 컨디셔닝 Prepend

```python
# ContinuousTransformer 내부:
x = project_in(x)              # Linear(64, 1024)
# (B, 256, 64) → (B, 256, 1024)

prepend = global_embed.unsqueeze(1)  # (B, 1024) → (B, 1, 1024)
x = cat([prepend, x], dim=1)        # (B, 1+256, 1024) = (B, 257, 1024)
```

글로벌 임베딩이 시퀀스 **맨 앞에 추가 토큰**으로 삽입됨.
Self-attention에서 모든 위치가 이 토큰에 접근 가능.

#### 단계 6: 트랜스포머 블록 (×16)

```
입력: x (B, 257, 1024), context (B, 66, 1024)

┌─── Self-Attention ─────────────────────────────┐
│ Q, K, V = to_q(x), to_k(x), to_v(x)          │
│ 형상: (B, 257, 1024) → (B, 8헤드, 257, 128)    │
│ 8개 어텐션 헤드, 헤드 차원=128                   │
│ RoPE 위치 인코딩이 Q, K에 적용                   │
│ Attention = softmax(QK^T/√128) × V            │
│ 출력: (B, 257, 1024)                           │
└────────────────────────────────────────────────┘
        │ + 잔차 연결
        ▼
┌─── Cross-Attention ────────────────────────────┐
│ Q = to_q(x)           → (B, 8, 257, 128)      │
│ K = to_k(context)     → (B, 8, 66, 128)       │
│ V = to_v(context)     → (B, 8, 66, 128)       │
│                                                │
│ 257개 잠재 위치 각각이 66개 컨디셔닝 토큰에       │
│ 어텐션을 계산:                                   │
│   [텍스트_1..텍스트_64, 재생시간, 스코어]         │
│                                                │
│ 스코어 토큰의 영향:                              │
│   attention_weight ≈ softmax(...)[..., 65]     │
│   64개 텍스트 + 1개 시간 토큰과 경쟁              │
│ 출력: (B, 257, 1024)                           │
└────────────────────────────────────────────────┘
        │ + 잔차 연결
        ▼
┌─── Feed-Forward ───────────────────────────────┐
│ Linear(1024, 4096) → GELU → Linear(4096, 1024)│
│ 출력: (B, 257, 1024)                           │
└────────────────────────────────────────────────┘
        │ + 잔차 연결
        ▼
출력: (B, 257, 1024)
```

16개 블록을 순차적으로 통과. 각 블록에서 스코어 토큰은 cross-attention을
통해 잠재 시퀀스에 영향을 미친다.

#### 단계 7: 출력 처리

```python
x = project_out(x)                     # Linear(1024, 64)
# (B, 257, 1024) → (B, 257, 64)

output = rearrange(x, "b t c -> b c t") # (B, 64, 257)
output = output[:, :, prepend_length:]   # prepend 제거: (B, 64, 256)
output = postprocess_conv(output) + output  # 제로 초기화 잔차
# 최종 출력: (B, 64, 256)
```

---

## 6. 스코어 프로젝션 전략

### 출처: `dit.py` 86-100행

### 현재 최선: 스코어 분리 프로젝션 (2026-03-17)

```python
# 텍스트/시간 토큰용 공유 프로젝션 (프리즈):
to_cond_embed = Sequential(Linear(768→1024), SiLU, Linear(1024→1024))

# 스코어 토큰 전용 프로젝션 (학습 가능, 1.8M):
to_score_embed = Sequential(Linear(768→1024), SiLU, Linear(1024→1024))
```

### 순방향 계산

```
입력: cross_attn_cond (B, 66, 768)
              │
       ┌──────┴──────┐
       ▼              ▼
  [:, :65, :]    [:, 65:, :]
  텍스트+시간       스코어
  (B, 65, 768)   (B, 1, 768)
       │              │
       ▼              ▼
  to_cond_embed   to_score_embed
  (프리즈)        (학습 가능)
  (B, 65, 1024)  (B, 1, 1024)
       │              │
       └──── cat ─────┘
              │
       (B, 66, 1024) → Cross-Attention K, V
```

### 왜 분리 프로젝션인가

| 접근법 | 학습 파라미터 | 텍스트 열화 | 오디오 품질 | Best Corr |
|--------|------------|-----------|-----------|-----------|
| to_cond_embed 완전 언프리즈 | 1.8M | 있음 (일부 장르 무음) | 대체로 OK | 0.357 |
| LoRA rank=8 | 802K | 전반적 노이즈 (flatness 0.21) | **나쁨** | 0.425 |
| **분리 to_score_embed** | **2.6M** | **없음 (프리즈된 텍스트 경로)** | **TBD** | **진행 중** |

LoRA와 완전 언프리즈 모두의 근본 문제: `to_cond_embed`가 66개 토큰을 공유.
1개 스코어 토큰을 위한 수정이 65개 텍스트/시간 토큰을 오염시킴.

분리 프로젝션은 이를 해결:
- **텍스트/시간**: 프리즈된 `to_cond_embed` → 열화 제로, 사전학습 품질 유지
- **스코어**: 전용 `to_score_embed` → 완전한 표현력, 간섭 없음

### 제로 초기화 원칙

```
to_score_embed.0 (Linear 768→1024):  작은 랜덤 초기화 (그래디언트 흐름)
to_score_embed.1 (SiLU):             활성 함수
to_score_embed.2 (Linear 1024→1024): 제로 초기화 (시작 시 효과 없음)
```

초기화 시 `to_score_embed.2.weight = 0` → 출력이 항상 0.
모델은 **정확히 사전학습된 동작**으로 시작. ControlNet 제로 컨볼루션 패턴.

### 이전 방식: LoRA (deprecated)

LoRA는 **공유** 프로젝션에 랭크 제한 잔차를 적용:
`출력 = to_cond_embed(x) + lora_B(lora_A(x))`. Corr=0.425를 달성했으나
전반적 노이즈 열화 발생 (spectral flatness 0.21 vs 기준 0.10).
LoRA 잔차가 66개 토큰 전부에 균일하게 영향을 주었기 때문.

---

## 7. 노이즈 스케줄 및 손실 함수

### 출처: `training/diffusion.py` 395-477행

### 정류 흐름 (Rectified Flow)

```python
# 타임스텝 샘플링
t = uniform(0, 1)                           # (B,)

# 순방향 프로세스 (데이터에 노이즈 추가)
alphas = 1 - t                              # (B, 1, 1)
sigmas = t                                  # (B, 1, 1)
noise = randn_like(잠재_벡터)                # (B, 64, 256)
노이즈_입력 = z₀ × alphas + noise × sigmas
# z_t = (1-t)·z₀ + t·ε

# 타겟: 속도 (velocity)
타겟 = noise - 잠재_벡터                     # v = ε - z₀
```

t=0: 순수 데이터 (노이즈 없음)
t=1: 순수 노이즈 (데이터 없음)
모델은 데이터에서 노이즈로 가는 **속도**를 예측한다.

### 손실 계산

```python
# MSE 손실
출력 = model(노이즈_입력, t, 컨디셔닝)       # (B, 64, 256)
loss = MSE(출력, 타겟)                       # 스칼라

# 스코어 가중 손실 (선택적, SA_SCORE_WEIGHTED_LOSS=1)
weight = (reward_score - score_min) / (score_max - score_min)
weight = weight.clamp(0.1, 1.0)             # 최소 가중치 0.1
loss = loss × weight.mean()                 # 고품질 샘플에 더 높은 가중치
```

스코어 가중 손실은 스코어 컨디셔닝과 **직교** — 아키텍처를 변경하지 않고
고품질 데이터에 학습을 집중시킨다.

---

## 8. 분류기 없는 가이던스 (CFG)

### 학습 시 드롭아웃

```python
# cfg_drop_rate 확률 (15% 또는 30%)로:
score_val = -999.0  # NULL 센티넬

# FourierScoreConditioner에서:
if score == -999.0:
    embedding = zeros(768)  # Null = 제로 벡터
```

이를 통해 모델이 두 가지 모드를 학습:
- **조건부** (학습의 70-85%): 스코어를 알고, 그에 맞게 생성
- **비조건부** (학습의 15-30%): 스코어 정보 없이 생성

### 추론 시 CFG

```python
# 배치 구성: [조건부, 비조건부]
batch_inputs = cat([x, x], dim=0)           # (2B, 64, 256)

# Cross-attention:
batch_cond = cat([
    cross_attn_cond,                         # (B, 66, 768) 실제 스코어 포함
    zeros_like(cross_attn_cond)              # (B, 66, 768) 전부 0
], dim=0)

# 모델을 한번에 두 번 실행 (조건부 + 비조건부)
cond_output, uncond_output = chunk(output, 2)

# CFG 공식:
최종 = uncond_output + cfg_scale × (cond_output - uncond_output)
```

**cfg_scale = 3.5** (기본값). 높을수록 컨디셔닝 신호를 더 강하게 증폭.
스코어 신호가 텍스트 신호와 함께 증폭된다.

**적용된 치명적 수정**: 이전에는 비조건부 패스가 조건부 패스와 **같은**
global_embed를 사용하여 스코어가 CFG에 투명(invisible)했음.
비조건부 패스에 제로를 사용하도록 수정됨.

---

## 9. 옵티마이저 및 스케줄러

### 출처: `finetune.py` 206-233행

```python
optimizer = AdamW(
    학습_가능_파라미터,
    lr=5e-5,            # 또는 LoRA 실험에서 1e-4
    weight_decay=1e-3
)

scheduler = CosineWithHardRestarts(
    warmup_steps=1000,          # 웜업 단계
    total_steps=300000,         # 총 학습 스텝
    num_cycles=18               # 코사인 사이클 수
)
```

### 유효 배치 크기

```
GPU당 배치:        2
그래디언트 누적:   4
GPU 수:            2
유효 배치:         2 × 4 × 2 = 16
```

---

## 10. 프리즈/언프리즈 전략

### 출처: `finetune.py` 22-39행, 142-203행

### 프로파일

| 프로파일 | 언프리즈 파라미터 | 학습 가능 수 |
|---------|-----------------|------------|
| `minimal` | continuous_score, score_bin | ~788K |
| `lora_xattn` | continuous_score, score_bin, cond_embed_lora | ~802K |
| `xattn` | continuous_score, score_bin, to_cond_embed | ~1.8M |
| **`separate_score_proj`** | **continuous_score, score_bin, to_score_embed** | **~2.6M** |
| `adaln` | continuous_score, to_global_embed, to_scale_shift_gate, global_cond_embedder | ~9.3M |

### 현재: `separate_score_proj`

```
언프리즈됨:
  conditioner.conditioners.continuous_score.*    788,096 파라미터
    ├── fourier_features.weight                  (128, 1)
    ├── mapper.0.weight                          (768, 256)
    ├── mapper.0.bias                            (768,)
    ├── mapper.2.weight                          (768, 768)
    └── mapper.2.bias                            (768,)

  model.model.to_score_embed.0.weight            (1024, 768)  = 786,432
  model.model.to_score_embed.2.weight            (1024, 1024) = 1,048,576
                                                 ──────────
  총 학습 가능:                                   ~2.6M / 497M (0.52%)

프리즈됨:
  나머지 전부 (496.2M 파라미터):
    VAE 인코더/디코더, T5, DiT 트랜스포머 블록,
    to_cond_embed, to_global_embed, to_timestep_embed,
    모든 self-attention, 모든 cross-attention, 모든 FFN
```

### 새 파라미터 가중치 초기화

ControlNet 제로 초기화 패턴:
- **출력 레이어**: 제로 초기화 (시작 시 효과 없음)
- **중간 레이어**: 작은 랜덤 값 (그래디언트 흐름 보장)

```python
KEEP_RANDOM = ["continuous_score", "score_concat", "cond_embed_lora_A"]

for name, param in model.named_parameters():
    if name not in 사전학습_키:
        if any(pat in name for pat in KEEP_RANDOM):
            param.data.normal_(0, 0.02)     # 작은 랜덤
        else:
            param.zero_()                   # 제로 초기화
```

---

## 11. 검증 및 보상 모니터링

### 출처: `training/reward_monitor.py` 158-338행

### 검증 흐름

```
5,000 스텝마다:
    100개 샘플 (10개 구간 × 구간당 10개):
        1. 검증 세트에서 프롬프트 선택
        2. 구간별 목표 스코어 설정
        3. CFG로 오디오 생성:
           - 양성: 프롬프트 + score=목표
           - 음성: "" + score=-999 (null)
           - cfg_scale=3.5, steps=50
        4. 생성 오디오에서 피처 추출:
           - MERT:       (1, 1024)   @ 24kHz
           - CLAP 오디오: (1, 512)    @ 48kHz
           - CLAP 텍스트: (1, 512)    프롬프트에서
           - 플래그:      (1, 1)      = 1.0
        5. 연결: [flag, clap_audio, mert, clap_text] = (1, 2049)
        6. 스코어 = MusicRankNet(피처)
        7. 목표 vs 측정 스코어 기록

    메트릭 계산:
        피어슨 상관계수(목표, 측정)
        쌍별 단조성
        구간별 평균 스코어
```

### Music-RankNet 아키텍처

```
입력: (1, 2049)
    [flag(1), CLAP_audio(512), MERT(1024), CLAP_text(512)]

Linear(2049, 1024) → BatchNorm → ReLU → Dropout(0.3)
Linear(1024, 512)  → BatchNorm → ReLU → Dropout(0.3)
Linear(512, 256)   → BatchNorm → ReLU → Dropout(0.3)
Linear(256, 128)   → ReLU
Linear(128, 1)     → 스코어 출력 (스칼라)
```

### 목표 스코어 구간

FMA 스코어 분포의 백분위수에서 10개 구간:

| 구간 | 목표 스코어 | 의미 |
|------|-----------|------|
| 상위 10% | 0.91 | 최고 품질 |
| 상위 20% | 0.60 | |
| 상위 50% | -0.02 | 중앙값 |
| 상위 100% | -1.52 | 최저 품질 |

---

## 12. 실험 히스토리

| 버전 | 경로 | 학습 파라미터 | 결과 | 근본 원인 |
|------|------|------------|------|----------|
| v1-v4 | prepend | 1.5K | Corr~0 | adaln 프로파일 + prepend 설정 = 무효 |
| v5-v10 | adaLN | 7.4-9.3M | Corr~0, NaN | sum-merge, 죽은 그래디언트, 약한 신호 |
| xattn v1-2 | cross-attention | 1.8M | Corr=0.357 | 작동! 단 일부 장르 무음 |
| LoRA v2 (r=8) | Fourier+LoRA | 802K | Corr=0.425 | 높은 Corr but **전반적 노이즈** (flatness 0.21) |
| LoRA+weighted+top50 | LoRA+가중손실 | 802K | Corr=0.325 | 중단: 가중 손실이 오히려 방해 |
| LoRA r=32 | Fourier+LoRA | 845K | — | 중단: 같은 노이즈 문제 예상 |
| **분리 프로젝션 v1** | **Fourier+전용 proj** | **2.6M** | **진행 중** | **교차 오염 없음** |

### 핵심 발견사항

- **adaLN은 스코어 컨디셔닝에 실패**: 6회 시도 (v5-v10), Corr≈0.
  원인: 스코어가 global_cond에 들어가면 seconds_total과 sum-merge되어
  모델이 두 신호를 분리할 수 없음.
- **Cross-attention은 작동** (Corr=0.357)하지만 `to_cond_embed` 언프리즈 시
  텍스트 생성 파괴 (Classical/Blues/Lo-Fi → 거의 무음).
- **LoRA on to_cond_embed**: Corr=0.425이지만 **전반적 오디오 품질 열화**.
  Spectral flatness가 0.10 (pretrained) → 0.21 (epoch 8)로 상승.
  LoRA 잔차가 66개 토큰 전부에 균일하게 적용되어 텍스트 프로젝션 오염. 중단.
- **분리 스코어 프로젝션** (현재): 전용 `to_score_embed`로 스코어 토큰만 처리.
  `to_cond_embed`는 프리즈 → 텍스트 열화 제로. 2.6M 학습, 가장 깔끔한 해법.
- **스코어 가중 손실은 역효과**: 0.325로 정점, 바닐라 LoRA보다 낮음.
- **스코어를 global+cross-attn 동시 투입**: 불가 (sum-merge 문제).
- **Demo config 버그 발견**: `continuous_score: 10.0`이 분포 밖이었음
  (FMA 범위: [-7.37, +2.73]). 실제 백분위 값으로 수정 완료.

### 현재 실행 상태 (2026-03-17)

| GPU | 실험 | 상태 |
|-----|------|------|
| 0 | FMA 캡션 생성 (Qwen2-Audio) | 84K/107K (79%) |
| 3-7 | **예약됨 / 사용 금지** | — |
| 8-9 | **분리 프로젝션 v1 (case3b_separate_score_proj_v1)** | 방금 시작 |

### 데이터 준비 상태

| 데이터셋 | 트랙 수 | 캡션 | 스코어 | 상태 |
|---------|--------|------|--------|------|
| FMA-Large | 106,574 | Qwen2-Audio (84K/107K) | 전부 스코어링 완료 | 학습 활용 중 |
| MTG-Jamendo | 54,330 | Qwen2-Audio (전부 완료) | 전부 스코어링 완료 (2026-03-17) | 준비 완료, 정규화 대기 |

---

## 13. 텐서 형상 완전 참조표

| 단계 | 변수 | 형상 | 비고 |
|------|------|------|------|
| **DataLoader** | 오디오 | (2, 2, 524288) | 스테레오, ~11.9초 |
| | continuous_score | (2,) | float, -999=null |
| **VAE 인코딩** | 잠재 벡터 | (2, 64, 256) | 2048배 압축 |
| **컨디셔너** | T5 프롬프트 임베딩 | (2, 64, 768) | 64개 텍스트 토큰 |
| | seconds_total 임베딩 | (2, 1, 768) | 1개 시간 토큰 |
| | 스코어 임베딩 | (2, 768) | 푸리에 + MLP |
| **라우팅** | cross_attn_cond | (2, 66, 768) | 모든 토큰 연결 |
| | global_cond | (2, 768) | seconds_total |
| **노이즈** | t | (2,) | U(0,1) |
| | 노이즈 ε | (2, 64, 256) | N(0,I) |
| | 노이즈 z_t | (2, 64, 256) | (1-t)z₀+tε |
| | 타겟 v | (2, 64, 256) | ε-z₀ |
| **DiT 프로젝션** | cross_attn_cond | (2, 66, 1024) | to_cond_embed(65)+to_score_embed(1) |
| | global_embed | (2, 1024) | to_global_embed |
| | timestep_embed | (2, 1024) | 푸리에+MLP |
| | global_embed | (2, 1024) | global+timestep 합 |
| **DiT 입력** | preprocess 후 | (2, 64, 256) | 제로 초기화 잔차 |
| | rearrange 후 | (2, 256, 64) | 채널 라스트 |
| | project_in 후 | (2, 256, 1024) | Linear(64,1024) |
| | prepend 포함 | (2, 257, 1024) | 시퀀스 차원 +1 |
| **트랜스포머** | Self-attn Q,K,V | (2, 8, 257, 128) | 8헤드 |
| | Cross-attn Q | (2, 8, 257, 128) | 잠재에서 |
| | Cross-attn K,V | (2, 8, 66, 128) | 컨디셔닝에서 |
| | 16블록 후 | (2, 257, 1024) | |
| **DiT 출력** | project_out | (2, 257, 64) | Linear(1024,64) |
| | prepend 제거 | (2, 256, 64) | 첫 토큰 삭제 |
| | rearrange | (2, 64, 256) | 채널 퍼스트 |
| | postprocess | (2, 64, 256) | 제로 초기화 잔차 |
| **손실** | 출력 v̂ | (2, 64, 256) | 모델 예측 |
| | MSE(v̂, v) | 스칼라 | 손실 값 |

### 추론: 프롬프트 + 스코어 → 오디오

| 단계 | 형상 | 비고 |
|------|------|------|
| 입력 프롬프트 | str | "A Rock song." |
| 입력 스코어 | float | 0.91 |
| T5 인코딩 | (1, 64, 768) | |
| 스코어 임베딩 | (1, 768) | |
| CFG 배치 2배 | (2, 64, 256) | [조건부, 비조건부] |
| 50회 디노이징 | (2, 64, 256) → ... | ODE 솔버 |
| CFG 결합 | (1, 64, 256) | uncond+scale*(cond-uncond) |
| VAE 디코딩 | (1, 2, 441000) | 10초 @ 44.1kHz 스테레오 |
| 출력 | WAV 파일 | 10초, 44.1kHz, 스테레오 |

---

*이 문서는 ISMIR 2026 및 ICME Challenge를 위한 SAO 스코어 컨디셔닝 프로젝트의 일부입니다.*
