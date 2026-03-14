# Conditioning Pipeline 분석 및 설계 검토

> 작성일: 2026-03-14
> 대상: SAO-Small Score Conditioning
> 목적: 각 conditioning 경로의 동작 원리, 현재 설계의 문제점, 최적 전략 검토

---

## 1. 컨디셔닝 경로 개요 (4가지)

SAO-Small DiT에는 4가지 컨디셔닝 경로가 존재한다:

```
                       ┌─────────────────────────────────────────────────────┐
  prompt (text)   ─────┤  cross_attention   │ (B, seq, 768) → Attention     │
                       ├────────────────────┤                                │
  seconds_total   ──┬──┤  global_cond       │ (B, 768) → Prepend or adaLN  │
                    │  ├────────────────────┤                                │
  continuous_score ─┼──┤  input_add         │ (B, 768, 1) → Conv1d → Add   │
                    │  ├────────────────────┤                                │
                    │  │  input_concat      │ (B, C, T) → Channel concat   │
                    │  └─────────────────────────────────────────────────────┘
                    │
                    └── global_cond_ids에도 포함 가능 (dual-path)
```

---

## 2. 각 경로의 상세 동작

### 2.1 Cross-Attention (prompt, seconds_total)

```
T5Conditioner("A jazz song.")  → (B, 64, 768)   텍스트 토큰 시퀀스
NumberConditioner(10.0)         → (B, 1, 768)    스칼라 → 프로젝션
                                     ↓ concat(dim=1)
                                (B, 65, 768)      → cross_attn_cond
                                     ↓
          TransformerBlock:  Q=latent, K,V=cross_attn_cond
          → 각 latent 위치가 텍스트 토큰에 independently attend
```

**특징**: 시퀀스 길이 방향으로 concatenation. 각 latent position이 어떤 텍스트 토큰에 attend할지 학습.

### 2.2 Global Conditioning — Prepend Mode (기본)

```
seconds_total     → NumberConditioner → (B, 768)
continuous_score  → ContinuousScoreConditioner → (B, 768, 1) → squeeze → (B, 768)
                                     ↓
               ★ 같은 dim이면 SUM: seconds_total + continuous_score = (B, 768)
                                     ↓
                              unsqueeze(1) → (B, 1, 768)
                                     ↓
                    Prepend to transformer sequence as extra token
                                     ↓
          latent (B, 256, 1024) → concat → (B, 257, 1024)
```

### 2.3 Global Conditioning — adaLN Mode

```
seconds_total + continuous_score (summed) → (B, 768)
           ↓
    global_cond_embedder:
        Linear(768 → 1024) → SiLU → Linear(1024 → 6144)
           ↓
    (B, 6144) → chunk(6) → 6 × (B, 1024)
        [scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff]
           ↓
    매 TransformerBlock에서:
        x = LayerNorm(x) * (1 + scale) + shift    ← 변조
        x = Attention(x) * sigmoid(1 - gate)       ← 게이팅
```

### 2.4 Input Add (continuous_score)

```
continuous_score (float, e.g., 0.91)
           ↓
    ContinuousScoreConditioner:
        Linear(1 → 768) → unsqueeze(-1) → (B, 768, 1)
        if score == -999.0: → zero vector (CFG null)
           ↓
    get_conditioning_inputs():
        그대로 전달 → (B, 768, 1)
           ↓
    DiT._forward():
        F.interpolate → (B, 768, 256)    ← 모든 temporal position에 동일 값 복제
        input_add_adapter: Conv1d(768 → 64, kernel_size=1) → (B, 64, 256)
        x = x + adapter_output           ← latent에 잔차 덧셈
```

### 2.5 Input Concat (현재 미사용)

```
conditioning → (B, C_extra, T)
           ↓
    x = torch.cat([x, input_concat_cond], dim=1)
    → (B, 64 + C_extra, T)
```

**input_add와의 차이**: add는 기존 64채널에 "더하기", concat은 채널 수 자체를 늘림. concat은 모델의 첫 번째 레이어 입력 차원이 바뀌므로 pretrained weight와 비호환.

---

## 3. 현재 설계의 문제점

### 3.1 (Critical) seconds_total과 continuous_score의 Sum-Merge

**코드 (`diffusion.py` get_conditioning_inputs):**
```python
# Multiple conditioners with same dim → element-wise SUM
if len(global_conds) > 1 and all(g.shape[-1] == global_conds[0].shape[-1] for g in global_conds):
    global_cond = sum(global_conds)
```

**문제**: `seconds_total`과 `continuous_score`가 모두 768차원이므로 **element-wise sum**됨.

```
seconds_total embedding:  [0.12, -0.34, 0.56, ...]  (768d)
score embedding:          [0.78, 0.11, -0.22, ...]  (768d)
                              ↓ SUM
global_cond:              [0.90, -0.23, 0.34, ...]  (768d)
```

모델이 이 합산 벡터에서 "10초짜리" vs "점수 0.91"를 분리해야 하는데, **선형 합이므로 정보 혼재**가 발생한다:
- 높은 score + 짧은 duration ≈ 낮은 score + 긴 duration (같은 합산 벡터 가능)
- 학습 초기에는 두 신호의 크기(norm)가 다를 수 있어 한쪽이 지배적

**영향 범위**:
- Prepend mode: sum된 벡터가 하나의 토큰으로 들어감
- adaLN mode: sum된 벡터가 global_cond_embedder를 거쳐 모든 블록의 scale/shift/gate를 결정

### 3.2 input_add의 제한된 표현력

**현재 구조**:
```
score (scalar) → Linear(1, 768) → Conv1d(768, 64, k=1) → add to latent
```

**문제**:
1. **단일 scalar에서 768차원 프로젝션**: Linear(1→768)은 사실상 weight vector * score. 즉, score의 크기에 비례하는 방향 벡터 하나만 생성.
2. **시간 무관 복제**: (B, 768, 1) → interpolate → (B, 768, 256). 모든 temporal position에 동일한 값이 더해짐. 오디오의 시간적 구조(도입부, 클라이맥스 등)에 대한 차별화 불가.
3. **Zero-init 수렴 속도**: adapter가 zero-init이므로 학습 초기에 gradient가 매우 작음. 유효한 컨디셔닝 효과가 나타나려면 상당한 step이 필요.

### 3.3 Dual-Path에서의 신호 중복

adaln config에서 `continuous_score`는 두 경로로 동시 주입:
1. `global_cond_ids` → adaLN (scale/shift/gate)
2. `input_add_ids` → latent residual

**잠재적 문제**:
- 같은 신호가 두 번 들어가므로 optimization landscape에서 redundancy 발생
- 두 경로의 gradient 방향이 충돌할 수 있음
- 다만, 경로가 질적으로 다르므로 (modulation vs addition) 상호보완 가능성도 존재

### 3.4 CFG에서의 Score Null 처리

```python
# ContinuousScoreConditioner: -999.0 → zero vector
if score == -999.0:
    embeds = 0.0

# CFG: output = uncond + scale * (cond - uncond)
#   cond:   score=0.91 → non-zero embedding
#   uncond: score=-999 → zero embedding
```

negative_conditioning에서 `continuous_score: -999.0`이면 zero vector가 들어간다. 이는 "점수 조건 없음"을 의미하며, CFG가 "점수가 있는 방향"으로 guidance를 주는 것은 의도된 동작. **이 부분은 정상**.

---

## 4. 대안적 컨디셔닝 전략 비교

### 4.1 전략 비교표

| 전략 | 위치 | 장점 | 단점 | 파라미터 |
|------|------|------|------|---------|
| **A. adaLN only** (현재 adaln profile) | 매 블록 scale/shift/gate | 모든 레이어에서 변조, DiT 논문에서 검증됨 | seconds_total과 sum-merge | 7.4M |
| **B. input_add only** (현재 adapter profile) | latent 입력 단계 | 심플, pretrained 보존 | 표현력 제한, scalar→768d 병목 | 50K |
| **C. Cross-attention** | 트랜스포머 CA | 텍스트와 함께 attend, 유연 | 스칼라에 과도한 구조 | ~0 (기존 CA 활용) |
| **D. Timestep embedding addition** | 시간 임베딩에 합산 | DiT 클래스 컨디셔닝 표준 방식 | 구현 필요, t와 혼재 | ~768 |
| **E. Separate adaLN head** | 별도 프로젝션 | sum-merge 해결 | 파라미터 2배, 구현 복잡 | ~14M |
| **F. FiLM (Feature-wise Linear Modulation)** | 특정 레이어만 | 경량, 타겟팅 가능 | 전체 변조 불가 | ~1M |

### 4.2 각 전략의 상세 분석

#### A. adaLN (현재 — sum-merge 문제 있음)

DiT 원 논문(Peebles & Xie, 2023)에서 **class conditioning**에 사용된 방식. class label은 하나의 embedding이므로 sum-merge 문제가 없었음. 현재 구현에서는 seconds_total과 score가 합산되어 들어가는 점이 원 논문과 다름.

**개선안**: `seconds_total`을 `global_cond_ids`에서 제거하고 `cross_attn_cond_ids`에만 남기면 sum-merge 문제 해결. score만 adaLN으로 들어감.

```json
// 개선된 config
"global_cond_ids": ["continuous_score"],          // score만
"cross_attn_cond_ids": ["prompt", "seconds_total"] // duration은 여기만
```

#### B. input_add (표현력 제한)

현재 adapter profile에서 사용. 50K 파라미터로 매우 경량이지만:
- Linear(1→768) → Conv1d(768→64): 실질적으로 score * W_linear * W_conv = score * W_combined
- 이는 **64차원 벡터 하나를 score에 비례하여 latent에 더하는 것**과 동일
- 비선형성이 없으므로 "높은 점수" vs "낮은 점수"의 효과가 정확히 반대 방향

이것이 나쁜 것은 아님 — 오히려 score가 연속값이므로 **선형적 반응이 자연스러울 수 있음**. 하지만 "좋은 음악의 latent feature"와 "나쁜 음악의 latent feature"의 차이가 단순히 선형이 아닐 가능성이 높음.

#### C. Cross-attention으로 score 주입

score를 cross-attention 시퀀스에 토큰으로 추가:

```
cross_attn_cond = [T5_tokens(64), seconds_total(1), score(1)] = (B, 66, 768)
```

**장점**: 텍스트와 score의 관계를 attention으로 학습 가능 (예: "jazz"에서 높은 점수 vs "noise"에서 높은 점수)
**단점**: 스칼라 하나에 cross-attention은 과도. 이미 cross-attention은 텍스트 정보에 최적화되어 있으므로, score 토큰 하나가 무시될 가능성.

#### D. Timestep embedding에 score 합산

DiT 원 논문의 class conditioning 방식과 유사:

```python
# 현재: t_embed = timestep_embed(t)
# 제안: t_embed = timestep_embed(t) + score_embed(score)
```

**장점**: 가장 자연스러운 DiT 컨디셔닝. timestep과 함께 모든 블록에 영향.
**단점**: timestep과 score가 같은 임베딩 공간에서 합산 → sum-merge와 유사한 문제. 하지만 DiT 논문에서 class+time을 이런 식으로 처리했고 잘 동작함.

#### E. Separate adaLN head (score 전용 프로젝션)

seconds_total과 score에 각각 독립적인 adaLN embedder를 부여:

```python
# 기존: 하나의 global_cond_embedder(768 → 6144)
# 제안:
self.time_embedder = nn.Sequential(Linear(768, 1024), SiLU, Linear(1024, 6144))
self.score_embedder = nn.Sequential(Linear(768, 1024), SiLU, Linear(1024, 6144))

# forward:
time_modulation = self.time_embedder(seconds_embed)
score_modulation = self.score_embedder(score_embed)
total_modulation = time_modulation + score_modulation  # 잠재 공간에서 합산
```

**장점**: 각 신호가 독립적으로 프로젝션되므로 정보 혼재 최소화
**단점**: 파라미터 2배 (~14M), 구현 복잡도 증가

---

## 5. 권장 전략

### 5.1 ISMIR Track (FMA, 파라미터 제한 없음)

**추천: adaLN + sum-merge 해소 (전략 A 개선)**

```
┌─ continuous_score ──→ global_cond (adaLN only) ──→ 매 블록 변조
│
│  seconds_total ──→ cross_attn_cond only (텍스트와 함께 attend)
│
│  prompt ──→ cross_attn_cond
│
└─ (input_add 제거 — adaLN만으로 충분)
```

**이유**:
1. `seconds_total`을 `global_cond_ids`에서 빼면 sum-merge 해소
2. `seconds_total`은 이미 `cross_attn_cond_ids`에 있으므로 정보 손실 없음
3. score가 adaLN 경로를 독점 → 더 순수한 score 변조 학습
4. input_add 제거로 dual-path redundancy 해소

**Config 변경**:
```json
{
    "diffusion": {
        "cross_attn_cond_ids": ["prompt", "seconds_total"],
        "global_cond_ids": ["continuous_score"],
        "input_add_ids": [],
        "config": {
            "global_cond_type": "adaLN",
            "input_add_dim": 0,
            "input_add_use_adapter": false
        }
    }
}
```

**학습 파라미터**: ~7.4M (현재와 동일)

### 5.2 ICME Track (500M 제한)

**추천: global profile (prepend mode) + sum-merge 해소**

```
continuous_score ──→ global_cond (prepend) ──→ 1개 추가 토큰
seconds_total ──→ cross_attn_cond only
```

**이유**: adapter(50K)는 너무 작아서 표현력 부족. global(1.8M)이 prepend 토큰으로 score 정보를 전달하면서 500M 이내. sum-merge를 해소하면 prepend 토큰이 순수하게 score만 반영.

### 5.3 향후 고려: Separate adaLN head (전략 E)

가장 깔끔한 해결이지만 구현 변경이 큼. ISMIR 실험에서 A 개선안의 성능이 부족하면 고려.

---

## 6. 현재 Config 검증

### 6.1 `model_config_with_score.json` (prepend mode)

```json
"cross_attn_cond_ids": ["prompt", "seconds_total"],
"global_cond_ids": ["seconds_total", "continuous_score"],
"input_add_ids": ["continuous_score"]
```

| 분석 | 결과 |
|------|------|
| seconds_total | cross_attn + global_cond 양쪽 | ★ 중복 주입 |
| continuous_score | global_cond + input_add 양쪽 | ★ 이중 경로 |
| global_cond sum-merge | seconds + score 합산 | ★ 정보 혼재 |
| adaLN 파라미터 | 없음 (prepend mode) | — |
| 실질 score 경로 | input_add만 유효 (global은 합산에 묻힘) | △ |

### 6.2 `model_config_with_score_adaln.json` (adaLN mode)

```json
"cross_attn_cond_ids": ["prompt", "seconds_total"],
"global_cond_ids": ["seconds_total", "continuous_score"],
"input_add_ids": ["continuous_score"]
```

| 분석 | 결과 |
|------|------|
| adaLN 활성화 | O (global_cond_type: "adaLN") | |
| sum-merge | seconds + score 합산 → adaLN 입력 | ★ 문제 |
| input_add 이중 경로 | score가 adaLN + input_add 동시 | △ redundancy |
| 총 파라미터 | 504.7M | ICME 초과 |

### 6.3 제안하는 개선 Config

```json
{
    "conditioning": {
        "configs": [
            {"id": "prompt", "type": "t5", ...},
            {"id": "seconds_total", "type": "number", ...},
            {"id": "continuous_score", "type": "continuous_score", ...}
        ]
    },
    "diffusion": {
        "cross_attn_cond_ids": ["prompt", "seconds_total"],
        "global_cond_ids": ["continuous_score"],
        "input_add_ids": [],
        "config": {
            "global_cond_type": "adaLN",
            "global_cond_dim": 768,
            "input_add_dim": 0,
            "input_add_use_adapter": false
        }
    }
}
```

**변경 사항 요약**:
1. `seconds_total`을 `global_cond_ids`에서 제거 → sum-merge 해소
2. `continuous_score`를 `input_add_ids`에서 제거 → 단일 경로(adaLN만)
3. `input_add_dim: 0`, adapter 비활성화

---

## 7. Tensor Shape 전체 흐름도

### 7.1 현재 (adaln config, dual-path + sum-merge)

```
                    ┌── Linear(1,768) ──→ (B,768,1) ── squeeze ──→ (B,768)
continuous_score ───┤                                                  │
                    └── Linear(1,768) ──→ (B,768,1) ─────────────────────── input_add_cond
                                                                       │
seconds_total ──── NumberConditioner ──→ (B,1,768) ── squeeze ──→ (B,768)
                                                                       │
                                                              SUM: (B,768)
                                                                       │
                                            global_cond_embedder: Linear→SiLU→Linear
                                                                       │
                                                                  (B, 6144)
                                                                       │
                                                        chunk(6) → 6 × (B,1024)
                                                                       │
                                                    ┌──────────────────┤
                                                    ↓                  ↓
                                           Self-Attn modulation  FF modulation
                                           scale, shift, gate    scale, shift, gate

Input latent: (B, 64, 256)
         │
         ├── + input_add_adapter(input_add_cond)   ← (B,64,256) added
         │
         ├── preprocess_conv → rearrange → (B, 256, 1024)
         │
         ├── 16 × TransformerBlock (adaLN modulation from global_cond)
         │          ↑ cross_attn from prompt+seconds_total: (B, 65, 768)
         │
         └── rearrange → postprocess_conv → (B, 64, 256)
```

### 7.2 제안 (adaLN only, sum-merge 해소)

```
continuous_score ─── Linear(1,768) ──→ (B,768)
                                          │
                         global_cond_embedder: Linear→SiLU→Linear
                                          │
                                     (B, 6144)
                                          │
                           chunk(6) → 6 × (B,1024)
                                          │
                              ┌───────────┤
                              ↓           ↓
                    Self-Attn modulation  FF modulation

Input latent: (B, 64, 256)
         │
         ├── preprocess_conv → rearrange → (B, 256, 1024)
         │
         ├── 16 × TransformerBlock (adaLN = score only)
         │          ↑ cross_attn from prompt+seconds_total: (B, 65, 768)
         │
         └── rearrange → postprocess_conv → (B, 64, 256)
```

**차이점**: input_add 경로 제거, global_cond에 score만 → 깔끔한 단일 경로

---

## 8. 참고: DiT 원 논문과의 비교

| 항목 | DiT (Peebles & Xie) | SAO-Small (현재) | SAO-Small (제안) |
|------|---------------------|-----------------|-----------------|
| Class conditioning | adaLN (class embed만) | adaLN (score + seconds 합산) | adaLN (score만) |
| Timestep | adaLN에 합산 | 별도 처리 (SAO 구조) | 변경 없음 |
| Text | 없음 | Cross-attention | 변경 없음 |
| Duration | 없음 | Cross-attn + Global (중복) | Cross-attn만 |
| 합산 문제 | 없음 (class 하나) | ★ 있음 | ★ 해소 |

DiT 논문에서 adaLN은 **하나의 conditioning signal**에 대해 설계되었다. 여러 신호를 합산해서 넣는 것은 원 설계 의도와 다르며, 각 신호의 독립적 학습을 방해한다.

---

## 9. 결론 및 Action Items

### 즉시 적용 가능 (config 변경만)
1. **`seconds_total`을 `global_cond_ids`에서 제거** — sum-merge 해소
2. **`continuous_score`를 `input_add_ids`에서 제거** — 단일 경로화

### 실험으로 검증 필요
3. adaLN-only (score 전용) vs adaLN+input_add (dual-path) 성능 비교
4. Prepend mode에서도 sum-merge 해소 시 성능 변화 확인

### 중장기 고려
5. Separate adaLN head (score와 seconds_total 각각 독립 embedder)
6. Score normalization ([0,1] 범위 정규화 vs raw score)
