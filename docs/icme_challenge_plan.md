# ICME 2026 NTU ATTM Grand Challenge — 실행 계획

> 작성일: 2026-03-15
> Challenge: Text-to-Music Generation (500M core model limit)
> 팀 목표: Score-conditioned SAO-Small로 FAD/CLAP/CCS 최적화

---

## 1. Challenge 규칙 요약

### 모델 제한
- **Core generative model ≤ 500M parameters**
- Core = 메인 생성 아키텍처 (DiT 등)
- **Auxiliary (제외)**: audio encoder/decoder (VAE), text encoder (T5), vocoder

### 우리 모델 파라미터 구성

| 구성 요소 | 파라미터 수 | ICME 분류 |
|-----------|-----------|----------|
| VAE Encoder (Oobleck) | 78.0M | Auxiliary (제외) |
| VAE Decoder (Oobleck) | 78.1M | Auxiliary (제외) |
| T5-base Text Encoder | ~109M | Auxiliary (제외) |
| **DiT Core** | **339.1M** | **Core** |
| NumberConditioner (seconds_total) | 0.2M | Core (conditioner) |
| FourierScoreConditioner | 0.8M | Core (conditioner) |
| ScoreInputConcatConditioner | 0.01M | Core (conditioner) |
| **Core 합계** | **~340M** | **500M 대비 68%, 160M 여유** |

### 평가 메트릭
1. **FAD (Fréchet Audio Distance)** — 생성 오디오 품질 (↓ better)
2. **CLAP Score** — 텍스트-오디오 의미적 일치 (↑ better)
3. **CCS (Concept Coverage Score, K/M)** — 프롬프트 내 개별 개념 존재 여부 (↑ better)
   - Audio LM이 블라인드 심사관으로 각 개념(장르, 악기, 무드 등) 감지
   - 프롬프트당 M개 개념 중 K개 감지 → K/M 점수

### 평가 조건
- 오디오 ≥ 10초, **첫 10초만 평가**
- 테스트 프롬프트: 100개
- 프롬프트 형식: Qwen2-Audio-7B가 태그에서 생성한 자연어 캡션
  - 예: `"An energetic rock track driven by a bold electric guitar, pulsing with intensity and raw power."`

---

## 2. 전략 개요

### 메트릭별 대응 전략

| 메트릭 | 핵심 요인 | 전략 |
|--------|---------|------|
| **FAD** | 생성 품질, 분포 일치 | Score conditioning → 추론 시 `score=max` 생성 |
| **CLAP** | 텍스트-오디오 일치 | 캡션 증강 (테스트 프롬프트 스타일에 맞춤) + text CFG 튜닝 |
| **CCS** | 개별 개념 커버리지 | 캡션에 구체적 태그(장르, 악기, 무드) 포함 학습 |

### 핵심 인사이트
1. **Score conditioning이 FAD에 직접 기여**: 고품질 점수로 생성하면 FAD↓
2. **캡션 형식이 CLAP/CCS에 결정적**: FMA의 `"A Rock, Electronic song"` vs 테스트의 `"An energetic rock track with electronic elements"` → 형식 불일치 시 CLAP/CCS↓
3. **160M 여유**: DiT를 키우거나 추가 conditioner 가능하지만, 학습 비용 대비 캡션 증강이 더 효과적일 수 있음

---

## 3. 실행 계획 (Phase별)

### Phase 1: 데이터 준비 (현재 진행 중)

**1-1. Jamendo Feature 추출**
- [x] 10_prep: 54,753 JSON 메타데이터 생성 완료
- [ ] 11_extract: MERT + CLAP audio + CLAP text 피처 추출 (GPU 0,1, ~27h)
- [ ] 12_scoring: Music-RankNet으로 스코어 부여
- [ ] 13_rankings: 스코어 분포 분석 및 threshold 계산

**1-2. 합산 데이터셋 구성**
- FMA-Large: 106K tracks (ISMIR primary)
- MTG-Jamendo: 54K tracks (ICME primary)
- 합산: ~160K tracks
- 각 트랙에 `reward_score` 부여 (Music-RankNet)

### Phase 2: 캡션 증강 (Critical for CLAP/CCS)

**문제**: FMA 메타데이터는 장르 태그만 (`"A Rock, Electronic song"`). 테스트 프롬프트는 Qwen2-Audio 스타일의 자연어 캡션 (`"An energetic rock track driven by a bold electric guitar"`).

**해결 방안**:

| 방법 | 구현 | 장점 | 단점 |
|------|------|------|------|
| **A. LLM 캡션 생성** | GPT-4/Qwen2로 태그→캡션 변환 | 테스트 형식과 동일 | API 비용, 160K 트랙 |
| **B. 템플릿 증강** | 규칙 기반 다양한 문장 구조 | 빠르고 저렴 | 다양성 제한 |
| **C. Qwen2-Audio 직접 사용** | 오디오 입력 → 캡션 생성 | 가장 정확 (테스트와 동일 모델) | 추론 비용 큼 |
| **D. A+B 혼합** | 핵심 데이터 LLM, 나머지 템플릿 | 균형 | 구현 복잡도 |

**권장**: 방법 D
- Jamendo (54K): 이미 풍부한 태그 보유 → 템플릿 증강
- FMA 상위 30% (32K): LLM 캡션 생성 (고품질 데이터 집중)
- FMA 하위 70% (74K): 템플릿 증강

**템플릿 예시**:
```
태그: [rock, electric guitar, energetic]
→ "An energetic rock track driven by a bold electric guitar, pulsing with intensity."
→ "A high-energy rock piece featuring powerful electric guitar riffs and driving rhythms."
→ "Electric guitar leads this energetic rock composition with raw power and momentum."
```

### Phase 3: Score-Conditioned 학습

**3-1. 기본 학습** (ISMIR과 공유)
- Config: `model_config_with_score_xattn.json` 또는 `xattn_concat`
- Profile: `xattn` (2.6M trainable)
- 데이터: FMA + Jamendo (합산)
- Score dropout: 30%
- Fourier score embedding

**3-2. ICME 특화 학습**
- 증강된 캡션으로 재학습
- 캡션 형식이 테스트 프롬프트와 일치하도록 보장
- 10초 생성에 최적화 (`seconds_total=10`)

### Phase 4: 추론 최적화

**4-1. Score 설정**
- `continuous_score = max_score` (고품질 유도 → FAD↓)
- 다양한 score 값에서 FAD 측정하여 최적값 탐색

**4-2. CFG Scale 튜닝**
- Text CFG scale: CLAP/CCS 최적화
- Score CFG scale: FAD 최적화 (score-specific CFG 구현 필요)
- Grid search: text_cfg ∈ {3, 5, 7} × score_cfg ∈ {1, 3, 5, 7}

**4-3. Sampling 설정**
- Steps: 50-100 (품질 우선)
- Sampler: DDIM or DPM++
- `seconds_total=10` 고정

---

## 4. 타임라인

| 주차 | 작업 | 의존성 |
|------|------|--------|
| W1 (현재) | Fourier+xattn 결과 확인, Jamendo 피처 추출 완료 | - |
| W2 | Jamendo 스코어링, 합산 데이터셋 구성, 캡션 증강 시작 | W1 |
| W3 | ICME 학습 시작 (증강 캡션 + score conditioning) | W2 |
| W4 | 추론 최적화 (CFG grid search, score 최적값) | W3 |
| W5 | 최종 제출 준비, ablation 정리 | W4 |

---

## 5. 리스크 및 대안

| 리스크 | 영향 | 대안 |
|--------|------|------|
| Score conditioning이 FAD를 유의미하게 낮추지 못함 | FAD 메트릭 미개선 | Data filtering (Case 4): 상위 30% 데이터만 사용 |
| 캡션 증강 품질이 낮음 | CLAP/CCS 저하 | Qwen2-Audio로 직접 오디오→캡션 생성 |
| Fourier+xattn correlation이 낮게 유지 | Score conditioning 무효 | 추론 시 score 없이 제출 (baseline) |
| 160M 여유 미활용 | 성능 미최적화 | DiT depth 증가 (16→20) 또는 LoRA adapter 추가 |

---

## 6. 160M 파라미터 여유 활용 방안

현재 Core = 340M. 160M 추가 가능.

| 방안 | 추가 파라미터 | 효과 | 리스크 |
|------|-------------|------|--------|
| DiT depth 16→20 | ~85M | 모델 용량↑ | 처음부터 사전학습 필요 |
| DiT embed 1024→1280 | ~105M | 표현력↑ | 처음부터 사전학습 필요 |
| LoRA (rank=64, 전 레이어) | ~20M | 미세 조정 효율↑ | 구현 필요 |
| 추가 conditioner (pitch, rhythm) | ~7M | 제어성↑ | LatCHs 스타일, 학습 필요 |
| **Cross-attention adapter** | ~10-30M | 캡션 이해력↑ | 비교적 안전 |

**권장**: LoRA adapter (저위험, 높은 효과) 또는 현재 340M 유지 (안전 우선)

---

*이 문서는 ICME 2026 NTU ATTM Grand Challenge 준비를 위한 실행 계획입니다.*
