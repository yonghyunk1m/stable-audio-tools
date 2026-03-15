# Score Conditioning 전략 후보 — 우선순위별 정리

> 작성일: 2026-03-15
> 목적: 현재 실험 결과를 바탕으로 다음 시도할 전략을 우선순위별로 정리
> 기준: 성공 가능성 × 구현 비용 × ICME/ISMIR 기여도

---

## 현재까지 확인된 사실

| 사실 | 근거 |
|------|------|
| adaLN은 약한 스코어 신호에 부적합 | v5-v10 (6회 실패, Corr~0) |
| Cross-attention은 작동함 | xattn v2: Corr=0.256, Mono=0.597 |
| Fourier embedding이 Linear보다 나을 것으로 기대 | 검증 중 (현재 학습 진행) |
| Input-concat은 추가적 경로로 가능 | DDP 버그 수정 후 학습 중 |
| Core model = 340M, **160M 여유** | ICME 500M 기준 |
| ICME 평가: FAD + CLAP + CCS | FAD=품질, CLAP=텍스트일치, CCS=개념커버리지 |

---

## 전략 후보 (우선순위순)

### Tier 1: 높은 확신, 즉시 시도 가능

#### 1-A. Score-specific CFG Scale (추론 시)

**아이디어**: 텍스트와 스코어에 서로 다른 CFG scale 적용.

```python
# 현재: 모든 조건에 동일한 cfg_scale
output = uncond + cfg_scale * (cond - uncond)

# 개선: 스코어 조건만 별도 증폭
# cross-attention에서 score 토큰의 기여를 분리하여 별도 스케일링
```

- **비용**: 코드 수정만, 학습 불필요 (현재 체크포인트에서 바로 실험 가능)
- **기대 효과**: 스코어 CFG 올리면 FAD↓, 텍스트 CFG 올리면 CLAP↑
- **리스크**: 낮음
- **ICME 기여**: FAD + CLAP 동시 최적화
- **우선순위**: ★★★★★ (비용 0, 효과 검증 즉시 가능)

#### 1-B. 캡션 증강 (Caption Augmentation)

**아이디어**: FMA/Jamendo의 태그 기반 프롬프트를 테스트 프롬프트 스타일로 변환.

```
현재:  "A Rock, Electronic, Indie-Rock song."
변환:  "An energetic rock track blending electronic elements with indie-rock sensibility,
       driven by distorted guitars and pulsing synthesizers."
```

- **비용**: LLM API 또는 템플릿 구현 (학습 데이터 재구성)
- **기대 효과**: CLAP↑, CCS↑ (테스트 프롬프트와 형식 일치)
- **리스크**: 낮음 (캡션 품질만 확보하면 됨)
- **ICME 기여**: CLAP + CCS (평가의 2/3)
- **우선순위**: ★★★★★ (ICME에서 가장 큰 영향)

#### 1-C. Data Filtering (Case 4)

**아이디어**: 스코어 상위 N%만으로 학습. Score conditioning 없이도 FAD↓.

- **비용**: 설정 변경만
- **기대 효과**: FAD↓ (저품질 데이터 제외)
- **리스크**: 데이터 다양성↓ → CLAP/CCS에 부정적일 수 있음
- **ICME 기여**: FAD
- **우선순위**: ★★★★ (score conditioning과 병행 가능)

---

### Tier 2: 합리적 기대, 구현 필요

#### 2-A. LoRA on Cross-Attention

**아이디어**: Cross-attention의 Q/K/V projection에 Low-Rank Adapter 추가.

```
기존: Q = W_q × x           (W_q frozen)
LoRA: Q = (W_q + B × A) × x  (B: d×r, A: r×d, r=32~64)

추가 파라미터: 16 blocks × 4 projections × 2 × r × d
r=32, d=1024: 16 × 4 × 2 × 32 × 1024 = ~4.2M
r=64: ~8.4M
```

- **비용**: LoRA 구현 필요 (peft 라이브러리 또는 직접)
- **기대 효과**:
  - Cross-attention이 스코어 토큰을 더 잘 처리 → Corr↑
  - 텍스트 이해도도 향상 → CLAP↑, CCS↑
- **리스크**: 중간 (LoRA rank 튜닝 필요)
- **ICME 기여**: FAD + CLAP + CCS (전 메트릭)
- **우선순위**: ★★★★ (cross-attention이 검증된 경로이므로 강화하는 것이 합리적)

#### 2-B. Inference-Time Guidance (LatCHs 방식)

**아이디어**: 별도의 경량 모델이 latent에서 품질을 예측하고, 추론 시 gradient guidance로 고품질 방향 유도.

```
학습: LatCH = Transformer(latent → quality_score)  // ~7M params, ~4h 학습
추론: z_t → DiT → z_0|t → LatCH(z_0|t) → quality_score
      ∇z_t(quality_score) → guidance gradient → z_t 업데이트
```

- **비용**: LatCH 학습 (7M params, 4h on 1 GPU) + 추론 코드 수정
- **기대 효과**: FAD↓ (추론 시 품질 직접 최적화)
- **리스크**: 추론 시간 증가 (selective TFG로 완화 가능)
- **ICME 기여**: FAD (직접적)
- **우선순위**: ★★★ (base model 수정 불필요, 독립적으로 추가 가능)
- **참고**: LatCHs 논문 (Novack et al., 2026)에서 SAO에서 검증됨

#### 2-C. FiLM Conditioning (Feature-wise Linear Modulation)

**아이디어**: 각 transformer block의 cross-attention 출력에 score-dependent scale/shift 적용.

```python
# 각 block에서:
gamma, beta = ScoreFiLM(score_embedding)  # Linear(768, 2*1024)
x = gamma * cross_attn_output + beta
```

- **비용**: 각 block에 FiLM layer 추가 (16 × 2 × 768 × 1024 = ~25M)
- **기대 효과**: adaLN보다 직접적이고 가벼움, cross-attention 출력을 직접 변조
- **리스크**: adaLN과 유사한 실패 가능성 (새 파라미터 학습 필요)
- **ICME 기여**: FAD
- **우선순위**: ★★★ (adaLN 실패 경험상 신중하게 접근)

---

### Tier 3: 실험적, 장기 고려

#### 3-A. Score를 텍스트 프롬프트에 삽입

**아이디어**: Score 값을 텍스트에 직접 인코딩. 새 파라미터 0개.

```
프롬프트: "A Rock song. [Quality: excellent, production: professional]"
vs
프롬프트: "A Rock song. [Quality: poor, production: amateur]"
```

- **비용**: 데이터 전처리만 (학습 시 score→텍스트 매핑)
- **기대 효과**: T5가 이미 품질 관련 언어를 이해 → 즉시 작동 가능
- **리스크**: 이산적 (continuous control 어려움), 텍스트 길이 증가
- **우선순위**: ★★ (간단하지만 세밀한 제어 어려움)

#### 3-B. DiT Depth/Width 확장

**아이디어**: 160M 여유를 활용해 DiT를 키움.

- depth 16→20: ~85M 추가
- embed_dim 1024→1280: ~105M 추가

- **비용**: 사전학습부터 다시 해야 함 (매우 높음)
- **리스크**: 매우 높음 (학습 시간, 데이터 부족)
- **우선순위**: ★ (현실적으로 어려움)

#### 3-C. Reward-Weighted Loss

**아이디어**: 디노이징 loss에 score 가중치 적용.

```python
loss = weight(score) * ||v_pred - v_target||²
# score가 높은 샘플에 더 큰 가중치 → 고품질 패턴 집중 학습
```

- **비용**: 코드 수정 (loss 함수)
- **기대 효과**: FAD↓ (고품질 데이터에 집중)
- **리스크**: 낮은 score 데이터 무시 → 다양성↓
- **우선순위**: ★★ (data filtering의 soft 버전)

---

## 추천 실행 순서

```
지금 (비용 0)
  └─ 1-A: Score-specific CFG scale 실험 (현재 체크포인트에서)

이번 주
  ├─ 1-B: 캡션 증강 시작 (ICME 최우선)
  ├─ 1-C: Data filtering 실험 (상위 30% FMA)
  └─ 현재 Fourier+xattn 결과 확인

다음 주
  ├─ 2-A: LoRA on cross-attention 구현 및 실험
  └─ 2-B: LatCH 학습 (독립적, 병렬 가능)

그 이후
  └─ 결과 기반 최종 조합 결정
```

---

## 최종 제출 구성 (예상)

ICME 제출 시 다음 조합이 가장 유력:

```
Base:     SAO-Small DiT (339M)
Score:    FourierScoreConditioner (0.8M) via cross-attention
          + (optional) LoRA on cross-attention (4-8M)
Data:     FMA + Jamendo, 증강 캡션, 상위 N% 필터링
Inference: Score-specific CFG + (optional) LatCH guidance
Total Core: ~344-348M (500M 이내)
```

---

*이 문서는 실험 결과에 따라 지속 업데이트됩니다.*
