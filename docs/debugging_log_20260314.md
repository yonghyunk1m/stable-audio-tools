# Debugging & Pipeline Alignment Log — 2026-03-14

> 작성일: 2026-03-14
> 대상: Case 3b (adaln profile) 학습의 Validation Monitoring Pipeline
> 요약: RewardMonitorCallback의 6개 버그를 수정하여 validation metric (correlation, monotonicity)이 정상 동작하도록 함

---

## 1. 문제 상황 (What was broken)

Case 3b v1 학습 실행 후 wandb에서 확인된 증상:
- `val/reward_correlation` = **0.0** (모든 epoch)
- `val/score_monotonicity` = **0.0** (모든 epoch)
- `val_audio` 패널이 **2개만** 표시 (100개 샘플 중)

원인: `RewardMonitorCallback`의 여러 단계에서 버그가 중첩되어, validation이 무의미한 데이터를 생성하고 있었음.

---

## 2. 수정된 버그 목록 (6 Fixes)

### Fix 1: Threshold Key Mismatch — 목표 점수가 전부 0.0

| 항목 | Before (broken) | After (fixed) |
|------|----------------|---------------|
| JSON 키 | `bin_{i}_median` (i=10→1) | `top_{i}_percent` (i=10,20,...100) |
| 결과 | `thresholds_data.get()` → default 0.0 | 정확한 중앙값 로드 |

```python
# Before: 존재하지 않는 키로 접근 → 전부 0.0
for i in range(10, 0, -1):
    scores.append(float(thresholds_data.get(f"bin_{i}_median", 0.0)))

# After: 실제 JSON 키와 매칭
for i in range(10, 110, 10):
    scores.append(float(thresholds_data.get(f"top_{i}_percent", 0.0)))
```

**영향**: 모든 target score가 0.0이므로 correlation/monotonicity 계산 불가 → 항상 0.

---

### Fix 2: Baseline Null Sample 낭비 — 첫 10개 샘플 버림

```python
# Before: 첫 10개 샘플을 null condition(-999.0)으로 강제 → target_scores_log에서 제외
if generated_count < 10:
    target_val = self.null_condition_value  # -999.0

# After: 모든 샘플이 target score 순환
if self.use_score_conditioning:
    target_val = float(self.target_score_list[generated_count % 10])
```

**영향**: 100개 중 10개 낭비 + top_100% bin이 한 번도 측정되지 않음.

---

### Fix 3: Wandb Audio Key 충돌 — 90개 샘플이 같은 키로 덮어씌워짐

```python
# Before: target이 전부 0.00이므로 키가 동일
key = f"val_audio/Target_{target_val:.2f}"  # "val_audio/Target_0.00" × 90

# After: bin index + percentile로 유니크 키 생성
bin_idx = generated_count % 10
pct = (bin_idx + 1) * 10
key = f"val_audio/top{pct}pct_{target_val:.2f}"
# → "val_audio/top10pct_0.91", "val_audio/top20pct_0.60", ...
```

**영향**: wandb에 2개 audio 패널만 표시 (마지막 덮어쓰기 결과).

---

### Fix 4: `sample_rate` 접근 오류

```python
# Before: pl_module에 sample_rate 속성이 없어 항상 fallback
model_sr = getattr(pl_module, "sample_rate", 44100)

# After: model_config dict에서 가져옴
model_config = getattr(pl_module, "model_config", {})
model_sr = model_config.get("sample_rate", 44100)
```

---

### Fix 5 (Critical): CLAP/Text Embedding Pipeline Mismatch

이전 구현이 pretrained scoring pipeline (04_extract_fma_features.py)과 완전히 다른 방식으로 embedding을 추출하여, **reward model 점수가 무의미**했던 문제.

#### 5a. CLAP Model 로딩

| 항목 | Before | After |
|------|--------|-------|
| 로딩 방식 | `model.load_state_dict(ckpt["model"], strict=False)` | `clap_load_state_dict()` + `model.load_state_dict()` |
| 차이 | 가중치 키 prefix 불일치로 일부만 로드 | factory 함수가 키 정규화 처리 |
| CLAP cosine sim | ~0.05 (사실상 랜덤) | ~0.97 (거의 동일) |

#### 5b. CLAP Audio Embedding

```python
# Before: get_audio_embedding_from_data() — 내부 전처리 파이프라인 다름
embedding = self.clap_model.get_audio_embedding_from_data(x=waveform, use_tensor=True)

# After: temp file로 저장 후 get_audio_embedding_from_filelist() — scoring pipeline과 동일
torchaudio.save(tmp_path, waveform.cpu(), target_sr)  # 48kHz
embedding = self.clap_model.get_audio_embedding_from_filelist(x=[tmp_path])
```

#### 5c. CLAP Text Embedding

```python
# Before: 수동 RobertaTokenizer → model 내부 메서드 직접 호출
text_data = self.tokenizer(texts, padding="max_length", ...)
embedding = self.clap_model.model.get_text_embedding(text_data)

# After: CLAP 공식 API 사용
embedding = self.clap_model.get_text_embedding(texts, tokenizer=_tokenizer_no_squeeze)
```

| Embedding | Before cosine sim | After cosine sim |
|-----------|------------------|-----------------|
| CLAP audio | 0.05 | 0.85–0.97 |
| CLAP text | -0.04 | 1.00 |
| Reward score diff | ~0.5 (무의미) | 0.005–0.31 (정상) |

---

### Fix 6: laion_clap 1.1.4 호환성 (sao 환경 전용)

`sao` conda env의 laion_clap==1.1.4와 `music-ranknet` env의 1.1.7 사이에 2가지 호환성 차이:

| 이슈 | 1.1.4 (sao) | 1.1.7 (music-ranknet) | 수정 방법 |
|------|------------|----------------------|----------|
| `position_ids` 키 | checkpoint에 있으나 모델이 거부 (`strict=True`) | `load_state_dict()`에서 자동 제거 | 수동으로 `state_dict.pop("text_branch.embeddings.position_ids", None)` |
| `tokenizer()` squeeze | `return {k: v.squeeze(0) ...}` → 1D 텐서 | `return result` → 2D 유지 | 커스텀 tokenizer를 `get_text_embedding()`에 전달 |

```python
# position_ids 수정
state_dict = clap_load_state_dict(self.clap_ckpt_path, skip_params=True)
state_dict.pop("text_branch.embeddings.position_ids", None)
self.clap_model.model.load_state_dict(state_dict)

# tokenizer squeeze 수정
def _tokenizer_no_squeeze(text):
    return self.clap_model.tokenize(
        text, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
    )
embedding = self.clap_model.get_text_embedding(texts, tokenizer=_tokenizer_no_squeeze)
```

---

### Fix 7: Threshold를 경계값(boundary)에서 중앙값(median)으로 변경

| Bin | Before (boundary, `np.percentile`) | After (median of bin) |
|-----|-----------------------------------|----------------------|
| Top 10% | 0.7305 | **0.9069** |
| Top 20% | 0.4679 | **0.5993** |
| Top 30% | 0.2855 | **0.3779** |
| Top 40% | 0.0664 | **0.1618** |
| Top 50% | -0.0992 | **-0.0210** |
| Top 60% | -0.2544 | **-0.1809** |
| Top 70% | -0.4326 | **-0.3444** |
| Top 80% | -0.5991 | **-0.5157** |
| Top 90% | -0.9574 | **-0.7782** |
| Top 100% | -5.7979 | **-1.5160** |

**근거**: 경계값은 bin의 "끝 지점"이므로, 해당 분위 그룹의 대표값으로는 중앙값이 더 적합. 특히 Top 100%의 경계(-5.80)는 극단적 이상치를 반영하여 비현실적.

---

## 3. 현재 Conditioning Pipeline (Data → Score → Generation)

### 3.1 Score 사전 계산 (offline, music-ranknet repo)

```
FMA-Large audio files
    ↓  04_extract_fma_features.py
    ├── CLAP audio embedding (512d) — load_ckpt() + get_audio_embedding_from_filelist()
    ├── MERT embedding (1024d) — m-a-p/MERT-v1-330M last_hidden_state mean pooling
    ├── CLAP text embedding (512d) — get_text_embedding() (genre → "A {genre} song.")
    └── FLAG (1d) = 1.0
    ↓  concat: [FLAG(1), CLAP(512), MERT(1024), TEXT(512)] = 2049d
    ↓  05_fma_large_scoring.py
    ↓  MusicRankNet.forward(features) → scalar score
    ↓
reward_score → JSON sidecar (audio 파일 옆에 저장)
    ↓  06_calculate_thresholds.py
    ↓  10-bin 중앙값 계산 → reward_thresholds.json
```

### 3.2 학습 시 Conditioning 주입 (online, stable-audio-tools)

```
DataLoader → batch = (audio, metadata)
    metadata["reward_score"] → ContinuousScoreConditioner
        ↓  Linear(1, 768) → conditioning tensor
        ↓  CFG dropout: 15% 확률로 -999.0 주입 → zero vector
        ↓
DiT에 2가지 경로로 주입:
    1. global_cond (prepend or adaLN) — seconds_total과 합산
    2. input_add — latent에 잔차 추가 (Conv1d 768→64)
```

### 3.3 추론 시 Score Conditioning (inference)

```
User 지정: target_score = 0.91 (Top 10% 중앙값)
    ↓
conditioning = [{"prompt": "A jazz song.", "seconds_total": 10.0, "continuous_score": 0.91}]
negative_conditioning = [{"prompt": "", "seconds_total": 10.0, "continuous_score": -999.0}]
    ↓
Classifier-Free Guidance (CFG scale=3.5):
    output = uncond + cfg_scale * (cond - uncond)
    ↓
    Selective CFG: sigma < 0.8이면 CFG 비활성화 (denoising 후반부)
```

---

## 4. Validation Monitoring (RewardMonitorCallback)

### 4.1 동작 흐름

```
매 5000 steps:
    ↓
100개 validation 샘플 생성 (10 bins × 10 repeats)
    target: [0.91, 0.60, 0.38, 0.16, -0.02, -0.18, -0.34, -0.52, -0.78, -1.52] × 10
    ↓
각 샘플에 대해:
    1. DiT로 오디오 생성 (50 steps, CFG 3.5)
    2. 생성 오디오에서 feature 추출 (on-the-fly):
       - CLAP audio embedding (48kHz, temp file → get_audio_embedding_from_filelist)
       - MERT embedding (24kHz, Wav2Vec2FeatureExtractor → last_hidden_state mean)
       - CLAP text embedding (get_text_embedding with custom tokenizer)
       - FLAG = 1.0
    3. MusicRankNet(concat_features) → measured_score
    ↓
Metric 계산 & wandb 로깅:
    - val/reward_correlation: Pearson correlation(target_scores, measured_scores)
    - val/score_monotonicity: pairwise ordering agreement ratio
    - val/measured_score_top_{10..100}_percent: bin별 평균 measured score
    - val_audio/top{pct}pct_{target:.2f}: 오디오 샘플 (100개 unique 키)
```

### 4.2 Monitoring Metrics 해석

| Metric | 의미 | 기대 추이 |
|--------|------|----------|
| `reward_correlation` | target과 measured score의 선형 상관 | 0 → 0.5+ (학습 진행 시) |
| `score_monotonicity` | target 순서대로 measured도 순서가 맞는 비율 | 0.5 (랜덤) → 0.7+ |
| `measured_score_top_10_percent` | 최고 품질 bin의 실제 점수 | 학습 초기: ~0.2 / 학습 후기: ~0.5+ |

### 4.3 Feature 추출 파이프라인 정합성 (Sanity Check 결과)

`04_extract_fma_features.py`의 사전 추출값 vs `RewardMonitorCallback`의 on-the-fly 추출값 비교:

| Feature | Method | Cosine Similarity | Score Diff |
|---------|--------|------------------|------------|
| CLAP audio | filelist (both) | 0.85–0.97 | — |
| CLAP text | get_text_embedding (both) | 1.00 | — |
| MERT | last_hidden_state mean (both) | — | — |
| Final score | MusicRankNet | — | 0.005–0.31 |

> CLAP audio cos가 1.0이 아닌 이유: 실시간 리샘플링(44.1→48kHz) vs 원본 48kHz 로딩의 미세 차이. 점수 차이 0.3 이내는 허용 범위.

---

## 5. 학습 이력 (Training Runs)

| Run | 날짜 | 상태 | 포함된 Fix | 결과 |
|-----|------|------|----------|------|
| case3b_adaln_v1 | 03-13 | completed | — | corr=0, mono=0, audio 2개만 |
| case3b_adaln_v2 | 03-14 00:00 | killed | Fix 1–4 | CLAP 로딩 실패 (`position_ids`) |
| case3b_adaln_v3 | 03-14 00:11 | crashed | Fix 1–5 | laion_clap 1.1.4 `position_ids` 에러 |
| case3b_adaln_v4 | 03-14 01:01 | killed | Fix 1–6a | `tuple index out of range` (tokenizer) |
| case3b_adaln_v4b | 03-14 01:05 | killed | Fix 1–6a + traceback | 에러 위치 확인 (RobertaModel.forward) |
| **case3b_adaln_v4c** | **03-14 01:08** | **running** | **Fix 1–7 (all)** | **100 samples, 0 errors, corr=0.03** |

### 현재 학습 상태 (v4c)
- **wandb**: `case3b_adaln_v4c` ([link](https://wandb.ai/yonghyunk1m/music-steerability-study))
- **GPU**: CUDA 8,9 (NVIDIA RTX A5000)
- **Profile**: adaln (7.4M trainable / 497M total)
- **Initial validation** (step 0): correlation=0.03, monotonicity=0.49
- **Train loss**: 1.3–2.1 범위 (정상)
- **다음 validation**: step 5000 (~30분 간격)

---

## 6. Git Commit History

```
2be308a  fix: resolve laion_clap 1.1.4 compat issues (position_ids + tokenizer squeeze)
f843a3f  fix: align CLAP/text extraction with pre-scoring pipeline
1fb5dd7  fix: correct reward thresholds, remove baseline null samples, stabilize sample_rate access
```

---

## 7. 남은 과제 (Next Steps)

- [ ] Step 5000 validation에서 correlation/monotonicity 개선 확인
- [ ] Case 2 (SFT baseline), Case 3a (adapter), Case 3c (hybrid) 실행
- [ ] ICME Track: 500M 파라미터 제한 → adapter(497.2M) 또는 global(499.0M) profile만 사용 가능
- [ ] Score normalization 전략 검토 (현재 raw score, 범위 ~[-5.8, +1.5])
- [ ] MTG-Jamendo 데이터셋 준비 (ICME Grand Challenge)
