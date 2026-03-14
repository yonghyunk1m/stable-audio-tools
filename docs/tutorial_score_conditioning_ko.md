# SAO-Small 스코어 조건부 오디오 생성: 완전 튜토리얼

**최종 수정일**: 2026-03-14

---

## 목차

1. [무엇을 만들고 있는가?](#1-무엇을-만들고-있는가)
2. [배경지식: 텍스트-투-오디오 생성의 원리](#2-배경지식-텍스트-투-오디오-생성의-원리)
3. [SAO-Small 아키텍처 상세 분석](#3-sao-small-아키텍처-상세-분석)
4. [스코어 컨디셔닝 문제 정의](#4-스코어-컨디셔닝-문제-정의)
5. [컨디셔닝 경로: 신호가 모델에 들어가는 4가지 방법](#5-컨디셔닝-경로-신호가-모델에-들어가는-4가지-방법)
6. [구현: 코드 분석](#6-구현-코드-분석)
7. [학습 파이프라인](#7-학습-파이프라인)
8. [검증 및 보상 모니터링](#8-검증-및-보상-모니터링)
9. [실험 기록: 시도한 것과 배운 것](#9-실험-기록-시도한-것과-배운-것)
10. [현재 최선의 접근법: Cross-Attention 스코어 컨디셔닝](#10-현재-최선의-접근법-cross-attention-스코어-컨디셔닝)
11. [핵심 디버깅 교훈](#11-핵심-디버깅-교훈)
12. [부록: 텐서 형상 참조표](#12-부록-텐서-형상-참조표)

---

## 1. 무엇을 만들고 있는가?

### 목표

텍스트 프롬프트뿐만 아니라 출력의 **품질**도 제어할 수 있는 텍스트-투-오디오 모델을 만들고자 한다.

입력 예시:
- 텍스트 프롬프트: *"재즈 피아노 솔로"*
- 품질 스코어: **9.0** (고품질) 또는 **2.0** (저품질)

모델은 텍스트에 맞으면서 동시에 원하는 품질 수준의 오디오를 생성해야 한다.

### 왜 중요한가

텍스트-투-오디오 모델은 텍스트 설명으로부터 오디오를 생성하지만, "품질"이라는 개념이 없다.
사용자가 최고 품질의 출력을 원해도 모델은 평범한 오디오를 생성할 수 있다.
품질 스코어로 컨디셔닝하면 다음이 가능해진다:

1. **품질 조향**: 원하는 수준의 고품질 오디오를 요청에 따라 생성
2. **커리큘럼 학습**: 저품질 데이터를 포함한 전체 데이터로 학습하되, 추론 시에는 고품질만 생성
3. **제어 가능한 생성**: 사용자가 품질과 다양성 사이의 균형을 조절

### 시스템 전체 구조

```
                                    스코어 컨디셔닝
                                    (우리가 추가한 부분)
                                         |
                                         v
  텍스트 프롬프트 --> [ T5 인코더 ] ------+---> [ DiT 트랜스포머 ] ---> [ VAE 디코더 ] ---> 오디오
                                         |            ^
  재생 시간 -------> [ Number 임베딩 ] ---+            |
                                                      |
  가우시안 노이즈 ---> (반복적 디노이징) ---------------+
```

우리는 기존의 텍스트-투-오디오 모델(Stable Audio Open Small, "SAO-Small")에
**품질 스코어**를 추가 컨디셔닝 신호로 넣는다.

---

## 2. 배경지식: 텍스트-투-오디오 생성의 원리

### 2.1 디퓨전 모델 60초 요약

디퓨전 모델은 **노이즈를 제거하는 방법**을 학습하여 데이터를 생성한다.

**학습 과정**: 실제 오디오에 노이즈를 추가하고, 모델이 그 노이즈를 예측하도록 학습한다.
```
깨끗한 오디오 x0 ---(노이즈 추가)---> 노이즈 오디오 xt ---(모델이 예측)---> 예측된 노이즈
                                                                              |
손실 = || 실제 노이즈 - 예측 노이즈 ||^2              <-------(최소화)----------+
```

**생성 과정**: 순수 노이즈에서 시작하여 반복적으로 노이즈를 제거해 깨끗한 오디오를 얻는다.
```
순수 노이즈 z_T --> (디노이즈) --> z_{T-1} --> (디노이즈) --> ... --> z_0 --> 깨끗한 오디오
```

### 2.2 잠재 디퓨전: 압축된 공간에서 작업하기

원시 오디오는 매우 크다 (44,100 샘플/초 * 2채널 * 12초 = ~100만 개의 숫자).
오디오에서 직접 디퓨전을 돌리면 너무 비싸다. 대신:

```
오디오 (2 x 524,288) ---> [ VAE 인코더 ] ---> 잠재 벡터 (64 x 256) ---> [ 디퓨전 ] ---> ...
      ~100만 개 숫자           압축              ~1.6만 개 숫자
                             2048배 작아짐
```

**변분 오토인코더(VAE)**가 오디오를 2048배 압축한다.
디퓨전은 이 작은 잠재 공간에서 동작하고, VAE 디코더가 다시 오디오로 변환한다.

### 2.3 정류 흐름 (SAO-Small의 학습 목표)

SAO-Small은 기존 디퓨전 대신 **정류 흐름(rectified flow)**을 사용한다.
핵심 아이디어: 노이즈에서 데이터로의 직선 경로를 학습한다.

```
순방향:  x_t = (1 - t) * x_0 + t * noise       (데이터와 노이즈를 섞음)
손실:    || model(x_t, t) - (noise - x_0) ||^2  ("속도"를 예측)
```

모델은 속도 `v = noise - x_0`를 예측한다. 이는 데이터에서 노이즈로 가는 방향
(또는 그 반대)을 알려준다. 추론 시에는 ODE 솔버를 사용하여 노이즈에서 데이터로
이 속도를 따라 역방향으로 이동한다.

### 2.4 분류기 없는 가이던스 (CFG)

CFG는 모델이 컨디셔닝을 더 강하게 따르도록 만드는 기법이다:

```python
output = uncond_output + cfg_scale * (cond_output - uncond_output)
```

**학습 중**: 15%의 확률로 조건을 드롭한다 (null로 대체).
이를 통해 모델이 조건부 생성과 비조건부 생성 모두를 학습한다.

**추론 시**: 모델을 두 번 실행하고 (조건 있음 / 없음) 차이를 증폭한다.
`cfg_scale=3.5`는 "자연스러운 것보다 3.5배 더 강하게 조건을 따르라"는 의미이다.

---

## 3. SAO-Small 아키텍처 상세 분석

### 3.1 모델 구성 요소

| 구성 요소 | 아키텍처 | 파라미터 수 | 역할 |
|-----------|---------|------------|------|
| VAE 인코더 | Oobleck (CNN) | ~78M | 오디오 -> 잠재 벡터 |
| VAE 디코더 | Oobleck (CNN) | ~78M | 잠재 벡터 -> 오디오 |
| 텍스트 인코더 | T5-base | ~109M | 텍스트 -> 임베딩 |
| DiT (디노이저) | 트랜스포머 | ~340M | 노이즈 예측 |
| **합계** | | **~497M** | |

### 3.2 DiT (디퓨전 트랜스포머)

DiT는 모델의 핵심이다. 디퓨전에 맞게 수정된 표준 트랜스포머 구조이다:

```
입력 잠재 벡터: (batch, 64, 256)        # 64채널, 256 타임스텝
                  |
          [패치 임베딩]                  # Conv1d: 64 -> 1024
                  |
          (batch, 1024, 256)             # 트랜스포머 차원으로 변환
                  |
    +--->[TransformerBlock x16]<---+
    |             |                |
    |     Self-Attention           |  텍스트 토큰이
    |     Cross-Attention   <------+  Cross-Attention으로 입력
    |     Feed-Forward             |
    |             |                |
    +-------------+                |
                  |                |
          [출력 프로젝션]            |
                  |                |
          (batch, 64, 256)         # 다시 잠재 차원으로
```

**SAO-Small 핵심 수치**:
- 임베딩 차원: 1024
- 어텐션 헤드 수: 8 (헤드 차원 = 128)
- 깊이: 16개 트랜스포머 블록
- 컨디셔닝 토큰 차원: 768 (T5-base 출력)
- 잠재 형상: (batch, 64, 256) = 샘플당 ~16K 값

### 3.3 원래 SAO의 컨디셔닝

원래 SAO-Small에는 두 가지 조건이 있다:

1. **텍스트 프롬프트** (T5-base 경유): `(batch, 64, 768)` 임베딩 생성
   - 각 트랜스포머 블록에서 **cross-attention**으로 입력
   - 잠재 벡터의 각 위치가 관련된 텍스트 토큰을 "바라볼" 수 있음

2. **재생 시간** (`seconds_total`): 단일 숫자 (예: 10.0)
   - `Linear(1, 768)` -> 768차원 임베딩으로 프로젝션
   - **cross-attention**과 **prepend/global 컨디셔닝** 모두로 입력

---

## 4. 스코어 컨디셔닝 문제 정의

### 4.1 스코어는 어디서 오는가?

우리는 학습된 품질 평가 모델인 **Music-RankNet**을 사용한다:

```
오디오 --> [MERT (1024d)] ----------+
       --> [CLAP 오디오 (512d)] ----+---> [RankNet MLP] ---> 스코어 (float)
       --> [CLAP 텍스트 (512d)] ----+         |
       --> [수공 피처 (1d)] --------+    [1024,512,256,128]
                                       샴 네트워크 구조
```

Music-RankNet은 각 오디오 파일에 대해 연속 스코어를 생성한다. 모든 학습 데이터
(FMA-Large, ~106K 트랙)에 대해 스코어를 사전 계산하여 메타데이터에 저장한다:

```jsonl
{"file": "000002.mp3", "prompt": "힙합 노래", "seconds_total": 30, "reward_score": 0.42}
{"file": "000005.mp3", "prompt": "록 노래",   "seconds_total": 30, "reward_score": -0.87}
```

스코어 분포: 대략 [-1.5, +0.9] 범위, 높을수록 좋은 품질.

### 4.2 왜 어려운가

사전학습된 모델에 새로운 컨디셔닝 신호를 추가하는 것이 어려운 이유:

1. **사전학습된 가중치는 민감하다**: 새 파라미터의 랜덤 초기화가 모델의 기존 능력을 망가뜨릴 수 있음
2. **모델은 스코어가 필요 없다**: 텍스트 + 오디오 잠재 벡터만으로 노이즈 예측이 가능. 스코어는 디노이징에 "잉여 정보"
3. **학습 가능한 파라미터가 적다**: 모델 대부분(~497M)을 프리징하고 스코어 관련 파라미터(~1-9M)만 학습

---

## 5. 컨디셔닝 경로: 신호가 모델에 들어가는 4가지 방법

DiT에는 컨디셔닝 신호를 위한 **4가지 경로**가 있다:

### 5.1 Cross-Attention (검증됨, 텍스트 + 재생시간이 사용하는 경로)

```
                    잠재 토큰              컨디셔닝 토큰
                   (batch, 256, 1024)     (batch, seq, 1024)
                         |                       |
                     [Q = W_q * x]          [K = W_k * c]
                         |                  [V = W_v * c]
                         |                       |
                    Attention(Q, K, V) ----------+
                         |
                   (batch, 256, 1024)   # 각 위치가 컨디셔닝에 어텐션
```

**작동 방식**: 잠재 시퀀스는 256개 위치를 가진다. 각 위치는 컨디셔닝 토큰
(텍스트 + 재생시간 + 스코어)에 대해 어텐션을 계산한다. 모델은 각 위치에서
어떤 컨디셔닝 토큰이 관련 있는지 학습한다.

**장점**: 검증된 경로, 유연함, 위치별 특화 컨디셔닝 가능.

### 5.2 Global Conditioning - Prepend 방식

```
    스코어 임베딩 (batch, 1, 1024)
              |
    시퀀스 앞에 추가:  [score_token, latent_1, latent_2, ..., latent_256]
              |
    Self-attention이 스코어 토큰을 시퀀스의 일부로 인식
```

**작동 방식**: 스코어 임베딩이 시퀀스 맨 앞에 추가 토큰이 된다.
Self-attention이 어느 위치에서든 이 토큰에 접근할 수 있다.

### 5.3 Global Conditioning - adaLN (적응적 레이어 정규화)

```
    스코어 임베딩 (batch, 768)
              |
    [global_cond_embedder]   Linear(768->1024) -> SiLU -> Linear(1024->6144)
              |
    (batch, 6144) -> 6 x (batch, 1024)로 분할
              |
    scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff
              |
    각 TransformerBlock에서:
        x = LayerNorm(x)
        x = x * (1 + scale) + shift        # 통계량 변조
        x = SelfAttention(x)
        x = x * sigmoid(1 - gate)          # 출력 게이팅
```

**작동 방식**: 스코어가 모든 레이어 활성화의 **통계량**(평균, 분산)을 변조한다.
원래 DiT 논문에서 ImageNet 클래스 레이블을 컨디셔닝하는 방식이다.

**발견된 문제**: `global_cond_embedder`가 제로 초기화되면 (사전학습 체크포인트에
없는 새 파라미터이므로), 출력이 항상 0이 된다. 스코어가 adaLN에 **전혀 영향을
미치지 않으며**, 그래디언트도 흐르지 않아 가중치 업데이트가 불가능하다.
[9장](#9-실험-기록-시도한-것과-배운-것)에서 상세히 설명한다.

### 5.4 Input-Add 어댑터 (채널별 잔차)

```
    스코어 임베딩 (batch, 768, 1)
              |
    [시퀀스 길이로 보간]  -> (batch, 768, 256)
              |
    [Conv1d(768, 64, kernel_size=1)]  -> (batch, 64, 256)   # 제로 초기화
              |
    잠재 벡터에 더하기:  latent = latent + adapter_output
```

**작동 방식**: 스코어가 시퀀스 전체에 브로드캐스트되어 잠재 벡터에 직접 더해진다.
Conv1d는 제로 초기화 (ControlNet 방식)되어 처음에는 영향 없이 시작한다.

**발견된 문제**: `Linear(1->768)` 이후 `Conv1d(768->64)`는 표현력이 매우 제한적이다.
스코어 값에 비례하는 하나의 "방향 벡터"만 추가할 수 있을 뿐이다.

### 5.5 비교표

| 경로 | 메커니즘 | 사전학습됨? | 새 파라미터 | 스코어에 적합? |
|------|---------|-----------|------------|---------------|
| Cross-attention | 토큰 어텐션 | O | ~769 | 현재 최선의 접근법 |
| Prepend | Self-attn 추가 토큰 | O | ~769 | 미검증 |
| adaLN | 레이어별 scale/shift/gate | X | ~9M | 실패 (v5-v10) |
| Input-add | 채널 잔차 | X | ~49K | 실패 (표현력 부족) |

---

## 6. 구현: 코드 분석

### 6.1 스코어 컨디셔너

가장 간단한 구성 요소: 스칼라 스코어를 768차원 임베딩으로 매핑한다.

```python
# stable_audio_tools/models/conditioners.py

class ContinuousScoreConditioner(nn.Module):
    def __init__(self, output_dim=768, cond_dim=1):
        super().__init__()
        self.mapper = nn.Linear(cond_dim, output_dim)   # 1 -> 768

    def forward(self, x, device):
        x = x.view(-1, 1)                              # (batch, 1)
        embeds = self.mapper(x)                         # (batch, 768)

        # Null 조건: score == -999.0 -> 제로 임베딩
        null_idx = (x.squeeze(-1) == -999.0)
        if null_idx.any():
            embeds[null_idx] = 0.0

        mask = torch.ones(embeds.shape[0], 1, device=device)
        return embeds, mask                             # (batch, 768), (batch, 1)
```

**왜 -999.0인가?** CFG 드롭아웃 시, 실제 스코어를 -999.0 (센티넬 값, 실제
데이터에는 절대 나오지 않는 값)으로 대체한다. 컨디셔너는 이 값에 대해 임베딩을
0으로 만든다. 이로써 모델은 "스코어 없음"이 어떤 것인지 학습한다. 추론 시
CFG는 "스코어 있음" vs "스코어 없음"을 비교하여 효과를 증폭한다.

### 6.2 모델 설정 (JSON)

설정 파일은 어떤 컨디셔너가 존재하고 어떤 경로를 사용하는지 지정한다:

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

`cross_attention_cond_ids`는 어떤 컨디셔너가 cross-attention에 입력되는지 결정한다.
`global_cond_ids`는 어떤 컨디셔너가 adaLN/prepend 글로벌 컨디셔닝에 입력되는지 결정한다.

### 6.3 컨디셔닝이 모델을 통과하는 과정

`diffusion.py`의 `get_conditioning_inputs()` 메서드가 모든 조건을 조립한다:

```python
def get_conditioning_inputs(self, conditioning_tensors):
    # Cross-attention: 모든 조건 토큰을 시퀀스 차원으로 연결
    cross_attention_input = []
    for key in self.cross_attn_cond_ids:       # ["prompt", "seconds_total", "continuous_score"]
        cond_in, mask = conditioning_tensors[key]

        if len(cond_in.shape) == 2:            # (B, 768) -> (B, 1, 768)
            cond_in = cond_in.unsqueeze(1)

        cross_attention_input.append(cond_in)

    cross_attention_input = torch.cat(cross_attention_input, dim=1)
    # 결과: (batch, 64 + 1 + 1, 768) = (batch, 66, 768)
    #        텍스트 토큰  초  스코어
```

결합된 컨디셔닝 시퀀스:
```
[text_tok_1, text_tok_2, ..., text_tok_64, seconds_tok, score_tok]
                                                           ^
                                                    우리가 추가한 부분!
```

### 6.4 CFG: 스코어 신호 증폭

`dit.py`에서 Classifier-Free Guidance는 모델을 두 번 실행한다:

```python
# 순방향 패스: 조건부 (스코어 있음) 와 비조건부 (스코어 없음)
batch_inputs = torch.cat([x, x], dim=0)                          # 배치 두 배로

if global_embed is not None:
    batch_global_cond = torch.cat([
        global_embed,                     # 조건부: 실제 스코어 임베딩
        torch.zeros_like(global_embed)    # 비조건부: 제로 (= 스코어 없음)
    ], dim=0)

# Cross-attention 조건도 두 배:
# 상위 절반: 실제 텍스트 + 실제 재생시간 + 실제 스코어
# 하위 절반: null 텍스트 + null 재생시간 + null 스코어 (제로)

output = model(batch_inputs, ...)

cond_output, uncond_output = output.chunk(2, dim=0)
final = uncond_output + cfg_scale * (cond_output - uncond_output)
```

**수정한 치명적 버그**: 이전에는 비조건부 패스가 조건부 패스와 **같은** 스코어
임베딩을 사용했다. 이는 CFG가 "스코어 있음"과 "스코어 없음"을 구별할 수 없었고,
스코어가 CFG에 완전히 투명(invisible)했다는 뜻이다!

수정: 비조건부 패스에 `torch.zeros_like(global_embed)`를 사용.

### 6.5 언프리즈 프로파일: 어떤 파라미터를 학습할 것인가

사전학습된 모델 전체를 프리징하고 스코어 관련 파라미터만 학습한다:

```python
UNFREEZE_PROFILES = {
    "xattn":   ["continuous_score", "to_cond_embed"],        # 1.8M 파라미터
    "adaln":   ["continuous_score", "to_global_embed",       # 9.3M 파라미터
                "to_scale_shift_gate", "global_cond_embedder"],
    "minimal": ["continuous_score"],                         # ~769 파라미터
}

def unfreeze_finetune_params(model):
    model.requires_grad_(False)                # 전부 프리징
    for name, param in model.named_parameters():
        if any(key in name for key in trainable_name_keys):
            param.requires_grad_(True)         # 매칭되는 파라미터만 언프리즈
```

`xattn` 프로파일이 언프리즈하는 것:
- `continuous_score`: `Linear(1, 768)` 매퍼 (769 파라미터)
- `to_cond_embed`: `Sequential(Linear(768,1024), SiLU, Linear(1024,1024))`
  cross-attention을 위한 컨디셔닝 토큰 프로젝션 (총 ~1.8M 파라미터)

### 6.6 가중치 초기화: ControlNet 제로 컨볼루션 패턴

사전학습 체크포인트에 없는 새 파라미터는 신중하게 초기화해야 한다:

```python
def zero_init_new_params(model, pretrained_keys):
    """
    ControlNet 제로 컨볼루션 원칙:
    - 출력 레이어: 제로 초기화 (새 경로가 영향 0에서 시작)
    - 중간 레이어: 소규모 랜덤 초기화 (그래디언트가 흐를 수 있도록)

    두 레이어 모두 제로 초기화하면:
      순방향:  h = W1 @ x = 0,  out = W2 @ h = 0     (항상 0)
      역방향:  dL/dW2 = dL/d(out) * h^T = ... * 0 = 0 (죽은 그래디언트!)
      네트워크가 절대 학습할 수 없다!

    출력 레이어만 제로로 하면:
      순방향:  h = W1 @ x != 0, out = W2 @ h = 0     (0에서 시작)
      역방향:  dL/dW2 = dL/d(out) * h^T != 0          (그래디언트 흐름!)
      W2가 업데이트되고, 이후 W1도 업데이트될 수 있다.
    """
    KEEP_RANDOM = ["continuous_score", "global_cond_embedder.0"]

    for name, param in model.named_parameters():
        if name not in pretrained_keys:
            if any(pat in name for pat in KEEP_RANDOM):
                param.data.normal_(0, 0.02)     # 소규모 랜덤
            else:
                param.zero_()                   # 제로
```

---

## 7. 학습 파이프라인

### 7.1 데이터 흐름

```
FMA-Large 오디오 파일 (106K 트랙)
         |
         v
[audio_dir Dataset] --> (audio_tensor, metadata_dict)
         |
         v
[ContinuousScoreDatasetWrapper]
    - metadata['reward_score'] 읽기
    - 15% CFG 드롭아웃: score -> -999.0 (null)
    - metadata['continuous_score'] = score_val 추가
         |
         v
[DataLoader] --> (audio, metadata) 배치
         |
         v
[학습 스텝]
    1. VAE 인코딩: 오디오 (2, 524288) -> 잠재 벡터 (64, 256)
    2. 노이즈 추가: latent_noisy = (1-t) * latent + t * noise
    3. 컨디셔너: metadata -> 컨디셔닝 텐서
    4. 모델 순방향: (latent_noisy, t, conditions)에서 속도 예측
    5. 손실 = MSE(예측 속도, 실제 속도)
    6. 역전파 & 언프리즌 파라미터만 업데이트
```

### 7.2 실행 스크립트

```bash
# 핵심 환경 변수
export SA_UNFREEZE_PROFILE=xattn          # 어떤 파라미터를 학습할지
export MODEL_CONFIG=./checkpoints/sao_small/model_config_with_score_xattn.json
export CUDA_VISIBLE_DEVICES=8,9           # GPU 2장
export SA_CFG_DROP_RATE=0.15              # 15% CFG 드롭아웃
export SA_LR=5e-5                         # 학습률
export SA_WARMUP_STEPS=1000               # 학습률 웜업
export SA_TOTAL_STEPS=300000              # 총 학습 스텝

# 실행
python finetune.py \
    --dataset-config ./configs/dataset_fma_scored.json \
    --model-config $MODEL_CONFIG \
    --pretrained-ckpt-path ./checkpoints/sao_small/model.safetensors \
    --batch-size 2 --accum-batches 4 \   # 유효 배치 = 2*4*2GPU = 16
    --precision 16-mixed \                # FP16 혼합 정밀도
    --checkpoint-every 5000 \
    --val-every 5000 \
    --logger wandb
```

### 7.3 학습률 스케줄

웜 리스타트를 가진 코사인 어닐링:
```
LR
 ^
 |  /\    /\    /\
 | /  \  /  \  /  \
 |/    \/    \/    \
 +-----|-----|-------> steps
  웜업  사이클1 사이클2
```

---

## 8. 검증 및 보상 모니터링

### 8.1 검증 작동 방식

5,000 스텝마다 `RewardMonitorCallback`이 실행된다:

1. **오디오 생성**: 서로 다른 목표 스코어에서 100개 오디오 클립 샘플링
   - 스코어 목표별 10개씩: 상위 10%, 20%, ..., 100% 백분위수
   - 검증 세트의 실제 텍스트 프롬프트 사용

2. **Music-RankNet으로 평가**: 각 생성 클립에 스코어 부여
   - MERT (1024d) + CLAP 오디오 (512d) + CLAP 텍스트 (512d) 피처 추출
   - 학습 스코어를 생성한 것과 같은 RankNet 통과

3. **메트릭 계산**:
   - **상관계수 (Correlation)**: 목표 스코어와 측정 스코어의 피어슨 상관관계
     - 완벽한 컨디셔닝: 상관계수 -> 1.0
     - 컨디셔닝 효과 없음: 상관계수 -> 0.0
   - **단조성 (Monotonicity)**: 높은 목표 -> 높은 측정인 쌍의 비율
     - 완벽: 1.0, 랜덤: 0.5

### 8.2 결과 해석 방법

```
Step 0 (학습 전):
  상관계수: 0.047   (사실상 랜덤)
  단조성:   0.510   (사실상 랜덤)
  -> 예상대로! 모델이 아직 스코어 사용법을 배우지 않았다.

Step 5000 (학습 후):
  상관계수: 0.35    (양의 상관관계!)
  단조성:   0.65    (랜덤보다 나음!)
  -> 모델이 스코어를 사용하는 법을 배우고 있다!

  구간별 측정 스코어:
    상위  10% (목표=0.91): 측정= +0.42   # 높은 목표 -> 높은 측정
    상위  50% (목표=0.25): 측정= -0.15   # 중간 목표 -> 중간 측정
    상위 100% (목표=-1.52): 측정= -0.88  # 낮은 목표 -> 낮은 측정
    -> 단조감소 = 스코어 컨디셔닝 작동!
```

---

## 9. 실험 기록: 시도한 것과 배운 것

### 실험 타임라인

| 버전 | 설정 | 경로 | 학습 파라미터 | 결과 | 근본 원인 |
|------|------|------|-------------|------|----------|
| v1-v4 | prepend | global_cond (prepend) | 1.5K | 효과 없음 | prepend 모드에서 adaLN 파라미터 부재 |
| v5 | adaln | global_cond (adaLN) | 7.4M | 상관=-0.03 | sum-merge + 이중 경로 중복 |
| v6 | adaln_pure | global_cond (adaLN만) | 7.4M | 상관=-0.12 | global_cond_embedder 죽은 그래디언트 |
| v7 | adaln_pure | adaLN + 소규모 랜덤 초기화 | 9.3M | 상관~0 | 디노이징 손실에서 스코어 잉여 |
| v8 | adaln_pure | adaLN + 전체 랜덤 초기화 | 9.3M | NaN | 큰 adaLN 값으로 인한 fp16 오버플로우 |
| v9 | adaln_pure | adaLN + to_global_embed | 9.3M | 상관~0 | 동일한 근본 문제 |
| v10 | adaln_pure | adaLN + CFG 수정 | 9.3M | 상관=-0.04 | adaLN이 품질 스코어에 부적합 |
| **v11** | **xattn** | **cross-attention** | **1.8M** | **진행 중** | **현재 실행 중** |

### 상세 실패 분석

#### 실패 1: adaLN 무효화 (v1-v4)

**설정**: `model_config_with_score.json` (prepend 모드) + `SA_UNFREEZE_PROFILE=adaln`.

**버그**: 설정 파일이 `global_cond_type: "prepend"`이었으므로 모델은 prepend
토큰을 사용하지, adaLN을 사용하지 않았다. adaLN 파라미터 (`to_scale_shift_gate`,
`global_cond_embedder`)는 prepend 모드에서는 아예 존재하지 않는다. 이들을
언프리즈하는 것은 무효 조작(no-op)이다 -- 실제로 학습된 것은 `Linear(1,768)`
매퍼의 1.5K 파라미터뿐이었다.

**교훈**: 언프리즈 프로파일이 모델 설정과 일치하는지 항상 확인할 것.

#### 실패 2: 제로 초기화의 죽은 그래디언트 (v5-v7)

**설정**: 올바른 adaLN 설정, global_cond_embedder 제로 초기화.

**버그**: `global_cond_embedder`는 2층 MLP이다:
```
Linear(1024, 1024) -> SiLU() -> Linear(1024, 6144)
```
두 레이어 모두 제로 초기화되었다. 이것은 죽은 네트워크를 만든다:

```
순방향:  h = W1 @ x + b1 = 0 + 0 = 0
         out = W2 @ SiLU(h) + b2 = W2 @ SiLU(0) + 0 = 0

역방향:  dL/dW2 = dL/d(out) * SiLU(h)^T = dL/d(out) * 0^T = 0
         dL/dW1 = (dL/d(out) * W2^T * SiLU'(h)) * x^T
                = (... * 0^T * ...) * x^T = 0

모든 그래디언트가 0이다. 네트워크가 절대 학습할 수 없다!
```

**수정**: 출력 레이어만 제로 초기화하고, 중간 레이어는 소규모 랜덤 값을 유지한다.
이것이 "ControlNet 제로 컨볼루션" 패턴이다.

#### 실패 3: 큰 초기화로 인한 NaN (v8)

**설정**: global_cond_embedder의 모든 레이어를 랜덤 초기화.

**버그**: 출력 레이어 (형상 1024x6144)에 `N(0, 0.02)` 초기화를 하면,
adaLN 변조 값이 너무 커진다:
```
output_norm ~ sqrt(1024) * 0.02 ~ 0.64
```
`to_scale_shift_gate` (역시 0이 아닌 값)와 합쳐지면, scale/shift 변조가
즉시 활성화 값을 fp16 범위 밖으로 밀어내어 NaN이 발생했다.

**교훈**: 새 컨디셔닝 경로의 출력 레이어는 반드시 0에서 시작해야 한다.

#### 실패 4: 디노이징에서의 스코어 잉여성 (v9-v10)

**설정**: 올바른 초기화, CFG 수정 적용, to_global_embed 언프리즈.

**여전히 실패한 이유**: 모든 것이 기술적으로 올바르더라도, adaLN 경로는
**디노이징 손실이 스코어 신호를 사용할 유인이 없기 때문에** 학습하지 않는다.

MSE 손실: `||v_pred(x_t, t, text, score) - v_target||^2`

모델은 `x_t`와 `text`만으로 이 손실을 완벽히 최소화할 수 있다.
스코어는 노이즈 예측에 **잉여 정보**이다. adaLN을 통해 스코어를 추가하면
(레이어 통계량을 변조하는 매우 간접적인 신호), 모델은 `global_cond_embedder`
출력을 0 근처로 유지하여 스코어를 무시하는 것이 가장 쉽다.

**왜 cross-attention은 다른가**: Cross-attention은 스코어를 모델이 어텐션할 수
있는 명시적 토큰으로 추가한다. 사전학습된 cross-attention 메커니즘은 이미
컨디셔닝 토큰(텍스트, 재생시간)에서 유용한 정보를 추출하도록 학습되어 있다.
스코어를 또 다른 토큰으로 추가하면 이 기존의 검증된 메커니즘을 활용하게 된다.

---

## 10. 현재 최선의 접근법: Cross-Attention 스코어 컨디셔닝

### 10.1 아키텍처

```
    텍스트 (64토큰, 각 768d)     재생시간 (1토큰, 768d)     스코어 (1토큰, 768d)
              |                           |                        |
              +---------- concat ---------+------------------------+
              |
        (batch, 66, 768)                     # 총 66개 컨디셔닝 토큰
              |
        [to_cond_embed]                      # 사전학습됨: Linear(768,1024) -> SiLU -> Linear(1024,1024)
              |
        (batch, 66, 1024)
              |
        각 TransformerBlock의 Cross-Attention에 입력
```

### 10.2 왜 작동해야 하는가

1. **검증된 경로**: 텍스트와 재생시간이 이미 cross-attention으로 성공적으로 작동
2. **사전학습된 프로젝션**: `to_cond_embed`이 이미 768d -> 1024d 매핑을 학습;
   스코어 임베딩도 768d이므로 호환됨
3. **어텐션의 유연성**: 각 잠재 위치가 학습된 어텐션 패턴에 따라 스코어 토큰에
   얼마나 어텐션할지 선택 가능
4. **CFG 증폭**: 비조건부 패스에 제로를 사용하면, CFG가 텍스트를 증폭하는 것처럼
   자연스럽게 스코어 신호도 증폭

### 10.3 설정

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

### 10.4 학습 설정

```bash
SA_UNFREEZE_PROFILE=xattn                  # 언프리즈: continuous_score + to_cond_embed
MODEL_CONFIG=model_config_with_score_xattn.json
# 학습 파라미터: 총 497M 중 ~1.8M (0.36%)
```

---

## 11. 핵심 디버깅 교훈

### 교훈 1: 실제로 무엇을 학습하고 있는지 확인하라

```python
# 항상 언프리즈된 파라미터와 형상을 출력한다
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  [UNFROZEN] {name}: {list(param.shape)}")
```

7.4M 학습 파라미터를 기대했는데 1.5K가 보인다면, 뭔가 잘못된 것이다.

### 교훈 2: 설정은 반드시 언프리즈 프로파일과 일치해야 한다

| 설정의 `global_cond_type` | 사용 가능한 파라미터 | `adaln` 프로파일 효과 |
|---------------------------|-------------------|---------------------|
| `"prepend"` | `to_global_embed`만 | **무효** (adaln 파라미터 부재) |
| `"adaLN"` | `to_global_embed` + `to_scale_shift_gate` + `global_cond_embedder` | 의도대로 작동 |

### 교훈 3: 제로 초기화는 레이어 선택적이어야 한다

```
나쁨:  W1 제로, W2 제로  ->  죽은 그래디언트, 네트워크가 절대 학습 불가
좋음:  W1 랜덤, W2 제로  ->  0 효과에서 시작하되, 그래디언트는 흐름
```

이것이 ControlNet 제로 컨볼루션 원칙이다.

### 교훈 4: CFG는 반드시 조건부와 비조건부를 구별해야 한다

```python
# 잘못됨: 두 패스 모두 같은 스코어 사용
batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
# CFG: cond - uncond = 스코어 성분에 대해 0!

# 올바름: 비조건부 패스는 제로 (null 스코어) 사용
batch_global_cond = torch.cat([global_embed, torch.zeros_like(global_embed)], dim=0)
# CFG: cond - uncond = 스코어 효과 (cfg_scale로 증폭됨)
```

### 교훈 5: 가능하면 검증된 경로를 사용하라

사전학습된 모델에 새 컨디셔닝을 추가할 때:
- 기존 임베딩 공간에 맞는 임베딩이 있다면 **cross-attention을 선호**하라
- adaLN은 레이어별 새 변조 파라미터를 처음부터 학습해야 한다
- 사전학습된 cross-attention은 이미 토큰에서 정보를 추출하는 방법을 알고 있다

### 교훈 6: NaN은 조기에 확인하라

```python
# 학습 로그에서 확인할 것:
train/loss=nan.0    # 즉시 NaN = 초기화 문제
                    # 보통 큰 값으로 인한 fp16 오버플로우
```

---

## 12. 부록: 텐서 형상 참조표

### A. 오디오 처리 파이프라인

```
원시 오디오:          (batch, 2, 524288)    # 2채널, ~11.9초 (44.1kHz)
VAE 인코딩 후:       (batch, 64, 256)      # 64 잠재 채널, 256 타임스텝
패치 임베딩 후:      (batch, 256, 1024)    # 256 토큰, 1024 임베딩 차원
트랜스포머 통과 후:  (batch, 256, 1024)
출력 프로젝션 후:    (batch, 256, 64)      # 잠재 공간으로 복원
잠재 벡터로 변환:    (batch, 64, 256)
VAE 디코딩 후:       (batch, 2, 524288)    # 오디오로 복원
```

### B. 컨디셔닝 형상

```
T5 텍스트 임베딩:    (batch, 64, 768)      # 64개 텍스트 토큰
재생시간 임베딩:     (batch, 768)          # 단일 스칼라 -> 768d
스코어 임베딩:       (batch, 768)          # 단일 스칼라 -> 768d

Cross-attn 결합:    (batch, 66, 768)      # 시퀀스 차원으로 연결
to_cond_embed 후:   (batch, 66, 1024)     # 모델 차원으로 프로젝션
```

### C. adaLN 형상 (참고용)

```
to_global_embed:     (batch, 768) -> (batch, 1024)
global_cond_embedder:(batch, 1024) -> (batch, 6144)
to_scale_shift_gate: (6144,) 블록당     # 총 16개 블록
6개로 분할:          6 x (batch, 1024)   # scale, shift, gate x2
```

### D. SAO-Small vs SAO-Original

| | SAO-Small | SAO-Original |
|---|---|---|
| DiT 깊이 | 16 | 24 |
| DiT embed_dim | 1024 | 1536 |
| DiT 파라미터 | ~340M | ~1.06B |
| 총 파라미터 | ~497M | ~1.2B |
| 샘플 크기 | 524,288 (~11.9초) | 2,097,152 (~47.5초) |
| 텍스트 인코더 | T5-base | T5-base |
| 잠재 차원 | 64 | 64 |
| 다운샘플링 | 2048배 | 2048배 |

---

## 용어 사전

- **adaLN**: 적응적 레이어 정규화. 샘플별로 LayerNorm 통계량을 변조.
- **CFG**: 분류기 없는 가이던스. 추론 시 컨디셔닝 신호를 증폭.
- **DiT**: 디퓨전 트랜스포머. 디퓨전 모델을 위한 트랜스포머 아키텍처.
- **FMA**: Free Music Archive. ~106K 음악 트랙의 오픈 데이터셋.
- **잠재 공간**: 오디오의 압축된 표현 (2048배 작음).
- **MERT**: Music Encoding and Representation with Transformers. 오디오 피처 추출기.
- **Music-RankNet**: 샴 RankNet을 사용하는 학습된 품질 평가 모델.
- **정류 흐름**: 노이즈->데이터의 직선 경로를 학습하는 학습 목표.
- **SAO**: Stable Audio Open. Stability AI의 오픈소스 텍스트-투-오디오 모델.
- **VAE**: 변분 오토인코더. 오디오를 잠재 공간으로/에서 압축/복원.

---

*이 문서는 ISMIR 2026을 위한 SAO 스코어 컨디셔닝 프로젝트의 일부입니다.*
