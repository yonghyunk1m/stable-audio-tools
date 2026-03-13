#!/usr/bin/env python3
"""
Score Conditioning 효과 검증 스크립트.

같은 프롬프트에 대해 score를 [0, 2, 5, 8, 10]으로 변화시키며 오디오를 생성하고,
생성 결과를 Music-RankNet으로 평가하여 score↔quality 상관관계를 측정합니다.

Usage:
    CUDA_VISIBLE_DEVICES=8 python scripts/eval_score_conditioning.py \
        --ckpt-path results/sao_small_case3/adaln/epoch=0-step=50000.ckpt \
        --output-dir results/eval_score_cond/

    또는 체크포인트 없이 현재 학습 중인 모델의 중간 체크포인트로:
    CUDA_VISIBLE_DEVICES=8 python scripts/eval_score_conditioning.py \
        --ckpt-path results/sao_small_case3/adaln/last.ckpt \
        --output-dir results/eval_score_cond/
"""

import argparse
import json
import os
import sys
import torch
import torchaudio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond


TEST_PROMPTS = [
    "A beautiful piano melody in C major",
    "Energetic drum breaks 140 BPM",
    "Ambient synth pad with reverb",
    "Acoustic guitar fingerpicking folk song",
    "Heavy bass drop dubstep wobble",
]

SCORE_VALUES = [0.0, 2.0, 5.0, 8.0, 10.0]


def load_model(ckpt_path, device="cuda"):
    """체크포인트에서 모델 로드."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # PL checkpoint에서 model_config 추출
    if "model_config" in ckpt:
        model_config = ckpt["model_config"]
    else:
        # fallback: 기본 config 사용
        config_path = os.path.join(os.path.dirname(ckpt_path), "model_config.json")
        if not os.path.exists(config_path):
            config_path = "checkpoints/sao_small/model_config_with_score_adaln.json"
        model_config = json.load(open(config_path))

    model = create_model_from_config(model_config)

    # state_dict 추출 (PL wrapper에서)
    if "state_dict" in ckpt:
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            # "diffusion_ema." 또는 "diffusion." prefix 제거
            if k.startswith("diffusion_ema."):
                new_k = k[len("diffusion_ema."):]
            elif k.startswith("diffusion."):
                new_k = k[len("diffusion."):]
            else:
                continue
            state_dict[new_k] = v
        copy_state_dict(model, state_dict)
    else:
        copy_state_dict(model, ckpt)

    model = model.to(device).eval()
    return model, model_config


def generate_with_score(model, model_config, prompt, score, device="cuda",
                        steps=50, cfg_scale=3.5, seconds_total=10.0):
    """주어진 prompt와 score로 오디오 생성."""
    conditioning = [{
        "prompt": prompt,
        "seconds_total": seconds_total,
        "continuous_score": score,
    }]

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    with torch.no_grad():
        output = generate_diffusion_cond(
            model,
            conditioning=conditioning,
            steps=steps,
            cfg_scale=cfg_scale,
            sample_size=sample_size,
            sample_rate=sample_rate,
            device=device,
        )

    # output: (1, channels, samples)
    return output.cpu(), sample_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True, help="체크포인트 경로")
    parser.add_argument("--output-dir", default="results/eval_score_cond/", help="출력 디렉토리")
    parser.add_argument("--steps", type=int, default=50, help="생성 스텝 수")
    parser.add_argument("--cfg-scale", type=float, default=3.5, help="CFG scale")
    parser.add_argument("--device", default="cuda", help="디바이스")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[*] Loading model from {args.ckpt_path}")
    model, model_config = load_model(args.ckpt_path, args.device)
    print(f"[*] Model loaded. Sample rate: {model_config['sample_rate']}")

    results = []

    for prompt_idx, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'='*60}")
        print(f"Prompt {prompt_idx+1}/{len(TEST_PROMPTS)}: {prompt}")
        print(f"{'='*60}")

        for score in SCORE_VALUES:
            print(f"  Score={score:.1f} ... ", end="", flush=True)

            audio, sr = generate_with_score(
                model, model_config, prompt, score,
                device=args.device, steps=args.steps, cfg_scale=args.cfg_scale,
            )

            # 파일 저장
            fname = f"p{prompt_idx}_score{score:.0f}.wav"
            fpath = os.path.join(args.output_dir, fname)
            torchaudio.save(fpath, audio.squeeze(0), sr)
            print(f"saved → {fname}")

            results.append({
                "prompt": prompt,
                "input_score": score,
                "file": fname,
            })

    # 결과 요약 저장
    summary_path = os.path.join(args.output_dir, "generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"생성 완료! 총 {len(results)}개 오디오")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"요약: {summary_path}")
    print(f"\n다음 단계:")
    print(f"  1. 오디오를 직접 들어보세요 — score=0 vs score=10 차이가 나는지")
    print(f"  2. Music-RankNet으로 품질 점수를 매겨 상관관계를 확인하세요")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
