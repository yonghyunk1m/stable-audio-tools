"""Quick evaluation: test different CFG scales on a checkpoint."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict
from stable_audio_tools.training.reward_monitor import RewardMonitorCallback

CKPT = sys.argv[1] if len(sys.argv) > 1 else "results/sao_small_case3/xattn/music-steerability-study/case3b_fourier_xattn_v1/checkpoints/epoch=9-step=65000.ckpt"
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/sao_small/model_config_with_score_xattn.json"
CFG_SCALES = [3.5, 5.0, 7.0]
NUM_SAMPLES = 100
DEVICE = f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '4').split(',')[0]}"

print(f"Loading config: {CONFIG}")
with open(CONFIG) as f:
    model_config = json.load(f)

print(f"Creating model...")
model = create_model_from_config(model_config)

print(f"Loading checkpoint: {CKPT}")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
state_dict = ckpt.get("state_dict", ckpt)
# Remove "diffusion." prefix from training wrapper
clean_sd = {}
for k, v in state_dict.items():
    clean_k = k.replace("diffusion.", "", 1) if k.startswith("diffusion.") else k
    clean_sd[clean_k] = v
copy_state_dict(model, clean_sd)
model = model.to(DEVICE).eval()

print(f"Device: {DEVICE}")
print(f"Samples per cfg: {NUM_SAMPLES}")

for cfg in CFG_SCALES:
    print(f"\n{'='*50}")
    print(f"  CFG Scale = {cfg}")
    print(f"{'='*50}")
    
    monitor = RewardMonitorCallback(
        model_config=model_config,
        num_samples=NUM_SAMPLES,
        gen_steps=50,
        cfg_scale=cfg,
        save_audio=False,
    )
    monitor.setup_on_device(DEVICE)
    
    results = monitor.evaluate(model, device=DEVICE)
    
    corr = results.get("reward_correlation", "?")
    mono = results.get("score_monotonicity", "?")
    print(f"\n  Correlation:  {corr:.4f}")
    print(f"  Monotonicity: {mono:.4f}")
    print(f"  Per-bin measured scores:")
    for i in range(10, 110, 10):
        k = f"measured_score_top_{i}_percent"
        v = results.get(k)
        if v is not None:
            print(f"    Top {i:3d}%: {v:+.4f}")

print("\nDone!")
