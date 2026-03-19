"""
Pilot: Compare audio-based vs tag-based captioning with Qwen2-Audio.

1. Audio → Caption: Feed actual FMA audio to Qwen2-Audio → grounded caption
2. Tags → Caption: Feed genre tags to Qwen2-Audio → natural language caption (ICME style)

Run: CUDA_VISIBLE_DEVICES=4 python scripts/pilot_caption_comparison.py
"""
import os, json, glob, random, torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import librosa

# --- Config ---
NUM_SAMPLES = 10
FMA_ROOT = "/home/yonghyun/fma/data/fma_large"
METADATA_PATH = "configs/metadata_fma_scored.jsonl"
DEVICE = "cuda:0"
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

# --- Load metadata ---
print("Loading metadata...")
meta_list = []
with open(METADATA_PATH) as f:
    for line in f:
        d = json.loads(line)
        if d.get("prompt") and d.get("file"):
            meta_list.append(d)

# Sample diverse tracks
random.seed(42)
samples = random.sample(meta_list, min(NUM_SAMPLES, len(meta_list)))

# --- Load model ---
print(f"Loading {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True
)
print("Model loaded.\n")

results = []

for i, sample in enumerate(samples):
    file_path = os.path.join(FMA_ROOT, sample["file"])
    tags = sample.get("prompt", "A music song.")
    score = sample.get("reward_score", 0.0)

    print(f"[{i+1}/{NUM_SAMPLES}] {sample['file']}")
    print(f"  Tags: {tags}")
    print(f"  Score: {score:.3f}")

    result = {"file": sample["file"], "tags": tags, "score": score}

    # --- Method 1: Audio → Caption ---
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=10.0)

        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": "Describe this music in detail. Include genre, instruments, mood, tempo, and production quality."}
            ]}
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio], sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=150)

        audio_caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        # Extract only the assistant's response
        if "assistant" in audio_caption.lower():
            audio_caption = audio_caption.split("assistant")[-1].strip()

        result["audio_caption"] = audio_caption
        print(f"  Audio→Caption: {audio_caption[:100]}...")
    except Exception as e:
        result["audio_caption"] = f"ERROR: {e}"
        print(f"  Audio→Caption: ERROR: {e}")

    # --- Method 2: Tags → Caption (ICME style) ---
    try:
        # Extract tag components
        tag_text = tags.replace("A ", "").replace(" song.", "").strip()

        conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": f"Given these musical tags: [{tag_text}], write a natural, descriptive music caption in one sentence. The caption should describe the musical qualities, mood, and characteristics implied by these tags. Write only the caption, nothing else."}
            ]}
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=100)

        tag_caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if "assistant" in tag_caption.lower():
            tag_caption = tag_caption.split("assistant")[-1].strip()

        result["tag_caption"] = tag_caption
        print(f"  Tags→Caption:  {tag_caption[:100]}...")
    except Exception as e:
        result["tag_caption"] = f"ERROR: {e}"
        print(f"  Tags→Caption:  ERROR: {e}")

    results.append(result)
    print()

# --- Save results ---
output_path = "logs/pilot_caption_comparison.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"Results saved to {output_path}")
print(f"{'='*60}")

# --- Summary ---
print("\n=== COMPARISON SUMMARY ===\n")
for r in results:
    print(f"File: {r['file']}")
    print(f"  Original tags:   {r['tags']}")
    print(f"  Audio→Caption:   {r.get('audio_caption', 'N/A')[:120]}")
    print(f"  Tags→Caption:    {r.get('tag_caption', 'N/A')[:120]}")
    print()
