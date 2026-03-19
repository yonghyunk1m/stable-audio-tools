# Config Files

## Active (Jamendo)

| File | Description |
|------|-------------|
| `dataset_jamendo_scored.json` | Dataset config pointing to Jamendo audio + scored metadata |
| `metadata_jamendo_scored_new.jsonl` | **Current**: 54,330 tracks, new 4-layer text-aware reward model, Qwen captions |
| `metadata_jamendo_scored.jsonl` | Legacy: old 5-layer reward model, original prompts |

## Legacy (FMA — audio deleted, configs kept for reference)

| File | Description |
|------|-------------|
| `dataset_fma_scored.json` | FMA dataset config (audio deleted, non-functional) |
| `metadata_fma_scored.jsonl` | FMA scores, old reward model |
| `metadata_fma_scored_fixed.jsonl` | FMA scores, fixed version |
| `metadata_fma_scored_top50.jsonl` | Top 50% filtered |
| `metadata_fma_scored_top30.jsonl` | Top 30% filtered |
| `metadata_fma_all.jsonl` | All FMA without scores |
| `metadata_fma_filtered.jsonl` | Quality-filtered FMA |

## Metadata JSONL format

```json
{"relpath": "/path/to/audio.mp3", "prompt": "A reggae track with...", "reward_score": 1.234}
```
