import json
import os
import sys
import time
from itertools import combinations

import laion_clap
import pytorch_lightning as pl
import torch
import torchaudio
import transformers.modeling_utils
import transformers.utils.import_utils
from transformers import AutoModel, Wav2Vec2FeatureExtractor

transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None


class RewardMonitorCallback(pl.Callback):
    def __init__(
        self,
        reward_model_path,
        clap_ckpt_path,
        thresholds_path,
        val_dl,
        num_samples=10,
        use_score_conditioning=False,
        music_ranknet_root=None,
        null_condition_value=-999.0,
        gen_steps=50,
        cfg_scale=3.5,
        save_audio=True,
    ):
        super().__init__()
        self.reward_model_path = reward_model_path
        self.clap_ckpt_path = clap_ckpt_path
        self.thresholds_path = thresholds_path
        self.val_dl = val_dl
        self.num_samples = num_samples
        self.use_score_conditioning = use_score_conditioning
        self.music_ranknet_root = music_ranknet_root
        self.null_condition_value = null_condition_value
        self.gen_steps = int(gen_steps)
        self.cfg_scale = float(cfg_scale)
        self.save_audio = bool(save_audio)

        self.reward_model = None
        self.mert_processor = None
        self.mert_model = None
        self.clap_model = None
        self.target_score_list = []

    def setup(self, trainer, pl_module, stage):
        device = pl_module.device
        print(f"[*] Setting up Reward Monitor on {device}...")

        self.target_score_list = self._load_threshold_targets()
        if not self.target_score_list:
            print(f"!!! FATAL: Thresholds file NOT FOUND at {self.thresholds_path}")
            self.use_score_conditioning = False

        if self.music_ranknet_root and self.music_ranknet_root not in sys.path:
            sys.path.append(self.music_ranknet_root)
        from models.music_ranknet import MusicRankNet

        self.reward_model = MusicRankNet(mode="RankNet", input_dim=2049)
        self.reward_model.load_state_dict(torch.load(self.reward_model_path, map_location=device))
        self.reward_model.eval().to(device).requires_grad_(False)

        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).eval().to(device).requires_grad_(False)

        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self._load_clap_checkpoint(device)
        self.clap_model.to(device).eval().requires_grad_(False)

    def _load_threshold_targets(self):
        if not os.path.exists(self.thresholds_path):
            return []
        with open(self.thresholds_path, "r") as f:
            thresholds_data = json.load(f)
        scores = []
        for i in range(10, 110, 10):
            scores.append(float(thresholds_data.get(f"top_{i}_percent", 0.0)))
        print(f"[*] Successfully loaded 10 Thresholds: {scores}")
        return scores

    def _load_clap_checkpoint(self, device):
        # Use load_ckpt() — same as 04_extract_fma_features.py
        # This initializes CLAP's internal preprocessing pipeline correctly.
        original_torch_load = torch.load

        def safe_load_wrapper(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        try:
            torch.load = safe_load_wrapper
            self.clap_model.load_ckpt(ckpt=self.clap_ckpt_path)
        finally:
            torch.load = original_torch_load

    def get_mert_embedding(self, waveform, sr, device):
        target_sr = 24000
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        if waveform.dim() == 3 and waveform.size(1) > 1:
            waveform = torch.mean(waveform, dim=1, keepdim=True)
        waveform = waveform.squeeze(1)

        inputs = self.mert_processor(waveform.cpu().numpy(), sampling_rate=target_sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.mert_model(**inputs, output_hidden_states=True)
        return outputs.last_hidden_state.mean(dim=1)

    def get_clap_audio_embedding(self, waveform, sr, save_dir=None, sample_idx=0):
        """Extract CLAP audio embedding via temp file — matches 04_extract pipeline."""
        target_sr = 48000
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)  # (B,C,T) → (C,T)
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Save to temp file and use get_audio_embedding_from_filelist (same as 04_extract)
        tmp_dir = save_dir or "/tmp"
        tmp_path = os.path.join(tmp_dir, f"_clap_tmp_{sample_idx}.wav")
        torchaudio.save(tmp_path, waveform.cpu(), target_sr)
        with torch.no_grad():
            embedding = self.clap_model.get_audio_embedding_from_filelist(x=[tmp_path])
        os.remove(tmp_path)
        return torch.from_numpy(embedding).float().to(waveform.device if waveform.is_cuda else "cpu")

    def get_clap_text_embedding(self, texts, device):
        """Extract CLAP text embedding — matches 04_extract pipeline."""
        if isinstance(texts, str):
            texts = [texts]
        with torch.no_grad():
            embedding = self.clap_model.get_text_embedding(texts)
        return torch.from_numpy(embedding).float().to(device)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.reward_model is None or self.val_dl is None:
            return
        if not trainer.is_global_zero:
            return

        device = pl_module.device
        pl_module.eval()
        val_start = time.perf_counter()

        target_scores_log, measured_scores_log = [], []
        bin_scores = {i: [] for i in range(10)}
        generated_count = 0
        error_count = 0
        audio_log_count = 0
        # Get sample_rate from model_config or fallback to model attribute
        model_config = getattr(pl_module, "model_config", {})
        model_sr = model_config.get("sample_rate", 44100)

        with torch.no_grad():
            for batch in self.val_dl:
                if generated_count >= self.num_samples:
                    break

                audio, metadata = batch
                current_batch_size = audio.shape[0]
                for i in range(current_batch_size):
                    if generated_count >= self.num_samples:
                        break

                    m = self._metadata_item(metadata, i)
                    p_val = m.get("prompt", "A music song.")
                    while isinstance(p_val, (list, tuple)) and len(p_val) > 0:
                        p_val = p_val[0]
                    clean_p = str(p_val)
                    clean_s = 10.0

                    # Cycle through all target scores (no baseline null samples)
                    if self.use_score_conditioning:
                        target_val = float(self.target_score_list[generated_count % 10])
                    else:
                        target_val = self.null_condition_value

                    single_cond_input = [{"prompt": clean_p, "seconds_total": clean_s, "continuous_score": target_val}]
                    negative_cond_input = [{"prompt": "", "seconds_total": clean_s, "continuous_score": self.null_condition_value}]

                    try:
                        gen_audio = pl_module.diffusion.generate(
                            conditioning=single_cond_input,
                            negative_conditioning=negative_cond_input,
                            steps=self.gen_steps,
                            cfg_scale=self.cfg_scale,
                            batch_size=1,
                            sample_size=int(model_sr * clean_s),
                        )

                        audio_to_save = torch.clamp(gen_audio[0].cpu().float(), -1.0, 1.0)
                        save_path = None
                        if self.save_audio:
                            save_dir = os.path.join(trainer.default_root_dir, "val_samples", f"epoch_{trainer.current_epoch}")
                            os.makedirs(save_dir, exist_ok=True)
                            safe_prompt = "".join(x for x in clean_p[:15] if x.isalnum() or x.isspace()).replace(" ", "")
                            save_path = os.path.join(save_dir, f"sample_{generated_count}_target_{target_val:.2f}_{safe_prompt}.wav")
                            torchaudio.save(save_path, audio_to_save, model_sr)

                        mert_emb = self.get_mert_embedding(gen_audio, model_sr, device)
                        clap_audio_emb = self.get_clap_audio_embedding(
                            gen_audio, model_sr,
                            save_dir=os.path.join(trainer.default_root_dir, "val_samples"),
                            sample_idx=generated_count,
                        )
                        clap_text_emb = self.get_clap_text_embedding(clean_p, device)
                        flag_tensor = torch.ones((1, 1), dtype=torch.float32, device=device)
                        concat_feat = torch.cat([flag_tensor, clap_audio_emb, mert_emb, clap_text_emb], dim=-1)
                        score = self.reward_model(concat_feat).item()

                        if target_val != self.null_condition_value:
                            target_scores_log.append(target_val)
                            measured_scores_log.append(score)
                            bin_scores[generated_count % 10].append(score)

                        if trainer.logger and isinstance(trainer.logger, pl.loggers.WandbLogger):
                            import wandb

                            caption = f"Prompt: {clean_p} | Target: {target_val:.2f} | Score: {score:.2f}"
                            if save_path and os.path.exists(save_path):
                                audio_obj = wandb.Audio(save_path, sample_rate=model_sr, caption=caption)
                            else:
                                audio_obj = wandb.Audio(audio_to_save.numpy().T, sample_rate=model_sr, caption=caption)

                            bin_idx = generated_count % 10
                            pct = (bin_idx + 1) * 10
                            key = f"val_audio/top{pct}pct_{target_val:.2f}"
                            trainer.logger.experiment.log(
                                {key: audio_obj},
                                commit=False,
                            )
                            audio_log_count += 1

                        print(f"[*] Generated Sample {generated_count} (Target: {target_val:.2f})")
                        generated_count += 1
                    except Exception as e:
                        print(f"!!! Error at sample {generated_count}: {e}")
                        error_count += 1
                        generated_count += 1

        if trainer.logger and isinstance(trainer.logger, pl.loggers.WandbLogger):
            # Ensure pending commit=False audio logs are flushed once per validation run.
            trainer.logger.experiment.log({}, commit=True)

        val_elapsed_sec = time.perf_counter() - val_start
        pl_module.log("val/reward_eval_time_sec", float(val_elapsed_sec), rank_zero_only=True)
        pl_module.log("val/reward_generated_count", int(generated_count), rank_zero_only=True)
        pl_module.log("val/reward_error_count", int(error_count), rank_zero_only=True)
        pl_module.log("val/reward_audio_log_count", int(audio_log_count), rank_zero_only=True)
        pl_module.log("val/reward_scored_count", int(len(target_scores_log)), rank_zero_only=True)
        print(
            f"[*] RewardMonitor summary: generated={generated_count}, "
            f"scored={len(target_scores_log)}, errors={error_count}, "
            f"audio_logged={audio_log_count}, time={val_elapsed_sec:.2f}s"
        )

        if self.use_score_conditioning and target_scores_log:
            stacked = torch.stack((torch.tensor(target_scores_log), torch.tensor(measured_scores_log)))
            corr_matrix = torch.corrcoef(stacked)
            correlation = corr_matrix[0, 1].item() if not torch.isnan(corr_matrix[0, 1]) else 0.0
            monotonicity = self._pairwise_monotonicity(target_scores_log, measured_scores_log)
            pl_module.log("val/reward_correlation", correlation, rank_zero_only=True)
            pl_module.log("val/score_monotonicity", monotonicity, rank_zero_only=True)
            print(f"\n[*] Epoch {trainer.current_epoch} - Correlation: {correlation:.4f}")
            print(f"[*] Epoch {trainer.current_epoch} - Monotonicity: {monotonicity:.4f}")
            print("-" * 50)

            for i in range(10):
                if len(bin_scores[i]) > 0:
                    avg_bin_score = sum(bin_scores[i]) / len(bin_scores[i])
                    target_score = self.target_score_list[i]
                    percentile = (i + 1) * 10
                    pl_module.log(f"val/measured_score_top_{percentile}_percent", avg_bin_score, rank_zero_only=True)
                    print(f"  - Target Top {percentile}% (Val: {target_score:.2f}) -> Measured Avg: {avg_bin_score:.4f}")
            print("-" * 50)

    @staticmethod
    def _metadata_item(metadata, idx):
        if isinstance(metadata, dict):
            item = {}
            for key, value in metadata.items():
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] > idx:
                    v = value[idx]
                    item[key] = v.item() if torch.is_tensor(v) and v.ndim == 0 else v
                elif isinstance(value, (list, tuple)) and len(value) > idx:
                    item[key] = value[idx]
                else:
                    item[key] = value
            return item
        if isinstance(metadata, (list, tuple)) and len(metadata) > idx:
            return metadata[idx]
        return metadata

    @staticmethod
    def _pairwise_monotonicity(targets, measured):
        """
        Returns the fraction of pairwise orderings that agree between targets and measured scores.
        Ties in targets are skipped.
        """
        if len(targets) < 2 or len(measured) < 2:
            return 0.0

        agree = 0
        total = 0
        for i, j in combinations(range(len(targets)), 2):
            dt = targets[i] - targets[j]
            if dt == 0:
                continue
            dm = measured[i] - measured[j]
            total += 1
            if dt * dm > 0:
                agree += 1

        return float(agree) / float(total) if total > 0 else 0.0
