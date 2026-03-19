# stable-audio-tools/finetune.py
"""Research fine-tuning entrypoint for continuous score conditioning."""

import torch
import json
import os
import pytorch_lightning as pl

import types
from prefigure.prefigure import get_all_args, push_wandb_config

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict
from stable_audio_tools.training import create_training_wrapper_from_config
from stable_audio_tools.training.reward_monitor import RewardMonitorCallback

NULL_CONDITION_VALUE = -999.0
DEFAULT_CFG_DROP_RATE = float(os.getenv("SA_CFG_DROP_RATE", "0.15"))
MUSIC_RANKNET_ROOT = os.getenv("MUSIC_RANKNET_ROOT", "/home/yonghyun/music-ranknet")
DEFAULT_UNFREEZE_PROFILE = os.getenv("SA_UNFREEZE_PROFILE", "hybrid")
UNFREEZE_PROFILES = {
    # All score-related routes: global_embed projection + adaLN (scale/shift/gate) + input_add adapter
    "hybrid": ["continuous_score", "score_bin", "to_global_embed", "to_scale_shift_gate", "global_cond_embedder", "input_add_adapter"],
    # adaLN only: scale/shift/gate per block + shared projection (requires global_cond_type=adaLN in config)
    "adaln": ["continuous_score", "score_bin", "to_global_embed", "to_scale_shift_gate", "global_cond_embedder"],
    # Channel-wise residual adapter only
    "adapter": ["continuous_score", "score_bin", "input_add_adapter"],
    # Prepend-mode global conditioning projection only
    "global": ["continuous_score", "score_bin", "to_global_embed"],
    # Only the conditioner heads (no backbone params)
    "minimal": ["continuous_score", "score_bin"],
    # Cross-attention: score as cross-attn token, unfreeze conditioner + cross-attn projection
    "xattn": ["continuous_score", "score_bin", "to_cond_embed"],
    # Dual pathway: cross-attention + input-concat, Fourier embedding
    "xattn_concat": ["continuous_score", "score_bin", "score_concat", "to_cond_embed", "preprocess_conv"],
    # LoRA on cross-attention projection: preserves pretrained text while adapting for score
    "lora_xattn": ["continuous_score", "score_bin", "cond_embed_lora"],
    # Separate score projection: dedicated to_score_embed for score token, to_cond_embed frozen for text/time
    "separate_score_proj": ["continuous_score", "score_bin", "to_score_embed"],
}


def ranknet_path(*parts: str) -> str:
    return os.path.join(MUSIC_RANKNET_ROOT, *parts)


class ContinuousScoreDatasetWrapper(torch.utils.data.Dataset):
    """
    Passes the raw, continuous reward score directly to the model.
    Applies CFG dropout, and robustly handles missing/corrupted FMA files to prevent DDP deadlocks.
    """
    def __init__(self, dataset, is_training=True, cfg_drop_rate=DEFAULT_CFG_DROP_RATE):
        self.dataset = dataset
        self.is_training = is_training
        self.cfg_drop_rate = cfg_drop_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            audio, metadata = self.dataset[idx]
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
        
        # Deep copy to prevent caching the null condition permanently
        metadata = metadata.copy()

        raw_score = metadata.get('reward_score', 0.0)
        if torch.is_tensor(raw_score): 
            raw_score = raw_score.item()
        elif isinstance(raw_score, list): 
            raw_score = raw_score[0]
        
        score_val = float(raw_score)
        
        # CFG dropout: replace condition with null token during training.
        if self.is_training and torch.rand(1).item() < self.cfg_drop_rate:
            score_val = NULL_CONDITION_VALUE
            
        metadata['continuous_score'] = score_val
        metadata['score_concat'] = score_val  # Same value for input-concat pathway
        return audio, metadata

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def load_json(path):
    with open(path) as f:
        return json.load(f)


def create_wrapped_dataloader(dataset_config, model_config, batch_size, num_workers, is_training):
    base_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        shuffle=is_training,
    )

    wrapped_set = ContinuousScoreDatasetWrapper(
        base_dl.dataset,
        is_training=is_training,
        cfg_drop_rate=DEFAULT_CFG_DROP_RATE if is_training else 0.0,
    )

    return torch.utils.data.DataLoader(
        wrapped_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=is_training,
        pin_memory=True,
        collate_fn=getattr(base_dl, "collate_fn", None),
    )


def resolve_trainable_name_keys():
    # Optional override for power users (comma-separated substrings)
    custom_keys = os.getenv("SA_TRAINABLE_NAME_KEYS", "").strip()
    if custom_keys:
        keys = [k.strip() for k in custom_keys.split(",") if k.strip()]
        if keys:
            return "custom", keys

    profile = DEFAULT_UNFREEZE_PROFILE.lower().strip()
    if profile not in UNFREEZE_PROFILES:
        print(f"[WARN] Unknown SA_UNFREEZE_PROFILE='{profile}', fallback to 'hybrid'")
        profile = "hybrid"
    return profile, UNFREEZE_PROFILES[profile]


def zero_init_new_params(model, pretrained_keys: set):
    """Zero-initialize NEW parameters following ControlNet zero-conv principle.

    Output layers are zero-init'd so new conditioning paths start with zero
    effect (preserving pretrained behavior).  Intermediate layers keep their
    random init so that gradients can flow through the chain.

    Without this distinction, consecutive zero-init'd layers create a dead
    gradient: dL/dW = dL/d(out) * input = nonzero * 0 = 0, and
    dL/d(input) = W^T * dL/d(out) = 0 * nonzero = 0.
    """
    # Intermediate layers: keep random init for gradient flow.
    # These feed INTO zero-init'd output layers, so their non-zero activations
    # let the output layer's weight gradient be non-zero.
    # Intermediate layers: small random init for gradient flow.
    # Output layers: zero-init so new paths start with zero effect.
    # This is the ControlNet zero-conv principle:
    #   intermediate (non-zero) → output (zero) → zero effect, but gradients flow.
    KEEP_RANDOM_PATTERNS = [
        "continuous_score",        # FourierScoreConditioner (Fourier + MLP)
        "score_concat",            # ScoreInputConcatConditioner (Fourier + MLP)
        "cond_embed_lora_A",       # LoRA down-projection (random init for gradient flow)
        "global_cond_embedder.0",  # FIRST linear of embedder (1024->1024), intermediate
        "to_score_embed.0",        # FIRST linear of score projection (intermediate, gradient flow)
        # cond_embed_lora_B is OUTPUT → zero-init'd (not listed here = zero by default)
        # global_cond_embedder.2 (1024->6144) is OUTPUT → zero-init'd (not listed here)
        # to_score_embed.2 is OUTPUT → zero-init'd (starts with zero effect)
    ]

    zero_count = 0
    keep_count = 0
    gate_count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in pretrained_keys:
                if "to_scale_shift_gate" in name:
                    # adaLN gate initialization for pretrained models.
                    # to_scale_shift_gate has 6*dim values: [scale_sa, shift_sa, gate_sa, scale_ff, shift_ff, gate_ff]
                    # scale=0 → (1+0)=1 (identity), shift=0 → no shift
                    # gate must be NEGATIVE so sigmoid(1 - gate) ≈ 1.0 (passthrough)
                    # gate=0 → sigmoid(1) = 0.73 → 27% signal loss per block → 0.73^16 ≈ 0.01
                    # gate=-10 → sigmoid(11) ≈ 1.0 → full passthrough ✓
                    dim = param.shape[0] // 6
                    param.zero_()
                    param[2*dim:3*dim] = -10.0  # gate_self
                    param[5*dim:6*dim] = -10.0  # gate_ff
                    gate_count += 1
                    print(f"  [GATE-INIT] {name} (scale=0, shift=0, gate=-10)")
                elif any(pat in name for pat in KEEP_RANDOM_PATTERNS):
                    # Scale down random init to avoid fp16 overflow while
                    # keeping non-zero values for gradient flow.
                    param.data.normal_(0, 0.02)
                    keep_count += 1
                    print(f"  [SMALL-INIT] {name} (shape={list(param.shape)})")
                else:
                    param.zero_()
                    zero_count += 1
                    print(f"  [ZERO-INIT] {name} (shape={list(param.shape)})")
    print(
        f"[*] Initialized {zero_count + keep_count + gate_count} new parameters: "
        f"{zero_count} zero-init, {keep_count} small-random-init, {gate_count} gate-init.\n"
    )


def unfreeze_finetune_params(model):
    profile, trainable_name_keys = resolve_trainable_name_keys()
    print(f"[*] Freezing backbone, unfreezing score-conditioning routes (profile={profile})...")
    model.requires_grad_(False)

    print(f"[*] Trainable name keys: {trainable_name_keys}")
    print("\n[*] --- List of unfrozen parameters ---")
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if any(key in name for key in trainable_name_keys):
            param.requires_grad_(True)
            unfrozen_count += 1
            print(f"  [UNFROZEN] {name} (shape={list(param.shape)})")
    print(f"[*] --- Total unfrozen tensors: {unfrozen_count} ---\n")


def attach_custom_optimizer(training_wrapper):
    def custom_configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found. Check freeze/unfreeze rules.")

        lr = float(os.getenv("SA_LR", "5e-5"))
        weight_decay = float(os.getenv("SA_WEIGHT_DECAY", "1e-3"))
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

        warmup_steps = int(os.getenv("SA_WARMUP_STEPS", "1000"))
        total_steps = int(os.getenv("SA_TOTAL_STEPS", "300000"))
        num_cycles = int(os.getenv("SA_NUM_CYCLES", "18"))
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=num_cycles,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    training_wrapper.configure_optimizers = types.MethodType(custom_configure_optimizers, training_wrapper)
    print("[*] Applied custom optimizer/scheduler for fine-tuning.")


def create_logger_and_checkpoint_dir(args, training_wrapper):
    logger = None
    checkpoint_dir = args.save_dir if args.save_dir else "checkpoints"

    if args.logger == "wandb":
        wandb_project = "music-steerability-study"
        run_name = args.name
        logger = pl.loggers.WandbLogger(project=wandb_project, name=run_name, group="Case3_FMA_Scored")
        logger.watch(training_wrapper)

        if args.save_dir:
            checkpoint_dir = os.path.join(args.save_dir, wandb_project, run_name, "checkpoints")

    return logger, checkpoint_dir


def create_reward_callback(val_dl, train_dl, use_score_conditioning):
    reward_ckpt_path = os.getenv(
        "REWARD_MODEL_CKPT",
        ranknet_path("checkpoints", "ultimate_train_all(brainmusic).pt"),
    )
    clap_ckpt_path = os.getenv(
        "CLAP_MODEL_CKPT",
        ranknet_path("checkpoints", "music_audioset_epoch_15_esc_90.14.pt"),
    )
    thresholds_path = os.getenv(
        "REWARD_THRESHOLDS_PATH",
        ranknet_path("data", "processed", "FMA_Scoring", "reward_thresholds.json"),
    )
    val_num_samples = int(os.getenv("SA_VAL_NUM_SAMPLES", "100"))
    val_gen_steps = int(os.getenv("SA_VAL_GEN_STEPS", "50"))
    val_cfg_scale = float(os.getenv("SA_VAL_CFG_SCALE", "3.5"))
    val_save_audio = os.getenv("SA_VAL_SAVE_AUDIO", "1").strip().lower() in ("1", "true", "yes")
    return RewardMonitorCallback(
        reward_model_path=reward_ckpt_path,
        clap_ckpt_path=clap_ckpt_path,
        thresholds_path=thresholds_path,
        val_dl=val_dl if val_dl else train_dl,
        num_samples=val_num_samples,
        use_score_conditioning=use_score_conditioning,
        music_ranknet_root=MUSIC_RANKNET_ROOT,
        null_condition_value=NULL_CONDITION_VALUE,
        gen_steps=val_gen_steps,
        cfg_scale=val_cfg_scale,
        save_audio=val_save_audio,
    )


def create_trainer(args, logger, callbacks):
    strategy = "ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto"
    val_args = {"check_val_every_n_epoch": None, "val_check_interval": args.val_every} if args.val_every > 0 else {}

    max_steps = int(os.getenv("SA_TOTAL_STEPS", "-1"))
    epoch_kwargs = {"max_steps": max_steps} if max_steps > 0 else {"max_epochs": 10000000}

    return pl.Trainer(
        devices="auto",
        accelerator="gpu",
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches,
        limit_val_batches=100,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=1,
        **epoch_kwargs,
        **val_args,
    )


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = get_all_args()

    seed = args.seed if args.seed is not None else 42
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))
    pl.seed_everything(seed, workers=True)

    model_config = load_json(args.model_config)
    dataset_config = load_json(args.dataset_config)
    has_score_cond = any(
        cond.get("id") in {"score_bin", "continuous_score"}
        for cond in model_config.get("model", {}).get("conditioning", {}).get("configs", [])
    )

    train_dl = create_wrapped_dataloader(
        dataset_config=dataset_config,
        model_config=model_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_training=True,
    )

    val_dl = None
    if args.val_dataset_config:
        val_dataset_config = load_json(args.val_dataset_config)
        val_dl = create_wrapped_dataloader(
            dataset_config=val_dataset_config,
            model_config=model_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_training=False,
        )

    model = create_model_from_config(model_config)
    if args.pretrained_ckpt_path:
        pretrained_keys = set(load_ckpt_state_dict(args.pretrained_ckpt_path).keys())
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))
        zero_init_new_params(model, pretrained_keys)

    unfreeze_finetune_params(model)
    training_wrapper = create_training_wrapper_from_config(model_config, model)
    attach_custom_optimizer(training_wrapper)

    save_top_k = int(os.getenv("SA_SAVE_TOP_K", "3"))
    logger, checkpoint_dir = create_logger_and_checkpoint_dir(args, training_wrapper)
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            every_n_train_steps=args.checkpoint_every,
            dirpath=checkpoint_dir,
            save_top_k=save_top_k,
            monitor="train/loss",
            mode="min",
        ),
        ExceptionCallback(),
        ModelConfigEmbedderCallback(model_config),
        create_reward_callback(val_dl, train_dl, has_score_cond),
    ]

    args_dict = vars(args)
    args_dict.update({"model_config": model_config, "dataset_config": dataset_config})
    if args.logger == "wandb":
        push_wandb_config(logger, args_dict)

    trainer = create_trainer(args, logger, callbacks)
    trainer.fit(training_wrapper, train_dl, val_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    main()


"""
PYTHONPATH=. python3 ./finetune.py \
  --dataset-config ./configs/dataset_fma_scored.json \
  --val-dataset-config ./configs/dataset_fma_scored.json \
  --model-config ./checkpoints/sao_small/model_config_with_score.json \
  --pretrained-ckpt-path ./checkpoints/sao_small/model.safetensors \
  --name "sao_small_case3_$(date +%Y%m%d_%H%M%S)" \
  --save-dir ./results/sao_small_case3 \
  --batch-size 2 \
  --accum-batches 4 \
  --precision 16-mixed \
  --checkpoint-every 1000 \
  --val-every 1000
"""