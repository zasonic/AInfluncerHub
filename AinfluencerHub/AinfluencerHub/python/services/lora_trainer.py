"""
services/lora_trainer.py -- Native LoRA training using diffusers + peft + accelerate.

Trains a LoRA adapter on the SDXL UNet using the user's dataset of
captioned images.  Replaces the previous ai-toolkit subprocess approach
with a fully native training loop.

Training produces .safetensors LoRA weights that are directly loadable
by the diffusion_pipeline for inference.

v2.1 training improvements:
  DoRA  -- Weight-Decomposed Low-Rank Adaptation (Liu et al., ICLR 2024).
            Decomposes LoRA updates into magnitude + direction components for
            better gradient flow and sharper face identity. Enabled when
            peft >= 0.9.0; falls back to standard LoRA silently.
  MinSNR -- Signal-to-noise ratio weighted loss (Hang et al. 2023 / CVPR 2024).
            Prevents high-noise timesteps from dominating training. Produces
            sharper detail and more consistent results without extra steps.
            gamma=5.0 (published optimal default). Falls back to standard MSE
            loss on any exception so training is never interrupted.

v2.2 training improvements:
  rsLoRA -- Rank-Stabilized LoRA (Kalajdzievski 2023, arXiv 2312.03732).
            Divides adapter outputs by sqrt(rank) instead of rank, preventing
            the effective learning rate from collapsing as rank increases.
            Combined with DoRA, this produces better face identity sharpness
            at higher ranks (32+). Enabled via use_rslora=True in LoraConfig
            (peft >= 0.9.0). Nested fallback: DoRA+rsLoRA -> DoRA -> standard.
"""

import logging
import shutil
import threading
from collections.abc import Callable
from pathlib import Path

from services.models import SDXL_BASE

log = logging.getLogger("hub.trainer")

BASE_MODEL_ID = SDXL_BASE.repo_id


def prepare_training_folder(
    dataset_dir: Path,
    captions_dir: Path,
) -> None:
    """
    Copy .txt captions from captions_dir into dataset_dir so that
    the training loop can find image + caption pairs in one folder.
    """
    for txt in captions_dir.glob("*.txt"):
        dest = dataset_dir / txt.name
        if not dest.exists():
            shutil.copy2(txt, dest)


def run_training(
    dataset_dir: Path,
    output_dir: Path,
    trigger_word: str = "",
    steps: int = 2000,
    rank: int = 16,
    learning_rate: float = 1e-4,
    hf_token: str = "",
    log_cb: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[bool, str]:
    """
    Train a LoRA adapter on the SDXL UNet using the user's image-caption
    dataset.

    Returns (success, message).
    """
    import os

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    def _emit(msg: str) -> None:
        log.info(msg)
        if log_cb:
            log_cb(msg)

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if device == "cuda" else torch.float32

    # -- Step 1: Collect image-caption pairs -----------------------------------

    _emit("Collecting dataset...")
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    pairs: list[tuple[Path, str]] = []
    for img_path in sorted(dataset_dir.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue
        txt_path = dataset_dir / f"{img_path.stem}.txt"
        caption = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
        if not caption and trigger_word:
            caption = trigger_word
        pairs.append((img_path, caption))

    if not pairs:
        return False, f"No image-caption pairs found in {dataset_dir}"

    _emit(f"Found {len(pairs)} image-caption pairs.")

    # -- Step 2: Load model components -----------------------------------------

    _emit("Loading SDXL model components (this may download ~6.5 GB on first run)...")

    try:
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        tokenizer_1 = CLIPTokenizer.from_pretrained(
            BASE_MODEL_ID, subfolder="tokenizer", token=hf_token or None,
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            BASE_MODEL_ID, subfolder="tokenizer_2", token=hf_token or None,
        )
        text_encoder_1 = CLIPTextModel.from_pretrained(
            BASE_MODEL_ID, subfolder="text_encoder",
            torch_dtype=weight_dtype, token=hf_token or None,
        ).to(device)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            BASE_MODEL_ID, subfolder="text_encoder_2",
            torch_dtype=weight_dtype, token=hf_token or None,
        ).to(device)
        vae = AutoencoderKL.from_pretrained(
            BASE_MODEL_ID, subfolder="vae",
            torch_dtype=weight_dtype, token=hf_token or None,
        ).to(device)
        unet = UNet2DConditionModel.from_pretrained(
            BASE_MODEL_ID, subfolder="unet",
            torch_dtype=weight_dtype, token=hf_token or None,
        ).to(device)
        noise_scheduler = DDPMScheduler.from_pretrained(
            BASE_MODEL_ID, subfolder="scheduler", token=hf_token or None,
        )
    except Exception as exc:
        return False, f"Failed to load model: {exc}"

    _emit("Model loaded. Configuring LoRA...")

    # Freeze base model
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # -- Step 3: Apply LoRA to UNet --------------------------------------------

    from peft import LoraConfig, get_peft_model

    _lora_base_kwargs = dict(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
    )

    # DoRA (Liu et al., ICLR 2024) + rsLoRA (Kalajdzievski 2023):
    # DoRA decomposes LoRA updates into magnitude + direction components,
    # improving gradient flow and face-identity sharpness.
    # rsLoRA divides adapter outputs by sqrt(rank) instead of rank, stabilizing
    # the effective learning rate as rank increases -- particularly beneficial
    # at rank 16+ where standard LoRA's alpha/rank scaling becomes too small.
    # Nested fallback ensures training never fails due to peft version:
    #   DoRA + rsLoRA  (peft >= 0.9.0 with use_rslora support)
    #   DoRA only      (peft >= 0.9.0 without use_rslora)
    #   Standard LoRA  (peft < 0.9.0)
    try:
        lora_config = LoraConfig(**_lora_base_kwargs, use_dora=True, use_rslora=True)
        _emit("LoRA adapter: DoRA + rsLoRA enabled -- best face identity (peft>=0.9.0)")
    except TypeError:
        try:
            lora_config = LoraConfig(**_lora_base_kwargs, use_dora=True)
            _emit("LoRA adapter: DoRA enabled -- improved identity consistency (upgrade peft for rsLoRA too)")
        except TypeError:
            lora_config = LoraConfig(**_lora_base_kwargs)
            _emit("LoRA adapter: standard LoRA (upgrade peft>=0.9.0 for DoRA+rsLoRA)")

    unet = get_peft_model(unet, lora_config)
    unet.train()

    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    _emit(f"LoRA: {trainable_params:,} trainable params / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")

    # Enable gradient checkpointing
    unet.enable_gradient_checkpointing()

    # -- Step 4: Build dataset & dataloader ------------------------------------

    class CaptionImageDataset(Dataset):
        def __init__(self, pairs, tokenizer_1, tokenizer_2, resolution=1024):
            self.pairs = pairs
            self.tokenizer_1 = tokenizer_1
            self.tokenizer_2 = tokenizer_2
            self.transform = transforms.Compose([
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            img_path, caption = self.pairs[idx]
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(image)

            tokens_1 = self.tokenizer_1(
                caption, max_length=77, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            tokens_2 = self.tokenizer_2(
                caption, max_length=77, padding="max_length",
                truncation=True, return_tensors="pt",
            )

            return {
                "pixel_values": pixel_values,
                "input_ids_1": tokens_1.input_ids.squeeze(0),
                "input_ids_2": tokens_2.input_ids.squeeze(0),
            }

    dataset = CaptionImageDataset(pairs, tokenizer_1, tokenizer_2)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True,
    )

    # -- Step 5: Optimizer & scheduler -----------------------------------------

    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(
            unet.parameters(),
            lr=learning_rate,
            weight_decay=1e-2,
        )
        _emit("Using AdamW8bit optimizer.")
    except ImportError:
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=learning_rate,
            weight_decay=1e-2,
        )
        _emit("Using standard AdamW optimizer (install bitsandbytes for 8-bit).")

    # Cosine LR scheduler with warmup
    warmup_steps = max(50, steps // 20)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=steps - warmup_steps, eta_min=learning_rate * 0.1)

    # -- Step 6: Training loop -------------------------------------------------

    _emit(f"Starting training: {steps} steps, rank {rank}, lr {learning_rate}")

    # Pre-compute alphas_cumprod for MinSNR-gamma weighting (done once, not per step)
    _alphas_cumprod_dev = noise_scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
    _snr_gamma = 5.0  # published optimal default (Hang et al. 2023)
    _emit(f"Loss: MinSNR-gamma weighted (gamma={_snr_gamma}) -- balances timestep contributions")

    gradient_accumulation_steps = 4
    global_step = 0
    running_loss = 0.0
    save_every = max(250, steps // 8)

    data_iter = iter(dataloader)

    while global_step < steps:
        if cancel_event and cancel_event.is_set():
            _emit("Training cancelled by user.")
            _cleanup(unet, vae, text_encoder_1, text_encoder_2)
            return False, "Training cancelled by user."

        # Get next batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
        input_ids_1 = batch["input_ids_1"].to(device)
        input_ids_2 = batch["input_ids_2"].to(device)

        # Encode images to latents
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        # Encode text
        with torch.no_grad():
            encoder_output_1 = text_encoder_1(input_ids_1, output_hidden_states=True)
            encoder_output_2 = text_encoder_2(input_ids_2, output_hidden_states=True)
            # SDXL uses penultimate hidden states
            text_embeds_1 = encoder_output_1.hidden_states[-2]
            text_embeds_2 = encoder_output_2.hidden_states[-2]
            prompt_embeds = torch.cat([text_embeds_1, text_embeds_2], dim=-1)
            # Pooled output from text_encoder_2
            pooled_prompt_embeds = encoder_output_2[0]

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device,
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Time IDs for SDXL (original_size + crop_coords + target_size)
        add_time_ids = torch.tensor(
            [[1024, 1024, 0, 0, 1024, 1024]], dtype=weight_dtype, device=device,
        )
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }

        # Forward pass
        model_pred = unet(
            noisy_latents, timesteps, prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # MinSNR-gamma weighted loss (Hang et al. 2023, gamma=5.0).
        # SNR(t) = alpha_t^2 / sigma_t^2 = alphas_cumprod[t] / (1 - alphas_cumprod[t]).
        # Per-step weight = min(SNR, gamma) / SNR, clipping the loss contribution of
        # high-noise (low-SNR) timesteps that would otherwise dominate training.
        # Falls back to plain MSE on any error so training is never interrupted.
        try:
            _snr = (
                _alphas_cumprod_dev[timesteps]
                / (1.0 - _alphas_cumprod_dev[timesteps]).clamp(min=1e-6)
            ).clamp(min=1e-6)
            _weights = (
                torch.stack([_snr, _snr_gamma * torch.ones_like(_snr)], dim=1)
                .min(dim=1)[0]
                / _snr
            )
            loss = (
                F.mse_loss(model_pred.float(), noise.float(), reduction="none")
                .mean([1, 2, 3]) * _weights
            ).mean()
        except Exception:
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        loss = loss / gradient_accumulation_steps
        loss.backward()

        running_loss += loss.item() * gradient_accumulation_steps
        global_step += 1

        # Gradient accumulation step
        if global_step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            if global_step > warmup_steps:
                scheduler.step()
            optimizer.zero_grad()

        # Warmup (linear)
        if global_step <= warmup_steps:
            warmup_lr = learning_rate * (global_step / warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # Logging
        if global_step % 10 == 0 or global_step == 1:
            avg_loss = running_loss / 10 if global_step > 10 else running_loss
            running_loss = 0.0
            current_lr = optimizer.param_groups[0]["lr"]
            _emit(f"step: {global_step}/{steps}  loss: {avg_loss:.4f}  lr: {current_lr:.2e}")

        # Save checkpoint
        if global_step % save_every == 0 and global_step < steps:
            ckpt_path = output_dir / f"checkpoint-{global_step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            unet.save_pretrained(ckpt_path)
            _emit(f"Checkpoint saved: {ckpt_path}")

    # -- Step 7: Save final LoRA -----------------------------------------------

    _emit("Saving final LoRA weights...")
    final_path = output_dir / f"lora_rank{rank}_steps{steps}"
    final_path.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(final_path)

    # Also save as a single safetensors file for easy loading
    try:
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file

        state_dict = get_peft_model_state_dict(unet)
        safetensors_path = output_dir / f"lora_rank{rank}_steps{steps}.safetensors"
        save_file(state_dict, str(safetensors_path))
        _emit(f"LoRA saved: {safetensors_path}")
    except Exception as exc:
        _emit(f"Warning: Could not save single safetensors file: {exc}")
        safetensors_path = final_path

    _cleanup(unet, vae, text_encoder_1, text_encoder_2)
    return True, str(safetensors_path if safetensors_path.exists() else final_path)


def _cleanup(*models) -> None:
    """Free GPU memory."""
    import torch

    for model in models:
        try:
            del model
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Training models unloaded.")
