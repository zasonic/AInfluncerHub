"""
services/lora_trainer.py — Native LoRA training using diffusers + peft + accelerate.

Trains a LoRA adapter on the SDXL UNet using the user's dataset of
captioned images.  Replaces the previous ai-toolkit subprocess approach
with a fully native training loop.

Training produces .safetensors LoRA weights that are directly loadable
by the diffusion_pipeline for inference.

v2.1 training improvements:
  DoRA  — Weight-Decomposed Low-Rank Adaptation (Liu et al., ICLR 2024).
            Decomposes LoRA updates into magnitude + direction components for
            better gradient flow and sharper face identity. Enabled when
            peft >= 0.9.0; falls back to standard LoRA silently.
  MinSNR — Signal-to-noise ratio weighted loss (Hang et al. 2023 / CVPR 2024).
            Prevents high-noise timesteps from dominating training. Produces
            sharper detail and more consistent results without extra steps.
            gamma=5.0 (published optimal default). Falls back to standard MSE
            loss on any exception so training is never interrupted.

v2.2 FLUX training improvements:
  Logit-normal timesteps — Esser et al. 2024 (SD3/FLUX paper). Biases
            sampling towards intermediate timesteps, which carry the most
            information for rectified flow / flow matching models. Falls back
            to uniform sampling on any error.
  Velocity target — FLUX uses FlowMatchEulerDiscreteScheduler (rectified
            flow), so the correct training objective is to predict the
            velocity field (noise - latents), not the epsilon noise.
  DoRA —    Weight-Decomposed Low-Rank Adaptation now applied to FLUX
            Transformer path as well (same peft>=0.9.0 fallback pattern as
            SDXL). Improves face-identity sharpness in FLUX-trained adapters.
"""

import logging
import shutil
import threading
from collections.abc import Callable
from pathlib import Path

from services.models import FLUX_BASE, SDXL_BASE

log = logging.getLogger("hub.trainer")

BASE_MODEL_ID = SDXL_BASE.repo_id
FLUX_MODEL_ID = FLUX_BASE.repo_id


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
    base_model: str = "sdxl",
    hf_token: str = "",
    log_cb: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[bool, str]:
    """
    Train a LoRA adapter on the SDXL UNet (default) or FLUX Transformer.
    Dispatches to the appropriate training path based on base_model.

    Returns (success, message).
    """
    if base_model == "flux":
        return _run_flux_training(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            trigger_word=trigger_word,
            steps=steps,
            rank=rank,
            learning_rate=learning_rate,
            hf_token=hf_token,
            log_cb=log_cb,
            cancel_event=cancel_event,
        )

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

    # ── Step 1: Collect image-caption pairs ──────────────────────────────────────────────

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

    # ── Step 2: Load model components ──────────────────────────────────────────────

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

    # ── Step 3: Apply LoRA to UNet ────────────────────────────────────────────────

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

    # DoRA (Liu et al., ICLR 2024): decomposes LoRA updates into magnitude +
    # direction components, improving gradient flow and face-identity sharpness.
    # Requires peft >= 0.9.0; older installs fall back to standard LoRA.
    try:
        lora_config = LoraConfig(**_lora_base_kwargs, use_dora=True)
        _emit("LoRA adapter: DoRA enabled (Weight-Decomposed, peft>=0.9.0) — improved identity consistency")
    except TypeError:
        lora_config = LoraConfig(**_lora_base_kwargs)
        _emit("LoRA adapter: standard LoRA (upgrade peft>=0.9.0 to enable DoRA for better results)")

    unet = get_peft_model(unet, lora_config)
    unet.train()

    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    _emit(f"LoRA: {trainable_params:,} trainable params / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")

    # Enable gradient checkpointing
    unet.enable_gradient_checkpointing()

    # ── Step 4: Build dataset & dataloader ───────────────────────────────────────────

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

    # ── Step 5: Optimizer & scheduler ────────────────────────────────────────────

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

    # ── Step 6: Training loop ─────────────────────────────────────────────────────────

    _emit(f"Starting training: {steps} steps, rank {rank}, lr {learning_rate}")

    # Pre-compute alphas_cumprod for MinSNR-gamma weighting (done once, not per step)
    _alphas_cumprod_dev = noise_scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
    _snr_gamma = 5.0  # published optimal default (Hang et al. 2023)
    _emit(f"Loss: MinSNR-γ weighted (gamma={_snr_gamma}) — balances timestep contributions")

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

    # ── Step 7: Save final LoRA ─────────────────────────────────────────────────────────────

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


def _run_flux_training(
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
    Train a LoRA adapter on the FLUX.1-dev Transformer (DiT architecture).

    FLUX uses a different model structure than SDXL:
      - FluxTransformer2DModel instead of UNet2DConditionModel
      - T5-XXL + CLIP-L dual encoders (vs SDXL's CLIP-L + CLIP-G)
      - T5-XXL is ~10 GB — loaded in 8-bit to fit within 16 GB VRAM
      - Different attention projection names for LoRA target_modules
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
    weight_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # ── Collect dataset ──────────────────────────────────────────────────────────────

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

    # ── Load model components ────────────────────────────────────────────────────────────

    _emit("Loading FLUX.1-dev components (first run downloads ~23 GB)...")

    try:
        from diffusers import FluxTransformer2DModel
        from transformers import (
            BitsAndBytesConfig,
            CLIPTextModel,
            CLIPTokenizer,
            T5EncoderModel,
            T5TokenizerFast,
        )

        tokenizer_1 = CLIPTokenizer.from_pretrained(
            FLUX_MODEL_ID, subfolder="tokenizer", token=hf_token or None,
        )
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            FLUX_MODEL_ID, subfolder="tokenizer_2", token=hf_token or None,
        )
        text_encoder_1 = CLIPTextModel.from_pretrained(
            FLUX_MODEL_ID, subfolder="text_encoder",
            torch_dtype=weight_dtype, token=hf_token or None,
        ).to(device)

        # T5-XXL is ~10 GB; load in 8-bit so it fits in 16 GB alongside the transformer
        t5_quant = BitsAndBytesConfig(load_in_8bit=True) if device == "cuda" else None
        text_encoder_2 = T5EncoderModel.from_pretrained(
            FLUX_MODEL_ID, subfolder="text_encoder_2",
            quantization_config=t5_quant,
            torch_dtype=weight_dtype if t5_quant is None else None,
            token=hf_token or None,
        )
        if t5_quant is None:
            text_encoder_2 = text_encoder_2.to(device)

        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
        vae = AutoencoderKL.from_pretrained(
            FLUX_MODEL_ID, subfolder="vae",
            torch_dtype=weight_dtype, token=hf_token or None,
        ).to(device)
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            FLUX_MODEL_ID, subfolder="scheduler", token=hf_token or None,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            FLUX_MODEL_ID, subfolder="transformer",
            torch_dtype=weight_dtype, token=hf_token or None,
        ).to(device)

    except Exception as exc:
        return False, f"Failed to load FLUX model: {exc}"

    _emit("FLUX model loaded. Configuring LoRA...")

    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    # ── Apply LoRA to FLUX transformer ─────────────────────────────────────────────

    from peft import LoraConfig, get_peft_model

    _flux_lora_base_kwargs = dict(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out",
            "add_k_proj", "add_q_proj", "add_v_proj", "add_out_proj",
        ],
    )

    # DoRA (Liu et al., ICLR 2024): decomposes LoRA updates into magnitude +
    # direction components, improving gradient flow and face-identity sharpness.
    # Requires peft >= 0.9.0; older installs fall back to standard LoRA.
    try:
        lora_config = LoraConfig(**_flux_lora_base_kwargs, use_dora=True)
        _emit("LoRA adapter: DoRA enabled (Weight-Decomposed, peft>=0.9.0) — improved identity consistency")
    except TypeError:
        lora_config = LoraConfig(**_flux_lora_base_kwargs)
        _emit("LoRA adapter: standard LoRA (upgrade peft>=0.9.0 to enable DoRA for better results)")

    transformer = get_peft_model(transformer, lora_config)
    transformer.train()

    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in transformer.parameters())
    _emit(
        f"LoRA: {trainable_params:,} trainable params / {total_params:,} total "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    transformer.enable_gradient_checkpointing()

    # ── Dataset ───────────────────────────────────────────────────────────────────

    class FluxDataset(Dataset):
        def __init__(self, pairs: list[tuple[Path, str]], resolution: int = 1024):
            self.pairs = pairs
            self.transform = transforms.Compose([
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int) -> dict:
            img_path, caption = self.pairs[idx]
            pixel_values = self.transform(Image.open(img_path).convert("RGB"))
            clip_tokens = tokenizer_1(
                caption, max_length=77, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            t5_tokens = tokenizer_2(
                caption, max_length=512, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            return {
                "pixel_values":  pixel_values,
                "clip_input_ids": clip_tokens.input_ids.squeeze(0),
                "t5_input_ids":   t5_tokens.input_ids.squeeze(0),
            }

    dataloader = DataLoader(
        FluxDataset(pairs), batch_size=1, shuffle=True, num_workers=0,
    )

    # ── Optimizer ───────────────────────────────────────────────────────────────────

    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(transformer.parameters(), lr=learning_rate, weight_decay=1e-2)
        _emit("Using AdamW8bit optimizer.")
    except ImportError:
        optimizer = torch.optim.AdamW(
            transformer.parameters(), lr=learning_rate, weight_decay=1e-2,
        )
        _emit("Using standard AdamW optimizer.")

    warmup_steps = max(50, steps // 20)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=steps - warmup_steps, eta_min=learning_rate * 0.1)

    # ── Training loop ─────────────────────────────────────────────────────────────────

    _emit(f"Starting FLUX LoRA training: {steps} steps, rank {rank}, lr {learning_rate}")
    _emit(
        "Timestep sampling: logit-normal (Esser et al. 2024) — "
        "biases towards informative intermediate timesteps for flow matching"
    )
    _emit("Loss target: velocity field (noise − latents) — correct objective for rectified flow")

    gradient_accumulation_steps = 4
    global_step = 0
    running_loss = 0.0
    save_every = max(250, steps // 8)
    data_iter = iter(dataloader)

    while global_step < steps:
        if cancel_event and cancel_event.is_set():
            _emit("Training cancelled by user.")
            _cleanup(transformer, vae, text_encoder_1, text_encoder_2)
            return False, "Training cancelled by user."

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

            clip_embeds = text_encoder_1(
                batch["clip_input_ids"].to(device), output_hidden_states=False,
            ).pooler_output
            t5_embeds = text_encoder_2(
                batch["t5_input_ids"].to(device), output_hidden_states=False,
            ).last_hidden_state

        noise = torch.randn_like(latents)

        # Logit-normal timestep sampling (Esser et al. 2024, SD3/FLUX paper).
        try:
            u = torch.randn(latents.shape[0])
            t_frac = torch.sigmoid(u).to(device)
            timesteps = (
                t_frac * noise_scheduler.config.num_train_timesteps
            ).long().clamp(0, noise_scheduler.config.num_train_timesteps - 1)
        except Exception:
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device,
            ).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        model_pred = transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=t5_embeds,
            pooled_projections=clip_embeds,
        ).sample

        # Velocity target for flow matching: model predicts (noise - latents).
        target = noise - latents
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss = loss / gradient_accumulation_steps
        loss.backward()

        running_loss += loss.item() * gradient_accumulation_steps
        global_step += 1

        if global_step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()
            if global_step > warmup_steps:
                lr_scheduler.step()
            optimizer.zero_grad()

        if global_step <= warmup_steps:
            warmup_lr = learning_rate * (global_step / warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        if global_step % 10 == 0 or global_step == 1:
            avg_loss = running_loss / 10 if global_step > 10 else running_loss
            running_loss = 0.0
            current_lr = optimizer.param_groups[0]["lr"]
            _emit(f"step: {global_step}/{steps}  loss: {avg_loss:.4f}  lr: {current_lr:.2e}")

        if global_step % save_every == 0 and global_step < steps:
            ckpt_path = output_dir / f"flux-checkpoint-{global_step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            transformer.save_pretrained(ckpt_path)
            _emit(f"Checkpoint saved: {ckpt_path}")

    # ── Save final LoRA ─────────────────────────────────────────────────────────────────

    _emit("Saving final FLUX LoRA weights...")
    try:
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file

        state_dict = get_peft_model_state_dict(transformer)
        safetensors_path = output_dir / f"flux_lora_rank{rank}_steps{steps}.safetensors"
        save_file(state_dict, str(safetensors_path))
        _emit(f"FLUX LoRA saved: {safetensors_path}")
    except Exception as exc:
        _emit(f"Warning: Could not save safetensors: {exc}")
        safetensors_path = output_dir

    _cleanup(transformer, vae, text_encoder_1, text_encoder_2)
    return True, str(safetensors_path)


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
