"""
services/diffusion_pipeline.py — Native image generation using HuggingFace diffusers.

Replaces ComfyUI for all image generation:
  - Face-consistent dataset generation via IP-Adapter-FaceID
  - Text-to-image with trained LoRA weights
  - Model management (auto-download, VRAM-aware loading/unloading)

All operations run locally on the user's GPU without external services.
"""

import logging
import random
import threading
from collections.abc import Callable
from pathlib import Path

from services.models import IP_ADAPTER, SDXL_BASE, SDXL_LIGHTNING

log = logging.getLogger("hub.diffusion")

# Lazy globals — loaded on demand, freed after use
_pipeline = None
_ip_adapter_loaded = False
_face_app = None
_lightning_pipeline = None


def _get_device_and_dtype():
    """Return the best available device and dtype."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _load_base_pipeline(hf_token: str = ""):
    """Load the SDXL base pipeline."""
    global _pipeline
    if _pipeline is not None:
        return

    from diffusers import StableDiffusionXLPipeline

    device, dtype = _get_device_and_dtype()
    log.info("Loading SDXL pipeline on %s ...", device)

    kwargs: dict = {"torch_dtype": dtype, "variant": "fp16", "use_safetensors": True}
    if hf_token:
        kwargs["token"] = hf_token

    _pipeline = StableDiffusionXLPipeline.from_pretrained(
        SDXL_BASE.repo_id, revision=SDXL_BASE.revision, **kwargs
    )
    _pipeline.to(device)

    # Enable memory optimizations
    if device == "cuda":
        try:
            _pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers optional

    log.info("SDXL pipeline loaded.")


def _load_lightning_pipeline(hf_token: str = "") -> bool:
    """
    Load SDXL-Lightning (4-step distilled UNet, ByteDance ECCV 2024).
    Returns True on success, False with a warning on any failure.

    Lightning bakes guidance into the distillation process, so it requires
    guidance_scale=0 and EulerDiscreteScheduler with timestep_spacing="trailing".
    It is NOT compatible with IP-Adapter face conditioning (which needs cfg > 0).
    Only used for text-to-image preview without LoRA.
    """
    global _lightning_pipeline
    if _lightning_pipeline is not None:
        return True

    device, dtype = _get_device_and_dtype()
    if device != "cuda":
        return False  # Lightning UNet is CUDA-only for practical speed gains

    try:
        from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as _load_safetensors

        log.info("Loading SDXL-Lightning UNet ...")
        unet = UNet2DConditionModel.from_config(
            SDXL_BASE.repo_id, subfolder="unet"
        ).to(device, dtype)

        ckpt_path = hf_hub_download(
            SDXL_LIGHTNING.repo_id,
            SDXL_LIGHTNING.weight_name,
            token=hf_token or None,
        )
        unet.load_state_dict(_load_safetensors(ckpt_path, device=device))

        _lightning_pipeline = StableDiffusionXLPipeline.from_pretrained(
            SDXL_BASE.repo_id,
            unet=unet,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
            token=hf_token or None,
        ).to(device)

        _lightning_pipeline.scheduler = EulerDiscreteScheduler.from_config(
            _lightning_pipeline.scheduler.config,
            timestep_spacing="trailing",
        )
        log.info("SDXL-Lightning pipeline loaded.")
        return True

    except Exception as exc:
        log.warning("SDXL-Lightning load failed — falling back to standard SDXL: %s", exc)
        _lightning_pipeline = None
        return False


def _load_ip_adapter(hf_token: str = ""):
    """Load IP-Adapter for face-consistent generation."""
    global _ip_adapter_loaded
    if _ip_adapter_loaded:
        return

    _load_base_pipeline(hf_token)

    log.info("Loading IP-Adapter face model...")
    _pipeline.load_ip_adapter(
        IP_ADAPTER.repo_id,
        subfolder=IP_ADAPTER.subfolder,
        weight_name=IP_ADAPTER.weight_name,
        token=hf_token or None,
    )
    _pipeline.set_ip_adapter_scale(0.7)
    _ip_adapter_loaded = True
    log.info("IP-Adapter loaded.")


def _get_face_app():
    """Initialize InsightFace for face embedding extraction."""
    global _face_app
    if _face_app is not None:
        return _face_app

    from insightface.app import FaceAnalysis

    _face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    _face_app.prepare(ctx_id=0, det_size=(640, 640))
    log.info("InsightFace face analysis loaded.")
    return _face_app


def _extract_face_embedding(image_path: Path):
    """Extract face embedding from a reference image."""
    import cv2

    app = _get_face_app()
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = app.get(img)
    if not faces:
        raise ValueError(f"No face detected in {image_path.name}")

    # Use the largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding


def _prepare_face_image(image_path: Path):
    """Load and prepare a face image for IP-Adapter."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    return img


def unload() -> None:
    """Release all GPU memory."""
    global _pipeline, _ip_adapter_loaded, _face_app, _lightning_pipeline

    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        _ip_adapter_loaded = False

    if _lightning_pipeline is not None:
        del _lightning_pipeline
        _lightning_pipeline = None

    if _face_app is not None:
        del _face_app
        _face_app = None

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("Diffusion pipeline unloaded.")


# ── Public API ───────────────────────────────────────────────────────────────────────────


def generate_dataset(
    reference_image: Path,
    prompts: list[str],
    trigger_word: str,
    output_dir: Path,
    hf_token: str = "",
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[Path]:
    """
    Generate face-consistent dataset images using IP-Adapter.

    Uses the reference image's face to guide generation, producing varied
    poses and settings while maintaining face identity.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(prompts)
    paths: list[Path] = []

    try:
        _load_ip_adapter(hf_token)
        face_image = _prepare_face_image(reference_image)

        for i, prompt in enumerate(prompts):
            if cancel_event and cancel_event.is_set():
                break

            full_prompt = f"{trigger_word}, {prompt}" if trigger_word else prompt
            label = prompt[:50] + "..." if len(prompt) > 50 else prompt
            if progress_cb:
                progress_cb(i, total, f"Generating {i + 1}/{total}: {label}")

            seed = random.randint(0, 2**32 - 1)
            import torch
            generator = torch.Generator(device=_pipeline.device).manual_seed(seed)

            try:
                result = _pipeline(
                    prompt=full_prompt,
                    negative_prompt="blurry, low quality, watermark, text, deformed",
                    ip_adapter_image=face_image,
                    num_inference_steps=20,
                    guidance_scale=4.0,
                    width=832,
                    height=1216,
                    generator=generator,
                )
                image = result.images[0]
                out_path = output_dir / f"dataset_{len(paths) + 1:03d}.jpg"
                image.save(out_path, quality=95)
                paths.append(out_path)

            except Exception as exc:
                log.error("Generation failed for prompt %d: %s", i, exc)
                continue

        if progress_cb:
            progress_cb(total, total, f"Generated {len(paths)} images.")

    finally:
        unload()

    return paths


def generate_image(
    positive_prompt: str,
    negative_prompt: str = "blurry, low quality, watermark, text",
    lora_path: str = "",
    lora_strength: float = 0.85,
    width: int = 832,
    height: int = 1216,
    steps: int = 20,
    cfg: float = 4.0,
    seed: int = -1,
    output_dir: Path | None = None,
    hf_token: str = "",
    progress_cb: Callable[[str], None] | None = None,
) -> list[Path]:
    """
    Generate an image using SDXL with optional LoRA.

    When no LoRA is provided and a CUDA GPU is available, uses SDXL-Lightning
    (4-step distilled, ~4x faster) with automatic fallback to standard SDXL.
    LoRA generation always uses standard SDXL (Lightning requires guidance_scale=0
    which disables the adapter conditioning).

    Returns list of output file paths.
    """
    import torch

    if output_dir is None:
        output_dir = Path("output") / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    _lora_loaded = False
    try:
        # Use Lightning for fast preview when no LoRA is requested.
        # Lightning bakes guidance into distillation (guidance_scale must be 0),
        # so it cannot be combined with LoRA (which needs cfg > 0 to apply).
        use_lightning = not bool(lora_path)
        if use_lightning:
            use_lightning = _load_lightning_pipeline(hf_token)

        if use_lightning:
            active_pipeline = _lightning_pipeline
            actual_steps = 4
            actual_cfg = 0.0
            if progress_cb:
                progress_cb("Loading pipeline (Lightning — 4-step fast preview)...")
        else:
            _load_base_pipeline(hf_token)
            active_pipeline = _pipeline
            actual_steps = steps
            actual_cfg = cfg
            if progress_cb:
                progress_cb("Loading pipeline...")

            if lora_path and Path(lora_path).exists():
                if progress_cb:
                    progress_cb("Loading LoRA weights...")
                active_pipeline.load_lora_weights(lora_path, adapter_name="user_lora")
                active_pipeline.set_adapters(["user_lora"], adapter_weights=[lora_strength])
                _lora_loaded = True
                log.info("LoRA loaded: %s (strength=%.2f)", lora_path, lora_strength)

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device=active_pipeline.device).manual_seed(seed)

        if progress_cb:
            progress_cb("Generating image...")

        result = active_pipeline(
            prompt=positive_prompt,
            negative_prompt=negative_prompt if not use_lightning else "",
            num_inference_steps=actual_steps,
            guidance_scale=actual_cfg,
            width=width,
            height=height,
            generator=generator,
        )

        paths: list[Path] = []
        for idx, image in enumerate(result.images):
            out_path = output_dir / f"gen_{seed}_{idx}.png"
            image.save(out_path)
            paths.append(out_path)

        if progress_cb:
            progress_cb(f"Generated {len(paths)} image(s).")

        return paths

    finally:
        if _lora_loaded and _pipeline is not None:
            try:
                _pipeline.unload_lora_weights()
            except Exception:
                pass
        unload()
