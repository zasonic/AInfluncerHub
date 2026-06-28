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

from services.models import FLUX_BASE, IP_ADAPTER, SDXL_BASE

log = logging.getLogger("hub.diffusion")

# Lazy globals — loaded on demand, freed after use
_pipeline = None
_flux_pipeline = None
_ip_adapter_loaded = False
_face_app = None


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


def _is_flux_lora(lora_path: str) -> bool:
    """Return True when the LoRA filename signals it was trained on FLUX.1-dev."""
    return "flux" in Path(lora_path).name.lower()


def _load_flux_pipeline(hf_token: str = ""):
    """Load FLUX.1-dev pipeline; use NF4 quantization when VRAM < 16 GB."""
    global _flux_pipeline
    if _flux_pipeline is not None:
        return

    import torch
    from diffusers import FluxPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1e9
        if torch.cuda.is_available()
        else 0.0
    )
    # NF4 quantization lets FLUX.1-dev run in ~10-12 GB instead of 24 GB
    use_nf4 = vram_gb > 0 and vram_gb < 16.0
    log.info(
        "Loading FLUX.1-dev on %s (VRAM=%.1f GB, NF4=%s) ...",
        device, vram_gb, use_nf4,
    )

    token = hf_token or None

    if use_nf4:
        from transformers import BitsAndBytesConfig
        from diffusers import FluxTransformer2DModel

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            FLUX_BASE.repo_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.float16,
            token=token,
        )
        _flux_pipeline = FluxPipeline.from_pretrained(
            FLUX_BASE.repo_id,
            transformer=transformer,
            torch_dtype=torch.float16,
            token=token,
        )
    else:
        _flux_pipeline = FluxPipeline.from_pretrained(
            FLUX_BASE.repo_id,
            torch_dtype=torch.bfloat16,
            token=token,
        )

    # CPU offload keeps peak VRAM low regardless of quantization choice
    _flux_pipeline.enable_model_cpu_offload()
    log.info("FLUX.1-dev pipeline loaded (NF4=%s).", use_nf4)


def _load_ip_adapter(hf_token: str = ""):
    """Load IP-Adapter for face-consistent generation."""
    global _ip_adapter_loaded
    if _ip_adapter_loaded:
        return

    _load_base_pipeline(hf_token)

    log.info("Loading IP-Adapter FaceID Plus V2 model...")
    _pipeline.load_ip_adapter(
        IP_ADAPTER.repo_id,
        subfolder=IP_ADAPTER.subfolder,
        weight_name=IP_ADAPTER.weight_name,
        image_encoder_folder="models/image_encoder",
        token=hf_token or None,
    )
    _pipeline.set_ip_adapter_scale(0.8)
    _ip_adapter_loaded = True
    log.info("IP-Adapter FaceID Plus V2 loaded.")


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
    global _pipeline, _flux_pipeline, _ip_adapter_loaded, _face_app

    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        _ip_adapter_loaded = False

    if _flux_pipeline is not None:
        del _flux_pipeline
        _flux_pipeline = None

    if _face_app is not None:
        del _face_app
        _face_app = None

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("Diffusion pipeline unloaded.")


# ── Public API ───────────────────────────────────────────────────────────────


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
        # FaceID Plus V2 uses ArcFace embeddings, not a raw PIL image
        face_embedding = _extract_face_embedding(reference_image)

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
                    ip_adapter_image_embeds=[face_embedding],
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
    Generate an image using SDXL or FLUX.1-dev with optional LoRA.

    When ``lora_path`` points to a FLUX LoRA (filename contains "flux"),
    the function loads FLUX.1-dev with NF4 quantization on GPUs < 16 GB.
    Otherwise it uses SDXL as before.  Returns list of output file paths.
    """
    import torch

    if output_dir is None:
        output_dir = Path("output") / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed < 0:
        seed = random.randint(0, 2**32 - 1)

    use_flux = bool(lora_path) and _is_flux_lora(lora_path)

    try:
        if use_flux:
            _load_flux_pipeline(hf_token)
            pipe = _flux_pipeline

            if progress_cb:
                progress_cb("Loading FLUX pipeline...")

            if lora_path and Path(lora_path).exists():
                if progress_cb:
                    progress_cb("Loading FLUX LoRA weights...")
                pipe.load_lora_weights(lora_path, adapter_name="user_lora")
                pipe.set_adapters(["user_lora"], adapter_weights=[lora_strength])
                log.info("FLUX LoRA loaded: %s (strength=%.2f)", lora_path, lora_strength)

            # CPU generator is safe with enable_model_cpu_offload
            generator = torch.Generator("cpu").manual_seed(seed)

            if progress_cb:
                progress_cb("Generating image with FLUX...")

            result = pipe(
                prompt=positive_prompt,
                num_inference_steps=steps,
                guidance_scale=3.5,
                width=width,
                height=height,
                generator=generator,
                max_sequence_length=512,
            )

            paths: list[Path] = []
            for idx, image in enumerate(result.images):
                out_path = output_dir / f"flux_{seed}_{idx}.png"
                image.save(out_path)
                paths.append(out_path)

        else:
            _load_base_pipeline(hf_token)
            pipe = _pipeline

            if progress_cb:
                progress_cb("Loading pipeline...")

            if lora_path and Path(lora_path).exists():
                if progress_cb:
                    progress_cb("Loading LoRA weights...")
                pipe.load_lora_weights(lora_path, adapter_name="user_lora")
                pipe.set_adapters(["user_lora"], adapter_weights=[lora_strength])
                log.info("LoRA loaded: %s (strength=%.2f)", lora_path, lora_strength)

            generator = torch.Generator(device=pipe.device).manual_seed(seed)

            if progress_cb:
                progress_cb("Generating image...")

            result = pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                generator=generator,
            )

            paths = []
            for idx, image in enumerate(result.images):
                out_path = output_dir / f"gen_{seed}_{idx}.png"
                image.save(out_path)
                paths.append(out_path)

        if progress_cb:
            progress_cb(f"Generated {len(paths)} image(s).")

        return paths

    finally:
        # Unload LoRA weights so they don't bleed into the next generation
        active_pipe = _flux_pipeline if use_flux else _pipeline
        if lora_path and active_pipe is not None:
            try:
                active_pipe.unload_lora_weights()
            except Exception:
                pass
        unload()
