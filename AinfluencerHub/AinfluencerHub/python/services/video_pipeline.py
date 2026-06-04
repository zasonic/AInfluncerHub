"""
services/video_pipeline.py — Native video generation using HuggingFace diffusers.

Replaces ComfyUI's Wan2.1 I2V workflow with a native diffusers pipeline.
Supports image-to-video generation using Wan2.1 or CogVideoX as a fallback.

All operations run locally on the user's GPU without external services.

VRAM-aware model selection (automatic cascade):
  Wan2.1-14B  — 24+ GB VRAM  (best quality)
  CogVideoX-5b — 12+ GB VRAM  (good quality, faster download)
  CogVideoX-2b —  8+ GB VRAM  (acceptable quality, works on budget GPUs)
  CPU          — no GPU       (very slow but functional)

If a caller passes an explicit model_id that beats the VRAM check we honour
it; if loading fails we step down the cascade automatically so the app never
leaves non-technical users staring at an out-of-memory error.
"""

import logging
import random
from collections.abc import Callable
from pathlib import Path

from services.models import COGVIDEO, COGVIDEO_2B, WAN_VIDEO

log = logging.getLogger("hub.video")

WAN_MODEL_ID      = WAN_VIDEO.repo_id
COGVIDEO_MODEL_ID = COGVIDEO.repo_id
COGVIDEO_2B_ID    = COGVIDEO_2B.repo_id

# VRAM thresholds (GiB) required to safely load each model with CPU offloading.
_VRAM_REQUIRED: dict[str, float] = {
    WAN_MODEL_ID:      24.0,
    COGVIDEO_MODEL_ID: 12.0,
    COGVIDEO_2B_ID:     8.0,
}

# Lazy global
_pipeline = None
_pipeline_type = None


def _get_device_and_dtype():
    """Return the best available device and dtype."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def _free_vram_gib() -> float:
    """Return free GPU memory in GiB, or 0.0 if no CUDA device is present."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:
        return 0.0


def _pick_model() -> str:
    """
    Choose the heaviest model that fits available VRAM.

    Falls back down the cascade (Wan → CogVideoX-5b → CogVideoX-2b) so
    non-technical users with budget GPUs still get a result instead of an
    out-of-memory crash.
    """
    free = _free_vram_gib()
    if free >= _VRAM_REQUIRED[WAN_MODEL_ID]:
        return WAN_MODEL_ID
    if free >= _VRAM_REQUIRED[COGVIDEO_MODEL_ID]:
        log.info(
            "VRAM %.1f GiB < %.0f GiB required for Wan2.1 — using CogVideoX-5b",
            free, _VRAM_REQUIRED[WAN_MODEL_ID],
        )
        return COGVIDEO_MODEL_ID
    # 2b works at 8 GiB; below that we still try and let CPU offloading handle it
    log.info(
        "VRAM %.1f GiB < %.0f GiB required for CogVideoX-5b — using CogVideoX-2b",
        free, _VRAM_REQUIRED[COGVIDEO_MODEL_ID],
    )
    return COGVIDEO_2B_ID


def _load_pipeline(model_id: str = "", hf_token: str = ""):
    """Load the video generation pipeline, stepping down the cascade on OOM."""
    global _pipeline, _pipeline_type

    if _pipeline is not None:
        return

    device, dtype = _get_device_and_dtype()

    # Auto-select if the caller didn't pin a model
    resolved_id = model_id or _pick_model()

    # Cascade: try the resolved model, then step down on failure
    cascade = [resolved_id]
    if resolved_id == WAN_MODEL_ID:
        cascade += [COGVIDEO_MODEL_ID, COGVIDEO_2B_ID]
    elif resolved_id == COGVIDEO_MODEL_ID:
        cascade += [COGVIDEO_2B_ID]

    last_exc: Exception | None = None
    for candidate in cascade:
        log.info("Loading video pipeline: %s on %s ...", candidate, device)
        kwargs: dict = {"torch_dtype": dtype}
        if hf_token:
            kwargs["token"] = hf_token
        try:
            if "CogVideo" in candidate:
                from diffusers import CogVideoXImageToVideoPipeline
                _pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
                    candidate, **kwargs
                )
                _pipeline_type = "cogvideo"
            else:
                from diffusers import AutoPipelineForVideoGeneration
                _pipeline = AutoPipelineForVideoGeneration.from_pretrained(
                    candidate, **kwargs
                )
                _pipeline_type = "wan"

            if device == "cuda":
                _pipeline.enable_model_cpu_offload()
            else:
                _pipeline.to(device)

            log.info("Video pipeline loaded: %s (%s)", _pipeline_type, candidate)
            return
        except Exception as exc:
            log.warning("Failed to load %s: %s — trying next fallback", candidate, exc)
            last_exc = exc
            _pipeline = None
            _pipeline_type = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    raise RuntimeError(
        f"All video model candidates failed. Last error: {last_exc}"
    )


def unload() -> None:
    """Release all GPU memory."""
    global _pipeline, _pipeline_type

    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        _pipeline_type = None

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("Video pipeline unloaded.")


def generate_video(
    image_path: Path,
    prompt: str,
    output_dir: Path,
    model_id: str = "",
    hf_token: str = "",
    num_frames: int = 81,
    steps: int = 20,
    cfg: float = 5.0,
    seed: int = -1,
    progress_cb: Callable[[str], None] | None = None,
) -> Path | None:
    """
    Generate a video from a still image.

    Args:
        image_path:   Source image to animate.
        prompt:       Motion/scene description.
        output_dir:   Where to save the output video.
        model_id:     HuggingFace model ID. Leave empty for automatic VRAM-aware
                      selection (Wan2.1 → CogVideoX-5b → CogVideoX-2b cascade).
        hf_token:     HuggingFace token for model download.
        num_frames:   Number of video frames to generate.
        steps:        Diffusion inference steps.
        cfg:          Guidance scale.
        seed:         Random seed (-1 for random).
        progress_cb:  Callback for status messages.

    Returns:
        Path to the generated video, or None on failure.
    """
    import torch
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        free_gib = _free_vram_gib()
        if progress_cb:
            vram_info = f" ({free_gib:.0f} GB free VRAM)" if free_gib > 0 else " (CPU mode)"
            progress_cb(
                f"Loading video model{vram_info} — this may take a few minutes on first run..."
            )

        _load_pipeline(model_id, hf_token)

        # Report which model was actually loaded so users understand what happened
        if progress_cb and _pipeline_type:
            model_names = {
                "wan":      "Wan2.1-14B (best quality)",
                "cogvideo": (
                    "CogVideoX-5b"
                    if _pipeline is not None and COGVIDEO_MODEL_ID in str(getattr(_pipeline, "config", {}).get("_name_or_path", ""))
                    else "CogVideoX-2b (budget GPU mode)"
                ),
            }
            progress_cb(f"Model ready: {model_names.get(_pipeline_type, _pipeline_type)}")

        if progress_cb:
            progress_cb("Preparing source image...")

        source = Image.open(image_path).convert("RGB")
        source = source.resize((832, 480), Image.LANCZOS)

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        if progress_cb:
            progress_cb("Generating video... this may take several minutes.")

        result = _pipeline(
            image=source,
            prompt=prompt,
            negative_prompt="",
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        )

        if progress_cb:
            progress_cb("Saving video...")

        from diffusers.utils import export_to_video

        out_path = output_dir / f"video_{image_path.stem}_{seed}.mp4"
        export_to_video(result.frames[0], str(out_path), fps=16)

        if progress_cb:
            progress_cb("Video created successfully.")

        return out_path

    except Exception as exc:
        log.error("Video generation failed: %s", exc)
        if progress_cb:
            progress_cb(f"Error: {str(exc)[:200]}")
        return None

    finally:
        unload()
