"""
services/video_pipeline.py — Native video generation using HuggingFace diffusers.

Replaces ComfyUI's Wan2.1 I2V workflow with a native diffusers pipeline.
Supports image-to-video generation with the following preference order:
  1. Wan2.1-I2V  (WanImageToVideoPipeline or AutoPipelineForVideoGeneration)
  2. Wan2.1-T2V  (AutoPipelineForVideoGeneration, for users with the T2V model)
  3. CogVideoX-5b-I2V  (last-resort fallback)

All operations run locally on the user's GPU without external services.
"""

import logging
import random
from collections.abc import Callable
from pathlib import Path

from services.models import COGVIDEO, WAN_I2V, WAN_VIDEO

log = logging.getLogger("hub.video")

WAN_I2V_MODEL_ID  = WAN_I2V.repo_id
WAN_MODEL_ID      = WAN_VIDEO.repo_id     # T2V fallback for users who downloaded it
COGVIDEO_MODEL_ID = COGVIDEO.repo_id

# Lazy global
_pipeline = None
_pipeline_type = None


def _get_device_and_dtype():
    """Return the best available device and dtype."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def _try_load_wan(model_id: str, kwargs: dict):
    """Attempt to load a Wan pipeline, preferring the dedicated I2V class."""
    try:
        from diffusers import WanImageToVideoPipeline  # added in diffusers ~0.33
        return WanImageToVideoPipeline.from_pretrained(model_id, **kwargs), "wan"
    except (ImportError, AttributeError):
        pass
    except Exception as exc:
        raise  # re-raise real loading errors (e.g. missing weights)
    # Older diffusers: AutoPipeline handles both I2V and T2V Wan variants
    from diffusers import AutoPipelineForVideoGeneration
    return AutoPipelineForVideoGeneration.from_pretrained(model_id, **kwargs), "wan"


def _load_pipeline(model_id: str = "", hf_token: str = ""):
    """Load the video generation pipeline.

    Tries models in preference order when no model_id is supplied:
      WAN_I2V → WAN_VIDEO (T2V fallback) → CogVideoX
    Raises RuntimeError with a user-friendly message only when every
    candidate fails so that generate_video() can surface it clearly.
    """
    global _pipeline, _pipeline_type

    if _pipeline is not None:
        return

    device, dtype = _get_device_and_dtype()

    kwargs: dict = {"torch_dtype": dtype}
    if hf_token:
        kwargs["token"] = hf_token

    # Build candidate list
    if model_id:
        candidates = [model_id]
    else:
        candidates = [WAN_I2V_MODEL_ID, WAN_MODEL_ID, COGVIDEO_MODEL_ID]

    last_exc: Exception | None = None
    for mid in candidates:
        log.info("Loading video pipeline: %s on %s ...", mid, device)
        try:
            if "CogVideo" in mid:
                from diffusers import CogVideoXImageToVideoPipeline
                pipe = CogVideoXImageToVideoPipeline.from_pretrained(mid, **kwargs)
                ptype = "cogvideo"
            else:
                pipe, ptype = _try_load_wan(mid, kwargs)

            # Move to device / enable offloading
            if device == "cuda":
                pipe.enable_model_cpu_offload()
                # Reduce peak VRAM during VAE decode; no quality impact
                try:
                    pipe.enable_vae_slicing()
                except AttributeError:
                    pass
                try:
                    pipe.enable_vae_tiling()
                except AttributeError:
                    pass
            else:
                pipe.to(device)

            _pipeline = pipe
            _pipeline_type = ptype
            log.info("Video pipeline loaded: %s (%s)", ptype, mid)
            return

        except Exception as exc:
            log.warning("Failed to load video model %s: %s — trying next candidate", mid, exc)
            last_exc = exc
            _pipeline = None
            _pipeline_type = None

    raise RuntimeError(
        "No video model available. "
        "Download Wan2.1-I2V (recommended) or CogVideoX from the Models page, "
        "then try again."
    ) from last_exc


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
        model_id:     HuggingFace model ID (defaults to Wan2.1-I2V).
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
        if progress_cb:
            progress_cb("Loading video model (this may take a few minutes on first run)...")

        _load_pipeline(model_id, hf_token)

        if progress_cb:
            progress_cb("Preparing source image...")

        # Load source image and resize to a standard Wan2.1 resolution while
        # preserving aspect ratio. Portrait input → 480×832; landscape/square
        # → 832×480. Forcing all input to 832×480 (the previous behaviour)
        # produced distorted, landscape-cropped videos for portrait photos.
        source = Image.open(image_path).convert("RGB")
        src_w, src_h = source.size
        if src_h > src_w:
            new_w, new_h = 480, 832   # portrait
        else:
            new_w, new_h = 832, 480   # landscape / square
        source = source.resize((new_w, new_h), Image.LANCZOS)

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
