"""
services/video_pipeline.py — Native video generation using HuggingFace diffusers.

Replaces ComfyUI's Wan2.1 I2V workflow with a native diffusers pipeline.
Supports image-to-video generation with the following priority order:
  1. LTX-Video (Lightricks, 2B params, ~9 GB — runs on consumer GPUs)
  2. Wan2.1-T2V-14B (28 GB — high-VRAM GPUs only)
  3. CogVideoX-5b-I2V (10 GB — final fallback)

All operations run locally on the user's GPU without external services.
"""

import logging
import random
from collections.abc import Callable
from pathlib import Path

from services.models import COGVIDEO, LTX_VIDEO, WAN_VIDEO

log = logging.getLogger("hub.video")

LTX_MODEL_ID = LTX_VIDEO.repo_id
WAN_MODEL_ID = WAN_VIDEO.repo_id
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


def _load_pipeline(model_id: str = "", hf_token: str = ""):
    """Load the video generation pipeline with LTX-Video first, then fallbacks."""
    global _pipeline, _pipeline_type

    if _pipeline is not None:
        return

    device, dtype = _get_device_and_dtype()

    if not model_id:
        model_id = LTX_MODEL_ID  # Default: LTX-Video (~9 GB, consumer-friendly)

    log.info("Loading video pipeline: %s on %s ...", model_id, device)

    kwargs: dict = {"torch_dtype": dtype}
    if hf_token:
        kwargs["token"] = hf_token

    if "CogVideo" in model_id:
        from diffusers import CogVideoXImageToVideoPipeline

        _pipeline = CogVideoXImageToVideoPipeline.from_pretrained(model_id, **kwargs)
        _pipeline_type = "cogvideo"

    elif "LTX" in model_id or "Lightricks" in model_id:
        try:
            from diffusers import LTXImageToVideoPipeline

            _pipeline = LTXImageToVideoPipeline.from_pretrained(model_id, **kwargs)
            _pipeline_type = "ltx"
            log.info("LTX-Video pipeline loaded.")
        except Exception as exc:
            log.warning("LTX-Video failed (%s) — trying Wan2.1 ...", exc)
            try:
                from diffusers import AutoPipelineForVideoGeneration

                _pipeline = AutoPipelineForVideoGeneration.from_pretrained(
                    WAN_MODEL_ID, **kwargs
                )
                _pipeline_type = "wan"
                log.info("Wan2.1 fallback pipeline loaded.")
            except Exception as exc2:
                log.warning("Wan2.1 failed (%s) — trying CogVideoX ...", exc2)
                from diffusers import CogVideoXImageToVideoPipeline

                _pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
                    COGVIDEO_MODEL_ID, **kwargs
                )
                _pipeline_type = "cogvideo"
                log.info("CogVideoX fallback pipeline loaded.")

    else:
        # Wan2.1 or other explicit model ID
        from diffusers import AutoPipelineForVideoGeneration

        _pipeline = AutoPipelineForVideoGeneration.from_pretrained(model_id, **kwargs)
        _pipeline_type = "wan"

    # Enable CPU offloading for large models
    if device == "cuda":
        _pipeline.enable_model_cpu_offload()
    else:
        _pipeline.to(device)

    log.info("Video pipeline loaded: %s", _pipeline_type)


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
        model_id:     HuggingFace model ID (defaults to LTX-Video).
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

        # Load and resize source image
        source = Image.open(image_path).convert("RGB")
        source = source.resize((832, 480), Image.LANCZOS)

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # LTX-Video architecture caps at 97 frames
        actual_frames = min(num_frames, 97) if _pipeline_type == "ltx" else num_frames

        if progress_cb:
            progress_cb("Generating video... this may take several minutes.")

        result = _pipeline(
            image=source,
            prompt=prompt,
            negative_prompt="",
            num_frames=actual_frames,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        )

        # Export frames to video
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
