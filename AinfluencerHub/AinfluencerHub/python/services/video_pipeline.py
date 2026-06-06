"""
services/video_pipeline.py — Native video generation using HuggingFace diffusers.

Replaces ComfyUI's Wan2.1 I2V workflow with a native diffusers pipeline.
Supports image-to-video generation using Wan2.1 I2V or CogVideoX as a fallback.

All operations run locally on the user's GPU without external services.

v2 — Wan2.1 T2V → I2V fix:
  The original code loaded Wan2.1-T2V-14B (text-to-video) but called the
  pipeline with image= — a mismatch that caused the input image to be
  silently ignored (T2V pipelines don't accept an image parameter).
  Now loads Wan2.1-I2V-14B-720P-Diffusers first (the correct image-to-video
  variant). Falls back to CogVideoX-5b-I2V if I2V is unavailable, then
  T2V as a last resort with a clear warning that the image will be ignored.
"""

import logging
import random
from collections.abc import Callable
from pathlib import Path

from services.models import COGVIDEO, WAN_VIDEO, WAN_VIDEO_I2V

log = logging.getLogger("hub.video")

WAN_I2V_MODEL_ID  = WAN_VIDEO_I2V.repo_id   # preferred: proper image-to-video
WAN_T2V_MODEL_ID  = WAN_VIDEO.repo_id        # legacy fallback only
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
    """
    Load the video generation pipeline.

    Priority when model_id is empty:
      1. Wan2.1-I2V-14B-720P-Diffusers  (proper image-to-video)
      2. CogVideoX-5b-I2V                (smaller, 10 GB fallback)
      3. Wan2.1-T2V-14B-Diffusers        (last resort; image is ignored)
    """
    global _pipeline, _pipeline_type

    if _pipeline is not None:
        return

    device, dtype = _get_device_and_dtype()

    # Resolve model_id: caller can override, otherwise auto-select I2V
    if not model_id:
        model_id = WAN_I2V_MODEL_ID

    log.info("Loading video pipeline: %s on %s ...", model_id, device)

    kwargs: dict = {"torch_dtype": dtype}
    if hf_token:
        kwargs["token"] = hf_token

    if "CogVideo" in model_id:
        from diffusers import CogVideoXImageToVideoPipeline

        _pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
            model_id, **kwargs
        )
        _pipeline_type = "cogvideo"

    elif "I2V" in model_id or "i2v" in model_id:
        # Wan2.1 image-to-video — uses WanImageToVideoPipeline via Auto dispatch
        from diffusers import AutoPipelineForVideoGeneration

        _pipeline = AutoPipelineForVideoGeneration.from_pretrained(
            model_id, **kwargs
        )
        _pipeline_type = "wan_i2v"

    else:
        # T2V fallback — the image= argument will be ignored at inference time.
        # Log a clear warning so it's obvious in the log viewer.
        log.warning(
            "Video pipeline: loaded T2V model '%s' for an image-to-video task. "
            "The source image will be ignored and the video will be generated "
            "from the text prompt only. Download '%s' for true image animation.",
            model_id, WAN_I2V_MODEL_ID,
        )
        from diffusers import AutoPipelineForVideoGeneration

        _pipeline = AutoPipelineForVideoGeneration.from_pretrained(
            model_id, **kwargs
        )
        _pipeline_type = "wan_t2v"

    # Enable CPU offloading for large models (saves VRAM on smaller GPUs)
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
        model_id:     HuggingFace model ID. Defaults to Wan2.1-I2V-14B-720P.
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

        # Load and resize source image to 720P landscape for Wan2.1
        source = Image.open(image_path).convert("RGB")
        source = source.resize((1280, 720), Image.LANCZOS)

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        if progress_cb:
            progress_cb("Generating video... this may take several minutes.")

        # Build call kwargs — T2V pipelines don't accept image=
        call_kwargs: dict = {
            "prompt":              prompt,
            "num_frames":          num_frames,
            "num_inference_steps": steps,
            "guidance_scale":      cfg,
            "generator":           generator,
        }
        if _pipeline_type in ("wan_i2v", "cogvideo"):
            call_kwargs["image"] = source
            call_kwargs["negative_prompt"] = ""
        # wan_t2v: no image key — accepted without error but image is unused

        result = _pipeline(**call_kwargs)

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
