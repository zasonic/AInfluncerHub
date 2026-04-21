"""
services/video_pipeline.py — Native video generation using HuggingFace diffusers.

Supports image-to-video generation using Wan2.1 I2V or CogVideoX as a fallback.
All operations run locally on the user's GPU without external services.
"""

import logging
import random
from collections.abc import Callable
from pathlib import Path

from services.models import COGVIDEO, WAN_VIDEO

log = logging.getLogger("hub.video")

WAN_MODEL_ID = WAN_VIDEO.repo_id
COGVIDEO_MODEL_ID = COGVIDEO.repo_id

_pipeline = None
_pipeline_type = None


def _get_device_and_dtype():
    """Return the best available device and dtype."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def _load_pipeline(model_id: str = "", hf_token: str = ""):
    """Load the image-to-video pipeline with fallback chain."""
    global _pipeline, _pipeline_type

    if _pipeline is not None:
        return

    device, dtype = _get_device_and_dtype()

    if not model_id:
        model_id = WAN_MODEL_ID

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
    else:
        try:
            from diffusers import WanImageToVideoPipeline

            _pipeline = WanImageToVideoPipeline.from_pretrained(
                model_id, **kwargs
            )
            _pipeline_type = "wan_i2v"
        except Exception as exc:
            log.warning(
                "WanImageToVideoPipeline not available (%s), "
                "falling back to CogVideoX",
                exc,
            )
            from diffusers import CogVideoXImageToVideoPipeline

            _pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
                COGVIDEO_MODEL_ID, **kwargs
            )
            _pipeline_type = "cogvideo"

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

        source = Image.open(image_path).convert("RGB")

        if _pipeline_type == "cogvideo":
            source = source.resize((720, 480), Image.LANCZOS)
        else:
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
