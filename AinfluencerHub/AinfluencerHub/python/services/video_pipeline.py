"""services/video_pipeline.py — Native video generation using HuggingFace diffusers.

Supports image-to-video generation. CogVideoX-5B-I2V is the default model
(10 GB, fits 16 GB VRAM); Wan2.1-I2V-14B-480P is an optional higher-quality
alternative that requires ~24 GB VRAM with CPU offloading.

Previous behaviour (now fixed):
  - Wan2.1-T2V was used by default; T2V pipelines don't accept an `image`
    argument so every video call raised TypeError at runtime.
  - Source images were always resized to landscape (832 x 480), squashing
    portrait influencer photos and distorting faces.
  - fps=16 produced noticeably choppy output.
"""

import logging
import random
from collections.abc import Callable
from pathlib import Path

from services.models import COGVIDEO, WAN_VIDEO

log = logging.getLogger("hub.video")

# Use CogVideoX as the default — it is I2V-native and fits in 16 GB VRAM.
# Wan I2V remains available as an explicit opt-in via model_id.
DEFAULT_MODEL_ID = COGVIDEO.repo_id
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
    """Load the video generation pipeline."""
    global _pipeline, _pipeline_type

    if _pipeline is not None:
        return

    if not model_id:
        model_id = DEFAULT_MODEL_ID  # CogVideoX-5B-I2V

    device, dtype = _get_device_and_dtype()
    log.info("Loading video pipeline: %s on %s ...", model_id, device)

    kwargs: dict = {"torch_dtype": dtype}
    if hf_token:
        kwargs["token"] = hf_token

    if "CogVideo" in model_id:
        from diffusers import CogVideoXImageToVideoPipeline
        _pipeline = CogVideoXImageToVideoPipeline.from_pretrained(model_id, **kwargs)
        _pipeline_type = "cogvideo"
    else:
        # Wan2.1 I2V — use the explicit class so the correct pipeline is always
        # selected regardless of what AutoPipeline resolves to.
        from diffusers import WanImageToVideoPipeline
        _pipeline = WanImageToVideoPipeline.from_pretrained(model_id, **kwargs)
        _pipeline_type = "wan"

    # CPU offloading for large models; moves layers to RAM between forward passes.
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


def _portrait_aware_resize(source, model_type: str):
    """
    Resize source image to a model-compatible resolution while preserving
    the portrait/landscape orientation of the original.

    CogVideoX training resolutions: 720x480 (landscape) or 480x720 (portrait).
    Wan I2V 480P: 832x480 (landscape) or 480x832 (portrait).
    Returning (resized_image, target_width, target_height).
    """
    from PIL import Image

    w, h = source.size
    is_portrait = h >= w

    if model_type == "cogvideo":
        target_w, target_h = (480, 720) if is_portrait else (720, 480)
    else:
        # Wan I2V 480P native resolution
        target_w, target_h = (480, 832) if is_portrait else (832, 480)

    resized = source.resize((target_w, target_h), Image.LANCZOS)
    return resized, target_w, target_h


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
        model_id:     HuggingFace model ID (defaults to CogVideoX-5B-I2V).
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

        source = Image.open(image_path).convert("RGB")
        source, target_w, target_h = _portrait_aware_resize(source, _pipeline_type or "cogvideo")

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
            height=target_h,
            width=target_w,
        )

        if progress_cb:
            progress_cb("Saving video...")

        from diffusers.utils import export_to_video

        out_path = output_dir / f"video_{image_path.stem}_{seed}.mp4"
        export_to_video(result.frames[0], str(out_path), fps=24)  # 24fps standard

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
