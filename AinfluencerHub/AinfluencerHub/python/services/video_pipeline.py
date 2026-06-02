"""
services/video_pipeline.py — Native video generation using HuggingFace diffusers.

Replaces ComfyUI's Wan2.1 I2V workflow with a native diffusers pipeline.
Supports image-to-video generation using Wan 2.2 TI2V-5B (preferred),
Wan 2.1 T2V-14B (fallback), or CogVideoX.

Model selection at runtime (when model_id is not supplied by the caller):
  1. Wan 2.2 TI2V-5B  — 5 B params, 8 GB VRAM, 720p.
                         Auto-selected only when already in the local HF cache
                         so no unexpected multi-GB download is triggered.
  2. Wan 2.1 T2V-14B  — 14 B params, 28 GB VRAM.
                         Used when Wan 2.2 is not cached locally.
  3. CogVideoX-5b-I2V — selected when model_id explicitly contains "CogVideo".

All operations run locally on the user's GPU without external services.
"""

import logging
import random
from collections.abc import Callable
from pathlib import Path

from services.models import COGVIDEO, WAN_VIDEO, WAN_VIDEO_22

log = logging.getLogger("hub.video")

WAN_MODEL_ID     = WAN_VIDEO.repo_id      # Wan 2.1 — fallback
WAN_MODEL_ID_V22 = WAN_VIDEO_22.repo_id   # Wan 2.2 TI2V-5B — preferred
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


def _best_wan_model_id() -> str:
    """
    Return Wan 2.2 TI2V-5B if it is already present in the local HF cache;
    otherwise return Wan 2.1.  Uses huggingface_hub.try_to_load_from_cache
    which returns None when the model is absent — no network request is made,
    so this probe never triggers an unintended download.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        probe = try_to_load_from_cache(WAN_MODEL_ID_V22, "model_index.json")
        if probe is not None:
            return WAN_MODEL_ID_V22
    except Exception:
        pass
    return WAN_MODEL_ID


def _load_pipeline(model_id: str = "", hf_token: str = "") -> None:
    """Load the video generation pipeline."""
    global _pipeline, _pipeline_type

    if _pipeline is not None:
        return

    device, dtype = _get_device_and_dtype()

    if not model_id:
        model_id = _best_wan_model_id()

    log.info("Loading video pipeline: %s on %s ...", model_id, device)

    kwargs: dict = {"torch_dtype": dtype}
    if hf_token:
        kwargs["token"] = hf_token

    if "CogVideo" in model_id:
        from diffusers import CogVideoXImageToVideoPipeline
        _pipeline = CogVideoXImageToVideoPipeline.from_pretrained(model_id, **kwargs)
        _pipeline_type = "cogvideo"
    else:
        from diffusers import AutoPipelineForVideoGeneration
        _pipeline = AutoPipelineForVideoGeneration.from_pretrained(model_id, **kwargs)
        _pipeline_type = "wan22" if model_id == WAN_MODEL_ID_V22 else "wan"

    if device == "cuda":
        _pipeline.enable_model_cpu_offload()
    else:
        _pipeline.to(device)

    log.info("Video pipeline loaded: %s (%s)", _pipeline_type, model_id)


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
        model_id:     HuggingFace model ID. Empty = auto-select best cached Wan model.
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
        resolved_id = model_id or _best_wan_model_id()
        model_label = (
            "Wan 2.2 TI2V-5B" if resolved_id == WAN_MODEL_ID_V22
            else resolved_id.split("/")[-1]
        )
        if progress_cb:
            progress_cb(f"Loading {model_label} (this may take a few minutes on first run)...")

        _load_pipeline(resolved_id, hf_token)

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
