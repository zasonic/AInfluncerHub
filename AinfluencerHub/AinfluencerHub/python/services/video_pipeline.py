"""
services/video_pipeline.py — Native video generation using HuggingFace diffusers.

v2.1 — SVD-XT as default primary model (replaces Wan2.1):

  Model priority  │ VRAM    │ Notes
  ────────────────┼─────────┼──────────────────────────────────────────────────
  SVD-XT          │  ~8 GB  │ Default. Smooth 25-frame, image-conditioned.
  CogVideoX-5B    │ ~10 GB  │ Auto-fallback if SVD-XT fails (e.g. OOM).
  Wan2.1-T2V-14B  │ ~28 GB  │ Explicit only — pass WAN_MODEL_ID as model_id.

Wan2.1 was the previous default but its 28 GB VRAM requirement made it
impractical for almost all consumer GPUs.  SVD-XT delivers smooth animations
on a typical gaming GPU (RTX 3080/4070 or better) and degrades gracefully to
CogVideoX if the GPU is tight on memory.

SVD-XT is image-conditioned only (no text prompt) — the `prompt` parameter
is passed to Wan2.1 and CogVideoX but silently ignored for SVD-XT.
"""

import logging
import random
from collections.abc import Callable
from pathlib import Path

from services.models import COGVIDEO, SVD_XT, WAN_VIDEO

log = logging.getLogger("hub.video")

SVD_MODEL_ID    = SVD_XT.repo_id
COGVIDEO_MODEL_ID = COGVIDEO.repo_id
WAN_MODEL_ID    = WAN_VIDEO.repo_id

# Lazy globals
_pipeline      = None
_pipeline_type = None   # "svd" | "cogvideo" | "wan"


def _get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_pipeline(model_id: str = "", hf_token: str = "") -> None:
    """Load the video pipeline.  Falls back SVD → CogVideoX on OOM."""
    global _pipeline, _pipeline_type

    if _pipeline is not None:
        return

    import torch

    device = _get_device()
    dtype  = torch.float16 if device == "cuda" else torch.float32

    if not model_id:
        model_id = SVD_MODEL_ID

    log.info("Loading video pipeline: %s on %s ...", model_id, device)

    kwargs: dict = {"torch_dtype": dtype}
    if hf_token:
        kwargs["token"] = hf_token

    if "stable-video-diffusion" in model_id:
        _load_svd(model_id, kwargs, device)
    elif "CogVideo" in model_id:
        _load_cogvideo(model_id, kwargs, device)
    else:
        _load_wan(model_id, kwargs, device)

    log.info("Video pipeline ready: %s", _pipeline_type)


def _load_svd(model_id: str, kwargs: dict, device: str) -> None:
    """Load Stable Video Diffusion XT; fall back to CogVideoX on failure."""
    global _pipeline, _pipeline_type
    try:
        from diffusers import StableVideoDiffusionPipeline

        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id, variant="fp16", **kwargs
        )
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        _pipeline      = pipe
        _pipeline_type = "svd"
    except Exception as exc:
        log.warning(
            "SVD-XT failed to load (%s) — falling back to CogVideoX.", exc
        )
        _load_cogvideo(COGVIDEO_MODEL_ID, kwargs, device)


def _load_cogvideo(model_id: str, kwargs: dict, device: str) -> None:
    global _pipeline, _pipeline_type
    from diffusers import CogVideoXImageToVideoPipeline

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, **kwargs)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    _pipeline      = pipe
    _pipeline_type = "cogvideo"


def _load_wan(model_id: str, kwargs: dict, device: str) -> None:
    global _pipeline, _pipeline_type
    from diffusers import AutoPipelineForVideoGeneration

    pipe = AutoPipelineForVideoGeneration.from_pretrained(model_id, **kwargs)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    _pipeline      = pipe
    _pipeline_type = "wan"


def unload() -> None:
    """Release all GPU memory."""
    global _pipeline, _pipeline_type

    if _pipeline is not None:
        del _pipeline
        _pipeline      = None
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
        prompt:       Motion/scene description (used by CogVideoX and Wan2.1;
                      ignored by SVD-XT which is image-conditioned only).
        output_dir:   Where to save the output video.
        model_id:     HuggingFace model ID.  Empty string → SVD-XT (default).
        hf_token:     HuggingFace token for model download.
        num_frames:   Frames to generate (SVD-XT always generates 25).
        steps:        Diffusion inference steps.
        cfg:          Guidance scale (unused for SVD-XT).
        seed:         Random seed (-1 = random).
        progress_cb:  Callback for status messages shown in the UI.

    Returns:
        Path to the generated MP4, or None on failure.
    """
    import torch
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_label = model_id.split("/")[-1] if model_id else "SVD-XT (default)"
        if progress_cb:
            progress_cb(
                f"Loading video model ({model_label}) — first run downloads model weights..."
            )

        _load_pipeline(model_id, hf_token)

        if progress_cb:
            progress_cb("Preparing source image...")

        source = Image.open(image_path).convert("RGB")

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        out_path = output_dir / f"video_{image_path.stem}_{seed}.mp4"

        if _pipeline_type == "svd":
            _generate_svd(source, out_path, steps, generator, progress_cb)
        elif _pipeline_type == "cogvideo":
            _generate_cogvideo(source, prompt, out_path, num_frames, steps, cfg, generator, progress_cb)
        else:
            _generate_wan(source, prompt, out_path, num_frames, steps, cfg, generator, progress_cb)

        if progress_cb:
            progress_cb("Video created successfully.")

        return out_path

    except Exception as exc:
        log.error("Video generation failed: %s", exc)
        if progress_cb:
            progress_cb(f"Error generating video: {str(exc)[:200]}")
        return None

    finally:
        unload()


def _generate_svd(
    source: "Image.Image",
    out_path: Path,
    steps: int,
    generator,
    progress_cb: Callable[[str], None] | None,
) -> None:
    """SVD-XT: image-conditioned, 25 frames at 7 fps (~3.5 s)."""
    from diffusers.utils import export_to_video

    if progress_cb:
        progress_cb("Generating video with SVD-XT (~25 frames)...")

    result = _pipeline(
        image=source,
        num_frames=25,
        decode_chunk_size=8,    # process in chunks to reduce VRAM peak
        motion_bucket_id=127,   # 0–255 motion intensity; 127 = medium
        noise_aug_strength=0.02,
        num_inference_steps=steps,
        generator=generator,
    )

    if progress_cb:
        progress_cb("Saving video...")

    export_to_video(result.frames[0], str(out_path), fps=7)


def _generate_cogvideo(
    source: "Image.Image",
    prompt: str,
    out_path: Path,
    num_frames: int,
    steps: int,
    cfg: float,
    generator,
    progress_cb: Callable[[str], None] | None,
) -> None:
    from diffusers.utils import export_to_video

    if progress_cb:
        progress_cb("Generating video with CogVideoX...")

    resized = source.resize((720, 480), Image.LANCZOS)

    result = _pipeline(
        image=resized,
        prompt=prompt or "a person, natural motion",
        negative_prompt="",
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
    )

    if progress_cb:
        progress_cb("Saving video...")

    export_to_video(result.frames[0], str(out_path), fps=16)


def _generate_wan(
    source: "Image.Image",
    prompt: str,
    out_path: Path,
    num_frames: int,
    steps: int,
    cfg: float,
    generator,
    progress_cb: Callable[[str], None] | None,
) -> None:
    from diffusers.utils import export_to_video

    if progress_cb:
        progress_cb("Generating video with Wan2.1 (this may take 10+ minutes)...")

    resized = source.resize((832, 480), Image.LANCZOS)

    result = _pipeline(
        image=resized,
        prompt=prompt or "a person, natural motion",
        negative_prompt="",
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
    )

    if progress_cb:
        progress_cb("Saving video...")

    export_to_video(result.frames[0], str(out_path), fps=16)
