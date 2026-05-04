"""
services/video_pipeline.py — Native video generation using HuggingFace diffusers.

Supports image-to-video generation with automatic model selection:

  1. LTX-Video (Lightricks/LTX-Video) — default when diffusers >= 0.32 is installed.
     ~9.5 GB, generates at 24 fps in portrait orientation (512x768).
     Works on a single 16 GB VRAM card without CPU offloading.
     num_frames=81 satisfies LTX-Video\'s 8N+1 frame constraint (≈3.4 s at 24 fps).

  2. CogVideoX-5b-I2V — ~10 GB, 16 fps, landscape. Used when model_id is
     explicitly set to the CogVideoX repo or when LTX is unavailable.

  3. Wan2.1-T2V-14B — ~28 GB, 16 fps. Used when explicitly requested or when
     neither LTX nor CogVideoX pipeline classes are importable.

All operations run locally on the user\'s GPU without external services.

Model selection:
  - Call generate_video() with model_id="" (default) to auto-select.
  - Pass an explicit model_id to override (e.g. for the Wan2.1 quality path).
"""

import logging
import random
from collections.abc import Callable
from pathlib import Path

from services.models import COGVIDEO, LTX_VIDEO, WAN_VIDEO

log = logging.getLogger("hub.video")

LTX_MODEL_ID      = LTX_VIDEO.repo_id        # Lightricks/LTX-Video
WAN_MODEL_ID      = WAN_VIDEO.repo_id        # Wan-AI/Wan2.1-T2V-14B-Diffusers
COGVIDEO_MODEL_ID = COGVIDEO.repo_id         # THUDM/CogVideoX-5b-I2V

# Detect LTX-Video pipeline class at import time without raising.
# LTXImageToVideoPipeline was added in diffusers ~0.32; older installs
# (requirements say >=0.30) fall back to CogVideoX / Wan2.1 automatically.
try:
    from diffusers import LTXImageToVideoPipeline as _LTXPipeline
    _LTX_DIFFUSERS_OK = True
except ImportError:
    _LTXPipeline = None  # type: ignore[assignment,misc]
    _LTX_DIFFUSERS_OK = False

# Lazy globals — loaded on demand, freed after each task
_pipeline      = None
_pipeline_type = None   # "ltx" | "cogvideo" | "wan"


def _get_device_and_dtype():
    """Return the best available device and dtype."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def _default_model_id() -> str:
    """
    Return the best model ID to use when the caller did not specify one.

    Priority:
      LTX-Video  — fast, portrait-native, fits on 16 GB VRAM (preferred).
      Wan2.1     — original default, largest quality, 28 GB.
    """
    if _LTX_DIFFUSERS_OK:
        return LTX_MODEL_ID
    return WAN_MODEL_ID


def _is_ltx_model(model_id: str) -> bool:
    return "LTX" in model_id or "Lightricks" in model_id


def _load_pipeline(model_id: str = "", hf_token: str = "") -> None:
    """Load the appropriate video generation pipeline into the global."""
    global _pipeline, _pipeline_type

    if _pipeline is not None:
        return

    if not model_id:
        model_id = _default_model_id()

    device, dtype = _get_device_and_dtype()
    log.info("Loading video pipeline: %s on %s ...", model_id, device)

    kwargs: dict = {"torch_dtype": dtype}
    if hf_token:
        kwargs["token"] = hf_token

    # ── LTX-Video ───────────────────────────────────────────────────────────
    if _is_ltx_model(model_id):
        if not _LTX_DIFFUSERS_OK:
            raise RuntimeError(
                "LTX-Video requires diffusers >= 0.32.0. "
                "Run: pip install --upgrade diffusers"
            )
        _pipeline = _LTXPipeline.from_pretrained(model_id, **kwargs)
        _pipeline_type = "ltx"
        # LTX-Video fits in 16 GB without CPU offloading; keep it on-device
        # for faster generation.
        _pipeline.to(device)

    # ── CogVideoX ──────────────────────────────────────────────────────────
    elif "CogVideo" in model_id:
        from diffusers import CogVideoXImageToVideoPipeline
        _pipeline = CogVideoXImageToVideoPipeline.from_pretrained(model_id, **kwargs)
        _pipeline_type = "cogvideo"
        if device == "cuda":
            _pipeline.enable_model_cpu_offload()
        else:
            _pipeline.to(device)

    # ── Wan2.1 (and any other model) ─────────────────────────────────────────
    else:
        from diffusers import AutoPipelineForVideoGeneration
        _pipeline = AutoPipelineForVideoGeneration.from_pretrained(model_id, **kwargs)
        _pipeline_type = "wan"
        if device == "cuda":
            _pipeline.enable_model_cpu_offload()
        else:
            _pipeline.to(device)

    log.info("Video pipeline loaded: type=%s model=%s", _pipeline_type, model_id)


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

    When model_id is empty the best available model is selected automatically:
      LTX-Video (~9.5 GB, 24 fps, portrait) if diffusers>=0.32 is installed,
      otherwise Wan2.1 (~28 GB).

    Args:
        image_path:   Source image to animate.
        prompt:       Motion/scene description.
        output_dir:   Where to save the output video.
        model_id:     HuggingFace model ID. Empty = auto-select.
        hf_token:     HuggingFace token for gated model download.
        num_frames:   Frames to generate. Default 81 works for all models
                      (satisfies LTX-Video\'s 8N+1 constraint; ≈3.4 s at 24 fps).
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

    selected    = model_id or _default_model_id()
    model_label = (
        "LTX-Video"  if _is_ltx_model(selected) else
        "CogVideoX"  if "CogVideo" in selected   else
        "Wan2.1"
    )

    try:
        if progress_cb:
            progress_cb(
                f"Loading {model_label} — first run downloads the model, please wait..."
            )

        _load_pipeline(model_id, hf_token)

        if progress_cb:
            progress_cb("Preparing source image...")

        # Per-model resolution and frame rate.
        # LTX-Video is portrait-native (9:16); CogVideoX and Wan2.1 are landscape.
        if _pipeline_type == "ltx":
            target_w, target_h = 512, 768
            fps = 24
        else:
            target_w, target_h = 832, 480
            fps = 16

        source = Image.open(image_path).convert("RGB")
        source = source.resize((target_w, target_h), Image.LANCZOS)

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        if progress_cb:
            duration_s = round(num_frames / fps, 1)
            progress_cb(
                f"Generating {num_frames}-frame video ({duration_s}s at {fps} fps) "
                f"with {model_label}..."
            )

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
        export_to_video(result.frames[0], str(out_path), fps=fps)

        if progress_cb:
            progress_cb(f"Video ready ({model_label}, {fps} fps).")

        return out_path

    except Exception as exc:
        log.error("Video generation failed: %s", exc)
        if progress_cb:
            progress_cb(f"Error: {str(exc)[:200]}")
        return None

    finally:
        unload()
