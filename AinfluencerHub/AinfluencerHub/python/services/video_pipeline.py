"""services/video_pipeline.py — Native image-to-video generation using HuggingFace diffusers.

Primary model: LTX-Video (Lightricks/LTX-Video) — diffusers-native, fits within 16 GB
VRAM with model CPU offload, and generates portrait-format video at 576×832 @ 24 fps.

CogVideoX-5B-I2V is retained as an automatic fallback for environments where
LTX-Video fails to load (e.g. CUDA not available, download error).

Portrait images are center-cropped (not squished) to preserve face proportions.

All operations run locally without external services.
"""

import logging
import random
from collections.abc import Callable
from pathlib import Path

from services.models import COGVIDEO, LTX_VIDEO

log = logging.getLogger("hub.video")

# Lazy globals — loaded on demand, freed after use
_pipeline = None
_pipeline_type = None


def _get_device_and_dtype():
    """Return the best available device and dtype."""
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def _load_pipeline(model_id: str = "", hf_token: str = "") -> None:
    """Load the video generation pipeline, with CogVideoX as automatic fallback."""
    global _pipeline, _pipeline_type
    if _pipeline is not None:
        return

    device, dtype = _get_device_and_dtype()
    if not model_id:
        model_id = LTX_VIDEO.repo_id

    log.info("Loading video pipeline: %s on %s ...", model_id, device)
    kwargs: dict = {"torch_dtype": dtype}
    if hf_token:
        kwargs["token"] = hf_token

    try:
        if "LTX" in model_id or "ltx" in model_id.lower():
            from diffusers import LTXImageToVideoPipeline
            _pipeline = LTXImageToVideoPipeline.from_pretrained(model_id, **kwargs)
            _pipeline_type = "ltx"
        elif "CogVideo" in model_id:
            from diffusers import CogVideoXImageToVideoPipeline
            _pipeline = CogVideoXImageToVideoPipeline.from_pretrained(model_id, **kwargs)
            _pipeline_type = "cogvideo"
        else:
            from diffusers import AutoPipelineForVideoGeneration
            _pipeline = AutoPipelineForVideoGeneration.from_pretrained(model_id, **kwargs)
            _pipeline_type = "generic"

        if device == "cuda":
            _pipeline.enable_model_cpu_offload()
        else:
            _pipeline.to(device)

        log.info("Video pipeline loaded: %s", _pipeline_type)

    except Exception as exc:
        log.warning(
            "Primary video model %s failed to load: %s — falling back to CogVideoX",
            model_id, exc,
        )
        try:
            from diffusers import CogVideoXImageToVideoPipeline
            _pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
                COGVIDEO.repo_id, **kwargs
            )
            _pipeline_type = "cogvideo"
            if device == "cuda":
                _pipeline.enable_model_cpu_offload()
            else:
                _pipeline.to(device)
            log.info("CogVideoX fallback pipeline loaded.")
        except Exception as exc2:
            raise RuntimeError(
                f"All video pipeline options failed. Last error: {exc2}"
            ) from exc2


def _fit_portrait(image, target_w: int = 576, target_h: int = 832):
    """
    Resize then center-crop to target_w × target_h without distorting the image.
    Portrait source images remain portrait; landscape images are cropped on the sides.
    Both dimensions are divisible by 32 as required by LTX-Video's VAE.
    """
    from PIL import Image as _Image
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    image = image.resize((new_w, new_h), _Image.LANCZOS)
    left = (new_w - target_w) // 2
    top  = (new_h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))


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
    num_frames: int = 121,
    steps: int = 50,
    cfg: float = 3.0,
    seed: int = -1,
    progress_cb: Callable[[str], None] | None = None,
) -> Path | None:
    """
    Generate a portrait image-to-video clip.

    LTX-Video (primary): 576 × 832, 121 frames @ 24 fps ≈ 5 seconds.
    CogVideoX (fallback): 720 × 480, limited to 49 frames @ 8 fps.

    Provides a safe fallback: if LTX-Video fails to load (first-run download
    error, CUDA OOM), CogVideoX is attempted automatically.
    """
    import torch
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if progress_cb:
            progress_cb("Loading video model (first run downloads ~14 GB)...")
        _load_pipeline(model_id, hf_token)

        if progress_cb:
            progress_cb("Preparing source image...")

        source = Image.open(image_path).convert("RGB")

        if _pipeline_type == "ltx":
            # Center-crop to portrait 576×832 without distorting the face
            source = _fit_portrait(source, target_w=576, target_h=832)
            out_w, out_h, fps = 576, 832, 24
        else:
            # CogVideoX prefers landscape format
            source = source.resize((720, 480), Image.LANCZOS)
            out_w, out_h, fps = 720, 480, 8

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        if progress_cb:
            progress_cb("Generating video... this may take several minutes.")

        if _pipeline_type == "ltx":
            result = _pipeline(
                image=source,
                prompt=prompt,
                negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
                width=out_w,
                height=out_h,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            )
        else:
            result = _pipeline(
                image=source,
                prompt=prompt,
                negative_prompt="",
                num_frames=min(num_frames, 49),  # CogVideoX hard limit
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
            progress_cb("Video created successfully.")

        return out_path

    except Exception as exc:
        log.error("Video generation failed: %s", exc)
        if progress_cb:
            progress_cb(f"Error: {str(exc)[:200]}")
        return None

    finally:
        unload()
