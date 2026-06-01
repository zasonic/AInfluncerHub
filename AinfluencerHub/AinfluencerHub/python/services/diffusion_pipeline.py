"""
services/diffusion_pipeline.py — Native image generation using HuggingFace diffusers.

Replaces ComfyUI for all image generation:
  - Face-consistent dataset generation via IP-Adapter-FaceID
  - Text-to-image with trained LoRA weights
  - Model management (auto-download, VRAM-aware loading/unloading)

All operations run locally on the user's GPU without external services.

v2.1 — IP-Adapter FaceID Plus V2:
  - generate_dataset() now attempts to load IP-Adapter FaceID Plus V2
    (h94/IP-Adapter-FaceID, ip-adapter-faceid-plusv2_sdxl.bin) which uses
    ArcFace identity embeddings + CLIP for face conditioning.  ArcFace is
    trained on face-recognition tasks, so the face *identity* (who the
    person is) is preserved across generated poses — not just the visual
    texture.  The existing _extract_face_embedding() helper was already
    calling InsightFace but its output was never passed to the pipeline;
    this wires it up properly.
  - Safe fallback: if the FaceID model is not downloaded or fails to load
    for any reason, the pipeline silently reverts to the original CLIP-only
    IP-Adapter Plus Face model.  No user action required.
  - Improved portrait defaults: guidance_scale raised from 4.0 → 7.0
    (SDXL sweet spot for photorealistic portraits), and a more comprehensive
    negative prompt replaces the sparse original.
"""

import logging
import random
import threading
from collections.abc import Callable
from pathlib import Path

from services.models import IP_ADAPTER, IP_ADAPTER_FACEID, SDXL_BASE

log = logging.getLogger("hub.diffusion")

# Portrait generation negative prompt — covers common SDXL artifacts
_PORTRAIT_NEG = (
    "blurry, low quality, watermark, text, deformed, ugly, bad anatomy, "
    "bad proportions, extra limbs, distorted face, low resolution, malformed, "
    "out of frame, poorly drawn face, mutation, gross proportions, disfigured"
)

# Lazy globals — loaded on demand, freed after use
_pipeline = None
_ip_adapter_loaded = False
_faceid_mode = False   # True when FaceID Plus V2 is active
_face_app = None


def _get_device_and_dtype():
    """Return the best available device and dtype."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _load_base_pipeline(hf_token: str = ""):
    """Load the SDXL base pipeline."""
    global _pipeline
    if _pipeline is not None:
        return

    from diffusers import StableDiffusionXLPipeline

    device, dtype = _get_device_and_dtype()
    log.info("Loading SDXL pipeline on %s ...", device)

    kwargs: dict = {"torch_dtype": dtype, "variant": "fp16", "use_safetensors": True}
    if hf_token:
        kwargs["token"] = hf_token

    _pipeline = StableDiffusionXLPipeline.from_pretrained(
        SDXL_BASE.repo_id, revision=SDXL_BASE.revision, **kwargs
    )
    _pipeline.to(device)

    # Enable memory optimizations
    if device == "cuda":
        try:
            _pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers optional

    log.info("SDXL pipeline loaded.")


def _load_ip_adapter(hf_token: str = ""):
    """Load IP-Adapter Plus Face (CLIP-based). Called as fallback when FaceID is unavailable."""
    global _ip_adapter_loaded, _faceid_mode
    if _ip_adapter_loaded:
        return

    _load_base_pipeline(hf_token)

    log.info("Loading IP-Adapter Plus Face (CLIP)...")
    _pipeline.load_ip_adapter(
        IP_ADAPTER.repo_id,
        subfolder=IP_ADAPTER.subfolder,
        weight_name=IP_ADAPTER.weight_name,
        token=hf_token or None,
    )
    _pipeline.set_ip_adapter_scale(0.7)
    _ip_adapter_loaded = True
    _faceid_mode = False
    log.info("IP-Adapter Plus Face loaded.")


def _load_faceid_adapter(hf_token: str = "") -> bool:
    """
    Try to load IP-Adapter FaceID Plus V2 (ArcFace + CLIP).

    FaceID Plus V2 uses ArcFace identity embeddings alongside CLIP image
    features.  ArcFace is a face-recognition embedding trained to distinguish
    *who* a person is, not just what they look like — so face identity is
    preserved across varied poses and lighting more reliably than CLIP alone.

    Returns True if FaceID loaded successfully, False if not (caller must
    then call _load_ip_adapter() for the CLIP-only fallback).
    """
    global _ip_adapter_loaded, _faceid_mode
    if _ip_adapter_loaded:
        return _faceid_mode

    _load_base_pipeline(hf_token)

    try:
        log.info("Loading IP-Adapter FaceID Plus V2 (ArcFace+CLIP)...")
        _pipeline.load_ip_adapter(
            IP_ADAPTER_FACEID.repo_id,
            subfolder=None,
            weight_name=IP_ADAPTER_FACEID.weight_name,
            image_encoder_folder="models/image_encoder",
            token=hf_token or None,
        )
        _pipeline.set_ip_adapter_scale(0.5)
        _ip_adapter_loaded = True
        _faceid_mode = True
        log.info("IP-Adapter FaceID Plus V2 loaded.")
        return True
    except Exception as exc:
        log.warning(
            "FaceID Plus V2 not available (%s) — falling back to IP-Adapter Plus Face.",
            exc,
        )
        return False


def _get_face_app():
    """Initialize InsightFace for face embedding extraction."""
    global _face_app
    if _face_app is not None:
        return _face_app

    from insightface.app import FaceAnalysis

    _face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    _face_app.prepare(ctx_id=0, det_size=(640, 640))
    log.info("InsightFace face analysis loaded.")
    return _face_app


def _extract_faceid_inputs(image_path: Path):
    """
    Extract ArcFace identity embedding + aligned face image for FaceID Plus V2.

    Returns (faceid_embeds, face_pil):
      - faceid_embeds: torch.Tensor shape [1, 512] — ArcFace identity vector
      - face_pil: PIL.Image — aligned face crop for the CLIP encoder component
    """
    import cv2
    import torch
    from PIL import Image

    app = _get_face_app()
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = app.get(img)
    if not faces:
        raise ValueError(f"No face detected in {image_path.name}")

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # ArcFace identity embedding tensor — shape [1, 512]
    faceid_embeds = torch.from_numpy(face.normed_embedding).unsqueeze(0)

    # Aligned face crop for the CLIP component of FaceID Plus
    try:
        from insightface.utils import face_align
        aligned_bgr = face_align.norm_crop(img, landmark=face.kps, image_size=224)
        face_pil = Image.fromarray(cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB))
    except Exception:
        # Fallback: simple bbox crop if keypoints unavailable
        x1, y1, x2, y2 = (max(0, int(c)) for c in face.bbox)
        crop_bgr = img[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else img
        face_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

    return faceid_embeds, face_pil


def _prepare_face_image(image_path: Path):
    """Load a face image as PIL for CLIP-only IP-Adapter (fallback path)."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    return img


def unload() -> None:
    """Release all GPU memory."""
    global _pipeline, _ip_adapter_loaded, _faceid_mode, _face_app

    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        _ip_adapter_loaded = False
        _faceid_mode = False

    if _face_app is not None:
        del _face_app
        _face_app = None

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("Diffusion pipeline unloaded.")


# ── Public API ───────────────────────────────────────────────────────────────


def generate_dataset(
    reference_image: Path,
    prompts: list[str],
    trigger_word: str,
    output_dir: Path,
    hf_token: str = "",
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[Path]:
    """
    Generate face-consistent dataset images.

    Attempts IP-Adapter FaceID Plus V2 (ArcFace + CLIP) for stronger face
    identity, falling back to IP-Adapter Plus Face (CLIP only) if FaceID
    is not downloaded or fails to load.
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(prompts)
    paths: list[Path] = []

    try:
        # Try FaceID Plus V2; fall back to CLIP-only Plus Face if unavailable
        use_faceid = _load_faceid_adapter(hf_token)
        if not use_faceid:
            _load_ip_adapter(hf_token)

        if use_faceid:
            faceid_embeds, face_pil = _extract_faceid_inputs(reference_image)
        else:
            face_pil = _prepare_face_image(reference_image)

        for i, prompt in enumerate(prompts):
            if cancel_event and cancel_event.is_set():
                break

            full_prompt = f"{trigger_word}, {prompt}" if trigger_word else prompt
            label = prompt[:50] + "..." if len(prompt) > 50 else prompt
            if progress_cb:
                progress_cb(i, total, f"Generating {i + 1}/{total}: {label}")

            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=_pipeline.device).manual_seed(seed)

            try:
                if use_faceid:
                    result = _pipeline(
                        prompt=full_prompt,
                        negative_prompt=_PORTRAIT_NEG,
                        ip_adapter_image=face_pil,
                        ip_adapter_image_embeds=[faceid_embeds],
                        num_inference_steps=20,
                        guidance_scale=7.0,
                        width=832,
                        height=1216,
                        generator=generator,
                    )
                else:
                    result = _pipeline(
                        prompt=full_prompt,
                        negative_prompt=_PORTRAIT_NEG,
                        ip_adapter_image=face_pil,
                        num_inference_steps=20,
                        guidance_scale=7.0,
                        width=832,
                        height=1216,
                        generator=generator,
                    )
                image = result.images[0]
                out_path = output_dir / f"dataset_{len(paths) + 1:03d}.jpg"
                image.save(out_path, quality=95)
                paths.append(out_path)

            except Exception as exc:
                log.error("Generation failed for prompt %d: %s", i, exc)
                continue

        if progress_cb:
            progress_cb(total, total, f"Generated {len(paths)} images.")

    finally:
        unload()

    return paths


def generate_image(
    positive_prompt: str,
    negative_prompt: str = _PORTRAIT_NEG,
    lora_path: str = "",
    lora_strength: float = 0.85,
    width: int = 832,
    height: int = 1216,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = -1,
    output_dir: Path | None = None,
    hf_token: str = "",
    progress_cb: Callable[[str], None] | None = None,
) -> list[Path]:
    """
    Generate an image using SDXL with optional LoRA.
    Returns list of output file paths.
    """
    import torch

    if output_dir is None:
        output_dir = Path("output") / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _load_base_pipeline(hf_token)

        if progress_cb:
            progress_cb("Loading pipeline...")

        # Load LoRA if provided
        if lora_path and Path(lora_path).exists():
            if progress_cb:
                progress_cb("Loading LoRA weights...")
            _pipeline.load_lora_weights(
                lora_path,
                adapter_name="user_lora",
            )
            _pipeline.set_adapters(["user_lora"], adapter_weights=[lora_strength])
            log.info("LoRA loaded: %s (strength=%.2f)", lora_path, lora_strength)

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device=_pipeline.device).manual_seed(seed)

        if progress_cb:
            progress_cb("Generating image...")

        result = _pipeline(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            generator=generator,
        )

        paths: list[Path] = []
        for idx, image in enumerate(result.images):
            out_path = output_dir / f"gen_{seed}_{idx}.png"
            image.save(out_path)
            paths.append(out_path)

        if progress_cb:
            progress_cb(f"Generated {len(paths)} image(s).")

        return paths

    finally:
        # Unload LoRA weights to avoid contaminating next generation
        if lora_path and _pipeline is not None:
            try:
                _pipeline.unload_lora_weights()
            except Exception:
                pass
        unload()
