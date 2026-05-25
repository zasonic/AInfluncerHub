"""
services/models.py — Central registry for every HuggingFace model the app uses.

All pipelines must import their model specs from this module instead of
hard-coding repo IDs. This gives a single place to pin/bump model revisions,
roll back safely, and drive download progress in the UI.

v2.1 model upgrades:
  JoyCaption Alpha Two — replaces beta-one; richer, more accurate training
    captions with better instruction following (Oct 2024, ~35k HF downloads).
    Same LLaVA/Llama-3 architecture, drop-in compatible with the same loader.
  SVD-XT — Stable Video Diffusion XT replaces Wan2.1 as the default video
    model. Wan2.1 requires 28 GB VRAM (impractical for consumer GPUs);
    SVD-XT delivers smooth 25-frame animations on ~8 GB VRAM.
    CogVideoX remains as the 10 GB fallback; Wan2.1 stays available for
    high-end users who explicitly select it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """A pinned HuggingFace model reference."""

    repo_id:     str
    purpose:     str
    size_gb:     float
    required:    bool = True
    revision:    str | None = None   # HF ref (branch / tag / commit); None = main
    subfolder:   str | None = None   # for sub-artifacts inside a repo
    weight_name: str | None = None   # specific weight file (e.g. IP-Adapter)


# ── Specs ────────────────────────────────────────────────────────────────────

SDXL_BASE = ModelSpec(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    purpose="Image generation + LoRA training base (Steps 2, 4, 5)",
    size_gb=6.5,
    required=True,
)

IP_ADAPTER = ModelSpec(
    repo_id="h94/IP-Adapter",
    purpose="Face-consistent dataset generation (Step 2)",
    size_gb=1.8,
    required=True,
    subfolder="sdxl_models",
    weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
)

FLORENCE_CAPTIONER = ModelSpec(
    repo_id="microsoft/Florence-2-large",
    purpose="Fast image captioning (Step 3)",
    size_gb=4.0,
    required=True,
)

# Alpha Two (Oct 2024) — improved accuracy and instruction following over beta-one.
# Same Llama-3.2 vision architecture; 4-bit quantization keeps VRAM at ~10 GB.
JOY_CAPTIONER = ModelSpec(
    repo_id="fancyfeast/llama-joycaption-alpha-two-hf-llava",
    purpose="Training-optimized captioning (Step 3)",
    size_gb=10.0,
    required=False,
)

# SVD-XT: primary video model (~8 GB VRAM, runs on most consumer GPUs).
# Generates 25 frames at 7 fps (~3.5 s), image-conditioned (no text prompt).
SVD_XT = ModelSpec(
    repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
    purpose="Image-to-video animation, primary — 8 GB VRAM (Step 5)",
    size_gb=8.0,
    required=False,
)

# CogVideoX: 10 GB fallback when SVD-XT fails or is explicitly selected.
COGVIDEO = ModelSpec(
    repo_id="THUDM/CogVideoX-5b-I2V",
    purpose="Image-to-video animation, fallback — 10 GB VRAM (Step 5)",
    size_gb=10.0,
    required=False,
)

# Wan2.1: high-end option for users with 28+ GB VRAM.
WAN_VIDEO = ModelSpec(
    repo_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    purpose="Image-to-video animation, high-end — 28 GB VRAM (Step 5)",
    size_gb=28.0,
    required=False,
)


# ── Public API ───────────────────────────────────────────────────────────────

ALL: dict[str, ModelSpec] = {
    "sdxl_base":        SDXL_BASE,
    "ip_adapter":       IP_ADAPTER,
    "florence":         FLORENCE_CAPTIONER,
    "joycaption":       JOY_CAPTIONER,
    "svd_xt":           SVD_XT,
    "cogvideo":         COGVIDEO,
    "wan_video":        WAN_VIDEO,
}


def get(key: str) -> ModelSpec:
    """Look up a model spec by registry key. Raises KeyError if absent."""
    return ALL[key]


def manifest() -> dict[str, dict]:
    """Serializable view of all specs, used by /api/models/status."""
    return {
        key: {
            "hf_id":    spec.repo_id,
            "size_gb":  spec.size_gb,
            "purpose":  spec.purpose,
            "required": spec.required,
            "revision": spec.revision,
        }
        for key, spec in ALL.items()
    }
