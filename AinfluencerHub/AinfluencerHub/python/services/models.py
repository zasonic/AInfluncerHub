"""
services/models.py — Central registry for every HuggingFace model the app uses.

All pipelines must import their model specs from this module instead of
hard-coding repo IDs. This gives a single place to pin/bump model revisions,
roll back safely, and drive download progress in the UI.
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

JOY_CAPTIONER = ModelSpec(
    repo_id="fancyfeast/llama-joycaption-beta-one-hf-llava",
    purpose="Training-optimized captioning (Step 3)",
    size_gb=10.0,
    required=False,
)

WAN_VIDEO = ModelSpec(
    repo_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    purpose="Image-to-video animation (Step 5)",
    size_gb=28.0,
    required=False,
)

COGVIDEO = ModelSpec(
    repo_id="THUDM/CogVideoX-5b-I2V",
    purpose="Fallback image-to-video model (Step 5)",
    size_gb=10.0,
    required=False,
)


# ── Public API ───────────────────────────────────────────────────────────────

ALL: dict[str, ModelSpec] = {
    "sdxl_base":        SDXL_BASE,
    "ip_adapter":       IP_ADAPTER,
    "florence":         FLORENCE_CAPTIONER,
    "joycaption":       JOY_CAPTIONER,
    "wan_video":        WAN_VIDEO,
    "cogvideo":         COGVIDEO,
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
