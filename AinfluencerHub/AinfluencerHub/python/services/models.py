"""services/models.py — Central registry for every HuggingFace model the app uses.

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
    size_gb=10.0,  # ~10 GB at 4-bit quantization; ~17 GB at bf16
    required=False,
)

# I2V primary — fits in 16 GB VRAM with enable_model_cpu_offload().
# The previous Wan2.1-T2V model was text-to-video only and crashed at
# runtime when called with image= (TypeError on WanPipeline.__call__).
WAN_VIDEO = ModelSpec(
    repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    purpose="Image-to-video animation — optional high-quality (Step 5)",
    size_gb=28.0,
    required=False,
)

# Default I2V model used when no model_id is supplied to generate_video().
# 10 GB fits comfortably in 16 GB VRAM without CPU offloading.
COGVIDEO = ModelSpec(
    repo_id="THUDM/CogVideoX-5b-I2V",
    purpose="Image-to-video animation — default for 16 GB VRAM (Step 5)",
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
