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


# ── Specs ────────────────────────────────────────────────────────────────────────────
SDXL_BASE = ModelSpec(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    purpose="Image generation + LoRA training base (Steps 2, 4, 5)",
    size_gb=6.5,
    required=True,
)

# ── IP-Adapter-FaceID-PlusV2 (primary) ───────────────────────────────────────────
# Uses face recognition embeddings (InsightFace/ArcFace) instead of CLIP image
# embeddings. Combines identity preservation (face ID) with structure guidance
# for significantly better face consistency across varied poses and backgrounds.
# Upgrade over ip-adapter-plus-face which relied on generic CLIP patch embeddings.
IP_ADAPTER_FACEID_V2 = ModelSpec(
    repo_id="h94/IP-Adapter-FaceID",
    purpose="Face-consistent dataset generation via recognition embeddings (Step 2)",
    size_gb=0.8,
    required=True,
    subfolder=None,
    weight_name="ip-adapter-faceid-plusv2_sdxl.bin",
)

# Companion LoRA — boosts face ID consistency when loaded alongside the adapter.
IP_ADAPTER_FACEID_LORA = ModelSpec(
    repo_id="h94/IP-Adapter-FaceID",
    purpose="ID-consistency LoRA for FaceID-PlusV2 (Step 2)",
    size_gb=0.2,
    required=False,
    subfolder=None,
    weight_name="ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
)

# ── IP-Adapter Plus Face SDXL (fallback) ─────────────────────────────────────────
# CLIP-based face adapter. Used when InsightFace fails to detect a face in the
# reference image, or when FaceID-PlusV2 weights cannot be loaded.
IP_ADAPTER_PLUS_FACE = ModelSpec(
    repo_id="h94/IP-Adapter",
    purpose="Face-consistent generation fallback (CLIP-based, Step 2)",
    size_gb=1.8,
    required=False,
    subfolder="sdxl_models",
    weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
)

# Legacy alias kept so any external code importing IP_ADAPTER still resolves.
IP_ADAPTER = IP_ADAPTER_PLUS_FACE

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


# ── Public API ─────────────────────────────────────────────────────────────────────────────
ALL: dict[str, ModelSpec] = {
    "sdxl_base":              SDXL_BASE,
    "ip_adapter_faceid_v2":   IP_ADAPTER_FACEID_V2,
    "ip_adapter_faceid_lora": IP_ADAPTER_FACEID_LORA,
    "ip_adapter_plus_face":   IP_ADAPTER_PLUS_FACE,
    "florence":               FLORENCE_CAPTIONER,
    "joycaption":             JOY_CAPTIONER,
    "wan_video":              WAN_VIDEO,
    "cogvideo":               COGVIDEO,
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
