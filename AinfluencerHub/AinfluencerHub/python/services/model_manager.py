"""
services/model_manager.py — Centralized model download and cache management.

Tracks which HuggingFace models are required, checks download status,
and provides download-with-progress for the UI.  All models use the
standard HuggingFace Hub cache (~/.cache/huggingface/).
"""

import logging
from collections.abc import Callable
from pathlib import Path

from services.models import manifest as _registry_manifest

log = logging.getLogger("hub.models")

# Kept for backwards-compatibility with any caller that expects a plain dict.
MODEL_MANIFEST: dict[str, dict] = {
    key: {
        "hf_id":    info["hf_id"],
        "size_gb":  info["size_gb"],
        "purpose":  info["purpose"],
        "required": info["required"],
    }
    for key, info in _registry_manifest().items()
}


def check_model_cached(hf_id: str) -> bool:
    """Check if a HuggingFace model is already cached locally."""
    try:
        from huggingface_hub import try_to_load_from_cache

        # Check for the config file — if present, model is cached
        result = try_to_load_from_cache(hf_id, "config.json")
        if result is not None and isinstance(result, (str, Path)):
            return True

        # Some models use model_index.json (diffusers pipelines)
        result = try_to_load_from_cache(hf_id, "model_index.json")
        if result is not None and isinstance(result, (str, Path)):
            return True

        return False
    except Exception:
        return False


def get_all_model_status() -> dict[str, dict]:
    """Return status of all models in the manifest."""
    status = {}
    for key, info in MODEL_MANIFEST.items():
        cached = check_model_cached(info["hf_id"])
        status[key] = {
            "hf_id":    info["hf_id"],
            "size_gb":  info["size_gb"],
            "purpose":  info["purpose"],
            "required": info["required"],
            "cached":   cached,
        }
    return status


def check_gpu() -> dict:
    """Check GPU availability and VRAM."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "ok":       False,
                "detail":   "No CUDA GPU detected",
                "vram_gb":  0,
                "device":   "cpu",
            }

        device = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        return {
            "ok":       True,
            "detail":   f"{device} ({vram:.1f} GB VRAM)",
            "vram_gb":  round(vram, 1),
            "device":   device,
        }
    except Exception as exc:
        return {
            "ok":       False,
            "detail":   f"GPU check failed: {str(exc)[:80]}",
            "vram_gb":  0,
            "device":   "unknown",
        }


def download_model(
    hf_id: str,
    hf_token: str = "",
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """
    Download a model from HuggingFace Hub.
    Returns (success, message).
    """
    try:
        from huggingface_hub import snapshot_download

        if progress_cb:
            progress_cb(f"Downloading {hf_id}...")

        kwargs: dict = {}
        if hf_token:
            kwargs["token"] = hf_token

        snapshot_download(hf_id, **kwargs)

        if progress_cb:
            progress_cb(f"Downloaded {hf_id} successfully.")
        return True, f"Model {hf_id} downloaded."

    except Exception as exc:
        msg = f"Download failed for {hf_id}: {str(exc)[:200]}"
        log.error(msg)
        return False, msg
