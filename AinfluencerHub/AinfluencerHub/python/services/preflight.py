"""
services/preflight.py — Service availability checks for native pipelines.

Checks performed on launch:
  1. GPU available with sufficient VRAM
  2. Required ML libraries importable (diffusers, peft, transformers)
  3. Required models cached locally
  4. HuggingFace token present (needed for gated model downloads)

All checks are non-blocking; results are returned as a dict.
"""

import logging

log = logging.getLogger("hub.preflight")

# Minimum VRAM (GB) each operation needs to run without OOM.
# Shown to users in the UI so they know why something can't run yet.
VRAM_REQUIREMENTS: dict[str, float] = {
    "generate_dataset":   8.0,   # SDXL + IP-Adapter
    "caption_joycaption": 10.0,  # JoyCaption 4-bit
    "train_lora":         14.0,  # SDXL UNet + LoRA + optimizer states
    "generate_studio":    8.0,   # SDXL + LoRA inference
    "animate_video":      16.0,  # Wan2.1 14B or CogVideoX-5B
}


# ── individual checks ────────────────────────────────────────────────────────

def check_gpu() -> dict:
    """Check GPU availability, VRAM, and per-operation feasibility."""
    try:
        from services.model_manager import check_gpu as _check_gpu
        result = _check_gpu()
        vram = result.get("vram_gb", 0)
        # Annotate which operations are feasible at the detected VRAM level.
        result["vram_requirements"] = VRAM_REQUIREMENTS
        result["feasible_ops"] = {
            op: vram >= req for op, req in VRAM_REQUIREMENTS.items()
        }
        if vram > 0 and vram < min(VRAM_REQUIREMENTS.values()):
            result["ok"] = False
            result["detail"] += (
                f" — {vram:.1f} GB VRAM is below the {min(VRAM_REQUIREMENTS.values()):.0f} GB "
                "minimum needed for any operation."
            )
        return result
    except Exception as exc:
        return {"ok": False, "detail": f"GPU check failed ({_short(exc)})"}


def check_ml_libraries() -> dict:
    """Check that required ML libraries are importable."""
    missing = []
    for lib in ["diffusers", "transformers", "peft", "accelerate", "torch"]:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)

    if missing:
        return {
            "ok": False,
            "detail": f"Missing libraries: {', '.join(missing)}. Run pip install -r requirements.txt",
        }
    return {"ok": True, "detail": "All ML libraries available"}


def check_models() -> dict:
    """Check that required models are cached locally."""
    try:
        from services.model_manager import get_all_model_status

        statuses = get_all_model_status()
        required_missing = [
            info["hf_id"]
            for info in statuses.values()
            if info["required"] and not info["cached"]
        ]
        if required_missing:
            return {
                "ok": False,
                "detail": f"{len(required_missing)} required model(s) not yet downloaded",
                "missing": required_missing,
            }
        return {"ok": True, "detail": "All required models cached"}
    except Exception as exc:
        return {"ok": False, "detail": f"Cannot check models ({_short(exc)})"}


def check_hf_token(token: str) -> dict:
    if token and len(token) > 10:
        return {"ok": True, "detail": "Token present"}
    return {"ok": False, "detail": "No token — needed for model downloads"}


def run_all(settings) -> dict[str, dict]:
    """Run every check and return a combined status dict."""
    return {
        "gpu":          check_gpu(),
        "ml_libraries": check_ml_libraries(),
        "models":       check_models(),
        "hf_token":     check_hf_token(settings.get("hf_token")),
    }


# ── helper ───────────────────────────────────────────────────────────────────

def _short(exc: Exception) -> str:
    return str(exc)[:80]
