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


# ── individual checks ────────────────────────────────────────────────────────

def check_gpu() -> dict:
    """Check GPU availability and VRAM."""
    try:
        from services.model_manager import check_gpu as _check_gpu
        return _check_gpu()
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


def check_disk_space(output_dir: str = "") -> dict:
    """Check that enough disk space is available for models and training output."""
    import shutil
    try:
        path = output_dir if output_dir else "."
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < 5:
            return {
                "ok": False,
                "detail": f"Only {free_gb:.1f} GB free. Training and models need at least 10 GB.",
                "free_gb": round(free_gb, 1),
            }
        if free_gb < 15:
            return {
                "ok": True,
                "detail": f"{free_gb:.1f} GB free. Consider freeing space — models can use 10-30 GB.",
                "free_gb": round(free_gb, 1),
            }
        return {"ok": True, "detail": f"{free_gb:.1f} GB free", "free_gb": round(free_gb, 1)}
    except Exception as exc:
        return {"ok": True, "detail": f"Could not check disk space ({_short(exc)})"}


def run_all(settings) -> dict[str, dict]:
    """Run every check and return a combined status dict."""
    return {
        "gpu":          check_gpu(),
        "ml_libraries": check_ml_libraries(),
        "models":       check_models(),
        "hf_token":     check_hf_token(settings.get("hf_token")),
        "disk_space":   check_disk_space(settings.get("output_dir", "")),
    }


# ── helper ───────────────────────────────────────────────────────────────────

def _short(exc: Exception) -> str:
    return str(exc)[:80]
