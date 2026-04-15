"""
services/preflight.py — Service availability checks and first-run setup.

Checks performed on launch:
  1. ComfyUI running at configured URL
  2. PuLID-FLUX custom nodes installed in ComfyUI
  3. Wan2.1 I2V video nodes installed in ComfyUI
  4. ai-toolkit directory exists (offers to clone if not)
  5. HuggingFace token present (needed for Z-Image model download)

All checks are non-blocking; results are returned as a dict.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable

import requests

log = logging.getLogger("hub.preflight")


# ── individual checks ────────────────���─────────────────────────────��─────────

def check_comfyui(url: str, timeout: int = 4) -> dict:
    """Ping ComfyUI's /system_stats endpoint."""
    try:
        r = requests.get(f"{url.rstrip('/')}/system_stats", timeout=timeout)
        if r.status_code == 200:
            return {"ok": True, "detail": "Connected"}
        return {"ok": False, "detail": f"HTTP {r.status_code}"}
    except Exception as exc:
        return {"ok": False, "detail": f"Not reachable ({_short(exc)})"}


def check_pulid_nodes(url: str, timeout: int = 4) -> dict:
    """Check that PuLID-FLUX custom nodes are installed in ComfyUI."""
    required = ["PulidFluxModelLoader", "ApplyPulidFlux"]
    try:
        for node in required:
            r = requests.get(
                f"{url.rstrip('/')}/object_info/{node}", timeout=timeout
            )
            if r.status_code != 200 or node not in r.json():
                return {
                    "ok": False,
                    "detail": f"Missing node: {node}. Install ComfyUI-PuLID-Flux.",
                }
        return {"ok": True, "detail": "PuLID-FLUX nodes found"}
    except Exception as exc:
        return {"ok": False, "detail": f"Cannot check ({_short(exc)})"}


def check_wan_video_nodes(url: str, timeout: int = 4) -> dict:
    """Check that Wan2.1 I2V nodes are available in ComfyUI."""
    try:
        r = requests.get(
            f"{url.rstrip('/')}/object_info/WanImageToVideo", timeout=timeout
        )
        if r.status_code == 200 and "WanImageToVideo" in r.json():
            return {"ok": True, "detail": "Wan2.1 I2V nodes found"}
        return {
            "ok": False,
            "detail": "Missing WanImageToVideo node. Update ComfyUI or install Wan video nodes.",
        }
    except Exception as exc:
        return {"ok": False, "detail": f"Cannot check ({_short(exc)})"}


def check_ai_toolkit(path: str) -> dict:
    """Check that the ai-toolkit run.py exists."""
    if not path:
        return {"ok": False, "detail": "Path not configured"}
    p = Path(path) / "run.py"
    if p.exists():
        return {"ok": True, "detail": path}
    return {"ok": False, "detail": f"run.py not found in {path}"}


def check_hf_token(token: str) -> dict:
    if token and len(token) > 10:
        return {"ok": True, "detail": "Token present"}
    return {"ok": False, "detail": "No token — needed for model downloads"}


def run_all(settings) -> dict[str, dict]:
    """Run every check and return a combined status dict."""
    comfyui_url = settings.get("comfyui_url")
    return {
        "comfyui":         check_comfyui(comfyui_url),
        "pulid_nodes":     check_pulid_nodes(comfyui_url),
        "wan_video_nodes": check_wan_video_nodes(comfyui_url),
        "ai_toolkit":      check_ai_toolkit(settings.get("ai_toolkit_path")),
        "hf_token":        check_hf_token(settings.get("hf_token")),
    }


# ── ai-toolkit installer ───────���────────────────────��────────────────────────

def clone_ai_toolkit(
    dest: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """
    Clone ostris/ai-toolkit into dest and install its requirements
    into the current venv.  Returns (success, message).
    """

    def _emit(msg: str) -> None:
        log.info(msg)
        if progress_cb:
            progress_cb(msg)

    try:
        import git  # gitpython
    except ImportError:
        return False, "gitpython is not installed — run pip install gitpython"

    dest = Path(dest)

    # Step 1: clone
    if not (dest / ".git").exists():
        _emit(f"Cloning ai-toolkit into {dest} ...")
        try:
            git.Repo.clone_from(
                "https://github.com/ostris/ai-toolkit.git",
                str(dest),
                depth=1,
            )
            _emit("Clone complete.")
        except Exception as exc:
            return False, f"Git clone failed: {exc}"
    else:
        _emit("ai-toolkit already cloned, skipping.")

    # Step 2: submodules
    try:
        repo = git.Repo(str(dest))
        _emit("Initialising submodules...")
        repo.git.submodule("update", "--init", "--recursive")
        _emit("Submodules ready.")
    except Exception as exc:
        _emit(f"Submodule warning (non-fatal): {exc}")

    # Step 3: install requirements into current venv
    req_file = dest / "requirements.txt"
    if req_file.exists():
        _emit("Installing ai-toolkit requirements (this may take a few minutes)...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file),
             "--quiet", "--prefer-binary"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            _emit(f"pip install failed: {result.stderr[:300]}")
            return False, f"pip install failed — check your Python environment."
        _emit("Requirements installed.")
    else:
        _emit("No requirements.txt found in ai-toolkit — skipping pip install.")

    return True, str(dest)


# ── helper ─��────────────────────���──────────────────────────────────────────

def _short(exc: Exception) -> str:
    return str(exc)[:80]
