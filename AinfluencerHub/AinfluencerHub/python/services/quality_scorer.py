"""
services/quality_scorer.py — Dataset image quality scoring.

Uses pyiqa to score images for face quality and aesthetics before training.
Rejects blurry, badly-lit, or poorly composed images that would degrade
LoRA quality.

Metrics:
  - topiq_nr-face: face-specific no-reference quality (0–1, higher = better)
  - nima: Neural Image Assessment for aesthetic quality (1–10, higher = better)
"""

import logging
from pathlib import Path

import torch

log = logging.getLogger("hub.quality")

_face_metric = None
_aesthetic_metric = None

# Thresholds — images below these are flagged as low quality
FACE_THRESHOLD = 0.4
AESTHETIC_THRESHOLD = 4.5


def load_models() -> None:
    global _face_metric, _aesthetic_metric
    import pyiqa

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading quality scoring models on %s", device)
    _face_metric = pyiqa.create_metric("topiq_nr-face", device=device)
    _aesthetic_metric = pyiqa.create_metric("nima", device=device)


def unload_models() -> None:
    global _face_metric, _aesthetic_metric
    _face_metric = None
    _aesthetic_metric = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Quality scoring models unloaded.")


def score_images(
    image_paths: list[Path],
    progress_cb=None,
) -> list[dict]:
    """Score images for face quality and aesthetics.

    Returns list of dicts:
      {path, filename, face_score, aesthetic_score, passed, reject_reason}

    reject_reason is None for passing images and a plain-English string for
    failing ones so the UI can explain why a photo was rejected.
    """
    if _face_metric is None:
        load_models()

    results = []
    total = len(image_paths)

    for i, img_path in enumerate(image_paths):
        try:
            face_score = float(_face_metric(str(img_path)).item())
            aesthetic_score = float(_aesthetic_metric(str(img_path)).item())
        except Exception as exc:
            log.warning("Failed to score %s: %s", img_path.name, exc)
            face_score = 0.0
            aesthetic_score = 0.0

        face_ok = face_score >= FACE_THRESHOLD
        aesthetic_ok = aesthetic_score >= AESTHETIC_THRESHOLD
        passed = face_ok and aesthetic_ok

        if not passed:
            if not face_ok and not aesthetic_ok:
                reject_reason = (
                    f"face quality too low ({face_score:.2f} < {FACE_THRESHOLD}) "
                    f"and aesthetic quality too low ({aesthetic_score:.1f} < {AESTHETIC_THRESHOLD})"
                )
            elif not face_ok:
                reject_reason = (
                    f"face quality too low ({face_score:.2f} < {FACE_THRESHOLD}) — "
                    "try a clearer, well-lit photo where the face is the main subject"
                )
            else:
                reject_reason = (
                    f"aesthetic quality too low ({aesthetic_score:.1f} < {AESTHETIC_THRESHOLD}) — "
                    "try a sharper, better-composed photo"
                )
        else:
            reject_reason = None

        results.append({
            "path": str(img_path),
            "filename": img_path.name,
            "face_score": round(face_score, 3),
            "aesthetic_score": round(aesthetic_score, 3),
            "passed": passed,
            "reject_reason": reject_reason,
        })

        if progress_cb:
            progress_cb(i + 1, total, f"Scoring {i + 1}/{total}")

    unload_models()
    return results
