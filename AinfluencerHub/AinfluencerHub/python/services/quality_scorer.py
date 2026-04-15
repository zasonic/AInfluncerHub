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

    Returns list of dicts: {path, face_score, aesthetic_score, passed}
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

        results.append({
            "path": str(img_path),
            "filename": img_path.name,
            "face_score": round(face_score, 3),
            "aesthetic_score": round(aesthetic_score, 3),
            "passed": face_score >= FACE_THRESHOLD and aesthetic_score >= AESTHETIC_THRESHOLD,
        })

        if progress_cb:
            progress_cb(i + 1, total, f"Scoring {i + 1}/{total}")

    unload_models()
    return results
