"""
services/fal_generator.py — Dataset image generation via fal.ai.

Uses FLUX.1-Kontext-dev to produce face-consistent variations of a
reference photo: different angles, poses, outfits, and backgrounds.

fal.ai is a cloud GPU service (~$0.025 per image).  A 25-image dataset
costs about $0.63.  Requires a FAL_API_KEY from https://fal.ai.
"""

import base64
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Callable

log = logging.getLogger("hub.fal_generator")

FAL_MODEL = "fal-ai/flux-kontext-dev"


def _image_to_data_url(path: Path) -> str:
    """Convert an image file to a base-64 data URL."""
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp"}.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


def generate_dataset(
    reference_image: Path,
    prompts: list[str],
    trigger_word: str,
    output_dir: Path,
    api_key: str,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_event=None,
) -> list[Path]:
    """
    Generate one image per prompt using FLUX.1-Kontext-dev.

    Args:
        reference_image: Path to the seed photo.
        prompts:         List of variation prompts (with TRIGGER placeholder).
        trigger_word:    Replaces TRIGGER in each prompt.
        output_dir:      Where to save generated images.
        api_key:         fal.ai API key.
        progress_cb:     Called as (completed, total, current_label).
        cancel_event:    threading.Event; checked between generations.

    Returns:
        List of Paths to saved images.
    """
    try:
        import fal_client
    except ImportError as exc:
        raise RuntimeError(
            "fal-client is not installed. "
            "Run: pip install fal-client"
        ) from exc

    os.environ["FAL_KEY"] = api_key

    image_url = _image_to_data_url(reference_image)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []

    total = len(prompts)
    for i, prompt_template in enumerate(prompts, 1):
        if cancel_event and cancel_event.is_set():
            log.info("Dataset generation cancelled at image %d/%d", i, total)
            break

        prompt = prompt_template.replace("TRIGGER", trigger_word)
        label = prompt[:60] + "..." if len(prompt) > 60 else prompt

        if progress_cb:
            progress_cb(i - 1, total, f"Generating {i}/{total}: {label}")

        try:
            result = fal_client.subscribe(
                FAL_MODEL,
                arguments={
                    "prompt":       prompt,
                    "image_url":    image_url,
                    "num_images":   1,
                    "image_size":   "portrait_4_3",
                    "guidance_scale": 3.5,
                    "num_inference_steps": 28,
                    "output_format": "jpeg",
                },
            )

            # fal result shape: {"images": [{"url": "...", ...}]}
            images = result.get("images") or []
            if not images:
                log.warning("No images returned for prompt %d", i)
                continue

            img_url = images[0].get("url", "")
            if img_url.startswith("data:"):
                # inline base64
                header, b64 = img_url.split(",", 1)
                img_bytes = base64.b64decode(b64)
            else:
                import requests
                r = requests.get(img_url, timeout=60)
                r.raise_for_status()
                img_bytes = r.content

            out_path = output_dir / f"dataset_{i:03d}.jpg"
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            results.append(out_path)
            log.info("Saved %s", out_path.name)

        except Exception as exc:
            log.error("Generation failed for prompt %d: %s", i, exc)
            # continue to next prompt rather than aborting entire batch

    if progress_cb:
        progress_cb(len(results), total, f"Done — {len(results)} images saved")

    return results
