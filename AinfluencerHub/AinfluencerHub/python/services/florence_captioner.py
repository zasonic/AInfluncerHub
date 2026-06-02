"""
services/florence_captioner.py — Local Florence-2 captioning with PromptGen v2.0.

Primary model: MiaoshouAI/Florence-2-large-PromptGen-v2.0
  Adds the MIXED_CAPTION task which produces combined tag + natural-language
  output — richer than DETAILED_CAPTION and specifically tuned for diffusion
  model LoRA training.  If the primary model cannot be loaded (network error,
  gated access, missing cache, etc.) the code falls back to
  microsoft/Florence-2-large with DETAILED_CAPTION so captioning always
  completes without user intervention.

Fallback model: microsoft/Florence-2-large
  Standard Florence-2-large.  Used automatically when PromptGen-v2.0 is
  unavailable.  Produces DETAILED_CAPTION output instead of MIXED_CAPTION.

Both models are ~4 GB and run on GPU.  The loaded model is kept in memory
for the duration of a captioning session then unloaded to free VRAM for
the LoRA training step.
"""

import logging
from collections.abc import Callable
from pathlib import Path

from services.models import FLORENCE_CAPTIONER, FLORENCE_CAPTIONER_FALLBACK

log = logging.getLogger("hub.captioner")

# Lazy globals — populated on first use, cleared after session
_model = None
_processor = None
_device = None
_active_repo_id: str | None = None  # tracks which model is currently loaded


def _load_model(hf_token: str | None = None) -> None:
    global _model, _processor, _device, _active_repo_id
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    load_kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if _device == "cuda" else torch.float32,
    }
    if hf_token:
        load_kwargs["token"] = hf_token

    # Try PromptGen-v2.0 first; fall back to Florence-2-large on any error.
    for spec in (FLORENCE_CAPTIONER, FLORENCE_CAPTIONER_FALLBACK):
        try:
            log.info("Loading %s on %s ...", spec.repo_id, _device)
            _model = AutoModelForCausalLM.from_pretrained(
                spec.repo_id, **load_kwargs
            ).to(_device)
            _processor = AutoProcessor.from_pretrained(
                spec.repo_id,
                trust_remote_code=True,
                token=hf_token or None,
            )
            _active_repo_id = spec.repo_id
            log.info("%s loaded successfully.", spec.repo_id)
            return
        except Exception as exc:
            log.warning(
                "Could not load %s: %s — %s",
                spec.repo_id,
                exc,
                "trying fallback." if spec is FLORENCE_CAPTIONER else "no further fallbacks.",
            )
            _model = None
            _processor = None

    raise RuntimeError(
        "No captioner model could be loaded. "
        "Check your HuggingFace token and internet connection."
    )


def _active_caption_task() -> str:
    """
    Return the best task token for the loaded model.
    PromptGen-v2.0 exposes MIXED_CAPTION (tags + natural language), which
    produces richer training captions than DETAILED_CAPTION.
    Falls back to DETAILED_CAPTION for vanilla Florence-2-large.
    """
    if _active_repo_id and "PromptGen" in _active_repo_id:
        return "MIXED_CAPTION"
    return "DETAILED_CAPTION"


def unload_model() -> None:
    """Release GPU memory after captioning is complete."""
    global _model, _processor, _active_repo_id
    if _model is not None:
        import torch
        del _model, _processor
        _model = None
        _processor = None
        _active_repo_id = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Florence captioner unloaded.")


def caption_image(
    image_path: Path,
    trigger_word: str = "",
    detail: str | None = None,
    hf_token: str | None = None,
) -> str:
    """
    Generate a caption for a single image.

    Args:
        image_path:   Path to the image file.
        trigger_word: Prepended to the caption if non-empty.
        detail:       Task name override (e.g. "DETAILED_CAPTION").
                      When None, auto-selects the best task for the loaded model
                      (MIXED_CAPTION for PromptGen-v2.0, else DETAILED_CAPTION).
        hf_token:     HuggingFace token for model download.

    Returns:
        Caption string formatted for SDXL / FLUX LoRA training.
    """
    import torch
    from PIL import Image

    _load_model(hf_token)

    task_name = detail if detail is not None else _active_caption_task()
    task_prompt = f"<{task_name}>"

    img = Image.open(image_path).convert("RGB")

    inputs = _processor(
        text=task_prompt,
        images=img,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        generated_ids = _model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
        )

    raw = _processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    caption = _processor.post_process_generation(
        raw, task=task_prompt, image_size=(img.width, img.height)
    )
    if isinstance(caption, dict):
        caption = caption.get(task_prompt, "")
    caption = str(caption).strip()

    if trigger_word:
        caption = f"{trigger_word}, {caption}"

    return caption


def caption_batch(
    image_paths: list[Path],
    trigger_word: str = "",
    hf_token: str | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_event=None,
    captions_dir: Path | None = None,
) -> dict[Path, str]:
    """
    Caption a list of images.  Saves a .txt file alongside each image
    in captions_dir (or next to the image if captions_dir is None).

    Returns mapping of image path -> caption string.
    """
    _load_model(hf_token)
    results: dict[Path, str] = {}
    total = len(image_paths)

    model_label = (
        "PromptGen-v2.0" if (_active_repo_id and "PromptGen" in _active_repo_id)
        else "Florence-2-large"
    )
    if progress_cb and total > 0:
        progress_cb(0, total, f"Using {model_label} — {total} images queued")

    for i, img_path in enumerate(image_paths, 1):
        if cancel_event and cancel_event.is_set():
            break
        if progress_cb:
            progress_cb(i - 1, total, f"Captioning {i}/{total}: {img_path.name}")
        try:
            cap = caption_image(img_path, trigger_word=trigger_word, hf_token=hf_token)
            results[img_path] = cap

            if captions_dir:
                captions_dir.mkdir(parents=True, exist_ok=True)
                txt_path = captions_dir / (img_path.stem + ".txt")
            else:
                txt_path = img_path.with_suffix(".txt")
            txt_path.write_text(cap, encoding="utf-8")

        except Exception as exc:
            log.error("Captioning failed for %s: %s", img_path.name, exc)
            results[img_path] = ""

    if progress_cb:
        progress_cb(
            len(results), total,
            f"Captioning complete — {len(results)}/{total} images captioned with {model_label}."
        )

    unload_model()
    return results
