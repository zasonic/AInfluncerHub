"""
services/florence_captioner.py — Local Florence2-large captioning.

Florence2-large is a ~4 GB model that runs on GPU.  It is loaded once
and kept in memory for the duration of a captioning session, then
unloaded to free VRAM for training.

Caption output is formatted for the target architecture.  For Z-Image
Turbo / Flux models this is rich natural-language with trigger word.

v2 improvements:
  - Default task changed to MORE_DETAILED_CAPTION for richer attribute
    descriptions (hair, clothing, expression, environment). Richer captions
    are consistently correlated with sharper face identity in LoRA training.
  - num_beams raised from 3 to 5: more coherent, grammatically complete
    captions for the same GPU cost per image.
  - max_new_tokens raised from 256 to 512: headroom for long captions;
    the CLIP tokenizer truncates to 77 tokens at training time so there
    is no text-encoder overflow risk.
  - _clean_caption() strips generic Florence2 prefixes ('The image shows',
    'A photo of', etc.) that dilute the 77-token CLIP budget and push
    meaningful tokens past the truncation boundary.
"""

import logging
import re
from collections.abc import Callable
from pathlib import Path

from services.models import FLORENCE_CAPTIONER

log = logging.getLogger("hub.captioner")

# Lazy globals — loaded on first use, cleared after session
_model = None
_processor = None
_device = None

# ── Caption cleanup ──────────────────────────────────────────────────────────

# Regex for common Florence2 filler prefixes that waste CLIP token budget.
# These phrases add no training signal and push meaningful tokens past the
# 77-token truncation boundary.
_CAPTION_PREFIXES = re.compile(
    r"^(the image (shows?|depicts?|features?|captures?|displays?|presents?|contains?)|?"
    r"a (close-?up |full-?body |portrait )?(photo(graph)?|picture|image|shot) (of|showing|featuring|depicting|with)|?"
    r"this (image|photo(graph)?|picture|shot) (shows?|depicts?|features?|captures?|contains?)|?"
    r"(here|here's|shown here|displayed here)[,: ]+|?"
    r"in (this|the) (image|photo(graph)?|picture|shot)[,: ]+)\s*",
    re.IGNORECASE,
)


def _clean_caption(text: str) -> str:
    """Strip generic Florence2 filler prefixes and capitalise the result."""
    cleaned = _CAPTION_PREFIXES.sub("", text.strip()).strip(" ,.").strip()
    if not cleaned:
        return text.strip()
    return cleaned[0].upper() + cleaned[1:]


def _load_model(hf_token: str | None = None) -> None:
    global _model, _processor, _device
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading Florence2-large on %s ...", _device)

    kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype":       torch.float16 if _device == "cuda" else torch.float32,
    }
    if hf_token:
        kwargs["token"] = hf_token

    _model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_CAPTIONER.repo_id, **kwargs
    ).to(_device)
    _processor = AutoProcessor.from_pretrained(
        FLORENCE_CAPTIONER.repo_id,
        trust_remote_code=True,
        token=hf_token or None,
    )
    log.info("Florence2 loaded.")


def unload_model() -> None:
    """Release GPU memory after captioning is complete."""
    global _model, _processor
    if _model is not None:
        import torch
        del _model, _processor
        _model = None
        _processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Florence2 unloaded.")


def caption_image(
    image_path: Path,
    trigger_word: str = "",
    detail: str = "MORE_DETAILED_CAPTION",
    hf_token: str | None = None,
) -> str:
    """
    Generate a caption for a single image.

    Args:
        image_path:   Path to the image file.
        trigger_word: Prepended to the caption if non-empty.
        detail:       Florence2 task token.
                      "MORE_DETAILED_CAPTION" (default) — richest attribute
                      descriptions; best for LoRA training.
                      "DETAILED_CAPTION" — shorter, faster alternative.
        hf_token:     HuggingFace token for model download.

    Returns:
        Caption string formatted for Z-Image Turbo / FLUX training.
    """
    import torch
    from PIL import Image

    _load_model(hf_token)

    img = Image.open(image_path).convert("RGB")
    task_prompt = f"<{detail}>"

    inputs = _processor(
        text=task_prompt,
        images=img,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        generated_ids = _model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=5,
        )

    raw = _processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # Strip Florence2's task-answer formatting
    caption = _processor.post_process_generation(
        raw, task=task_prompt, image_size=(img.width, img.height)
    )
    if isinstance(caption, dict):
        caption = caption.get(task_prompt, "")
    caption = _clean_caption(str(caption))

    # Prefix with trigger word for training
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

    Returns mapping of image path → caption string.
    """
    _load_model(hf_token)
    results: dict[Path, str] = {}
    total = len(image_paths)

    for i, img_path in enumerate(image_paths, 1):
        if cancel_event and cancel_event.is_set():
            break
        if progress_cb:
            progress_cb(i - 1, total, f"Captioning {i}/{total}: {img_path.name}")
        try:
            cap = caption_image(img_path, trigger_word=trigger_word, hf_token=hf_token)
            results[img_path] = cap

            # Save .txt file
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
        progress_cb(len(results), total, "Captioning complete.")

    unload_model()
    return results
