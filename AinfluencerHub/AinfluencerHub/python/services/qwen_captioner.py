"""
services/qwen_captioner.py — Qwen2.5-VL image captioning for LoRA training.

Qwen2.5-VL-7B produces higher-quality structured captions than Florence-2 and
fits in less VRAM than JoyCaption (4-bit quantized ~4 GB vs 10 GB).
It describes subject appearance, clothing, pose, expression, lighting, and
background with the detail needed for effective SDXL/FLUX LoRA training.
"""

import logging
from collections.abc import Callable
from pathlib import Path

from services.models import QWEN_CAPTIONER

log = logging.getLogger("hub.qwen_captioner")

_model = None
_processor = None
_device = None

CAPTION_PROMPT = (
    "Describe this image in precise detail for use as a LoRA training caption. "
    "Include: subject appearance, clothing, pose, expression, lighting, background, "
    "and photographic style. Be specific and concise. Do not include subjective opinions."
)


def _load_model(hf_token: str | None = None) -> None:
    global _model, _processor, _device
    if _model is not None:
        return

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading Qwen2.5-VL-7B on %s (4-bit quantized)...", _device)

    quant_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        if _device == "cuda"
        else None
    )

    kwargs: dict = {}
    if hf_token:
        kwargs["token"] = hf_token
    if quant_config:
        kwargs["quantization_config"] = quant_config
    else:
        kwargs["torch_dtype"] = torch.float32

    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_CAPTIONER.repo_id,
        device_map="auto" if _device == "cuda" else None,
        **kwargs,
    )
    _processor = AutoProcessor.from_pretrained(
        QWEN_CAPTIONER.repo_id,
        token=hf_token or None,
    )
    log.info("Qwen2.5-VL loaded.")


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
        log.info("Qwen2.5-VL unloaded.")


def caption_image(
    image_path: Path,
    trigger_word: str = "",
    hf_token: str | None = None,
) -> str:
    """
    Generate a training caption for a single image.

    Args:
        image_path:   Path to the image file.
        trigger_word: Prepended to the caption if non-empty.
        hf_token:     HuggingFace token for model download.

    Returns:
        Caption string formatted for SDXL / FLUX LoRA training.
    """
    import torch
    from PIL import Image

    _load_model(hf_token)

    img = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]

    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _processor(
        text=[text],
        images=[img],
        return_tensors="pt",
        padding=True,
    ).to(_model.device)

    with torch.no_grad():
        generated_ids = _model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )

    output_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    caption = _processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

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
    Caption a list of images. Saves a .txt file alongside each image
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
