"""services/joy_captioner.py — JoyCaption image captioning for LoRA training.

JoyCaption is purpose-built for generating natural-language captions optimised
for diffusion model training.  It produces richer, more descriptive captions
than Florence-2, which directly translates to better LoRA prompt responsiveness.

Model: JoyCaption Alpha Two (fancyfeast/llama-joycaption-alpha-two-hf-llava)
  - ~10 GB in 4-bit (fits within 16 GB VRAM alongside other pipeline components)
  - Fixes caption truncation bug present in Beta One
  - Supports subject naming: the prompt can instruct the model to refer to the
    person by the project trigger word, so captions already embed the activation
    token instead of requiring a prepend pass afterward.
"""

import logging
from collections.abc import Callable
from pathlib import Path

from services.models import JOY_CAPTIONER

log = logging.getLogger("hub.joycaption")

_model = None
_processor = None
_device = None

JOYCAPTION_MODEL_ID = JOY_CAPTIONER.repo_id


def _build_caption_prompt(trigger_word: str = "") -> str:
    """
    Build a JoyCaption Alpha Two caption prompt.

    When a trigger word is supplied, the prompt instructs the model to name
    the subject using that word directly in the caption.  This leverages
    Alpha Two's named-entity feature so captions are self-contained and the
    trigger word appears naturally rather than being prepended as a tag.
    """
    base = (
        "Write a detailed description of this image for use as a training caption "
        "for an AI image generation model. Describe the person's appearance, pose, "
        "expression, clothing, and the setting. Be specific and use natural language. "
        "Do not start with 'The image shows' or 'This is'. Just describe what you see."
    )
    if trigger_word:
        base += (
            f" Refer to the person in the image as '{trigger_word}' "
            "(use this name consistently; do not substitute pronouns)."
        )
    return base


def _load_model(hf_token: str | None = None) -> None:
    global _model, _processor, _device
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading JoyCaption Alpha Two on %s ...", _device)

    kwargs: dict = {"trust_remote_code": True}
    if hf_token:
        kwargs["token"] = hf_token

    # Use 4-bit quantization on GPU to fit within 10–12 GB VRAM
    if _device == "cuda":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32

    _model = AutoModelForCausalLM.from_pretrained(
        JOYCAPTION_MODEL_ID, **kwargs
    )
    _processor = AutoProcessor.from_pretrained(
        JOYCAPTION_MODEL_ID,
        trust_remote_code=True,
        token=hf_token or None,
    )
    log.info("JoyCaption Alpha Two loaded.")


def unload_model() -> None:
    global _model, _processor
    if _model is not None:
        import torch
        del _model, _processor
        _model = None
        _processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("JoyCaption unloaded.")


def caption_image(
    image_path: Path,
    trigger_word: str = "",
    hf_token: str | None = None,
) -> str:
    """Generate a training-optimised caption for a single image."""
    import torch
    from PIL import Image

    _load_model(hf_token)

    img = Image.open(image_path).convert("RGB")

    # Alpha Two: pass the trigger word as a naming instruction in the prompt
    caption_prompt = _build_caption_prompt(trigger_word)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": caption_prompt},
        ]},
    ]

    prompt_text = _processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    inputs = _processor(
        text=prompt_text,
        images=img,
        return_tensors="pt",
    ).to(_model.device)

    with torch.no_grad():
        generated_ids = _model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    # Decode only the new tokens (skip the prompt)
    new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
    caption = _processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    # Safety fallback: if the model didn't embed the trigger word despite the
    # instruction, prepend it so the caption is always activation-ready.
    if trigger_word and trigger_word not in caption:
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
    """Caption a list of images using JoyCaption Alpha Two.

    Same interface as florence_captioner.caption_batch for drop-in replacement.
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
            log.error("JoyCaption failed for %s: %s", img_path.name, exc)
            results[img_path] = ""

    if progress_cb:
        progress_cb(len(results), total, "Captioning complete.")

    unload_model()
    return results
