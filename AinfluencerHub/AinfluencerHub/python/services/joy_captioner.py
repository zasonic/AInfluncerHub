"""
services/joy_captioner.py — JoyCaption image captioning for LoRA training.

JoyCaption is purpose-built for generating natural-language captions optimized
for diffusion model training.  It produces richer, more descriptive captions
than Florence-2, which directly translates to better LoRA fidelity.

Model: fancyfeast/llama-joycaption-alpha-two-hf-llava (~10 GB 4-bit on GPU).
Alpha Two (Oct 2024) improves instruction following and caption accuracy over
beta-one, producing more precise training captions for better LoRA quality.
The system-message format is required for Alpha Two; a bare user-only fallback
is tried automatically for older or differently-configured checkpoints.
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

# Alpha Two's system prompt activates its training-caption mode.
_SYSTEM_PROMPT = "You are a helpful image captioner."

# Descriptive prompt that takes advantage of Alpha Two's improved instruction
# following — asks for structured detail useful in diffusion training captions.
CAPTION_PROMPT = (
    "Write a detailed training caption for this image. Describe the subject's "
    "physical appearance (hair, eyes, skin tone, facial features), expression, "
    "pose, clothing and accessories, and the background/setting. Use natural, "
    "specific language suitable for a text-to-image model. Begin directly with "
    "the description — no preamble like 'The image shows' or 'This is a photo'."
)


def _load_model(hf_token: str | None = None) -> None:
    global _model, _processor, _device
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading JoyCaption on %s ...", _device)

    kwargs: dict = {"trust_remote_code": True}
    if hf_token:
        kwargs["token"] = hf_token

    # Use 4-bit quantization on GPU to fit in 10-12 GB VRAM
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
    log.info("JoyCaption loaded.")


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


def _build_prompt_text(img) -> str:
    """
    Build the tokenised prompt for Alpha Two using a system + user message.

    Falls back to the user-only format if the processor's chat template does
    not support system roles (e.g., older or non-Llama-3 checkpoints).  Either
    way the final prompt is identical to what the model expects at generation.
    """
    messages_with_system = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": CAPTION_PROMPT},
        ]},
    ]
    try:
        return _processor.apply_chat_template(
            messages_with_system, add_generation_prompt=True
        )
    except Exception:
        # Older processor / chat template that doesn't support system roles.
        messages_user_only = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": CAPTION_PROMPT},
            ]},
        ]
        return _processor.apply_chat_template(
            messages_user_only, add_generation_prompt=True
        )


def caption_image(
    image_path: Path,
    trigger_word: str = "",
    hf_token: str | None = None,
) -> str:
    """Generate a training-optimized caption for a single image."""
    import torch
    from PIL import Image

    _load_model(hf_token)

    img = Image.open(image_path).convert("RGB")

    prompt_text = _build_prompt_text(img)
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
    """Caption a list of images using JoyCaption.

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
