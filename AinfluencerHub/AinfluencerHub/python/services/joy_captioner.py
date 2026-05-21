"""
services/joy_captioner.py — JoyCaption image captioning for LoRA training.

JoyCaption is purpose-built for generating natural-language captions optimized
for diffusion model training.  It produces richer, more descriptive captions
than Florence-2, which directly translates to better LoRA fidelity.

v1.1 — JoyCaption Two upgrade with automatic fallback:
  Primary model: fancyfeast/llama-joycaption-two-hf (JoyCaption 2)
    - More detailed, training-specific descriptions
    - Better face-identity language (useful for influencer LoRA)
    - Same HuggingFace AutoModelForCausalLM interface
  Fallback model: fancyfeast/llama-joycaption-beta-one-hf-llava (Beta One)
    - Used automatically when JoyCaption 2 fails to load (older transformers,
      insufficient VRAM, download error, etc.)
    - Zero user intervention required — the caption step continues with the
      best available model and logs which one is active.

Model sizes: ~10 GB bf16, ~10 GB 4-bit quantized (GPU), both versions.
"""

import logging
from collections.abc import Callable
from pathlib import Path

from services.models import JOY_CAPTIONER, JOY_CAPTIONER_V1

log = logging.getLogger("hub.joycaption")

_model = None
_processor = None
_device = None
_loaded_model_id: str | None = None  # tracks which version is actually in memory

JOYCAPTION_MODEL_ID = JOY_CAPTIONER.repo_id
JOYCAPTION_V1_MODEL_ID = JOY_CAPTIONER_V1.repo_id

# System prompt for training-style captions.
# JoyCaption 2 responds well to this directive; Beta One uses the same prompt.
CAPTION_PROMPT = (
    "Write a detailed description of this image for use as a training caption "
    "for an AI image generation model. Describe the person's appearance, pose, "
    "expression, clothing, and the setting. Be specific and use natural language. "
    "Do not start with 'The image shows' or 'This is'. Just describe what you see."
)


def _load_model(hf_token: str | None = None) -> None:
    """Load JoyCaption Two, falling back to Beta One on any load failure."""
    global _model, _processor, _device, _loaded_model_id
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def _try_load(model_id: str) -> bool:
        """Attempt to load a specific JoyCaption model. Returns True on success."""
        global _model, _processor, _loaded_model_id
        log.info("Loading JoyCaption on %s: %s ...", _device, model_id)
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

        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=hf_token or None,
            )
            _model = model
            _processor = processor
            _loaded_model_id = model_id
            log.info("JoyCaption loaded: %s", model_id)
            return True
        except Exception as exc:
            log.warning("JoyCaption load failed for %s: %s", model_id, exc)
            return False

    # Try JoyCaption Two first; fall back to Beta One if it fails.
    if not _try_load(JOYCAPTION_MODEL_ID):
        log.warning(
            "JoyCaption Two unavailable — falling back to JoyCaption Beta One. "
            "Captions will still be generated but may be less detailed."
        )
        if not _try_load(JOYCAPTION_V1_MODEL_ID):
            raise RuntimeError(
                "Both JoyCaption Two and Beta One failed to load. "
                "Check your HuggingFace token, disk space, and GPU VRAM."
            )


def unload_model() -> None:
    global _model, _processor, _loaded_model_id
    if _model is not None:
        import torch
        del _model, _processor
        _model = None
        _processor = None
        _loaded_model_id = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("JoyCaption unloaded.")


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

    # Build the chat-style prompt (works for both JoyCaption 2 and Beta One)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": CAPTION_PROMPT},
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
    Falls back to JoyCaption Beta One automatically if JoyCaption Two fails
    to load — the batch continues without interruption.
    """
    _load_model(hf_token)
    results: dict[Path, str] = {}
    total = len(image_paths)

    # Log which model version is active so users/support can confirm.
    if _loaded_model_id:
        version = "JoyCaption 2" if "joycaption-two" in _loaded_model_id else "JoyCaption Beta One"
        log.info("Captioning %d images with %s.", total, version)

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
