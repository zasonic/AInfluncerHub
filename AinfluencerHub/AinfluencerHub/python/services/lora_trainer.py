"""
services/lora_trainer.py — ai-toolkit LoRA training wrapper.

Generates a YAML config for Z-Image Turbo training and launches
ai-toolkit's run.py as a subprocess, streaming stdout/stderr back
through a callback so the UI can display a live log.

ai-toolkit must be cloned before calling this.  Use
services/preflight.py:clone_ai_toolkit() for first-run setup.
"""

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

import yaml

log = logging.getLogger("hub.trainer")

ZIMAGE_MODEL_ID  = "Ostris/zimage_turbo"
ZIMAGE_ADAPTER   = "ostris/zimage_turbo_training_adapter"


def build_yaml_config(
    project_name: str,
    trigger_word: str,
    dataset_dir: Path,
    captions_dir: Path,
    output_dir: Path,
    steps: int = 2000,
    rank: int = 16,
    learning_rate: str = "1e-4",
    hf_token: str = "",
    sample_prompt: str = "",
) -> dict:
    """
    Return an ai-toolkit YAML config dict for Z-Image Turbo LoRA training.
    """
    # Caption files live in captions_dir but images are in dataset_dir.
    # ai-toolkit expects image + .txt in the same folder.
    # We use dataset_dir as the training folder and symlink/copy captions there.
    # Simplest: set caption_ext and let ai-toolkit look for .txt next to images.
    # We copy captions into dataset_dir during the training-prep phase.

    sample_prompts = []
    if sample_prompt:
        sample_prompts.append(f"{trigger_word}, {sample_prompt}")
    else:
        sample_prompts.append(
            f"{trigger_word}, portrait photo of a person, "
            "professional photography, sharp focus"
        )

    config = {
        "job": "extension",
        "config": {
            "name": project_name,
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": str(output_dir),
                    "device": "cuda:0",
                    "trigger_word": trigger_word,

                    "network": {
                        "type": "lora",
                        "linear": rank,
                        "linear_alpha": rank,
                    },

                    "save": {
                        "dtype": "float16",
                        "save_every": max(250, steps // 8),
                        "max_step_saves_to_keep": 4,
                    },

                    "datasets": [
                        {
                            "folder_path": str(dataset_dir),
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": True,
                            "resolution": [1024, 1024],
                        }
                    ],

                    "train": {
                        "batch_size": 1,
                        "steps": steps,
                        "gradient_accumulation_steps": 4,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "learning_rate": float(learning_rate),
                        "loraplus_lr_ratio": 16.0,
                        "lr_scheduler": "cosine_with_restarts",
                        "lr_warmup_steps": max(50, steps // 20),
                        "max_grad_norm": 1.0,
                        "ema_config": {
                            "use_ema": True,
                            "ema_decay": 0.99,
                        },
                    },

                    "model": {
                        "name_or_path": ZIMAGE_MODEL_ID,
                        "assistant_lora_path": ZIMAGE_ADAPTER,
                        "is_flux": False,
                        "quantize": False,
                    },

                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": max(250, steps // 8),
                        "width": 1024,
                        "height": 1024,
                        "prompts": sample_prompts,
                        "neg": "",
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 3.5,
                        "sample_steps": 8,
                    },
                }
            ],
        },
        "meta": {
            "name": "@file",
            "version": "1.0",
        },
    }
    return config


def prepare_training_folder(
    dataset_dir: Path,
    captions_dir: Path,
) -> None:
    """
    Copy .txt captions from captions_dir into dataset_dir so that
    ai-toolkit can find image + caption pairs in one folder.
    """
    import shutil
    for txt in captions_dir.glob("*.txt"):
        dest = dataset_dir / txt.name
        if not dest.exists():
            shutil.copy2(txt, dest)


def run_training(
    ai_toolkit_path: Path,
    config_path: Path,
    log_cb: Callable[[str], None] | None = None,
    cancel_event=None,
    hf_token: str = "",
) -> tuple[bool, str]:
    """
    Launch ai-toolkit run.py with the given config and stream output
    line by line through log_cb.

    Returns (success, final_message).
    """
    run_py = Path(ai_toolkit_path) / "run.py"
    if not run_py.exists():
        return False, f"run.py not found at {ai_toolkit_path}"

    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    cmd = [sys.executable, str(run_py), str(config_path)]

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ai_toolkit_path),
            env=env,
        )

        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log.debug("[ai-toolkit] %s", line)
                if log_cb:
                    log_cb(line)
            if cancel_event and cancel_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return False, "Training cancelled by user."

        proc.wait(timeout=60)

        if proc.returncode == 0:
            return True, "Training completed successfully."
        else:
            return False, f"Training exited with code {proc.returncode}."

    except subprocess.TimeoutExpired:
        if proc:
            proc.kill()
        return False, "Training process timed out waiting to exit."
    except Exception as exc:
        return False, f"Failed to launch training: {exc}"
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
