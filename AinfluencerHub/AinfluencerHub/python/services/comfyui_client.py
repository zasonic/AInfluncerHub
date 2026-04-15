"""
services/comfyui_client.py — ComfyUI REST API client.

Used in Step 5 (Content Studio) to generate images with the trained LoRA
loaded into whatever model the user has running in ComfyUI.

The workflow JSON is submitted to /prompt, then polled via /history
until complete.  Output images are retrieved from /view.
"""

import base64
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Callable

import requests

log = logging.getLogger("hub.comfyui")


class ComfyUIClient:
    """Thin wrapper around the ComfyUI HTTP API."""

    def __init__(self, base_url: str = "http://localhost:8188") -> None:
        self.base_url = base_url.rstrip("/")
        self._client_id = str(uuid.uuid4())

    # ── connectivity ─────────────────────────────────────────────────────────

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/system_stats", timeout=4)
            return r.status_code == 200
        except Exception:
            return False

    def list_checkpoints(self) -> list[str]:
        """Return available checkpoint filenames from ComfyUI's object_info."""
        try:
            r = requests.get(
                f"{self.base_url}/object_info/CheckpointLoaderSimple", timeout=8
            )
            info = r.json()
            return (
                info.get("CheckpointLoaderSimple", {})
                .get("input", {})
                .get("required", {})
                .get("ckpt_name", [[]])[0]
            )
        except Exception:
            return []

    def list_loras(self) -> list[str]:
        """Return available LoRA filenames."""
        try:
            r = requests.get(
                f"{self.base_url}/object_info/LoraLoader", timeout=8
            )
            info = r.json()
            return (
                info.get("LoraLoader", {})
                .get("input", {})
                .get("required", {})
                .get("lora_name", [[]])[0]
            )
        except Exception:
            return []

    # ── generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str = "blurry, low quality, watermark, text",
        lora_path: str = "",
        lora_strength: float = 0.85,
        width: int = 832,
        height: int = 1216,
        steps: int = 20,
        cfg: float = 4.0,
        seed: int = -1,
        checkpoint: str = "zimage_turbo.safetensors",
        output_dir: Path | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> list[Path]:
        """
        Submit a txt2img workflow to ComfyUI, poll until done,
        save outputs to output_dir, return list of Paths.
        """
        import random
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        workflow = self._build_workflow(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            lora_filename=Path(lora_path).name if lora_path else "",
            lora_strength=lora_strength,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            checkpoint=checkpoint,
        )

        if progress_cb:
            progress_cb("Submitting to ComfyUI...")

        prompt_id = self._submit(workflow)
        if not prompt_id:
            raise RuntimeError("ComfyUI did not accept the workflow.")

        if progress_cb:
            progress_cb(f"Generating... (job {prompt_id[:8]})")

        output_images = self._poll(prompt_id, timeout=300)

        if output_dir is None:
            output_dir = Path("output") / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for node_id, imgs in output_images.items():
            for img_info in imgs:
                fname = img_info.get("filename", "output.png")
                subfolder = img_info.get("subfolder", "")
                img_bytes = self._download_image(fname, subfolder)
                out_path = output_dir / f"gen_{fname}"
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                paths.append(out_path)

        return paths

    # ── workflow builder ──────────────────────────────────────────────────────

    def _build_workflow(
        self,
        positive_prompt: str,
        negative_prompt: str,
        lora_filename: str,
        lora_strength: float,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        checkpoint: str,
    ) -> dict:
        """
        Build a minimal ComfyUI API workflow dict.
        Supports an optional single LoRA loader node.
        The workflow uses CLIP + VAE embedded in the checkpoint.
        """
        # Node IDs as strings (ComfyUI convention)
        workflow: dict = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": positive_prompt,
                    "clip": ["1", 1],
                },
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["1", 1],
                },
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed":          seed,
                    "steps":         steps,
                    "cfg":           cfg,
                    "sampler_name":  "euler",
                    "scheduler":     "simple",
                    "denoise":       1.0,
                    "model":         ["1", 0],
                    "positive":      ["2", 0],
                    "negative":      ["3", 0],
                    "latent_image":  ["4", 0],
                },
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {"images": ["6", 0], "filename_prefix": "hub_gen"},
            },
        }

        if lora_filename:
            # Insert LoRA loader between checkpoint and KSampler
            workflow["8"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name":          lora_filename,
                    "strength_model":     lora_strength,
                    "strength_clip":      lora_strength,
                    "model":              ["1", 0],
                    "clip":               ["1", 1],
                },
            }
            # Redirect KSampler and CLIP encoders to use LoRA-patched model/clip
            workflow["5"]["inputs"]["model"] = ["8", 0]
            workflow["2"]["inputs"]["clip"]  = ["8", 1]
            workflow["3"]["inputs"]["clip"]  = ["8", 1]

        return workflow

    # ── API helpers ───────────────────────────────────────────────────────────

    def _submit(self, workflow: dict) -> str | None:
        payload = {"prompt": workflow, "client_id": self._client_id}
        try:
            r = requests.post(
                f"{self.base_url}/prompt",
                json=payload,
                timeout=15,
            )
            r.raise_for_status()
            return r.json().get("prompt_id")
        except Exception as exc:
            log.error("ComfyUI submit error: %s", exc)
            return None

    def _poll(self, prompt_id: str, timeout: int = 300) -> dict:
        """Poll /history/{prompt_id} until the job completes."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(
                    f"{self.base_url}/history/{prompt_id}", timeout=8
                )
                history = r.json()
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    # Collect image-output nodes
                    images = {}
                    for node_id, node_out in outputs.items():
                        if "images" in node_out:
                            images[node_id] = node_out["images"]
                    return images
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(f"ComfyUI job {prompt_id} did not finish within {timeout}s")

    def _download_image(self, filename: str, subfolder: str = "") -> bytes:
        params = {"filename": filename, "type": "output"}
        if subfolder:
            params["subfolder"] = subfolder
        r = requests.get(f"{self.base_url}/view", params=params, timeout=30)
        r.raise_for_status()
        return r.content
