"""
services/comfyui_client.py — ComfyUI REST API client.

Unified backend for all GPU inference:
  - Dataset generation via PuLID-FLUX (face-consistent variations)
  - Image generation with trained LoRA
  - Video animation via Wan2.1 Image-to-Video

The workflow JSON is submitted to /prompt, then polled via /history
until complete.  Output images/videos are retrieved from /view.
"""

import json
import logging
import random
import threading
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

    # ── node detection ───────────────────────────────────────────────────────

    def check_custom_nodes(self, group: str) -> bool:
        """Check if required custom nodes are installed in ComfyUI.

        group: "pulid" or "wan_video"
        """
        required: dict[str, list[str]] = {
            "pulid": [
                "PulidFluxModelLoader",
                "PulidFluxInsightFaceLoader",
                "PulidFluxEvaClipLoader",
                "ApplyPulidFlux",
            ],
            "wan_video": [
                "WanImageToVideo",
            ],
        }
        nodes = required.get(group, [])
        if not nodes:
            return False
        try:
            for node_name in nodes:
                r = requests.get(
                    f"{self.base_url}/object_info/{node_name}", timeout=8
                )
                if r.status_code != 200 or node_name not in r.json():
                    return False
            return True
        except Exception:
            return False

    def upload_image(self, image_path: Path) -> str:
        """Upload a local image to ComfyUI. Returns the filename on the server."""
        with open(image_path, "rb") as f:
            r = requests.post(
                f"{self.base_url}/upload/image",
                files={"image": (image_path.name, f, "image/png")},
                timeout=30,
            )
        r.raise_for_status()
        data = r.json()
        return data.get("name", image_path.name)

    # ── dataset generation (PuLID-FLUX) ──────────────────────────────────────

    def generate_dataset(
        self,
        reference_image: Path,
        prompts: list[str],
        trigger_word: str,
        checkpoint: str,
        output_dir: Path,
        progress_cb: Callable[[int, int, str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[Path]:
        """Generate face-consistent dataset images using PuLID-FLUX.

        Uploads the reference image, then generates one image per prompt
        using the user's chosen checkpoint with PuLID face injection.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Upload reference to ComfyUI
        ref_name = self.upload_image(reference_image)
        total = len(prompts)
        paths: list[Path] = []

        for i, prompt in enumerate(prompts):
            if cancel_event and cancel_event.is_set():
                break

            full_prompt = f"{trigger_word}, {prompt}" if trigger_word else prompt
            label = prompt[:50] + "..." if len(prompt) > 50 else prompt
            if progress_cb:
                progress_cb(i, total, f"Generating {i + 1}/{total}: {label}")

            seed = random.randint(0, 2**32 - 1)
            workflow = self._build_pulid_workflow(
                checkpoint=checkpoint,
                reference_image=ref_name,
                prompt=full_prompt,
                seed=seed,
            )

            prompt_id = self._submit(workflow)
            if not prompt_id:
                log.error("ComfyUI rejected workflow for prompt %d", i)
                continue

            try:
                output_images = self._poll(prompt_id, timeout=600)
            except TimeoutError:
                log.error("Timeout waiting for prompt %d", i)
                continue

            for node_id, imgs in output_images.items():
                for img_info in imgs:
                    fname = img_info.get("filename", "output.png")
                    subfolder = img_info.get("subfolder", "")
                    img_bytes = self._download_image(fname, subfolder)
                    out_path = output_dir / f"dataset_{len(paths) + 1:03d}.jpg"
                    out_path.write_bytes(img_bytes)
                    paths.append(out_path)

        if progress_cb:
            progress_cb(total, total, f"Generated {len(paths)} images.")
        return paths

    def _build_pulid_workflow(
        self,
        checkpoint: str,
        reference_image: str,
        prompt: str,
        negative_prompt: str = "blurry, low quality, watermark, text, deformed",
        seed: int = -1,
        width: int = 832,
        height: int = 1216,
        steps: int = 20,
        cfg: float = 4.0,
    ) -> dict:
        """Build a PuLID-FLUX workflow for face-consistent generation."""
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint},
            },
            "2": {
                "class_type": "PulidFluxModelLoader",
                "inputs": {},
            },
            "3": {
                "class_type": "PulidFluxEvaClipLoader",
                "inputs": {},
            },
            "4": {
                "class_type": "PulidFluxInsightFaceLoader",
                "inputs": {"provider": "CPU"},
            },
            "5": {
                "class_type": "LoadImage",
                "inputs": {"image": reference_image},
            },
            "6": {
                "class_type": "ApplyPulidFlux",
                "inputs": {
                    "weight": 0.9,
                    "start_at": 0.0,
                    "end_at": 1.0,
                    "model": ["1", 0],
                    "pulid_flux": ["2", 0],
                    "eva_clip": ["3", 0],
                    "face_analysis": ["4", 0],
                    "image": ["5", 0],
                },
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["1", 1]},
            },
            "8": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["1", 1]},
            },
            "9": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "10": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["6", 0],
                    "positive": ["7", 0],
                    "negative": ["8", 0],
                    "latent_image": ["9", 0],
                },
            },
            "11": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["10", 0], "vae": ["1", 2]},
            },
            "12": {
                "class_type": "SaveImage",
                "inputs": {"images": ["11", 0], "filename_prefix": "hub_dataset"},
            },
        }

    # ── video generation (Wan2.1 I2V) ────────────────────────────────────────

    def generate_video(
        self,
        image_path: Path,
        prompt: str,
        output_dir: Path,
        wan_checkpoint: str = "wan2.1_i2v_480p_bf16.safetensors",
        progress_cb: Callable[[str], None] | None = None,
    ) -> Path | None:
        """Generate a video from a still image using Wan2.1 I2V in ComfyUI."""
        output_dir.mkdir(parents=True, exist_ok=True)

        img_name = self.upload_image(image_path)

        if progress_cb:
            progress_cb("Building Wan2.1 I2V workflow...")

        workflow = self._build_wan_video_workflow(
            source_image=img_name,
            prompt=prompt,
            wan_checkpoint=wan_checkpoint,
        )

        prompt_id = self._submit(workflow)
        if not prompt_id:
            raise RuntimeError("ComfyUI did not accept the video workflow.")

        if progress_cb:
            progress_cb("Generating video... this may take a few minutes.")

        # Video generation is slower; use a longer timeout
        output_data = self._poll(prompt_id, timeout=900)

        # Look for video/gif outputs (VHS nodes output gifs key)
        for node_id, node_out in output_data.items():
            for img_info in node_out:
                fname = img_info.get("filename", "")
                if fname.endswith((".mp4", ".webm", ".gif")):
                    subfolder = img_info.get("subfolder", "")
                    video_bytes = self._download_image(fname, subfolder)
                    out_path = output_dir / f"video_{image_path.stem}.mp4"
                    out_path.write_bytes(video_bytes)
                    return out_path

        # Fallback: try to find any output file
        for node_id, imgs in output_data.items():
            for img_info in imgs:
                fname = img_info.get("filename", "output.png")
                subfolder = img_info.get("subfolder", "")
                video_bytes = self._download_image(fname, subfolder)
                out_path = output_dir / f"video_{image_path.stem}.mp4"
                out_path.write_bytes(video_bytes)
                return out_path

        return None

    def _build_wan_video_workflow(
        self,
        source_image: str,
        prompt: str,
        wan_checkpoint: str = "wan2.1_i2v_480p_bf16.safetensors",
        negative_prompt: str = "",
        num_frames: int = 81,
        steps: int = 20,
        cfg: float = 5.0,
        seed: int = -1,
    ) -> dict:
        """Build a Wan2.1 Image-to-Video workflow."""
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        return {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": source_image},
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "umt5_xxl_fp8_e4m3fn.safetensors",
                    "type": "wan",
                },
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["2", 0]},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["2", 0]},
            },
            "5": {
                "class_type": "WanImageToVideo",
                "inputs": {
                    "width": 832,
                    "height": 480,
                    "length": num_frames,
                    "batch_size": 1,
                    "image": ["1", 0],
                },
            },
            "6": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": wan_checkpoint,
                    "weight_dtype": "default",
                },
            },
            "7": {
                "class_type": "ModelSamplingWan",
                "inputs": {
                    "shift": 5.0,
                    "model": ["6", 0],
                },
            },
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "uni_pc_bh2",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["7", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                },
            },
            "9": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
            },
            "10": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["8", 0], "vae": ["9", 0]},
            },
            "11": {
                "class_type": "SaveAnimatedWEBP",
                "inputs": {
                    "filename_prefix": "hub_video",
                    "fps": 16.0,
                    "lossless": False,
                    "quality": 85,
                    "method": "default",
                    "images": ["10", 0],
                },
            },
        }

    # ── image generation (LoRA) ──────────────────────────────────────────────

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
