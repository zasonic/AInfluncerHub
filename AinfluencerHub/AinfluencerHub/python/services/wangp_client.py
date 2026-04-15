"""
services/wangp_client.py — WanGP Gradio API client for I2V animation.

WanGP is the user's Pinokio-hosted WAN 2.2 interface.  It exposes a
Gradio API that we call programmatically via gradio_client.

The exact API shape can vary between WanGP versions, so we try a few
known endpoint names and fall back gracefully.
"""

import logging
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger("hub.wangp")


class WanGPClient:
    """
    Connects to a running WanGP instance and submits image-to-video jobs.
    """

    def __init__(self, url: str = "http://localhost:7860") -> None:
        self.url = url.rstrip("/")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from gradio_client import Client
            self._client = Client(self.url, verbose=False)
        return self._client

    def ping(self) -> bool:
        try:
            import requests
            r = requests.get(self.url, timeout=4)
            return r.status_code < 500
        except Exception:
            return False

    def list_endpoints(self) -> list[str]:
        try:
            client = self._get_client()
            return [ep["api_name"] for ep in client.view_api(print_info=False)]
        except Exception:
            return []

    def generate_video(
        self,
        image_path: Path,
        prompt: str,
        output_dir: Path,
        steps: int = 20,
        duration_seconds: float = 4.0,
        progress_cb: Callable[[str], None] | None = None,
    ) -> Path | None:
        """
        Submit an image-to-video job to WanGP.

        Args:
            image_path:  Source image for animation.
            prompt:      Motion/scene prompt (e.g. "the person smiles and turns").
            output_dir:  Where to save the resulting video.
            steps:       Diffusion steps (20 = fast, 30 = quality).
            duration_seconds: Approximate clip length.
            progress_cb: Called with status strings.

        Returns:
            Path to saved .mp4, or None on failure.
        """
        try:
            client = self._get_client()
        except Exception as exc:
            log.error("Could not connect to WanGP at %s: %s", self.url, exc)
            raise RuntimeError(
                f"Cannot connect to WanGP. Make sure it is running in Pinokio "
                f"(expected at {self.url})."
            ) from exc

        if progress_cb:
            progress_cb("Submitting to WanGP...")

        # WanGP API endpoint names vary by version — try known names
        endpoint_candidates = ["/predict", "/generate_video", "/run", "/i2v"]
        endpoints = self.list_endpoints()
        chosen_endpoint = "/predict"
        for candidate in endpoint_candidates:
            if candidate in endpoints:
                chosen_endpoint = candidate
                break

        # Build frames argument from duration (24fps)
        num_frames = max(17, int(duration_seconds * 24))
        # Round to nearest odd number (WAN requirement)
        if num_frames % 2 == 0:
            num_frames += 1

        try:
            # WanGP Gradio I2V call — argument order matches WanGP's standard UI
            result = client.predict(
                str(image_path),   # image input
                prompt,            # text prompt
                "",                # negative prompt
                num_frames,        # number of frames
                steps,             # inference steps
                7.0,               # guidance scale
                -1,                # seed (-1 = random)
                api_name=chosen_endpoint,
            )
        except Exception:
            # Fallback: try with just image + prompt (minimal signature)
            try:
                result = client.predict(
                    str(image_path),
                    prompt,
                    api_name=chosen_endpoint,
                )
            except Exception as exc2:
                log.error("WanGP generation failed: %s", exc2)
                raise RuntimeError(
                    f"WanGP returned an error: {exc2}\n\n"
                    "Make sure the WAN 2.2 Image-to-Video model is loaded in WanGP."
                ) from exc2

        if progress_cb:
            progress_cb("Downloading video from WanGP...")

        # result is typically a file path or (file_path, ...) tuple
        if isinstance(result, (list, tuple)):
            video_src = result[0]
        else:
            video_src = result

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"video_{image_path.stem}.mp4"

        if isinstance(video_src, str) and video_src.startswith("http"):
            import requests
            r = requests.get(video_src, timeout=120)
            r.raise_for_status()
            out_path.write_bytes(r.content)
        elif isinstance(video_src, str) and Path(video_src).exists():
            import shutil
            shutil.copy2(video_src, out_path)
        else:
            raise RuntimeError(f"Unexpected WanGP result type: {type(video_src)}")

        if progress_cb:
            progress_cb(f"Video saved: {out_path.name}")

        return out_path
