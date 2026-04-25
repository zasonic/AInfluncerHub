"""
python/server.py — AinfluencerHub FastAPI backend.

Launched by Tauri (Rust) before the WebView opens.
All long-running operations stream progress via Server-Sent Events.

This backend is fully self-contained — all GPU inference, training,
and captioning runs natively using HuggingFace diffusers, peft,
and transformers.  No external services (ComfyUI, ai-toolkit) required.

Usage:
    python server.py --port 8765
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Ensure python/ is importable as the root package path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core.project import Project
from core.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("hub.server")

# ── Global singletons ─────────────────────────────────────────────────────────

settings = Settings()

# GPU lock — one task at a time on 16 GB VRAM
_gpu_lock = threading.Lock()

# Per-project training cancel events
_cancel_lock = threading.Lock()
_cancel_events: dict[str, threading.Event] = {}

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="AinfluencerHub", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse_event(data: dict) -> dict:
    return {"data": json.dumps(data)}


class SSEQueue:
    """
    Thread-safe queue with async drain semantics.

    Worker threads push via `put()`; the FastAPI endpoint awaits items via the
    async generator returned by `drain()`. Items push from any thread — pushes
    are bounced through the running event loop so the awaiter wakes up
    immediately without polling.
    """

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def put(self, item: dict[str, Any]) -> None:
        self._loop.call_soon_threadsafe(self._q.put_nowait, item)

    async def drain(self) -> AsyncGenerator[dict, None]:
        while True:
            item = await self._q.get()
            yield _sse_event(item)
            if item.get("type") in ("done", "error"):
                break


def _drain_queue(q: SSEQueue) -> AsyncGenerator[dict, None]:
    """Legacy-compatible alias so existing callers keep working."""
    return q.drain()

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True}

# ── Preflight ─────────────────────────────────────────────────────────────────

@app.get("/api/preflight")
def preflight():
    from services.preflight import run_all
    return run_all(settings)

# ── Projects ──────────────────────────────────────────────────────────────────

@app.get("/api/projects")
def list_projects():
    output_dir = settings.resolve_output_dir()
    return [p.to_dict() for p in Project.list_all(output_dir)]


class CreateProjectBody(BaseModel):
    name:         str
    trigger_word: str
    gender:       str = "female"


@app.post("/api/projects")
def create_project(body: CreateProjectBody):
    output_dir = settings.resolve_output_dir()
    proj = Project.create(output_dir, body.name, body.trigger_word, body.gender)
    return proj.to_dict()


@app.get("/api/projects/{slug}")
def get_project(slug: str):
    proj = _load_project(slug)
    return proj.to_dict()


@app.delete("/api/projects/{slug}")
def delete_project(slug: str):
    proj = _load_project(slug)
    proj.delete()
    return {"ok": True}


@app.post("/api/projects/{slug}/references")
async def upload_references(slug: str, files: list[UploadFile] = File(...)):
    proj = _load_project(slug)
    proj.reference_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in files[:3]:
        suffix = Path(f.filename or "ref.jpg").suffix or ".jpg"
        dest   = proj.reference_dir / f"ref_{count + 1:02d}{suffix}"
        content = await f.read()
        dest.write_bytes(content)
        count += 1
    return {"count": count}

# ── Dataset ───────────────────────────────────────────────────────────────────

@app.get("/api/dataset/{slug}/images")
def get_dataset_images(slug: str):
    proj = _load_project(slug)
    return {"images": [str(p) for p in proj.dataset_images()]}


@app.post("/api/dataset/{slug}/upload")
async def upload_dataset(slug: str, files: list[UploadFile] = File(...)):
    proj = _load_project(slug)
    proj.dataset_dir.mkdir(parents=True, exist_ok=True)
    existing = len(proj.dataset_images())
    count = 0
    for f in files:
        suffix = Path(f.filename or "img.jpg").suffix or ".jpg"
        dest   = proj.dataset_dir / f"dataset_{existing + count + 1:03d}{suffix}"
        content = await f.read()
        dest.write_bytes(content)
        count += 1
    proj.set("dataset_count", existing + count)
    proj.save()
    return {"count": count}


@app.get("/api/dataset/{slug}/generate")
async def generate_dataset_images(
    slug:  str,
    count: int = Query(25),
):
    """Generate face-consistent dataset images using native diffusers pipeline."""
    proj = _load_project(slug)
    refs = proj.reference_images()
    if not refs:
        raise HTTPException(400, "No reference images found for this influencer.")

    gender      = proj.gender
    prompt_file = ROOT / "assets" / "prompts" / (
        "male_variations.json" if gender == "male" else "female_variations.json"
    )
    with open(prompt_file) as f:
        all_prompts = json.load(f)
    prompts = [p["prompt"] for p in all_prompts[:count]]

    q      = SSEQueue()
    cancel = threading.Event()
    with _cancel_lock:
        _cancel_events[slug] = cancel

    def _run():
        try:
            from services.diffusion_pipeline import generate_dataset
            hf_token = settings.get("hf_token", "")
            generate_dataset(
                reference_image=refs[0],
                prompts=prompts,
                trigger_word=proj.trigger_word,
                output_dir=proj.dataset_dir,
                hf_token=hf_token,
                progress_cb=lambda done, total, msg: q.put(
                    {"type": "progress", "done": done, "total": total, "message": msg}
                ),
                cancel_event=cancel,
            )
            proj.set("dataset_count", len(proj.dataset_images()))
            proj.save()
            q.put({"type": "done", "message": "Dataset generation complete."})
        except Exception as exc:
            q.put({"type": "error", "message": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return EventSourceResponse(_drain_queue(q))


@app.post("/api/dataset/{slug}/score")
def score_dataset(slug: str):
    proj = _load_project(slug)
    images = proj.dataset_images()
    if not images:
        raise HTTPException(400, "No dataset images to score.")
    from services.quality_scorer import score_images
    results = score_images(images)
    passed = [r for r in results if r["passed"]]
    return {"scores": results, "passed": len(passed), "total": len(results)}

# ── Captions ──────────────────────────────────────────────────────────────────

@app.get("/api/captions/{slug}")
def get_captions(slug: str):
    proj    = _load_project(slug)
    result: dict[str, str] = {}
    for txt in proj.captions_dir.glob("*.txt"):
        result[txt.stem] = txt.read_text(encoding="utf-8").strip()
    # Also pick up any .txt files sitting beside dataset images
    for txt in proj.dataset_dir.glob("*.txt"):
        if txt.stem not in result:
            result[txt.stem] = txt.read_text(encoding="utf-8").strip()
    return result


class CaptionBody(BaseModel):
    text: str


@app.put("/api/captions/{slug}/{stem}")
def update_caption(slug: str, stem: str, body: CaptionBody):
    if "/" in stem or "\\" in stem or ".." in stem:
        raise HTTPException(400, "Invalid caption name.")
    proj = _load_project(slug)
    proj.captions_dir.mkdir(parents=True, exist_ok=True)
    txt_path = proj.captions_dir / f"{stem}.txt"
    txt_path.write_text(body.text, encoding="utf-8")
    return {"ok": True}


@app.post("/api/captions/{slug}/inject-trigger")
def inject_trigger(slug: str):
    proj    = _load_project(slug)
    tw      = proj.trigger_word
    changed = 0
    proj.captions_dir.mkdir(parents=True, exist_ok=True)
    for img in proj.dataset_images():
        txt_path = proj.captions_dir / f"{img.stem}.txt"
        cap      = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
        if tw not in cap:
            cap = f"{tw}, {cap}" if cap else tw
            txt_path.write_text(cap, encoding="utf-8")
            changed += 1
    return {"updated": changed}


@app.get("/api/captions/{slug}/run")
async def run_captioning(
    slug:     str,
    hf_token: str = Query(""),
    captioner: str = Query("florence2"),
):
    proj   = _load_project(slug)
    images = proj.dataset_images()
    if not images:
        raise HTTPException(400, "No dataset images found.")

    if not _gpu_lock.acquire(blocking=False):
        raise HTTPException(409, "GPU is busy. Try again shortly.")

    # Prefer token from saved settings; fall back to query param for backwards compat
    resolved_token = hf_token.strip() or settings.get("hf_token", "")

    q      = SSEQueue()
    cancel = threading.Event()
    with _cancel_lock:
        _cancel_events[slug] = cancel

    def _run():
        try:
            if captioner == "joycaption":
                from services.joy_captioner import caption_batch
            else:
                from services.florence_captioner import caption_batch
            caption_batch(
                image_paths=images,
                trigger_word=proj.trigger_word,
                hf_token=resolved_token or None,
                progress_cb=lambda done, total, msg: q.put(
                    {"type": "progress", "done": done, "total": total, "message": msg}
                ),
                cancel_event=cancel,
                captions_dir=proj.captions_dir,
            )
            q.put({"type": "done", "message": "Captioning complete."})
        except Exception as exc:
            q.put({"type": "error", "message": str(exc)})
        finally:
            _gpu_lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return EventSourceResponse(_drain_queue(q))

# ── Training ──────────────────────────────────────────────────────────────────

@app.get("/api/training/{slug}/start")
async def start_training(
    slug:     str,
    hf_token: str = Query(""),
    steps:    int = Query(2000),
    rank:     int = Query(16),
    lr:       str = Query("1e-4"),
):
    """Start native LoRA training using diffusers + peft."""
    proj = _load_project(slug)
    # Prefer token from saved settings; fall back to query param for backwards compat
    resolved_token = hf_token.strip() or settings.get("hf_token", "")
    if not resolved_token:
        raise HTTPException(400, "hf_token is required. Set it in Settings first.")
    if not _gpu_lock.acquire(blocking=False):
        raise HTTPException(409, "GPU is busy. Wait for the current task to finish.")

    q      = SSEQueue()
    cancel = threading.Event()
    with _cancel_lock:
        _cancel_events[slug] = cancel

    # Save settings for next time (store resolved token, never log it)
    settings.update({
        "hf_token":        resolved_token,
        "training_steps":  steps,
        "lora_rank":       rank,
        "learning_rate":   lr,
    })

    def _run():
        try:
            from services.lora_trainer import prepare_training_folder, run_training

            prepare_training_folder(proj.dataset_dir, proj.captions_dir)

            q.put({"type": "log", "line": f"Starting training: {steps} steps, rank {rank}, lr {lr}"})

            success, message = run_training(
                dataset_dir=proj.dataset_dir,
                output_dir=proj.lora_dir,
                trigger_word=proj.trigger_word,
                steps=steps,
                rank=rank,
                learning_rate=float(lr),
                hf_token=resolved_token,
                log_cb=lambda line: q.put({"type": "log", "line": line}),
                cancel_event=cancel,
            )

            if success:
                loras = sorted(
                    proj.lora_dir.glob("**/*.safetensors"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                lora_path = str(loras[0]) if loras else message
                proj.set("lora_path", lora_path)
                proj.mark_step_done(4)
                proj.save()
                q.put({"type": "done", "message": "Training complete.", "payload": {"path": lora_path}})
            else:
                q.put({"type": "error", "message": message})
        except Exception as exc:
            q.put({"type": "error", "message": str(exc)})
        finally:
            _gpu_lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return EventSourceResponse(_drain_queue(q))


@app.post("/api/training/{slug}/cancel")
def cancel_training(slug: str):
    with _cancel_lock:
        ev = _cancel_events.get(slug)
    if ev:
        ev.set()
    return {"ok": True}

# ── Model management ─────────────────────────────────────────────────────────

@app.get("/api/models/status")
def get_model_status():
    """Report which models are downloaded and their sizes."""
    from services.model_manager import get_all_model_status
    return get_all_model_status()


@app.get("/api/models/download")
async def download_model(model_hf_id: str = Query("")):
    """Download a model from HuggingFace Hub with SSE progress."""
    if not model_hf_id.strip():
        raise HTTPException(400, "model_hf_id is required.")

    q = SSEQueue()

    def _run():
        from services.model_manager import download_model as _download
        hf_token = settings.get("hf_token", "")
        ok, msg = _download(
            model_hf_id.strip(),
            hf_token=hf_token,
            progress_cb=lambda m: q.put({"type": "log", "line": m}),
        )
        if ok:
            q.put({"type": "done", "message": msg})
        else:
            q.put({"type": "error", "message": msg})

    threading.Thread(target=_run, daemon=True).start()
    return EventSourceResponse(_drain_queue(q))

# ── Studio — image generation ─────────────────────────────────────────────────

@app.get("/api/studio/{slug}/generate")
async def generate_image(
    slug:          str,
    prompt:        str   = Query(""),
    lora_strength: float = Query(0.85),
):
    """Generate an image using native diffusers pipeline with optional LoRA."""
    proj = _load_project(slug)
    if not _gpu_lock.acquire(blocking=False):
        raise HTTPException(409, "GPU is busy.")

    q = SSEQueue()

    def _run():
        try:
            lora = proj.lora_path()
            from services.diffusion_pipeline import generate_image as _generate
            hf_token = settings.get("hf_token", "")
            q.put({"type": "progress", "done": 0, "total": 1, "message": "Starting generation..."})
            paths = _generate(
                positive_prompt=prompt,
                lora_path=str(lora) if lora else "",
                lora_strength=lora_strength,
                output_dir=proj.generated_dir,
                hf_token=hf_token,
                progress_cb=lambda msg: q.put(
                    {"type": "progress", "done": 0, "total": 1, "message": msg}
                ),
            )
            q.put({"type": "done", "message": "Image generated.", "payload": {"paths": [str(p) for p in paths]}})
        except Exception as exc:
            q.put({"type": "error", "message": str(exc)})
        finally:
            _gpu_lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return EventSourceResponse(_drain_queue(q))


@app.get("/api/studio/{slug}/images")
def get_generated_images(slug: str):
    proj  = _load_project(slug)
    exts  = {".jpg", ".jpeg", ".png", ".webp"}
    imgs  = sorted(
        [p for p in proj.generated_dir.iterdir() if p.suffix.lower() in exts],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return {
        "images": [
            {
                "path":     str(img),
                "filename": img.name,
                "url":      f"/api/files/image?path={img}",
            }
            for img in imgs
        ]
    }

# ── Studio — video ────────────────────────────────────────────────────────────

@app.get("/api/studio/{slug}/animate")
async def animate_image(
    slug:          str,
    image_path:    str = Query(""),
    motion_prompt: str = Query(""),
):
    """Generate a video from a still image using native diffusers pipeline."""
    proj = _load_project(slug)
    if not image_path:
        raise HTTPException(400, "image_path is required.")
    if not _gpu_lock.acquire(blocking=False):
        raise HTTPException(409, "GPU is busy.")

    q = SSEQueue()

    def _run():
        try:
            from services.video_pipeline import generate_video
            from services.models import WAN_VIDEO, COGVIDEO
            hf_token = settings.get("hf_token", "")
            video_model = settings.get("video_model", "wan2.1")
            model_id = COGVIDEO.repo_id if video_model == "cogvideo" else WAN_VIDEO.repo_id
            q.put({"type": "progress", "done": 0, "total": 1, "message": "Starting video generation..."})
            video = generate_video(
                image_path=Path(image_path),
                prompt=motion_prompt,
                output_dir=proj.videos_dir,
                model_id=model_id,
                hf_token=hf_token,
                progress_cb=lambda m: q.put(
                    {"type": "progress", "done": 0, "total": 1, "message": m}
                ),
            )
            q.put({"type": "done", "message": "Video created.",
                   "payload": {"path": str(video) if video else ""}})
        except Exception as exc:
            q.put({"type": "error", "message": str(exc)})
        finally:
            _gpu_lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return EventSourceResponse(_drain_queue(q))


@app.get("/api/studio/{slug}/videos")
def get_videos(slug: str):
    proj = _load_project(slug)
    vids = sorted(
        proj.videos_dir.glob("*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return {
        "videos": [{"path": str(v), "filename": v.name} for v in vids]
    }

# ── Settings ──────────────────────────────────────────────────────────────────

@app.get("/api/settings")
def get_settings():
    return settings.to_dict()


@app.put("/api/settings")
def update_settings(body: dict):
    settings.update(body)
    return {"ok": True}

# ── File serving ──────────────────────────────────────────────────────────────

@app.get("/api/files/image")
def serve_image(path: str = Query(...)):
    p = Path(path).resolve()
    output_dir = settings.resolve_output_dir().resolve()
    if not str(p).startswith(str(output_dir) + os.sep) and p != output_dir:
        raise HTTPException(403, "Access denied.")
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "Image not found.")
    return FileResponse(p)

# ── Internal helper ───────────────────────────────────────────────────────────

def _load_project(slug: str) -> Project:
    output_dir = settings.resolve_output_dir()
    root       = output_dir / slug
    try:
        return Project.load(root)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Project '{slug}' not found.") from exc

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    log.info("AinfluencerHub backend starting on %s:%d", args.host, args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",   # suppress access log spam
    )
