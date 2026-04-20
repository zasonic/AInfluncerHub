# AInfluencerHub

A desktop app that turns one reference photo into a fully-trained AI
influencer: varied dataset, LoRA weights, generated images, and animated
video — all running locally on your GPU, no external services.

## Architecture

- **Frontend** — React + TypeScript + Vite, wrapped in a Tauri 2 shell.
- **Backend** — FastAPI + diffusers + peft + transformers, launched by the
  Tauri shell before the UI opens. All GPU work (image generation, LoRA
  training, captioning, video) runs natively in-process.

Top-level layout:

```
AinfluencerHub/AinfluencerHub/
├── src/                 React UI (5-step workflow + Studio)
│   └── hooks/           useSSE, useAsyncOperation
├── python/
│   ├── server.py        FastAPI entrypoint (SSE progress streaming)
│   ├── core/            Settings + Project model
│   └── services/        diffusion / video / lora_trainer / captioners
│                        — model IDs live in services/models.py
├── src-tauri/           Rust shell (launches backend, opens webview)
└── package.json
```

## Prerequisites

- Python 3.10+
- Node.js 20+
- Rust (stable) + Tauri prerequisites for your OS
- ~16 GB VRAM recommended for SDXL + IP-Adapter + JoyCaption

## Install

```bash
cd AinfluencerHub/AinfluencerHub

# Frontend
npm install

# Backend
cd python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# From AinfluencerHub/AinfluencerHub
npm run tauri:dev
```

For backend-only iteration:

```bash
cd AinfluencerHub/AinfluencerHub/python
python server.py --port 8765
```

## HuggingFace token

A read token from <https://huggingface.co/settings/tokens> is required for
gated model downloads (SDXL, IP-Adapter, JoyCaption). Set it in the in-app
**Settings** modal or via `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`. See
`AinfluencerHub/AinfluencerHub/python/.env.example` for env var names.

## Tests & lint

```bash
# Backend
cd AinfluencerHub/AinfluencerHub/python
pytest -v
ruff check .

# Frontend
cd AinfluencerHub/AinfluencerHub
npx tsc --noEmit
npx vitest run
```

CI runs these on every PR — see `.github/workflows/ci.yml`.
