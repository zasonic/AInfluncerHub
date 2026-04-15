"""
core/settings.py — Persistent JSON settings store.
Shared path with the previous build so user settings carry over.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("hub.settings")

DEFAULTS: dict[str, Any] = {
    "comfyui_url":       "http://localhost:8188",
    "wangp_url":         "http://localhost:7860",
    "lm_studio_url":     "http://localhost:1234",
    "ai_toolkit_path":   "",
    "output_dir":        "",
    "fal_api_key":       "",
    "hf_token":          "",
    "dataset_method":    "fal",
    "training_steps":    2000,
    "lora_rank":         16,
    "learning_rate":     "1e-4",
    "caption_model":     "florence2",
    "theme":             "dark",
    "last_project":      "",
    "setup_complete":    False,
}


class Settings:
    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            path = Path.home() / ".ai_influencer_hub" / "settings.json"
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        self._load()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, DEFAULTS.get(key, default))

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
        self._save()

    def update(self, mapping: dict[str, Any]) -> None:
        with self._lock:
            self._data.update(mapping)
        self._save()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {**DEFAULTS, **self._data}

    def resolve_output_dir(self) -> Path:
        raw = self.get("output_dir", "")
        p = Path(raw) if raw else Path(__file__).parent.parent.parent / "output" / "influencers"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as exc:
                log.warning("Could not read settings: %s", exc)
                self._data = {}

    def _save(self) -> None:
        try:
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                with self._lock:
                    json.dump(self._data, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as exc:
            log.error("Could not save settings: %s", exc)
