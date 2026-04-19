"""
core/settings.py — Persistent settings store backed by a pydantic schema.

The schema lives here so typos are rejected at the boundary (PUT /api/settings)
and type mismatches (e.g. learning_rate="foo") never reach the pipelines.

Public API is stable: callers use .get / .set / .update / .to_dict exactly as
before. Unknown keys are logged and dropped rather than silently persisted.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

log = logging.getLogger("hub.settings")


class SettingsModel(BaseModel):
    """Typed settings schema. Add fields here — nowhere else."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    output_dir:      str                              = ""
    hf_token:        str                              = ""
    dataset_method:  Literal["local", "manual"]       = "local"
    training_steps:  int                              = Field(2000, ge=100, le=20_000)
    lora_rank:       int                              = Field(16, ge=2, le=256)
    learning_rate:   str                              = "1e-4"
    preferred_model: str                              = "sdxl"
    video_model:     str                              = "wan2.1"
    theme:           Literal["dark"]                  = "dark"
    last_project:    str                              = ""
    setup_complete:  bool                             = False


DEFAULTS: dict[str, Any] = SettingsModel().model_dump()


class Settings:
    """Thread-safe JSON-backed settings store with pydantic validation."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            path = Path.home() / ".ai_influencer_hub" / "settings.json"
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._model = SettingsModel()
        self._load()

    # ── Public API ──────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if hasattr(self._model, key):
                return getattr(self._model, key)
            return default

    def set(self, key: str, value: Any) -> None:
        self.update({key: value})

    def update(self, mapping: dict[str, Any]) -> None:
        """Apply a partial update. Unknown / invalid keys are dropped with a log."""
        with self._lock:
            clean = self._sanitize(mapping)
            if not clean:
                return
            merged = {**self._model.model_dump(), **clean}
            try:
                self._model = SettingsModel(**merged)
            except ValidationError as exc:
                log.warning("Settings update rejected: %s", exc)
                return
        self._save()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return self._model.model_dump()

    def resolve_output_dir(self) -> Path:
        raw = self.get("output_dir", "")
        p = Path(raw) if raw else Path(__file__).parent.parent.parent / "output" / "influencers"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Internals ───────────────────────────────────────────────────────────

    def _sanitize(self, mapping: dict[str, Any]) -> dict[str, Any]:
        allowed = set(SettingsModel.model_fields.keys())
        out: dict[str, Any] = {}
        for k, v in mapping.items():
            if k in allowed:
                out[k] = v
            else:
                log.warning("Ignoring unknown setting: %s", k)
        return out

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            log.warning("Could not read settings (%s); using defaults.", exc)
            return

        if not isinstance(data, dict):
            log.warning("Settings file corrupted (not a dict); using defaults.")
            return

        # Drop unknown keys from older versions, then validate.
        clean = self._sanitize(data)
        try:
            self._model = SettingsModel(**clean)
        except ValidationError as exc:
            log.warning("Settings file invalid (%s); using defaults.", exc)

    def _save(self) -> None:
        try:
            tmp = self._path.with_suffix(".tmp")
            with self._lock:
                payload = self._model.model_dump()
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as exc:
            log.error("Could not save settings: %s", exc)
