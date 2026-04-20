"""
core/project.py — Influencer project data model.
Salvaged from previous build; added to_dict() for FastAPI responses.
"""

import json
import logging
import os
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("hub.project")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.lower()).strip()
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug or "influencer"


class Project:
    def __init__(self, root: Path, data: dict[str, Any]) -> None:
        self.root  = root
        self._lock = threading.Lock()
        self._d    = data

    # ── factories ─────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, output_dir: Path, name: str, trigger_word: str,
               gender: str = "female") -> "Project":
        slug = _slugify(name)
        root = output_dir / slug
        suffix = 1
        while root.exists():
            root = output_dir / f"{slug}_{suffix}"
            suffix += 1
        actual_slug = root.name
        for sub in ("reference", "dataset", "captions", "lora", "generated", "videos"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        data = {
            "name":          name,
            "slug":          actual_slug,
            "trigger_word":  trigger_word,
            "gender":        gender,
            "created_at":    datetime.now().isoformat(),
            "steps_done":    [],
            "dataset_count": 0,
            "lora_path":     "",
            "status":        "new",
        }
        proj = cls(root, data)
        proj.save()
        return proj

    @classmethod
    def load(cls, root: Path) -> "Project":
        path = root / "project.json"
        if not path.exists():
            raise FileNotFoundError(f"No project.json in {root}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(root, data)

    @classmethod
    def list_all(cls, output_dir: Path) -> list["Project"]:
        projects: list[Project] = []
        if not output_dir.exists():
            return projects
        for child in output_dir.iterdir():
            if child.is_dir() and (child / "project.json").exists():
                try:
                    projects.append(cls.load(child))
                except Exception as exc:
                    log.warning("Skipping project %s: %s", child, exc)
        projects.sort(key=lambda p: p.get("created_at", ""), reverse=True)
        return projects

    # ── accessors ─────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._d.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._d[key] = value

    def update(self, mapping: dict[str, Any]) -> None:
        with self._lock:
            self._d.update(mapping)

    def mark_step_done(self, step_num: int) -> None:
        with self._lock:
            done = self._d.get("steps_done", [])
            if step_num not in done:
                done.append(step_num)
            self._d["steps_done"] = done

    def step_done(self, n: int) -> bool:
        with self._lock:
            return n in self._d.get("steps_done", [])

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def name(self)         -> str: return self._d.get("name", "")
    @property
    def slug(self)         -> str: return self._d.get("slug", "")
    @property
    def trigger_word(self) -> str: return self._d.get("trigger_word", "")
    @property
    def gender(self)       -> str: return self._d.get("gender", "female")

    @property
    def reference_dir(self)  -> Path: return self.root / "reference"
    @property
    def dataset_dir(self)    -> Path: return self.root / "dataset"
    @property
    def captions_dir(self)   -> Path: return self.root / "captions"
    @property
    def lora_dir(self)       -> Path: return self.root / "lora"
    @property
    def generated_dir(self)  -> Path: return self.root / "generated"
    @property
    def videos_dir(self)     -> Path: return self.root / "videos"

    def dataset_images(self) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        return sorted(p for p in self.dataset_dir.iterdir() if p.suffix.lower() in exts)

    def reference_images(self) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        return sorted(p for p in self.reference_dir.iterdir() if p.suffix.lower() in exts)

    def lora_path(self) -> Path | None:
        raw = self._d.get("lora_path", "")
        if raw:
            p = Path(raw)
            return p if p.exists() else None
        for f in self.lora_dir.glob("**/*.safetensors"):
            return f
        return None

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            d = dict(self._d)
        lora = self.lora_path()
        d["lora_path"] = str(lora) if lora else ""
        return d

    def save(self) -> None:
        path = self.root / "project.json"
        tmp  = path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                with self._lock:
                    json.dump(self._d, f, indent=2)
            os.replace(tmp, path)
        except Exception as exc:
            log.error("Failed to save project %s: %s", self.root, exc)

    def delete(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
