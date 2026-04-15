/**
 * api.ts — Typed HTTP client for the Python FastAPI backend.
 *
 * The backend URL is supplied by Tauri via `invoke("get_backend_url")`.
 * Falls back to localhost:8765 for dev convenience.
 */

import type {
  AppSettings,
  CaptionMap,
  GeneratedImage,
  GeneratedVideo,
  PreflightResult,
  Project,
} from "./types";

let _baseUrl = "http://localhost:8765";

export async function initBackendUrl(): Promise<void> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const url = await invoke<string>("get_backend_url");
    if (url) _baseUrl = url;
  } catch {
    // Running outside Tauri (browser dev mode) — keep default
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown
): Promise<T> {
  const res = await fetch(`${_baseUrl}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body:    body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Health ─────────────────────────────────────────────────────────────────

export const health = () =>
  request<{ ok: boolean }>("GET", "/health");

// ── Preflight ──────────────────────────────────────────────────────────────

export const getPreflight = () =>
  request<PreflightResult>("GET", "/api/preflight");

// ── Projects ───────────────────────────────────────────────────────────────

export const listProjects = () =>
  request<Project[]>("GET", "/api/projects");

export const createProject = (data: {
  name:         string;
  trigger_word: string;
  gender:       string;
}) => request<Project>("POST", "/api/projects", data);

export const getProject = (slug: string) =>
  request<Project>("GET", `/api/projects/${slug}`);

export const deleteProject = (slug: string) =>
  request<void>("DELETE", `/api/projects/${slug}`);

export const uploadReferences = async (
  slug: string,
  files: File[]
): Promise<{ count: number }> => {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${_baseUrl}/api/projects/${slug}/references`, {
    method: "POST",
    body:   form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};

// ── Dataset ────────────────────────────────────────────────────────────────

export const startDatasetGen = (
  slug:       string,
  count:      number,
  checkpoint: string
): EventSource =>
  new EventSource(
    `${_baseUrl}/api/dataset/${slug}/generate?count=${count}&checkpoint=${encodeURIComponent(checkpoint)}`
  );

export const uploadDatasetImages = async (
  slug:  string,
  files: File[]
): Promise<{ count: number }> => {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${_baseUrl}/api/dataset/${slug}/upload`, {
    method: "POST",
    body:   form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};

export const getDatasetImages = (slug: string) =>
  request<{ images: string[] }>("GET", `/api/dataset/${slug}/images`);

export interface ImageScore {
  path:             string;
  filename:         string;
  face_score:       number;
  aesthetic_score:  number;
  passed:           boolean;
}

export const scoreDataset = (slug: string) =>
  request<{ scores: ImageScore[]; passed: number; total: number }>(
    "POST", `/api/dataset/${slug}/score`
  );

/** Returns an EventSource streaming SSE progress events */
export const startCaptioning = (
  slug: string,
  hf_token: string,
  captioner: "florence2" | "joycaption" = "florence2"
): EventSource =>
  new EventSource(
    `${_baseUrl}/api/captions/${slug}/run?hf_token=${encodeURIComponent(hf_token)}&captioner=${captioner}`
  );

export const getCaptions = (slug: string) =>
  request<CaptionMap>("GET", `/api/captions/${slug}`);

export const updateCaption = (
  slug:  string,
  stem:  string,
  text:  string
) => request<void>("PUT", `/api/captions/${slug}/${stem}`, { text });

export const injectTrigger = (slug: string) =>
  request<{ updated: number }>("POST", `/api/captions/${slug}/inject-trigger`);

// ── Training ───────────────────────────────────────────────────────────────

export const startTraining = (
  slug:            string,
  ai_toolkit_path: string,
  hf_token:        string,
  steps:           number,
  rank:            number,
  learning_rate:   string
): EventSource =>
  new EventSource(
    `${_baseUrl}/api/training/${slug}/start?` +
    `ai_toolkit_path=${encodeURIComponent(ai_toolkit_path)}&` +
    `hf_token=${encodeURIComponent(hf_token)}&` +
    `steps=${steps}&rank=${rank}&lr=${encodeURIComponent(learning_rate)}`
  );

export const cancelTraining = (slug: string) =>
  request<void>("POST", `/api/training/${slug}/cancel`);

export const cloneAiToolkit = (dest: string): EventSource =>
  new EventSource(
    `${_baseUrl}/api/toolkit/clone?dest=${encodeURIComponent(dest)}`
  );

// ── Studio ─────────────────────────────────────────────────────────────────

export const generateImage = (
  slug:           string,
  prompt:         string,
  lora_strength:  number,
  checkpoint:     string
): EventSource =>
  new EventSource(
    `${_baseUrl}/api/studio/${slug}/generate?` +
    `prompt=${encodeURIComponent(prompt)}&` +
    `lora_strength=${lora_strength}&` +
    `checkpoint=${encodeURIComponent(checkpoint)}`
  );

export const animateImage = (
  slug:         string,
  image_path:   string,
  motion_prompt: string
): EventSource =>
  new EventSource(
    `${_baseUrl}/api/studio/${slug}/animate?` +
    `image_path=${encodeURIComponent(image_path)}&` +
    `motion_prompt=${encodeURIComponent(motion_prompt)}`
  );

export const getGeneratedImages = (slug: string) =>
  request<{ images: GeneratedImage[] }>("GET", `/api/studio/${slug}/images`);

export const getVideos = (slug: string) =>
  request<{ videos: GeneratedVideo[] }>("GET", `/api/studio/${slug}/videos`);

// ── ComfyUI ────────────────────────────────────────────────────

export const getCheckpoints = () =>
  request<{ checkpoints: string[] }>("GET", "/api/comfyui/checkpoints");

export const getComfyUIStatus = () =>
  request<{ online: boolean; has_pulid: boolean; has_wan_video: boolean }>(
    "GET", "/api/comfyui/status"
  );

// ── Settings ───────────────────────────────────────────────────────────────

export const getSettings = () =>
  request<AppSettings>("GET", "/api/settings");

export const updateSettings = (data: Partial<AppSettings>) =>
  request<void>("PUT", "/api/settings", data);

// ── Image URL helper ───────────────────────────────────────────────────────

export const imageUrl = (path: string) =>
  `${_baseUrl}/api/files/image?path=${encodeURIComponent(path)}`;

// ── SSE helper ─────────────────────────────────────────────────────────────

/**
 * Wraps an EventSource in a cleaner callback interface.
 * Handles JSON parsing and cleanup automatically.
 */
export function listenSSE(
  source:     EventSource,
  onEvent:    (e: StreamEvent) => void,
  onComplete: () => void
): () => void {
  source.onmessage = (ev) => {
    try {
      const event = JSON.parse(ev.data) as StreamEvent;
      onEvent(event);
      if (event.type === "done" || event.type === "error") {
        source.close();
        onComplete();
      }
    } catch {
      // ignore malformed events
    }
  };

  source.onerror = () => {
    source.close();
    onComplete();
  };

  return () => source.close();
}
