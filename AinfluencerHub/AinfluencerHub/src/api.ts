/**
 * api.ts — Typed HTTP client for the Python FastAPI backend.
 *
 * The backend URL is supplied by Tauri via `invoke("get_backend_url")`.
 * Falls back to localhost:8765 for dev convenience.
 *
 * All GPU operations (generation, training, captioning) run natively
 * on the backend — no external services required.
 *
 * All mutating operations use POST (not GET) per HTTP semantics.
 * Sensitive tokens are sent in POST bodies, never in URLs.
 */

import type {
  AppSettings,
  CaptionMap,
  GeneratedImage,
  GeneratedVideo,
  ModelStatusMap,
  PreflightResult,
  Project,
  StreamEvent,
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

// ── POST-based SSE helper ─────────────────────────────────────────────────

export interface SSEHandle {
  abort: () => void;
}

/**
 * Start a POST-based SSE stream. Uses fetch() instead of EventSource so
 * we can send JSON bodies (and keep tokens out of URLs).
 */
export function startPostSSE(
  path:       string,
  body:       unknown,
  onEvent:    (e: StreamEvent) => void,
  onComplete: () => void,
): SSEHandle {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${_baseUrl}${path}`, {
        method:  "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body:    JSON.stringify(body),
        signal:  controller.signal,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => res.statusText);
        onEvent({ type: "error", message: text || `HTTP ${res.status}` });
        onComplete();
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) { onComplete(); return; }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith("data:")) continue;
          const jsonStr = trimmed.slice(5).trim();
          if (!jsonStr) continue;
          try {
            const event = JSON.parse(jsonStr) as StreamEvent;
            onEvent(event);
            if (event.type === "done" || event.type === "error") {
              reader.cancel();
              onComplete();
              return;
            }
          } catch {
            // ignore malformed
          }
        }
      }
      onComplete();
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        onEvent({ type: "error", message: String(err) });
      }
      onComplete();
    }
  })();

  return { abort: () => controller.abort() };
}

// ── Dataset ────────────────────────────────────────────────────────────────

export function startDatasetGen(
  slug:  string,
  count: number,
  onEvent: (e: StreamEvent) => void,
  onComplete: () => void,
): SSEHandle {
  return startPostSSE(
    `/api/dataset/${slug}/generate`,
    { count },
    onEvent,
    onComplete,
  );
}

export const cancelDatasetGen = (slug: string) =>
  request<void>("POST", `/api/dataset/${slug}/cancel`);

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

// ── Captions ──────────────────────────────────────────────────────────────

export function startCaptioning(
  slug:      string,
  hf_token:  string,
  captioner: "florence2" | "joycaption",
  onEvent:   (e: StreamEvent) => void,
  onComplete: () => void,
): SSEHandle {
  return startPostSSE(
    `/api/captions/${slug}/run`,
    { hf_token, captioner },
    onEvent,
    onComplete,
  );
}

export const cancelCaptioning = (slug: string) =>
  request<void>("POST", `/api/captions/${slug}/cancel`);

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

export function startTraining(
  slug:          string,
  hf_token:      string,
  steps:         number,
  rank:          number,
  learning_rate: string,
  onEvent:       (e: StreamEvent) => void,
  onComplete:    () => void,
): SSEHandle {
  return startPostSSE(
    `/api/training/${slug}/start`,
    { hf_token, steps, rank, learning_rate },
    onEvent,
    onComplete,
  );
}

export const cancelTraining = (slug: string) =>
  request<void>("POST", `/api/training/${slug}/cancel`);

// ── Model management ──────────────────────────────────────────────────────

export const getModelStatus = () =>
  request<ModelStatusMap>("GET", "/api/models/status");

export const downloadModel = (modelHfId: string): EventSource =>
  new EventSource(
    `${_baseUrl}/api/models/download?model_hf_id=${encodeURIComponent(modelHfId)}`
  );

// ── Studio ─────────────────────────────────────────────────────────────────

export function generateImage(
  slug:           string,
  prompt:         string,
  lora_strength:  number,
  onEvent:        (e: StreamEvent) => void,
  onComplete:     () => void,
): SSEHandle {
  return startPostSSE(
    `/api/studio/${slug}/generate`,
    { prompt, lora_strength },
    onEvent,
    onComplete,
  );
}

export function animateImage(
  slug:          string,
  image_path:    string,
  motion_prompt: string,
  onEvent:       (e: StreamEvent) => void,
  onComplete:    () => void,
): SSEHandle {
  return startPostSSE(
    `/api/studio/${slug}/animate`,
    { image_path, motion_prompt },
    onEvent,
    onComplete,
  );
}

export const getGeneratedImages = (slug: string) =>
  request<{ images: GeneratedImage[] }>("GET", `/api/studio/${slug}/images`);

export const getVideos = (slug: string) =>
  request<{ videos: GeneratedVideo[] }>("GET", `/api/studio/${slug}/videos`);

// ── Settings ───────────────────────────────────────────────────────────────

export const getSettings = () =>
  request<AppSettings>("GET", "/api/settings");

export const updateSettings = (data: Partial<AppSettings>) =>
  request<void>("PUT", "/api/settings", data);

// ── Image URL helper ───────────────────────────────────────────────────────

export const imageUrl = (path: string) =>
  `${_baseUrl}/api/files/image?path=${encodeURIComponent(path)}`;

// ── Legacy SSE helper (for GET-based EventSource like model downloads) ────

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
