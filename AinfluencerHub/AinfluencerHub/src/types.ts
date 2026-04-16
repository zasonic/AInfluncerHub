// ── Project ───────────────────────────────────────────────────────────────

export interface Project {
  slug:          string;
  name:          string;
  trigger_word:  string;
  gender:        "female" | "male" | "neutral";
  created_at:    string;
  steps_done:    number[];
  dataset_count: number;
  lora_path:     string;
  status:        "new" | "in_progress" | "complete";
}

// ── Settings ──────────────────────────────────────────────────────────────

export interface AppSettings {
  hf_token:            string;
  output_dir:          string;
  dataset_method:      "local" | "manual";
  training_steps:      number;
  lora_rank:           number;
  learning_rate:       string;
  preferred_model:     string;
  video_model:         string;
  theme:               "dark";
  last_project:        string;
  setup_complete:      boolean;
}

// ── Preflight ─────────────────────────────────────────────────────────────

export interface PreflightItem {
  ok:     boolean;
  detail: string;
}

export interface PreflightResult {
  gpu:            PreflightItem;
  ml_libraries:   PreflightItem;
  models:         PreflightItem;
  hf_token:       PreflightItem;
}

// ── Model status ──────────────────────────────────────────────────────────

export interface ModelStatus {
  hf_id:    string;
  size_gb:  number;
  purpose:  string;
  required: boolean;
  cached:   boolean;
}

export type ModelStatusMap = Record<string, ModelStatus>;

// ── API responses ─────────────────────────────────────────────────────────

export interface ApiOk<T = void> {
  ok:   true;
  data: T;
}

export interface ApiErr {
  ok:    false;
  error: string;
}

export type ApiResult<T = void> = ApiOk<T> | ApiErr;

// ── SSE events ────────────────────────────────────────────────────────────

export interface ProgressEvent {
  type:    "progress";
  done:    number;
  total:   number;
  message: string;
}

export interface LogEvent {
  type: "log";
  line: string;
}

export interface DoneEvent {
  type:    "done";
  message: string;
  payload?: unknown;
}

export interface ErrorEvent {
  type:    "error";
  message: string;
}

export type StreamEvent = ProgressEvent | LogEvent | DoneEvent | ErrorEvent;

// ── Caption ───────────────────────────────────────────────────────────────

export interface CaptionMap {
  [image_stem: string]: string;
}

// ── Generation result ─────────────────────────────────────────────────────

export interface GeneratedImage {
  path:     string;
  filename: string;
  url:      string;
}

export interface GeneratedVideo {
  path:     string;
  filename: string;
}
