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
  comfyui_url:         string;
  lm_studio_url:       string;
  hf_token:            string;
  ai_toolkit_path:     string;
  output_dir:          string;
  dataset_method:      "local" | "manual";
  dataset_checkpoint:  string;
  training_steps:      number;
  lora_rank:           number;
  learning_rate:       string;
  caption_model:       "florence2" | "lm_studio";
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
  comfyui:         PreflightItem;
  pulid_nodes:     PreflightItem;
  wan_video_nodes: PreflightItem;
  ai_toolkit:      PreflightItem;
  hf_token:        PreflightItem;
}

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
