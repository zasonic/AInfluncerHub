import { useState, useEffect } from "react";
import { X } from "lucide-react";
import { useStore } from "../store";
import * as api from "../api";
import type { AppSettings } from "../types";

interface Props {
  onClose: () => void;
}

const DEFAULTS: Partial<AppSettings> = {
  comfyui_url:     "http://localhost:8188",
  wangp_url:       "http://localhost:7860",
  lm_studio_url:   "http://localhost:1234",
  training_steps:  2000,
  lora_rank:       16,
  learning_rate:   "1e-4",
  dataset_method:  "fal",
};

export function SettingsModal({ onClose }: Props) {
  const { settings, setSettings } = useStore();
  const [form, setForm]     = useState<Partial<AppSettings>>(settings ?? DEFAULTS);
  const [saving, setSaving] = useState(false);
  const [saved,  setSaved]  = useState(false);

  useEffect(() => {
    if (settings) setForm(settings);
  }, [settings]);

  const field = (key: keyof AppSettings) => ({
    value:    String(form[key] ?? ""),
    onChange: (e: React.ChangeEvent<HTMLInputElement>) =>
      setForm((f) => ({ ...f, [key]: e.target.value })),
  });

  const save = async () => {
    setSaving(true);
    try {
      await api.updateSettings(form);
      const updated = await api.getSettings();
      setSettings(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch { /* ignore */ }
    setSaving(false);
  };

  const browseToolkit = async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const path = await open({ directory: true, title: "Select ai-toolkit folder" });
      if (path) setForm((f) => ({ ...f, ai_toolkit_path: String(path) }));
    } catch { /* no Tauri in dev mode */ }
  };

  const browseOutput = async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const path = await open({ directory: true, title: "Select output folder" });
      if (path) setForm((f) => ({ ...f, output_dir: String(path) }));
    } catch { /* no Tauri in dev mode */ }
  };

  return (
    <div className="overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal">
        <div className="flex items-center justify-between mb-16">
          <h2>Settings</h2>
          <button className="btn btn-ghost btn-sm" onClick={onClose}>
            <X size={14} />
          </button>
        </div>

        {/* Service endpoints */}
        <h4 style={{ color: "var(--text-muted)", marginBottom: 12, fontSize: 10, letterSpacing: "0.8px", textTransform: "uppercase" }}>
          Service Endpoints
        </h4>

        <div className="field">
          <label>ComfyUI URL</label>
          <input type="url" placeholder="http://localhost:8188" {...field("comfyui_url")} />
        </div>
        <div className="field">
          <label>WanGP URL</label>
          <input type="url" placeholder="http://localhost:7860" {...field("wangp_url")} />
        </div>
        <div className="field">
          <label>LM Studio URL</label>
          <input type="url" placeholder="http://localhost:1234" {...field("lm_studio_url")} />
        </div>

        <div className="divider" />

        <h4 style={{ color: "var(--text-muted)", marginBottom: 12, fontSize: 10, letterSpacing: "0.8px", textTransform: "uppercase" }}>
          API Keys
        </h4>

        <div className="field">
          <label>fal.ai API key</label>
          <input
            type="password"
            placeholder="Required for automatic dataset generation"
            {...field("fal_api_key")}
          />
          <span className="field-hint">Get a free key at fal.ai — about $0.025 per image.</span>
        </div>
        <div className="field">
          <label>HuggingFace token</label>
          <input
            type="password"
            placeholder="Required to download training models"
            {...field("hf_token")}
          />
          <span className="field-hint">Generate a token at huggingface.co/settings/tokens (read access is enough).</span>
        </div>

        <div className="divider" />

        <h4 style={{ color: "var(--text-muted)", marginBottom: 12, fontSize: 10, letterSpacing: "0.8px", textTransform: "uppercase" }}>
          Folder Paths
        </h4>

        <div className="field">
          <label>ai-toolkit folder</label>
          <div className="flex gap-8">
            <input type="text" placeholder="Path to the cloned ai-toolkit repository" {...field("ai_toolkit_path")} />
            <button className="btn btn-ghost btn-sm" style={{ flexShrink: 0 }} onClick={browseToolkit}>
              Browse
            </button>
          </div>
        </div>

        <div className="field">
          <label>Output folder</label>
          <div className="flex gap-8">
            <input
              type="text"
              placeholder="Leave blank for default (./output/influencers)"
              {...field("output_dir")}
            />
            <button className="btn btn-ghost btn-sm" style={{ flexShrink: 0 }} onClick={browseOutput}>
              Browse
            </button>
          </div>
        </div>

        <div className="flex justify-between mt-24">
          <button className="btn btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={save} disabled={saving}>
            {saved ? "Saved" : saving ? "Saving..." : "Save settings"}
          </button>
        </div>
      </div>
    </div>
  );
}
