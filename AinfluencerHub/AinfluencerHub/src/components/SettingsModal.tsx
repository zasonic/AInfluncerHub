import { useState, useEffect } from "react";
import { X } from "lucide-react";
import { useStore } from "../store";
import * as api from "../api";
import type { AppSettings } from "../types";

interface Props {
  onClose: () => void;
}

const DEFAULTS: Partial<AppSettings> = {
  training_steps:  2000,
  lora_rank:       16,
  learning_rate:   "1e-4",
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

        <h4 style={{ color: "var(--text-muted)", marginBottom: 12, fontSize: 10, letterSpacing: "0.8px", textTransform: "uppercase" }}>
          Authentication
        </h4>

        <div className="field">
          <label>HuggingFace token</label>
          <input
            type="password"
            placeholder="Required to download AI models"
            {...field("hf_token")}
          />
          <span className="field-hint">Generate a token at huggingface.co/settings/tokens (read access is enough).</span>
        </div>

        <div className="divider" />

        <h4 style={{ color: "var(--text-muted)", marginBottom: 12, fontSize: 10, letterSpacing: "0.8px", textTransform: "uppercase" }}>
          Storage
        </h4>

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

        <div className="divider" />

        <h4 style={{ color: "var(--text-muted)", marginBottom: 12, fontSize: 10, letterSpacing: "0.8px", textTransform: "uppercase" }}>
          Models
        </h4>

        <div className="field">
          <label>Image model</label>
          <select
            value={String(form.preferred_model ?? "sdxl")}
            onChange={(e) => setForm((f) => ({ ...f, preferred_model: e.target.value }))}
            style={{ width: "100%", fontSize: 12 }}
          >
            <option value="sdxl">Stable Diffusion XL</option>
            <option value="flux">FLUX.1</option>
          </select>
          <span className="field-hint">Base model for image generation and training.</span>
        </div>

        <div className="field">
          <label>Video model</label>
          <select
            value={String(form.video_model ?? "wan2.1")}
            onChange={(e) => setForm((f) => ({ ...f, video_model: e.target.value }))}
            style={{ width: "100%", fontSize: 12 }}
          >
            <option value="wan2.1">Wan 2.1 (best quality)</option>
            <option value="cogvideo">CogVideoX (lighter)</option>
          </select>
          <span className="field-hint">Model for image-to-video animation.</span>
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
