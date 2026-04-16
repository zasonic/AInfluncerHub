import { useState, useRef, useEffect } from "react";
import { useStore } from "../store";
import * as api from "../api";
import type { StreamEvent } from "../types";

interface Props {
  onAdvance: () => void;
}

export function Step4Training({ onAdvance }: Props) {
  const { activeProject, settings, refreshProject } = useStore();

  const [hfToken,   setHfToken]   = useState(settings?.hf_token ?? "");
  const [steps,     setSteps]     = useState(settings?.training_steps ?? 2000);
  const [rank,      setRank]      = useState(settings?.lora_rank ?? 16);
  const [lr,        setLr]        = useState(settings?.learning_rate ?? "1e-4");
  const [logLines,  setLogLines]  = useState<{ text: string; type: "normal" | "error" | "warn" }[]>([]);
  const [running,   setRunning]   = useState(false);
  const [progress,  setProgress]  = useState(0);
  const [status,    setStatus]    = useState("");
  const [statusOk,  setStatusOk]  = useState(true);
  const [done,      setDone]      = useState(false);
  const [error,     setError]     = useState("");
  const [eta,       setEta]       = useState("");
  const [vramWarn,  setVramWarn]  = useState("");

  const sourceRef  = useRef<EventSource | null>(null);
  const logRef     = useRef<HTMLDivElement>(null);
  const stepTimes  = useRef<number[]>([]);
  const lastStep   = useRef<{ step: number; time: number } | null>(null);
  const slug       = activeProject?.slug ?? "";

  // Check if already trained
  useEffect(() => {
    if (activeProject?.steps_done.includes(4)) setDone(true);
  }, [activeProject]);

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  const addLog = (text: string, type: "normal" | "error" | "warn" = "normal") => {
    setLogLines((prev) => [...prev.slice(-500), { text, type }]);
  };

  const validate = (): string | null => {
    if (!hfToken.trim()) return "HuggingFace token is required to download the training model.";
    return null;
  };

  // Check VRAM on mount
  useEffect(() => {
    api.getPreflight().then((result: Record<string, { ok: boolean; detail?: string }>) => {
      const gpu = result?.gpu;
      if (gpu && !gpu.ok) {
        setVramWarn("No GPU detected. Training will be extremely slow on CPU.");
      } else if (gpu?.detail) {
        const vramMatch = gpu.detail.match(/([\d.]+)\s*GB/i);
        if (vramMatch && parseFloat(vramMatch[1]) < 8) {
          setVramWarn(`Low VRAM (${vramMatch[1]} GB). Training may fail or be very slow. 8+ GB recommended.`);
        }
      }
    }).catch(() => {});
  }, []);

  const startTraining = () => {
    const err = validate();
    if (err) { setError(err); return; }
    setError("");
    setRunning(true);
    setDone(false);
    setProgress(0);
    setEta("");
    stepTimes.current = [];
    lastStep.current = null;
    setLogLines([]);
    addLog(`Starting training: ${steps} steps, rank ${rank}, lr ${lr}`);

    const es = api.startTraining(slug, hfToken.trim(), steps, rank, lr);
    sourceRef.current = es;
    api.listenSSE(
      es,
      (event: StreamEvent) => {
        if (event.type === "log") {
          const text = event.line;
          const type = text.toLowerCase().includes("error") ? "error"
            : text.toLowerCase().includes("warn")  ? "warn"
            : "normal";
          addLog(text, type);

          // Parse step progress from log line e.g. "step: 120/2000"
          const m = text.match(/(\d+)\s*\/\s*(\d+)/);
          if (m) {
            const current = Number(m[1]);
            const total = Number(m[2]);
            const pct = current / total;
            if (pct >= 0 && pct <= 1) setProgress(pct);

            // ETA calculation from step timings
            const now = Date.now();
            if (lastStep.current && current > lastStep.current.step) {
              const elapsed = (now - lastStep.current.time) / 1000;
              const stepsCompleted = current - lastStep.current.step;
              stepTimes.current.push(elapsed / stepsCompleted);
              if (stepTimes.current.length > 20) stepTimes.current.shift();
            }
            lastStep.current = { step: current, time: now };

            if (stepTimes.current.length >= 2) {
              const avgSecPerStep = stepTimes.current.reduce((a, b) => a + b, 0) / stepTimes.current.length;
              const remaining = (total - current) * avgSecPerStep;
              if (remaining > 3600) {
                setEta(`~${(remaining / 3600).toFixed(1)}h remaining`);
              } else if (remaining > 60) {
                setEta(`~${Math.round(remaining / 60)}m remaining`);
              } else {
                setEta(`~${Math.round(remaining)}s remaining`);
              }
            }
          }
        } else if (event.type === "done") {
          setRunning(false);
          setDone(true);
          setProgress(1);
          setStatus("Training complete. LoRA saved.");
          setStatusOk(true);
          refreshProject(slug);
          addLog("Training finished successfully.");
        } else if (event.type === "error") {
          setRunning(false);
          setStatus(event.message);
          setStatusOk(false);
          addLog(event.message, "error");
        }
      },
      () => {
        if (!done) setRunning(false);
      }
    );
  };

  const cancelTraining = () => {
    sourceRef.current?.close();
    api.cancelTraining(slug).catch(() => {});
    setRunning(false);
    addLog("Training cancelled.");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div className="step-header">
        <h1>Train LoRA model</h1>
        <p>
          This teaches the AI to recognise {activeProject?.name}. Training takes 1-3 hours
          depending on your GPU.
        </p>
      </div>

      <div className="step-container">
        <div
          style={{
            display:   "grid",
            gridTemplateColumns: "280px 1fr",
            gap:       20,
            padding:   "24px 32px",
            height:    "100%",
            boxSizing: "border-box",
          }}
        >
          {/* Config panel */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16, overflowY: "auto" }}>
            <div className="card">
              <div className="card-title mb-16">HuggingFace token</div>
              <div className="field">
                <input
                  type="password"
                  placeholder="Required to download base model"
                  value={hfToken}
                  onChange={(e) => setHfToken(e.target.value)}
                />
                <span className="field-hint">
                  Get a read token at huggingface.co/settings/tokens
                </span>
              </div>
            </div>

            <div className="card">
              <div className="card-title mb-16">Training parameters</div>

              <div className="field">
                <label>Steps: {steps.toLocaleString()}</label>
                <div className="slider-row">
                  <input
                    type="range" min={500} max={4000} step={100}
                    value={steps}
                    onChange={(e) => setSteps(Number(e.target.value))}
                  />
                  <span className="slider-value">{steps}</span>
                </div>
              </div>

              <div className="field">
                <label>LoRA rank</label>
                <div className="flex gap-8">
                  {[8, 16, 32, 64].map((r) => (
                    <label
                      key={r}
                      style={{
                        display:     "flex",
                        alignItems:  "center",
                        gap:         5,
                        cursor:      "pointer",
                        fontSize:    12,
                        color:       rank === r ? "var(--text-primary)" : "var(--text-secondary)",
                      }}
                    >
                      <input
                        type="radio" name="rank" value={r}
                        checked={rank === r}
                        onChange={() => setRank(r)}
                        style={{ accentColor: "var(--accent)" }}
                      />
                      {r}
                    </label>
                  ))}
                </div>
              </div>

              <div className="field">
                <label>Learning rate</label>
                <div className="flex gap-8">
                  {["5e-5", "1e-4", "2e-4"].map((l) => (
                    <label
                      key={l}
                      style={{
                        display:     "flex",
                        alignItems:  "center",
                        gap:         5,
                        cursor:      "pointer",
                        fontSize:    12,
                        color:       lr === l ? "var(--text-primary)" : "var(--text-secondary)",
                        fontFamily:  "var(--font-mono)",
                      }}
                    >
                      <input
                        type="radio" name="lr" value={l}
                        checked={lr === l}
                        onChange={() => setLr(l)}
                        style={{ accentColor: "var(--accent)" }}
                      />
                      {l}
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Log pane */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12, overflow: "hidden" }}>
            <div className="flex items-center justify-between">
              <h3>Training log</h3>
              {running && (
                <span className="badge badge-warning">Running</span>
              )}
              {done && (
                <span className="badge badge-success">Complete</span>
              )}
            </div>

            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
            </div>

            <div
              ref={logRef}
              className="log-viewer"
              style={{ flex: 1, maxHeight: "none" }}
            >
              {logLines.length === 0 ? (
                <span style={{ color: "var(--text-muted)" }}>
                  Log output will appear here when training starts...
                </span>
              ) : (
                logLines.map((l, i) => (
                  <div key={i} className={l.type === "error" ? "log-line-error" : l.type === "warn" ? "log-line-warning" : ""}>
                    {l.text}
                  </div>
                ))
              )}
            </div>

            {status && (
              <div className={`alert ${statusOk ? "alert-success" : "alert-error"}`}>
                {status}
              </div>
            )}
            {error && <div className="alert alert-error">{error}</div>}
            {vramWarn && !running && !done && (
              <div className="alert alert-warning" style={{ padding: "8px 12px", fontSize: 12 }}>
                {vramWarn}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="step-footer">
        <div className="step-footer-left">
          <span className="text-sm text-muted">Step 4 of 5</span>
          {running && (
            <span className="text-xs text-muted">
              {Math.round(progress * 100)}% complete{eta ? ` — ${eta}` : ""}
            </span>
          )}
        </div>
        <div className="step-footer-right">
          {running ? (
            <button className="btn btn-ghost" onClick={cancelTraining}>
              Cancel training
            </button>
          ) : (
            <button
              className="btn btn-ghost"
              onClick={startTraining}
              disabled={running}
            >
              Start training
            </button>
          )}
          <button
            className="btn btn-primary"
            onClick={onAdvance}
            disabled={!done}
          >
            Continue to Studio
          </button>
        </div>
      </div>
    </div>
  );
}
