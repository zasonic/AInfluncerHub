import { useState, useRef, useEffect } from "react";
import { useStore } from "../store";
import * as api from "../api";
import { useAsyncOperation } from "../hooks/useAsyncOperation";
import { useSSE } from "../hooks/useSSE";

interface Props {
  onAdvance: () => void;
}

export function Step4Training({ onAdvance }: Props) {
  const { activeProject, settings, refreshProject } = useStore();

  const [hfToken,  setHfToken]  = useState(settings?.hf_token ?? "");
  const [steps,    setSteps]    = useState(settings?.training_steps ?? 2000);
  const [rank,     setRank]     = useState(settings?.lora_rank ?? 32);
  const [lr,       setLr]       = useState(settings?.learning_rate ?? "1e-4");
  const [logLines, setLogLines] = useState<{ text: string; type: "normal" | "error" | "warn" }[]>([]);
  const [done,     setDone]     = useState(false);
  const [eta,      setEta]      = useState("");

  const logRef      = useRef<HTMLDivElement>(null);
  const startTimeRef = useRef<number | null>(null);
  const slug        = activeProject?.slug ?? "";

  const op  = useAsyncOperation();
  const sse = useSSE({
    onEvent: (event) => {
      if (event.type === "log") {
        const text = event.line;
        const type: "normal" | "error" | "warn" = text.toLowerCase().includes("error") ? "error"
          : text.toLowerCase().includes("warn")  ? "warn"
          : "normal";
        addLog(text, type);

        const m = text.match(/(\d+)\s*\/\s*(\d+)/);
        if (m) {
          const total    = Number(m[2]);
          const doneStep = Number(m[1]);
          if (total > 0 && doneStep >= 0 && doneStep <= total) {
            op.setProgress(doneStep, total, text);

            if (doneStep > 0 && startTimeRef.current) {
              const elapsed     = (Date.now() - startTimeRef.current) / 1000;
              const secsPerStep = elapsed / doneStep;
              const remaining   = Math.round(secsPerStep * (total - doneStep));
              const h           = Math.floor(remaining / 3600);
              const min         = Math.floor((remaining % 3600) / 60);
              setEta(h > 0 ? `~${h}h ${min}m remaining` : `~${min}m remaining`);
            }
          }
        }
      } else if (event.type === "done") {
        setDone(true);
        setEta("");
        op.succeed("Training complete. LoRA saved.");
        refreshProject(slug);
        addLog("Training finished successfully.");
      } else if (event.type === "error") {
        op.fail(event.message);
        setEta("");
        addLog(event.message, "error");
      }
    },
  });

  const { running, progress, error, status, statusKind } = op.state;
  const progressPct = progress.total > 0 ? progress.done / progress.total : 0;

  useEffect(() => {
    if (activeProject?.steps_done.includes(4)) setDone(true);
  }, [activeProject]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  const addLog = (text: string, type: "normal" | "error" | "warn" = "normal") => {
    setLogLines((prev) => [...prev.slice(-500), { text, type }]);
  };

  const startTraining = () => {
    if (!hfToken.trim()) {
      op.fail("HuggingFace token is required to download the training model.");
      return;
    }
    setDone(false);
    setLogLines([]);
    setEta("");
    startTimeRef.current = Date.now();
    op.start(`Starting training: ${steps} steps, rank ${rank}, lr ${lr}`, steps);
    addLog(`Starting training: ${steps} steps, rank ${rank}, lr ${lr}`);
    sse.start(api.startTraining(slug, hfToken.trim(), steps, rank, lr));
  };

  const cancelTraining = () => {
    sse.stop();
    api.cancelTraining(slug).catch(() => {});
    setEta("");
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
              <div className="progress-fill" style={{ width: `${progressPct * 100}%` }} />
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
              <div className={`alert ${statusKind === "ok" ? "alert-success" : "alert-error"}`}>
                {status}
              </div>
            )}
            {error && <div className="alert alert-error">{error}</div>}
          </div>
        </div>
      </div>

      <div className="step-footer">
        <div className="step-footer-left">
          <span className="text-sm text-muted">Step 4 of 5</span>
          {running && (
            <span className="text-xs text-muted">
              {Math.round(progressPct * 100)}% complete
              {eta ? ` · ${eta}` : ""}
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
              className="btn btn-primary"
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
