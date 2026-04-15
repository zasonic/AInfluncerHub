import { useState, useRef, useEffect } from "react";
import { Upload } from "lucide-react";
import { useStore } from "../store";
import * as api from "../api";
import { ImageGallery } from "../components/ImageGallery";
import type { StreamEvent } from "../types";

interface Props {
  onAdvance: () => void;
}

type Method = "local" | "manual";

const METHODS: { id: Method; title: string; desc: string }[] = [
  {
    id:    "local",
    title: "Automatic — local generation",
    desc:  "Uses PuLID-FLUX in ComfyUI to create varied poses from your reference photo. Free, runs on your GPU. Requires ComfyUI with PuLID-FLUX nodes.",
  },
  {
    id:    "manual",
    title: "Manual upload",
    desc:  "You provide 20 or more photos yourself. Best control — use real photos or images you already have.",
  },
];

export function Step2Dataset({ onAdvance }: Props) {
  const { activeProject, settings, refreshProject } = useStore();

  const [method,      setMethod]      = useState<Method>(
    settings?.dataset_method === "manual" ? "manual" : "local"
  );
  const [checkpoint,  setCheckpoint]  = useState(settings?.dataset_checkpoint ?? "");
  const [checkpoints, setCheckpoints] = useState<string[]>([]);
  const [count,       setCount]       = useState(25);
  const [images,      setImages]      = useState<{ path: string; filename: string }[]>([]);
  const [running,     setRunning]     = useState(false);
  const [progress,    setProgress]    = useState({ done: 0, total: 0, message: "" });
  const [error,       setError]       = useState("");

  const fileRef    = useRef<HTMLInputElement>(null);
  const sourceRef  = useRef<EventSource | null>(null);

  // Load dataset images and available checkpoints
  useEffect(() => {
    if (activeProject) {
      api.getDatasetImages(activeProject.slug).then(({ images: imgs }) => {
        setImages(imgs.map((p) => ({ path: p, filename: p.split(/[\\/]/).pop() ?? p })));
      }).catch(() => {});
    }
    api.getCheckpoints().then(({ checkpoints: ckpts }) => {
      setCheckpoints(ckpts);
      if (!checkpoint && ckpts.length > 0) setCheckpoint(ckpts[0]);
    }).catch(() => {});
    return () => { sourceRef.current?.close(); };
  }, [activeProject]);

  const startGeneration = () => {
    if (!activeProject) return;
    setError("");

    if (method !== "local") {
      return;
    }
    if (!checkpoint.trim()) {
      setError("Select a checkpoint first. Make sure ComfyUI is running.");
      return;
    }

    setRunning(true);
    setProgress({ done: 0, total: count, message: "Starting..." });

    const es = api.startDatasetGen(activeProject.slug, count, checkpoint.trim());

    sourceRef.current = es;
    api.listenSSE(
      es,
      (event: StreamEvent) => {
        if (event.type === "progress") {
          setProgress({ done: event.done, total: event.total, message: event.message });
        } else if (event.type === "done") {
          setRunning(false);
          refreshProject(activeProject.slug);
          api.getDatasetImages(activeProject.slug).then(({ images: imgs }) => {
            setImages(imgs.map((p) => ({ path: p, filename: p.split(/[\\/]/).pop() ?? p })));
          }).catch(() => {});
        } else if (event.type === "error") {
          setError(event.message);
          setRunning(false);
        }
      },
      () => setRunning(false)
    );
  };

  const uploadManual = async (files: FileList | null) => {
    if (!files || !activeProject) return;
    setRunning(true);
    setError("");
    try {
      const result = await api.uploadDatasetImages(activeProject.slug, Array.from(files));
      await refreshProject(activeProject.slug);
      const { images: imgs } = await api.getDatasetImages(activeProject.slug);
      setImages(imgs.map((p) => ({ path: p, filename: p.split(/[\\/]/).pop() ?? p })));
      setProgress({ done: result.count, total: result.count, message: `${result.count} images uploaded.` });
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  };

  const progressPct = progress.total > 0 ? (progress.done / progress.total) * 100 : 0;
  const canAdvance  = images.length >= 5;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div className="step-header">
        <h1>Generate dataset</h1>
        <p>
          {activeProject?.name} needs 20-30 varied photos to learn from.
          Choose how to create them.
        </p>
      </div>

      <div className="step-container">
        <div className="step-body">

          {/* Method selection */}
          <div className="radio-group mb-20">
            {METHODS.map((m) => (
              <div
                key={m.id}
                className={`radio-option ${method === m.id ? "selected" : ""}`}
                onClick={() => setMethod(m.id)}
              >
                <div className="radio-option-title">
                  <div className="radio-dot" />
                  {m.title}
                </div>
                <div className="radio-option-desc">{m.desc}</div>
              </div>
            ))}
          </div>

          {/* Method-specific config */}
          {method === "local" && (
            <div className="card mb-20">
              <div className="card-title mb-16">ComfyUI generation</div>
              <div className="field">
                <label>Checkpoint</label>
                <select
                  value={checkpoint}
                  onChange={(e) => setCheckpoint(e.target.value)}
                  style={{ width: "100%", fontSize: 12 }}
                >
                  {checkpoints.length === 0 && (
                    <option value="">No checkpoints found — is ComfyUI running?</option>
                  )}
                  {checkpoints.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
                <span className="field-hint">
                  Select any FLUX-compatible checkpoint from your ComfyUI models folder
                </span>
              </div>
              <div className="field">
                <label>Images to generate: {count}</label>
                <div className="slider-row">
                  <input
                    type="range"
                    min={10} max={30} step={1}
                    value={count}
                    onChange={(e) => setCount(Number(e.target.value))}
                  />
                  <span className="slider-value">{count} images</span>
                </div>
              </div>
            </div>
          )}

          {method === "manual" && (
            <div className="card mb-20">
              <div className="card-title mb-8">Upload your photos</div>
              <p className="card-desc mb-16">
                Minimum 20 recommended. More variety means better training results.
              </p>
              <div
                className="drop-zone"
                onClick={() => fileRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add("drag-over"); }}
                onDragLeave={(e) => e.currentTarget.classList.remove("drag-over")}
                onDrop={(e) => {
                  e.preventDefault();
                  e.currentTarget.classList.remove("drag-over");
                  uploadManual(e.dataTransfer.files);
                }}
              >
                <Upload size={20} style={{ color: "var(--text-muted)", marginBottom: 6 }} />
                <h4>Drop photos or click to browse</h4>
                <p style={{ margin: 0 }}>Select as many as you have</p>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".jpg,.jpeg,.png,.webp"
                  multiple
                  style={{ display: "none" }}
                  onChange={(e) => uploadManual(e.target.files)}
                />
              </div>
            </div>
          )}

          {/* Progress */}
          {(running || progress.message) && (
            <div className="card mb-20">
              <div
                className="flex items-center justify-between"
                style={{ marginBottom: 10 }}
              >
                <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>
                  {progress.message}
                </span>
                <span className="text-sm text-muted">
                  {progress.done}/{progress.total}
                </span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>
          )}

          {error && <div className="alert alert-error mb-16">{error}</div>}

          {/* Gallery */}
          {images.length > 0 && (
            <>
              <div
                className="flex items-center justify-between"
                style={{ marginBottom: 12 }}
              >
                <h3>{images.length} images in dataset</h3>
                {images.length < 20 && (
                  <span className="badge badge-warning">
                    Aim for at least 20 for best results
                  </span>
                )}
                {images.length >= 20 && (
                  <span className="badge badge-success">
                    Ready to train
                  </span>
                )}
              </div>
              <ImageGallery
                images={images}
                emptyLabel="No dataset images yet"
              />
            </>
          )}
        </div>
      </div>

      <div className="step-footer">
        <div className="step-footer-left">
          <span className="text-sm text-muted">Step 2 of 5</span>
          {images.length > 0 && (
            <span className="badge badge-neutral">{images.length} images</span>
          )}
        </div>
        <div className="step-footer-right">
          {method !== "manual" && !running && (
            <button className="btn btn-ghost" onClick={startGeneration} disabled={running}>
              Generate dataset
            </button>
          )}
          {running && (
            <button className="btn btn-ghost" onClick={() => sourceRef.current?.close()}>
              Cancel
            </button>
          )}
          <button
            className="btn btn-primary"
            onClick={onAdvance}
            disabled={!canAdvance}
          >
            Continue to Captions
          </button>
        </div>
      </div>
    </div>
  );
}
