import { useState, useEffect, useRef } from "react";
import { useStore } from "../store";
import * as api from "../api";
import { ImageGallery } from "../components/ImageGallery";
import type { StreamEvent, CaptionMap } from "../types";

interface Props {
  onAdvance: () => void;
}

export function Step3Captions({ onAdvance }: Props) {
  const { activeProject, settings, refreshProject } = useStore();

  const [images,      setImages]      = useState<{ path: string; filename: string }[]>([]);
  const [captions,    setCaptions]    = useState<CaptionMap>({});
  const [selected,    setSelected]    = useState<{ path: string; filename: string } | null>(null);
  const [editText,    setEditText]    = useState("");
  const [dirty,       setDirty]       = useState(false);
  const [running,     setRunning]     = useState(false);
  const [progress,    setProgress]    = useState({ done: 0, total: 0, message: "" });
  const [status,      setStatus]      = useState("");
  const [statusType,  setStatusType]  = useState<"ok" | "warn" | "error">("ok");
  const [error,       setError]       = useState("");
  const [captioner,   setCaptioner]   = useState<"florence2" | "joycaption">("florence2");

  const sourceRef = useRef<EventSource | null>(null);
  const slug      = activeProject?.slug ?? "";

  useEffect(() => {
    if (!slug) return;
    api.getDatasetImages(slug)
      .then(({ images: imgs }) =>
        setImages(imgs.map((p) => ({ path: p, filename: p.split(/[\\/]/).pop() ?? p })))
      ).catch(() => {});
    api.getCaptions(slug).then(setCaptions).catch(() => {});
    return () => sourceRef.current?.close();
  }, [slug]);

  const selectImage = (img: { path: string; filename: string }) => {
    if (dirty && selected) saveCaption();
    setSelected(img);
    const stem = img.filename.replace(/\.[^/.]+$/, "");
    setEditText(captions[stem] ?? "");
    setDirty(false);
  };

  const saveCaption = async () => {
    if (!selected || !slug) return;
    const stem = selected.filename.replace(/\.[^/.]+$/, "");
    setCaptions((prev) => ({ ...prev, [stem]: editText }));
    try {
      await api.updateCaption(slug, stem, editText);
    } catch { /* ignore */ }
    setDirty(false);
    setStatus("Caption saved.");
    setStatusType("ok");
  };

  const runAutocaption = () => {
    if (!slug) return;
    const hf_token = settings?.hf_token ?? "";
    setRunning(true);
    setError("");
    const modelLabel = captioner === "joycaption" ? "JoyCaption" : "Florence2";
    setProgress({ done: 0, total: images.length, message: `Loading ${modelLabel}...` });

    const es = api.startCaptioning(slug, hf_token, captioner);
    sourceRef.current = es;
    api.listenSSE(
      es,
      (event: StreamEvent) => {
        if (event.type === "progress") {
          setProgress({ done: event.done, total: event.total, message: event.message });
        } else if (event.type === "done") {
          setRunning(false);
          api.getCaptions(slug).then(setCaptions).catch(() => {});
          setStatus("Auto-captioning complete.");
          setStatusType("ok");
          if (selected) {
            const stem = selected.filename.replace(/\.[^/.]+$/, "");
            api.getCaptions(slug).then((caps) => {
              setEditText(caps[stem] ?? "");
            }).catch(() => {});
          }
        } else if (event.type === "error") {
          setError(event.message);
          setRunning(false);
        }
      },
      () => setRunning(false)
    );
  };

  const injectTrigger = async () => {
    try {
      const { updated } = await api.injectTrigger(slug);
      await api.getCaptions(slug).then(setCaptions);
      setStatus(`Trigger word added to ${updated} caption(s).`);
      setStatusType("ok");
      if (selected) {
        const stem = selected.filename.replace(/\.[^/.]+$/, "");
        const caps = await api.getCaptions(slug);
        setEditText(caps[stem] ?? editText);
      }
    } catch (e) {
      setError(String(e));
    }
  };

  const captionedCount = images.filter(
    (img) => captions[img.filename.replace(/\.[^/.]+$/, "")]
  ).length;

  const canAdvance = images.length >= 5 && captionedCount >= Math.ceil(images.length * 0.5);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div className="step-header">
        <h1>Caption review</h1>
        <p>
          Auto-caption all images, then review and edit individual captions.
          The trigger word is added automatically.
        </p>
      </div>

      {/* Progress bar (shown while running) */}
      {running && (
        <div
          style={{
            padding:      "10px 32px",
            background:   "var(--bg-surface)",
            borderBottom: "1px solid var(--border-subtle)",
          }}
        >
          <div
            className="flex items-center justify-between"
            style={{ marginBottom: 6 }}
          >
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>
              {progress.message}
            </span>
            <span className="text-xs text-muted">
              {progress.done}/{progress.total}
            </span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: `${progress.total > 0 ? (progress.done / progress.total) * 100 : 0}%`,
              }}
            />
          </div>
        </div>
      )}

      {/* Two-pane layout */}
      <div
        style={{
          flex:     1,
          display:  "grid",
          gridTemplateColumns: "300px 1fr",
          overflow: "hidden",
          gap:      0,
        }}
      >
        {/* Gallery pane */}
        <div
          style={{
            borderRight:  "1px solid var(--border-subtle)",
            overflowY:    "auto",
            padding:      "12px 12px",
            scrollbarWidth: "thin",
          }}
        >
          <div
            className="flex items-center justify-between"
            style={{ marginBottom: 10 }}
          >
            <span className="text-xs text-muted">
              {captionedCount}/{images.length} captioned
            </span>
          </div>
          <ImageGallery
            images={images}
            selected={selected?.path}
            onSelect={selectImage}
            columns={2}
          />
        </div>

        {/* Editor pane */}
        <div
          style={{
            display:        "flex",
            flexDirection:  "column",
            padding:        "20px 24px",
            gap:            16,
            overflow:       "hidden",
          }}
        >
          {!selected ? (
            <div
              style={{
                flex:           1,
                display:        "flex",
                alignItems:     "center",
                justifyContent: "center",
                color:          "var(--text-muted)",
                fontSize:       13,
              }}
            >
              Select an image to edit its caption
            </div>
          ) : (
            <>
              <div>
                <h4 style={{ marginBottom: 4 }}>{selected.filename}</h4>
                <span className="text-xs text-muted">
                  {captions[selected.filename.replace(/\.[^/.]+$/, "")]
                    ? "Caption exists"
                    : "No caption yet"}
                </span>
              </div>

              <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
                <label style={{ fontSize: 12, color: "var(--text-secondary)", fontWeight: 500 }}>
                  Caption text
                </label>
                <textarea
                  value={editText}
                  onChange={(e) => { setEditText(e.target.value); setDirty(true); }}
                  style={{ flex: 1, minHeight: 120, fontFamily: "var(--font-sans)", fontSize: 13 }}
                  placeholder="Caption will appear here after auto-captioning..."
                />
              </div>

              <div className="flex gap-8">
                <button
                  className="btn btn-primary btn-sm"
                  onClick={saveCaption}
                  disabled={!dirty}
                >
                  Save caption
                </button>
                {dirty && (
                  <button
                    className="btn btn-ghost btn-sm"
                    onClick={() => {
                      const stem = selected.filename.replace(/\.[^/.]+$/, "");
                      setEditText(captions[stem] ?? "");
                      setDirty(false);
                    }}
                  >
                    Discard
                  </button>
                )}
              </div>
            </>
          )}

          {/* Bulk actions */}
          <div
            style={{
              padding:       "16px",
              background:    "var(--bg-panel)",
              borderRadius:  "var(--radius-md)",
              border:        "1px solid var(--border-subtle)",
            }}
          >
            <h4 style={{ marginBottom: 10 }}>Batch actions</h4>
            <div className="field" style={{ margin: "0 0 10px 0" }}>
              <label style={{ fontSize: 11 }}>Captioning model</label>
              <div className="flex gap-8">
                {(["florence2", "joycaption"] as const).map((c) => (
                  <label
                    key={c}
                    style={{
                      display: "flex", alignItems: "center", gap: 5,
                      cursor: "pointer", fontSize: 12,
                      color: captioner === c ? "var(--text-primary)" : "var(--text-secondary)",
                    }}
                  >
                    <input
                      type="radio" name="captioner" value={c}
                      checked={captioner === c}
                      onChange={() => setCaptioner(c)}
                      style={{ accentColor: "var(--accent)" }}
                    />
                    {c === "florence2" ? "Florence-2 (fast, 4 GB)" : "JoyCaption (best, 10 GB)"}
                  </label>
                ))}
              </div>
            </div>
            <div className="flex gap-8">
              <button
                className="btn btn-ghost btn-sm"
                onClick={runAutocaption}
                disabled={running}
              >
                {running ? "Captioning..." : "Auto-caption all images"}
              </button>
              <button
                className="btn btn-ghost btn-sm"
                onClick={injectTrigger}
                disabled={running}
                title={`Adds "${activeProject?.trigger_word}" to the start of every caption`}
              >
                Add trigger word to all
              </button>
            </div>
            <p className="text-xs text-muted mt-8">
              Trigger word: <span className="mono">{activeProject?.trigger_word}</span>
            </p>
          </div>

          {error  && <div className="alert alert-error">{error}</div>}
          {status && !error && (
            <div className={`alert alert-${statusType === "ok" ? "success" : "warning"}`}>
              {status}
            </div>
          )}
        </div>
      </div>

      <div className="step-footer">
        <div className="step-footer-left">
          <span className="text-sm text-muted">Step 3 of 5</span>
          <span className="badge badge-neutral">{captionedCount}/{images.length} captioned</span>
        </div>
        <div className="step-footer-right">
          {!canAdvance && (
            <span className="text-xs text-muted" style={{ marginRight: 8 }}>
              Auto-caption at least half your images to continue
            </span>
          )}
          <button
            className="btn btn-primary"
            onClick={onAdvance}
            disabled={!canAdvance}
          >
            Continue to Training
          </button>
        </div>
      </div>
    </div>
  );
}
