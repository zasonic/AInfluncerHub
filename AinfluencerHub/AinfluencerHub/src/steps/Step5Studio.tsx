import { useState, useEffect, useRef } from "react";
import { FolderOpen, Play } from "lucide-react";
import { useStore } from "../store";
import * as api from "../api";
import { ImageGallery } from "../components/ImageGallery";
import type { GeneratedImage, GeneratedVideo, StreamEvent } from "../types";

interface Props {
  onAdvance: () => void;
}

export function Step5Studio({ onAdvance }: Props) {
  const { activeProject, settings } = useStore();

  const [genImages,   setGenImages]   = useState<GeneratedImage[]>([]);
  const [videos,      setVideos]      = useState<GeneratedVideo[]>([]);
  const [prompt,      setPrompt]      = useState("");
  const [strength,    setStrength]    = useState(0.85);
  const [checkpoint,  setCheckpoint]  = useState("zimage_turbo.safetensors");
  const [motionPrompt, setMotionPrompt] = useState("The person smiles gently and turns their head slightly.");
  const [selectedImg, setSelectedImg] = useState<GeneratedImage | null>(null);
  const [generating,  setGenerating]  = useState(false);
  const [animating,   setAnimating]   = useState(false);
  const [genStatus,   setGenStatus]   = useState("");
  const [vidStatus,   setVidStatus]   = useState("");
  const [genError,    setGenError]    = useState("");
  const [vidError,    setVidError]    = useState("");

  const genSourceRef = useRef<EventSource | null>(null);
  const vidSourceRef = useRef<EventSource | null>(null);
  const slug         = activeProject?.slug ?? "";

  // Pre-fill prompt with trigger word
  useEffect(() => {
    if (activeProject) {
      setPrompt(
        `${activeProject.trigger_word}, portrait photo, professional photography, sharp focus, beautiful lighting`
      );
    }
  }, [activeProject]);

  // Load existing generated content
  useEffect(() => {
    if (!slug) return;
    api.getGeneratedImages(slug).then(({ images }) => setGenImages(images)).catch(() => {});
    api.getVideos(slug).then(({ videos: vids }) => setVideos(vids)).catch(() => {});
    return () => {
      genSourceRef.current?.close();
      vidSourceRef.current?.close();
    };
  }, [slug]);

  const generateImage = () => {
    if (!slug) return;
    setGenError("");
    setGenerating(true);
    setGenStatus("Submitting to ComfyUI...");

    const es = api.generateImage(slug, prompt, strength, checkpoint);
    genSourceRef.current = es;
    api.listenSSE(
      es,
      (event: StreamEvent) => {
        if (event.type === "progress" || event.type === "log") {
          setGenStatus(event.type === "progress" ? event.message : (event as { line: string }).line);
        } else if (event.type === "done") {
          setGenerating(false);
          setGenStatus("Image generated.");
          api.getGeneratedImages(slug).then(({ images }) => setGenImages(images)).catch(() => {});
        } else if (event.type === "error") {
          setGenError(event.message);
          setGenerating(false);
        }
      },
      () => setGenerating(false)
    );
  };

  const animateImage = () => {
    if (!selectedImg || !slug) return;
    setVidError("");
    setAnimating(true);
    setVidStatus("Sending to WanGP...");

    const es = api.animateImage(slug, selectedImg.path, motionPrompt);
    vidSourceRef.current = es;
    api.listenSSE(
      es,
      (event: StreamEvent) => {
        if (event.type === "progress" || event.type === "log") {
          setVidStatus(event.type === "progress" ? event.message : (event as { line: string }).line);
        } else if (event.type === "done") {
          setAnimating(false);
          setVidStatus("Video created.");
          api.getVideos(slug).then(({ videos: vids }) => setVideos(vids)).catch(() => {});
        } else if (event.type === "error") {
          setVidError(event.message);
          setAnimating(false);
        }
      },
      () => setAnimating(false)
    );
  };

  const openFolder = async (path: string) => {
    try {
      const { openPath } = await import("@tauri-apps/plugin-opener");
      await openPath(path);
    } catch {
      // Dev mode fallback
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div className="step-header">
        <h1>Content Studio</h1>
        <p>
          Generate new images with your trained LoRA, then animate them into video.
        </p>
      </div>

      <div
        style={{
          flex:     1,
          display:  "grid",
          gridTemplateColumns: "1fr 360px",
          overflow: "hidden",
          gap:      0,
        }}
      >
        {/* Left: image generation */}
        <div
          style={{
            display:        "flex",
            flexDirection:  "column",
            borderRight:    "1px solid var(--border-subtle)",
            overflow:       "hidden",
          }}
        >
          {/* Prompt bar */}
          <div
            style={{
              padding:      "16px 20px",
              borderBottom: "1px solid var(--border-subtle)",
              background:   "var(--bg-surface)",
              display:      "flex",
              flexDirection: "column",
              gap:          10,
            }}
          >
            <div className="field" style={{ margin: 0 }}>
              <label>Scene prompt</label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={2}
                style={{ resize: "none", minHeight: "unset" }}
              />
            </div>

            <div
              style={{
                display:     "grid",
                gridTemplateColumns: "1fr 1fr auto",
                gap:         10,
                alignItems:  "end",
              }}
            >
              <div className="field" style={{ margin: 0 }}>
                <label>Checkpoint</label>
                <input
                  type="text"
                  value={checkpoint}
                  onChange={(e) => setCheckpoint(e.target.value)}
                  style={{ fontSize: 12 }}
                />
              </div>

              <div className="field" style={{ margin: 0 }}>
                <label>LoRA strength: {strength.toFixed(2)}</label>
                <div className="slider-row" style={{ marginTop: 8 }}>
                  <input
                    type="range"
                    min={0.4} max={1.2} step={0.05}
                    value={strength}
                    onChange={(e) => setStrength(Number(e.target.value))}
                  />
                  <span className="slider-value">{strength.toFixed(2)}</span>
                </div>
              </div>

              <button
                className="btn btn-primary"
                onClick={generateImage}
                disabled={generating || !prompt.trim()}
                style={{ height: 36, alignSelf: "flex-end" }}
              >
                {generating ? "Generating..." : "Generate"}
              </button>
            </div>

            {genStatus && !genError && (
              <span className="text-xs" style={{ color: "var(--text-muted)" }}>{genStatus}</span>
            )}
            {genError && <div className="alert alert-error" style={{ padding: "8px 12px" }}>{genError}</div>}
          </div>

          {/* Generated gallery */}
          <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px" }}>
            <div className="flex items-center justify-between mb-16">
              <h3>{genImages.length} generated image{genImages.length !== 1 ? "s" : ""}</h3>
              {genImages.length > 0 && (
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => openFolder(genImages[0]?.path.split(/[\\/]/).slice(0, -1).join("/"))}
                >
                  <FolderOpen size={13} />
                  Open folder
                </button>
              )}
            </div>

            <ImageGallery
              images={genImages.map((img) => ({ path: img.path, filename: img.filename, url: img.url }))}
              selected={selectedImg?.path}
              onSelect={(img) => {
                const found = genImages.find((g) => g.path === img.path);
                if (found) setSelectedImg(found);
              }}
              emptyLabel="No images yet — generate one above"
            />
          </div>
        </div>

        {/* Right: video animation */}
        <div
          style={{
            display:       "flex",
            flexDirection: "column",
            overflow:      "hidden",
          }}
        >
          <div
            style={{
              padding:      "16px 20px",
              borderBottom: "1px solid var(--border-subtle)",
              background:   "var(--bg-surface)",
            }}
          >
            <h3 style={{ marginBottom: 6 }}>Animate to video</h3>
            <p className="text-xs text-muted">
              Select an image on the left, describe the motion, then animate via ComfyUI.
            </p>
          </div>

          <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px", display: "flex", flexDirection: "column", gap: 14 }}>
            {/* Selected image preview */}
            <div>
              <label
                style={{
                  fontSize:    12,
                  fontWeight:  500,
                  color:       "var(--text-secondary)",
                  display:     "block",
                  marginBottom: 6,
                }}
              >
                Source image
              </label>
              {selectedImg ? (
                <div
                  style={{
                    width:        "100%",
                    aspectRatio:  "3/4",
                    borderRadius: "var(--radius-md)",
                    overflow:     "hidden",
                    background:   "var(--bg-panel)",
                    border:       "1px solid var(--border-base)",
                    maxHeight:    180,
                  }}
                >
                  <img
                    src={selectedImg.url}
                    alt={selectedImg.filename}
                    style={{ width: "100%", height: "100%", objectFit: "cover" }}
                  />
                </div>
              ) : (
                <div
                  style={{
                    padding:      "20px 12px",
                    textAlign:    "center",
                    color:        "var(--text-muted)",
                    fontSize:     12,
                    border:       "1px dashed var(--border-base)",
                    borderRadius: "var(--radius-md)",
                  }}
                >
                  Select an image to animate
                </div>
              )}
            </div>

            {/* Motion prompt */}
            <div className="field" style={{ margin: 0 }}>
              <label>Motion description</label>
              <textarea
                value={motionPrompt}
                onChange={(e) => setMotionPrompt(e.target.value)}
                rows={3}
                style={{ resize: "none", minHeight: "unset" }}
                placeholder="Describe how the person should move..."
              />
            </div>

            <button
              className="btn btn-primary w-full"
              onClick={animateImage}
              disabled={animating || !selectedImg}
            >
              <Play size={14} />
              {animating ? "Animating..." : "Animate this image"}
            </button>

            {vidStatus && !vidError && (
              <span className="text-xs text-muted">{vidStatus}</span>
            )}
            {vidError && <div className="alert alert-error" style={{ padding: "8px 12px" }}>{vidError}</div>}

            {/* Videos list */}
            {videos.length > 0 && (
              <div>
                <h4 style={{ marginBottom: 8, fontSize: 13 }}>
                  {videos.length} video{videos.length !== 1 ? "s" : ""}
                </h4>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {videos.map((v) => (
                    <div
                      key={v.path}
                      style={{
                        display:       "flex",
                        alignItems:    "center",
                        justifyContent: "space-between",
                        padding:       "8px 12px",
                        background:    "var(--bg-panel)",
                        borderRadius:  "var(--radius-md)",
                        border:        "1px solid var(--border-subtle)",
                      }}
                    >
                      <span
                        style={{
                          fontSize:      12,
                          color:         "var(--text-secondary)",
                          overflow:      "hidden",
                          textOverflow:  "ellipsis",
                          whiteSpace:    "nowrap",
                          flex:          1,
                          marginRight:   8,
                        }}
                      >
                        {v.filename}
                      </span>
                      <button
                        className="btn btn-ghost btn-sm"
                        onClick={() => openFolder(v.path.split(/[\\/]/).slice(0, -1).join("/"))}
                        style={{ flexShrink: 0 }}
                      >
                        Open
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="step-footer">
        <div className="step-footer-left">
          <span className="text-sm text-muted">Step 5 of 5 — Content Studio</span>
        </div>
        <div className="step-footer-right">
          <button
            className="btn btn-ghost"
            onClick={() => openFolder(settings?.output_dir ?? "")}
          >
            <FolderOpen size={13} />
            Open output folder
          </button>
        </div>
      </div>
    </div>
  );
}
