import { useState, useEffect } from "react";
import { FolderOpen, Play } from "lucide-react";
import { useStore } from "../store";
import * as api from "../api";
import { ImageGallery } from "../components/ImageGallery";
import { useSSE } from "../hooks/useSSE";
import type { GeneratedImage, GeneratedVideo } from "../types";

interface Props {
  onAdvance: () => void;
}

export function Step5Studio(_props: Props) {
  const { activeProject, settings } = useStore();

  const [genImages,    setGenImages]    = useState<GeneratedImage[]>([]);
  const [videos,       setVideos]       = useState<GeneratedVideo[]>([]);
  const [prompt,       setPrompt]       = useState("");
  const [strength,     setStrength]     = useState(0.85);
  const [genSeed,      setGenSeed]      = useState(-1);
  const [motionPrompt, setMotionPrompt] = useState("The person smiles gently and turns their head slightly.");
  const [selectedImg,  setSelectedImg]  = useState<GeneratedImage | null>(null);
  const [generating,   setGenerating]   = useState(false);
  const [animating,    setAnimating]    = useState(false);
  const [genStatus,    setGenStatus]    = useState("");
  const [vidStatus,    setVidStatus]    = useState("");
  const [genError,     setGenError]     = useState("");
  const [vidError,     setVidError]     = useState("");

  const slug = activeProject?.slug ?? "";

  const genSSE = useSSE({
    onEvent: (event) => {
      if (event.type === "progress" || event.type === "log") {
        setGenStatus(event.type === "progress" ? event.message : event.line);
      } else if (event.type === "done") {
        setGenerating(false);
        setGenStatus("Image generated.");
        api.getGeneratedImages(slug).then(({ images }) => setGenImages(images)).catch(() => {});
      } else if (event.type === "error") {
        setGenError(event.message);
        setGenerating(false);
      }
    },
    onTerminate: () => setGenerating(false),
  });

  const vidSSE = useSSE({
    onEvent: (event) => {
      if (event.type === "progress" || event.type === "log") {
        setVidStatus(event.type === "progress" ? event.message : event.line);
      } else if (event.type === "done") {
        setAnimating(false);
        setVidStatus("Video created.");
        api.getVideos(slug).then(({ videos: vids }) => setVideos(vids)).catch(() => {});
      } else if (event.type === "error") {
        setVidError(event.message);
        setAnimating(false);
      }
    },
    onTerminate: () => setAnimating(false),
  });

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
  }, [slug]);

  const generateImage = () => {
    if (!slug) return;
    setGenError("");
    setGenerating(true);
    setGenStatus("Starting generation...");
    genSSE.start(api.generateImage(slug, prompt, strength, genSeed));
  };

  const animateImage = () => {
    if (!selectedImg || !slug) return;
    setVidError("");
    setAnimating(true);
    setVidStatus("Starting video generation...");
    vidSSE.start(api.animateImage(slug, selectedImg.path, motionPrompt));
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
                gridTemplateColumns: "1fr auto auto",
                gap:         10,
                alignItems:  "end",
              }}
            >
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

              <div className="field" style={{ margin: 0 }}>
                <label style={{ whiteSpace: "nowrap" }}>Seed</label>
                <input
                  type="number"
                  value={genSeed}
                  onChange={(e) => setGenSeed(Number(e.target.value))}
                  title="-1 = random seed"
                  style={{ width: 90, textAlign: "right" }}
                />
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
              Select an image on the left, describe the motion, then animate it.
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

            {/* Videos list with inline player */}
            {videos.length > 0 && (
              <div>
                <h4 style={{ marginBottom: 8, fontSize: 13 }}>
                  {videos.length} video{videos.length !== 1 ? "s" : ""}
                </h4>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  {videos.map((v) => (
                    <div
                      key={v.path}
                      style={{
                        background:   "var(--bg-panel)",
                        borderRadius: "var(--radius-md)",
                        border:       "1px solid var(--border-subtle)",
                        overflow:     "hidden",
                      }}
                    >
                      <video
                        src={api.videoUrl(v.path)}
                        controls
                        style={{ width: "100%", display: "block", maxHeight: 240 }}
                      />
                      <div
                        style={{
                          display:        "flex",
                          alignItems:     "center",
                          justifyContent: "space-between",
                          padding:        "6px 10px",
                        }}
                      >
                        <span
                          style={{
                            fontSize:     11,
                            color:        "var(--text-muted)",
                            overflow:     "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace:   "nowrap",
                            flex:         1,
                            marginRight:  8,
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
