import { useState, useRef } from "react";
import { Upload, X } from "lucide-react";
import { useStore } from "../store";
import * as api from "../api";

interface Props {
  onCreated: () => void;
}

export function Step1Setup({ onCreated }: Props) {
  const { loadProjects, selectProject } = useStore();

  const [name,        setName]        = useState("");
  const [trigger,     setTrigger]     = useState("");
  const [gender,      setGender]      = useState<"female" | "male" | "neutral">("female");
  const [refs,        setRefs]        = useState<File[]>([]);
  const [previews,    setPreviews]    = useState<string[]>([]);
  const [creating,    setCreating]    = useState(false);
  const [error,       setError]       = useState("");

  const fileRef = useRef<HTMLInputElement>(null);

  const addFiles = (files: FileList | null) => {
    if (!files) return;
    const next = [...refs, ...Array.from(files)].slice(0, 3);
    setRefs(next);
    setPreviews(next.map((f) => URL.createObjectURL(f)));
  };

  const removeRef = (i: number) => {
    URL.revokeObjectURL(previews[i]);
    const nextFiles    = refs.filter((_, idx) => idx !== i);
    const nextPreviews = previews.filter((_, idx) => idx !== i);
    setRefs(nextFiles);
    setPreviews(nextPreviews);
  };

  const validate = (): string | null => {
    if (!name.trim())                 return "Please enter a name.";
    if (trigger.trim().length < 3)    return "Trigger word must be at least 3 characters.";
    if (trigger.includes(" "))        return "Trigger word must not contain spaces.";
    if (refs.length === 0)            return "Upload at least one reference photo.";
    return null;
  };

  const submit = async () => {
    const err = validate();
    if (err) { setError(err); return; }
    setError("");
    setCreating(true);

    try {
      const project = await api.createProject({
        name:         name.trim(),
        trigger_word: trigger.trim(),
        gender,
      });
      await api.uploadReferences(project.slug, refs);
      await loadProjects();
      selectProject(project.slug);
      onCreated();
    } catch (e) {
      setError(String(e));
    } finally {
      setCreating(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div className="step-header">
        <h1>Create your AI influencer</h1>
        <p>Fill in the details below and upload at least one clear, front-facing photo.</p>
      </div>

      <div className="step-container">
        <div className="step-body">

          {/* Identity */}
          <div className="card mb-20">
            <div className="card-title">Identity</div>
            <div className="card-desc mb-16">
              Choose a name and a unique trigger word the AI will associate with this person.
            </div>

            <div className="two-col">
              <div className="field">
                <label>Display name</label>
                <input
                  type="text"
                  placeholder="e.g. Sophia"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>
              <div className="field">
                <label>Trigger word</label>
                <input
                  type="text"
                  placeholder="e.g. s0ph1a"
                  value={trigger}
                  onChange={(e) => setTrigger(e.target.value.replace(/\s/g, ""))}
                />
                <span className="field-hint">
                  A unique nonsense word — avoids clashing with the model's existing knowledge.
                </span>
              </div>
            </div>

            <div className="field">
              <label>Gender hint (improves prompt library selection)</label>
              <div className="flex gap-12" style={{ marginTop: 4 }}>
                {(["female", "male", "neutral"] as const).map((g) => (
                  <label
                    key={g}
                    style={{
                      display:     "flex",
                      alignItems:  "center",
                      gap:         8,
                      cursor:      "pointer",
                      fontSize:    13,
                      color:       gender === g ? "var(--text-primary)" : "var(--text-secondary)",
                    }}
                  >
                    <input
                      type="radio"
                      name="gender"
                      value={g}
                      checked={gender === g}
                      onChange={() => setGender(g)}
                      style={{ accentColor: "var(--accent)" }}
                    />
                    {g.charAt(0).toUpperCase() + g.slice(1)}
                  </label>
                ))}
              </div>
            </div>
          </div>

          {/* Reference photos */}
          <div className="card mb-20">
            <div className="card-title">Reference photos</div>
            <div className="card-desc mb-16">
              Upload 1 to 3 clear photos. Good lighting, face clearly visible, no sunglasses.
              These are used to generate the training dataset.
            </div>

            <div
              className="drop-zone"
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add("drag-over"); }}
              onDragLeave={(e) => e.currentTarget.classList.remove("drag-over")}
              onDrop={(e) => {
                e.preventDefault();
                e.currentTarget.classList.remove("drag-over");
                addFiles(e.dataTransfer.files);
              }}
            >
              <Upload size={24} style={{ color: "var(--text-muted)", marginBottom: 8 }} />
              <h4>Drop photos here or click to browse</h4>
              <p style={{ margin: 0 }}>JPG, PNG or WEBP — up to 3 photos</p>
              <input
                ref={fileRef}
                type="file"
                accept=".jpg,.jpeg,.png,.webp"
                multiple
                style={{ display: "none" }}
                onChange={(e) => addFiles(e.target.files)}
              />
            </div>

            {previews.length > 0 && (
              <div className="flex gap-8 mt-12">
                {previews.map((src, i) => (
                  <div
                    key={i}
                    style={{
                      position:       "relative",
                      width:          80,
                      height:         80,
                      borderRadius:   "var(--radius-md)",
                      overflow:       "hidden",
                      border:         "1px solid var(--border-base)",
                    }}
                  >
                    <img
                      src={src}
                      alt={`Reference ${i + 1}`}
                      style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    />
                    <button
                      onClick={() => removeRef(i)}
                      style={{
                        position:    "absolute",
                        top:         3,
                        right:       3,
                        width:       20,
                        height:      20,
                        borderRadius: "50%",
                        background:  "rgba(0,0,0,0.75)",
                        border:      "none",
                        color:       "#fff",
                        cursor:      "pointer",
                        display:     "flex",
                        alignItems:  "center",
                        justifyContent: "center",
                      }}
                    >
                      <X size={10} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {error && <div className="alert alert-error mb-16">{error}</div>}

        </div>
      </div>

      <div className="step-footer">
        <div className="step-footer-left">
          <span className="text-sm" style={{ color: "var(--text-muted)" }}>Step 1 of 5</span>
        </div>
        <div className="step-footer-right">
          <button
            className="btn btn-primary"
            onClick={submit}
            disabled={creating}
          >
            {creating ? "Creating..." : "Continue to Dataset Generation"}
          </button>
        </div>
      </div>
    </div>
  );
}
