import { useState } from "react";
import { Settings, Plus, ChevronRight, Trash2 } from "lucide-react";
import { useStore } from "../store";
import * as api from "../api";

const STEPS = [
  { n: 1, label: "Setup" },
  { n: 2, label: "Dataset" },
  { n: 3, label: "Captions" },
  { n: 4, label: "Training" },
  { n: 5, label: "Studio" },
];

interface SidebarProps {
  onNewProject:    () => void;
  onOpenSettings:  () => void;
}

export function Sidebar({ onNewProject, onOpenSettings }: SidebarProps) {
  const {
    projects,
    activeProject,
    activeStep,
    selectProject,
    setActiveStep,
    setActiveProject,
    loadProjects,
  } = useStore();

  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const handleDelete = async (slug: string) => {
    try {
      await api.deleteProject(slug);
      if (activeProject?.slug === slug) {
        setActiveProject(null);
      }
      await loadProjects();
    } catch { /* ignore */ }
    setConfirmDelete(null);
  };

  const stepState = (n: number) => {
    if (!activeProject) return "pending";
    if (activeProject.steps_done.includes(n)) return "done";
    if (activeStep === n) return "active";
    return "pending";
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          Ainfluencer<span>Hub</span>
        </div>
      </div>

      <div className="sidebar-section">Influencers</div>

      <div className="sidebar-list">
        {projects.length === 0 ? (
          <p style={{ padding: "8px 10px", fontSize: 12, color: "var(--text-muted)" }}>
            No influencers yet
          </p>
        ) : (
          projects.map((proj) => {
            const isActive = activeProject?.slug === proj.slug;
            return (
              <div
                key={proj.slug}
                className={`sidebar-item ${isActive ? "active" : ""}`}
                onClick={() => selectProject(proj.slug)}
              >
                {isActive && <span />}
                <span className="sidebar-item-name">{proj.name}</span>
                <span className="sidebar-item-badge">
                  {proj.steps_done.length}/5
                </span>
                <button
                  className="sidebar-delete-btn"
                  title="Delete this influencer"
                  onClick={(e) => {
                    e.stopPropagation();
                    setConfirmDelete(proj.slug);
                  }}
                  style={{
                    opacity: 0, background: "none", border: "none",
                    cursor: "pointer", padding: "2px 4px", marginLeft: 4,
                    color: "var(--text-muted)", transition: "opacity 0.15s",
                  }}
                  onMouseEnter={(e) => { (e.target as HTMLElement).style.opacity = "1"; }}
                  onMouseLeave={(e) => { (e.target as HTMLElement).style.opacity = "0"; }}
                >
                  <Trash2 size={12} />
                </button>
              </div>
            );
          })
        )}
      </div>

      {/* Step progress — shown only when a project is active */}
      {activeProject && (
        <>
          <div className="sidebar-section">Progress</div>
          <div className="step-progress">
            {STEPS.map(({ n, label }) => {
              const state = stepState(n);
              return (
                <div
                  key={n}
                  className="step-row"
                  onClick={() => setActiveStep(n)}
                  title={`Go to step ${n}: ${label}`}
                >
                  <div className={`step-circle ${state}`}>
                    {state === "done" ? "✓" : n}
                  </div>
                  <span className={`step-label ${state}`}>{label}</span>
                </div>
              );
            })}
          </div>
        </>
      )}

      <div style={{ flex: 1 }} />

      <div className="sidebar-footer">
        <button
          className="sidebar-item w-full"
          onClick={onNewProject}
          style={{ borderRadius: "var(--radius-md)", marginBottom: 4 }}
        >
          <Plus size={14} />
          <span className="sidebar-item-name">New influencer</span>
        </button>

        <button
          className="sidebar-item w-full"
          onClick={onOpenSettings}
          style={{ borderRadius: "var(--radius-md)" }}
        >
          <Settings size={14} />
          <span className="sidebar-item-name">Settings</span>
          <ChevronRight size={12} style={{ color: "var(--text-muted)" }} />
        </button>
      </div>
      {confirmDelete && (
        <div
          style={{
            position:       "fixed",
            inset:          0,
            background:     "rgba(0,0,0,0.6)",
            display:        "flex",
            alignItems:     "center",
            justifyContent: "center",
            zIndex:         1000,
          }}
          onClick={() => setConfirmDelete(null)}
        >
          <div
            style={{
              background:     "var(--bg-surface)",
              borderRadius:   "var(--radius-lg)",
              padding:        "24px",
              maxWidth:       360,
              width:          "90%",
              border:         "1px solid var(--border-base)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ marginBottom: 8 }}>Delete influencer?</h3>
            <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 20 }}>
              This will permanently delete all dataset images, captions, trained models, and generated content for this influencer. This cannot be undone.
            </p>
            <div className="flex gap-8" style={{ justifyContent: "flex-end" }}>
              <button className="btn btn-ghost" onClick={() => setConfirmDelete(null)}>
                Cancel
              </button>
              <button
                className="btn"
                style={{ background: "var(--error)", color: "#fff" }}
                onClick={() => handleDelete(confirmDelete)}
              >
                Delete permanently
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
