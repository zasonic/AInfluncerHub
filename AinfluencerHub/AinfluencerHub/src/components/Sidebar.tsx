import { Settings, Plus, ChevronRight } from "lucide-react";
import { useStore } from "../store";

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
  } = useStore();

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
    </aside>
  );
}
