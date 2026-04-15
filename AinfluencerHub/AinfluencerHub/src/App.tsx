import { useEffect, useState } from "react";
import { Sidebar }        from "./components/Sidebar";
import { SettingsModal }  from "./components/SettingsModal";
import { Welcome }        from "./steps/Welcome";
import { Step1Setup }     from "./steps/Step1Setup";
import { Step2Dataset }   from "./steps/Step2Dataset";
import { Step3Captions }  from "./steps/Step3Captions";
import { Step4Training }  from "./steps/Step4Training";
import { Step5Studio }    from "./steps/Step5Studio";
import { useStore }       from "./store";
import * as api           from "./api";

export function App() {
  const {
    activeProject,
    activeStep,
    projects,
    loadProjects,
    loadSettings,
    setActiveStep,
    setActiveProject,
    setError,
  } = useStore();

  const [showSettings,   setShowSettings]   = useState(false);
  const [backendReady,   setBackendReady]   = useState(false);
  const [backendError,   setBackendError]   = useState("");

  // ── Bootstrap ────────────────────────────────────────────────────────────

  useEffect(() => {
    let attempts = 0;
    const max    = 60;   // 6 seconds total

    const poll = async () => {
      try {
        await api.health();
        setBackendReady(true);
        await loadSettings();
        await loadProjects();
      } catch {
        attempts++;
        if (attempts < max) {
          setTimeout(poll, 100);
        } else {
          setBackendError(
            "Could not connect to the Python backend. " +
            "Make sure Python is installed and try restarting the application."
          );
        }
      }
    };

    api.initBackendUrl().then(poll);
  }, []);

  // ── Step navigation ───────────────────────────────────────────────────────

  const advance = () => {
    const next = Math.min((activeStep ?? 1) + 1, 5);
    setActiveStep(next);
  };

  const handleNewProject = () => {
    setActiveProject(null);
    setActiveStep(1);
  };

  // ── Render guard ──────────────────────────────────────────────────────────

  if (backendError) {
    return (
      <div
        style={{
          height:          "100vh",
          display:         "flex",
          alignItems:      "center",
          justifyContent:  "center",
          padding:         40,
          flexDirection:   "column",
          gap:             16,
        }}
      >
        <h2 style={{ color: "var(--error)" }}>Startup error</h2>
        <p style={{ textAlign: "center", maxWidth: 420, color: "var(--text-secondary)" }}>
          {backendError}
        </p>
        <button
          className="btn btn-primary"
          onClick={() => window.location.reload()}
        >
          Retry
        </button>
      </div>
    );
  }

  if (!backendReady) {
    return (
      <div
        style={{
          height:         "100vh",
          display:        "flex",
          alignItems:     "center",
          justifyContent: "center",
          flexDirection:  "column",
          gap:            16,
          color:          "var(--text-secondary)",
        }}
      >
        <div
          style={{
            width:        32,
            height:       32,
            borderRadius: "50%",
            border:       "3px solid var(--border-base)",
            borderTopColor: "var(--accent)",
            animation:    "spin 0.8s linear infinite",
          }}
        />
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        <span style={{ fontSize: 13 }}>Starting up...</span>
      </div>
    );
  }

  // ── Step routing ──────────────────────────────────────────────────────────

  const renderStep = () => {
    if (activeStep === 0 || (!activeProject && activeStep !== 1)) {
      return <Welcome onNewProject={handleNewProject} />;
    }
    switch (activeStep) {
      case 1: return <Step1Setup    onCreated={advance} />;
      case 2: return <Step2Dataset  onAdvance={advance} />;
      case 3: return <Step3Captions onAdvance={advance} />;
      case 4: return <Step4Training onAdvance={advance} />;
      case 5: return <Step5Studio   onAdvance={advance} />;
      default: return <Welcome onNewProject={handleNewProject} />;
    }
  };

  return (
    <>
      <div className="app-layout">
        <Sidebar
          onNewProject={handleNewProject}
          onOpenSettings={() => setShowSettings(true)}
        />
        <div className="content-area">{renderStep()}</div>
      </div>

      {showSettings && (
        <SettingsModal onClose={() => setShowSettings(false)} />
      )}
    </>
  );
}
