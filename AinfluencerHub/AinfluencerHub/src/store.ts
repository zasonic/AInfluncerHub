/**
 * store.ts — Zustand global state.
 *
 * Keeps UI state in sync across components without prop-drilling.
 * Does NOT duplicate server state — it caches the last-known values
 * and provides actions that call the API then update local state.
 */

import { create } from "zustand";
import type { AppSettings, Project } from "./types";
import * as api from "./api";

interface AppState {
  // Projects
  projects:       Project[];
  activeProject:  Project | null;
  activeStep:     number;   // 1–5, 0 = welcome

  // Loading
  loading:        boolean;
  error:          string;

  // Settings (cached)
  settings:       AppSettings | null;

  // GPU lock state (read from backend)
  gpuBusy:        boolean;
  gpuHolder:      string;

  // Actions
  setProjects:     (p: Project[]) => void;
  setActiveProject:(p: Project | null) => void;
  setActiveStep:   (n: number) => void;
  setLoading:      (b: boolean) => void;
  setError:        (msg: string) => void;
  setSettings:     (s: AppSettings) => void;
  setGpuBusy:      (busy: boolean, holder?: string) => void;

  // Async actions
  loadProjects:    () => Promise<void>;
  selectProject:   (slug: string) => void;
  refreshProject:  (slug: string) => Promise<void>;
  loadSettings:    () => Promise<void>;
}

export const useStore = create<AppState>((set, get) => ({
  projects:       [],
  activeProject:  null,
  activeStep:     0,
  loading:        false,
  error:          "",
  settings:       null,
  gpuBusy:        false,
  gpuHolder:      "",

  setProjects:     (projects) => set({ projects }),
  setActiveProject:(activeProject) => set({ activeProject }),
  setActiveStep:   (activeStep) => set({ activeStep }),
  setLoading:      (loading) => set({ loading }),
  setError:        (error) => set({ error }),
  setSettings:     (settings) => set({ settings }),
  setGpuBusy:      (gpuBusy, gpuHolder = "") => set({ gpuBusy, gpuHolder }),

  loadProjects: async () => {
    set({ loading: true, error: "" });
    try {
      const projects = await api.listProjects();
      set({ projects, loading: false });

      // If no active project, show welcome
      if (!get().activeProject && projects.length === 0) {
        set({ activeStep: 0 });
      }
    } catch (err) {
      set({ loading: false, error: String(err) });
    }
  },

  selectProject: (slug: string) => {
    const { projects } = get();
    const proj = projects.find((p) => p.slug === slug);
    if (!proj) return;

    // Jump to first incomplete step
    const done = new Set(proj.steps_done);
    let step = 5;
    for (let n = 1; n <= 5; n++) {
      if (!done.has(n)) { step = n; break; }
    }
    set({ activeProject: proj, activeStep: step });
  },

  refreshProject: async (slug: string) => {
    try {
      const updated = await api.getProject(slug);
      set((state) => ({
        projects: state.projects.map((p) =>
          p.slug === slug ? updated : p
        ),
        activeProject:
          state.activeProject?.slug === slug ? updated : state.activeProject,
      }));
    } catch { /* ignore */ }
  },

  loadSettings: async () => {
    try {
      const settings = await api.getSettings();
      set({ settings });
    } catch { /* ignore */ }
  },
}));
