import { useCallback, useState } from "react";

export type StatusKind = "ok" | "warning" | "error";

export interface AsyncOpState {
  running:  boolean;
  progress: { done: number; total: number; message: string };
  status:   string;
  statusKind: StatusKind;
  error:    string;
}

const INITIAL: AsyncOpState = {
  running:    false,
  progress:   { done: 0, total: 0, message: "" },
  status:     "",
  statusKind: "ok",
  error:      "",
};

export interface AsyncOpControls {
  state:       AsyncOpState;
  start:       (initialMessage?: string, total?: number) => void;
  setProgress: (done: number, total: number, message: string) => void;
  setStatus:   (message: string, kind?: StatusKind) => void;
  fail:        (message: string) => void;
  succeed:     (message?: string) => void;
  reset:       () => void;
}

/**
 * Consolidated state-container for the running / progress / status / error
 * pattern duplicated across every step component. A single hook means one
 * place to add things like toast notifications or optimistic UI later.
 */
export function useAsyncOperation(): AsyncOpControls {
  const [state, setState] = useState<AsyncOpState>(INITIAL);

  const start = useCallback((initialMessage = "", total = 0) => {
    setState({
      ...INITIAL,
      running:  true,
      progress: { done: 0, total, message: initialMessage },
    });
  }, []);

  const setProgress = useCallback((done: number, total: number, message: string) => {
    setState((s) => ({ ...s, progress: { done, total, message } }));
  }, []);

  const setStatus = useCallback((message: string, kind: StatusKind = "ok") => {
    setState((s) => ({ ...s, status: message, statusKind: kind, error: "" }));
  }, []);

  const fail = useCallback((message: string) => {
    setState((s) => ({ ...s, running: false, error: message, status: "", statusKind: "error" }));
  }, []);

  const succeed = useCallback((message = "") => {
    setState((s) => ({
      ...s,
      running:    false,
      error:      "",
      status:     message,
      statusKind: "ok",
      progress:   { ...s.progress, done: s.progress.total || s.progress.done },
    }));
  }, []);

  const reset = useCallback(() => setState(INITIAL), []);

  return { state, start, setProgress, setStatus, fail, succeed, reset };
}
