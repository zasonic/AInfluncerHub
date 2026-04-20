import { useCallback, useEffect, useRef } from "react";
import type { StreamEvent } from "../types";

interface Options {
  onEvent:     (event: StreamEvent) => void;
  onTerminate?: () => void;
}

interface Controller {
  start: (source: EventSource) => void;
  stop:  () => void;
}

/**
 * Shared lifecycle for SSE-driven operations.
 *
 * Owns a single EventSource ref so the component doesn't have to. Parses
 * each payload into a StreamEvent and routes it to the caller. Tears down
 * automatically on unmount, on a terminal `done` / `error` event, or on
 * transport error. Consumers call `start(source)` to attach an EventSource
 * produced by the api module and `stop()` for user-initiated cancel.
 */
export function useSSE(options: Options): Controller {
  const { onEvent, onTerminate } = options;
  const sourceRef = useRef<EventSource | null>(null);

  const onEventRef     = useRef(onEvent);
  const onTerminateRef = useRef(onTerminate);
  useEffect(() => { onEventRef.current     = onEvent; },     [onEvent]);
  useEffect(() => { onTerminateRef.current = onTerminate; }, [onTerminate]);

  const close = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
  }, []);

  const terminate = useCallback(() => {
    close();
    onTerminateRef.current?.();
  }, [close]);

  const start = useCallback((source: EventSource) => {
    close();
    sourceRef.current = source;
    source.onmessage = (ev) => {
      let event: StreamEvent;
      try {
        event = JSON.parse(ev.data) as StreamEvent;
      } catch {
        return;
      }
      onEventRef.current(event);
      if (event.type === "done" || event.type === "error") {
        terminate();
      }
    };
    source.onerror = () => {
      terminate();
    };
  }, [close, terminate]);

  // Clean up on unmount.
  useEffect(() => close, [close]);

  return { start, stop: terminate };
}
