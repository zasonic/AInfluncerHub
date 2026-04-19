import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, renderHook } from "@testing-library/react";

import { useSSE } from "../hooks/useSSE";
import type { StreamEvent } from "../types";

class MockEventSource {
  static instances: MockEventSource[] = [];

  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror:   ((ev: Event) => void) | null        = null;
  closed = false;

  constructor() {
    MockEventSource.instances.push(this);
  }

  emit(event: StreamEvent) {
    this.onmessage?.({ data: JSON.stringify(event) } as MessageEvent);
  }

  emitRaw(data: string) {
    this.onmessage?.({ data } as MessageEvent);
  }

  errorOut() {
    this.onerror?.(new Event("error"));
  }

  close() {
    this.closed = true;
  }
}

describe("useSSE", () => {
  beforeEach(() => {
    MockEventSource.instances = [];
  });

  it("routes parsed events to onEvent", () => {
    const onEvent = vi.fn();
    const { result } = renderHook(() => useSSE({ onEvent }));
    const source = new MockEventSource();

    act(() => result.current.start(source as unknown as EventSource));
    act(() => source.emit({ type: "progress", done: 1, total: 5, message: "hi" }));

    expect(onEvent).toHaveBeenCalledWith({ type: "progress", done: 1, total: 5, message: "hi" });
  });

  it("terminates on done and closes the source", () => {
    const onTerminate = vi.fn();
    const { result } = renderHook(() =>
      useSSE({ onEvent: () => {}, onTerminate }),
    );
    const source = new MockEventSource();

    act(() => result.current.start(source as unknown as EventSource));
    act(() => source.emit({ type: "done", message: "ok" }));

    expect(source.closed).toBe(true);
    expect(onTerminate).toHaveBeenCalledTimes(1);
  });

  it("terminates on transport error", () => {
    const onTerminate = vi.fn();
    const { result } = renderHook(() =>
      useSSE({ onEvent: () => {}, onTerminate }),
    );
    const source = new MockEventSource();

    act(() => result.current.start(source as unknown as EventSource));
    act(() => source.errorOut());

    expect(source.closed).toBe(true);
    expect(onTerminate).toHaveBeenCalledTimes(1);
  });

  it("ignores malformed payloads", () => {
    const onEvent = vi.fn();
    const { result } = renderHook(() => useSSE({ onEvent }));
    const source = new MockEventSource();

    act(() => result.current.start(source as unknown as EventSource));
    act(() => source.emitRaw("{not json"));

    expect(onEvent).not.toHaveBeenCalled();
    expect(source.closed).toBe(false);
  });

  it("closes the previous source when a new start happens", () => {
    const { result } = renderHook(() => useSSE({ onEvent: () => {} }));
    const first  = new MockEventSource();
    const second = new MockEventSource();

    act(() => result.current.start(first  as unknown as EventSource));
    act(() => result.current.start(second as unknown as EventSource));

    expect(first.closed).toBe(true);
    expect(second.closed).toBe(false);
  });

  it("closes the source on unmount", () => {
    const { result, unmount } = renderHook(() => useSSE({ onEvent: () => {} }));
    const source = new MockEventSource();

    act(() => result.current.start(source as unknown as EventSource));
    unmount();

    expect(source.closed).toBe(true);
  });
});
