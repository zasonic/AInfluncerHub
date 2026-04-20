import { describe, it, expect } from "vitest";
import { act, renderHook } from "@testing-library/react";

import { useAsyncOperation } from "../hooks/useAsyncOperation";

describe("useAsyncOperation", () => {
  it("initializes idle", () => {
    const { result } = renderHook(() => useAsyncOperation());
    expect(result.current.state.running).toBe(false);
    expect(result.current.state.error).toBe("");
  });

  it("start → progress → succeed transitions", () => {
    const { result } = renderHook(() => useAsyncOperation());

    act(() => result.current.start("Warming up", 10));
    expect(result.current.state.running).toBe(true);
    expect(result.current.state.progress).toEqual({ done: 0, total: 10, message: "Warming up" });

    act(() => result.current.setProgress(3, 10, "Working"));
    expect(result.current.state.progress.done).toBe(3);

    act(() => result.current.succeed("All done"));
    expect(result.current.state.running).toBe(false);
    expect(result.current.state.status).toBe("All done");
    expect(result.current.state.statusKind).toBe("ok");
  });

  it("fail clears running and sets error", () => {
    const { result } = renderHook(() => useAsyncOperation());
    act(() => result.current.start());
    act(() => result.current.fail("Boom"));
    expect(result.current.state.running).toBe(false);
    expect(result.current.state.error).toBe("Boom");
    expect(result.current.state.statusKind).toBe("error");
  });

  it("reset returns to initial state", () => {
    const { result } = renderHook(() => useAsyncOperation());
    act(() => result.current.start("Start"));
    act(() => result.current.fail("X"));
    act(() => result.current.reset());
    expect(result.current.state.running).toBe(false);
    expect(result.current.state.error).toBe("");
    expect(result.current.state.progress).toEqual({ done: 0, total: 0, message: "" });
  });
});
