import type { ReactNode } from "react";
import type { AsyncState } from "../hooks";
import { BUSY_OPACITY, C } from "../theme";

export function StatusMessage({ children, height }: { children: ReactNode; height?: number }) {
  return (
    <div
      className="flex items-center justify-center text-xs text-center px-4 leading-relaxed"
      style={{ color: C.muted, minHeight: height }}
    >
      {children}
    </div>
  );
}

interface LoadableProps<T> {
  state: AsyncState<T>;
  height: number;
  children: (data: T) => ReactNode;
}

/** Per-panel loading and error handling; keeps stale data visible while reloading. */
export function Loadable<T>({ state, height, children }: LoadableProps<T>) {
  switch (state.status) {
    case "idle":
      return <StatusMessage height={height}>Waiting for the run to load…</StatusMessage>;
    case "error":
      return <StatusMessage height={height}>Could not load: {state.error}</StatusMessage>;
    case "loading":
      if (state.previous === null) {
        return <StatusMessage height={height}>Loading…</StatusMessage>;
      }
      return (
        <div style={{ opacity: BUSY_OPACITY, transition: "opacity 120ms" }} aria-busy="true">
          {children(state.previous)}
        </div>
      );
    case "ok":
      return <div>{children(state.data)}</div>;
  }
}
