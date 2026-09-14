import { useEffect, useRef, useState } from "react";

export type AsyncState<T> =
  | { status: "idle" }
  | { status: "loading"; previous: T | null }
  | { status: "error"; error: string }
  | { status: "ok"; data: T };

/**
 * Loads data whenever `key` changes; `null` means "not ready to ask yet".
 * The previous result stays available while a new request is in flight so
 * charts do not flash empty while a slider moves.
 */
export function useApi<T>(key: string | null, load: (signal: AbortSignal) => Promise<T>): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ status: "idle" });
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    if (key === null) {
      setState({ status: "idle" });
      return;
    }
    const controller = new AbortController();
    setState((prev) => ({
      status: "loading",
      previous: prev.status === "ok" ? prev.data : prev.status === "loading" ? prev.previous : null,
    }));
    loadRef
      .current(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setState({ status: "ok", data });
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        const message = err instanceof Error ? err.message : String(err);
        setState({ status: "error", error: message });
      });
    return () => controller.abort();
  }, [key]);

  return state;
}

/** The current data, or the previous data while reloading. */
export function dataOf<T>(state: AsyncState<T>): T | null {
  if (state.status === "ok") return state.data;
  if (state.status === "loading") return state.previous;
  return null;
}

/** True while a new request is in flight and the previous result is still shown. */
export function isReloading<T>(state: AsyncState<T>): boolean {
  return state.status === "loading" && state.previous !== null;
}

/** True while the viewport is narrower than Tailwind's `sm` breakpoint. */
export function useNarrow(): boolean {
  const query = "(max-width: 639px)";
  const [narrow, setNarrow] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const media = window.matchMedia(query);
    const onChange = () => setNarrow(media.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);
  return narrow;
}

export function useDebounced<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}
