import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ApiError } from "./api";

export const DEFAULT_SIZE = "large_v2";

/** The selected size lives in the URL (?size=), so a page can be shared. */
export function useSize(): [string, (size: string) => void] {
  const [params, setParams] = useSearchParams();
  const set = useCallback((size: string) => setParams((p) => {
    const next = new URLSearchParams(p);
    next.set("size", size);
    return next;
  }), [setParams]);
  return [params.get("size") ?? DEFAULT_SIZE, set];
}

export interface Loaded<T> { data?: T; error?: ApiError; loading: boolean; retry: () => void }

/** Calls fn(...args) whenever args change; retry() calls it again. `fn` must be a stable (module-level) function. */
export function useApi<A extends unknown[], T>(fn: (...args: A) => Promise<T>, ...args: A): Loaded<T> {
  const [attempt, setAttempt] = useState(0);
  const key = JSON.stringify([args, attempt]);
  const [state, setState] = useState<{ key?: string; data?: T; error?: ApiError }>({});
  useEffect(() => {
    let live = true;
    const [a] = JSON.parse(key) as [A];
    fn(...a).then(
      (data) => live && setState({ key, data }),
      (e: unknown) => live && setState({
        key, error: e instanceof ApiError ? e : new ApiError(0, "client", String(e)),
      }),
    );
    return () => { live = false; };
  }, [fn, key]);
  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  const current = state.key === key;
  return { data: current ? state.data : undefined, error: current ? state.error : undefined, loading: !current, retry };
}

/** Two loads as one: data when both arrived, the first error, retry both. */
export function combine<A, B>(a: Loaded<A>, b: Loaded<B>): Loaded<[A, B]> {
  return {
    data: a.data !== undefined && b.data !== undefined ? [a.data, b.data] : undefined,
    error: a.error ?? b.error, loading: a.loading || b.loading,
    retry: () => { if (a.error) a.retry(); if (b.error) b.retry(); },
  };
}
