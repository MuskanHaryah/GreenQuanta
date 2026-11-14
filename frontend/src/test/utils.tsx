// Test helpers: a fetch mock that answers API paths from the real fixtures, and rendering inside a router.
import type { ReactElement } from "react";
import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { vi } from "vitest";
import Layout from "../components/Layout";
import results from "./fixtures/results_small_v2.json";
import compare from "./fixtures/compare_small_v2.json";
import placementQuantum from "./fixtures/placement_small_v2_quantum.json";
import placementPabfd from "./fixtures/placement_small_v2_pabfd.json";
import trace from "./fixtures/gas_trace_small_v2_seed0.json";
import claims from "./fixtures/claims_small_v2.json";
import sizes from "./fixtures/sizes.json";
import health from "./fixtures/health.json";

export const fixtures = { results, compare, placementQuantum, placementPabfd, trace, claims, sizes, health };

export const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

type Handler = (init?: RequestInit) => Response | Promise<Response>;

/** Mocks fetch: exact path (with query) → handler or body. Unknown paths answer 404 like the API. */
export function mockApi(extra: Record<string, unknown | Handler> = {}) {
  const routes: Record<string, unknown | Handler> = {
    "/api/health": health,
    "/api/sizes": sizes,
    "/api/results/small_v2": results,
    "/api/results/small_v2/compare": compare,
    "/api/results/small_v2/placement/quantum": placementQuantum,
    "/api/results/small_v2/placement/pabfd": placementPabfd,
    "/api/gas-trace/small_v2?seed=0": trace,
    "/api/claims?size=small_v2": claims,
    ...extra,
  };
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    const route = routes[path];
    if (route === undefined) return json({ error: { code: "not_found", message: `no fixture for ${path}` } }, 404);
    return typeof route === "function" ? (route as Handler)(init) : json(route);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

/** Renders a page inside the real Layout at `path` (default size small_v2). */
export function renderPage(page: ReactElement, path = "/?size=small_v2") {
  const route = path.split("?")[0];
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route element={<Layout />}><Route path={route} element={page} /></Route>
      </Routes>
    </MemoryRouter>,
  );
}
